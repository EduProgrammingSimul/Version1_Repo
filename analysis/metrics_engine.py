"""Metric computation suite with strict NaN avoidance."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np

from analysis.frequency import ObservabilityError, summarize_frequency


@dataclass
class MetricResult:
    name: str
    value: Optional[float]
    unit: str
    higher_is_better: bool
    missing_reason: Optional[str] = None


class _PreflightError(RuntimeError):
    pass


class MetricsEngine:
    """Compatibility wrapper offering class-based API."""

    def __init__(self, registry: Dict[str, Iterable[Dict[str, object]]], strict: bool = False):
        self.registry = registry
        self.strict = strict

    def compute(self, timeseries: Dict[str, np.ndarray]) -> List[MetricResult]:
        return compute_all(timeseries, self.registry, self.strict)


def _ensure_timeseries(timeseries: Dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
    if "time" not in timeseries:
        raise _PreflightError("Missing 'time' array")
    time = np.asarray(timeseries["time"], dtype=float)
    if time.ndim != 1 or time.size < 4:
        raise _PreflightError("Insufficient time samples")
    diffs = np.diff(time)
    if not np.all(diffs > 0):
        raise _PreflightError("Time must be strictly increasing")
    dt = float(np.median(diffs))
    if not np.allclose(diffs, dt, rtol=1e-3, atol=1e-9):
        raise _PreflightError("Time-step must be uniform")
    length = time.size
    for key, arr in timeseries.items():
        arr_np = np.asarray(arr)
        if arr_np.shape != (length,):
            raise _PreflightError(f"Timeseries '{key}' has ragged shape {arr_np.shape}")
    return time, dt


def _safe_mean(arr: np.ndarray) -> float:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("no finite samples")
    return float(np.mean(finite))


def _safe_integral(values: np.ndarray, dt: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("no finite samples")
    return float(np.sum(finite) * dt)


def _safe_max_abs(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("no finite samples")
    return float(np.max(np.abs(finite)))


def _mask_ratio(mask: np.ndarray) -> float:
    if mask.size == 0:
        raise ValueError("empty mask")
    return float(np.mean(mask.astype(float)) * 100.0)


def _settling_time(time: np.ndarray, signal: np.ndarray, tol: float = 0.02) -> float:
    finite = np.abs(signal[np.isfinite(signal)])
    if finite.size == 0:
        raise ValueError("no finite samples")
    within = np.abs(signal) <= tol
    if not np.any(within):
        raise ValueError("never settled")
    if time.size < 2:
        return 0.0
    dt = float(np.median(np.diff(time)))
    window = max(5, int(round(1.0 / max(dt, 1e-9))))
    window = min(window, within.size)
    kernel = np.ones(window, dtype=int)
    hits = np.convolve(within.astype(int), kernel, mode="valid")
    valid_indices = np.where(hits == window)[0]
    if valid_indices.size == 0:
        raise ValueError("never settled")
    idx = int(valid_indices[0])
    return float(time[idx] - time[0])


def _overshoot(signal: np.ndarray) -> float:
    finite = signal[np.isfinite(signal)]
    if finite.size == 0:
        raise ValueError("no finite samples")
    return float(np.max(finite))


def _steady_state_error(signal: np.ndarray) -> float:
    tail = signal[-max(5, signal.size // 10) :]
    finite = tail[np.isfinite(tail)]
    if finite.size == 0:
        raise ValueError("no finite samples")
    return float(np.mean(finite))


def _jerk(control: np.ndarray, dt: float) -> float:
    if control.size < 3:
        raise ValueError("not enough samples")
    diffs = np.diff(control)
    finite = diffs[np.isfinite(diffs)]
    if finite.size == 0:
        raise ValueError("no finite diffs")
    return float(np.mean(np.abs(finite)) / max(dt, 1e-9))


def _valve_travel(control: np.ndarray) -> float:
    diffs = np.diff(control)
    finite = diffs[np.isfinite(diffs)]
    if finite.size == 0:
        raise ValueError("no finite diffs")
    return float(np.sum(np.abs(finite)))


def _make_result(
    name: str,
    unit: str,
    higher_is_better: bool,
    value: Optional[float],
    reason: Optional[str],
) -> MetricResult:
    if value is not None and not np.isfinite(value):
        value = None
        reason = reason or "UNOBSERVED:non-finite"
    if value is None and reason is None:
        reason = "UNOBSERVED:missing"
    return MetricResult(name=name, value=value, unit=unit, higher_is_better=higher_is_better, missing_reason=reason)


def compute_all(
    timeseries: Dict[str, np.ndarray],
    registry: Dict[str, Iterable[Dict[str, object]]],
    strict: bool,
) -> List[MetricResult]:
    try:
        time, dt = _ensure_timeseries(timeseries)
    except _PreflightError as exc:
        if strict:
            raise
        results: List[MetricResult] = []
        for bucket in registry.values():
            for entry in bucket:
                results.append(
                    _make_result(
                        name=str(entry["name"]),
                        unit=str(entry.get("unit", "")),
                        higher_is_better=bool(entry.get("higher_is_better", False)),
                        value=None,
                        reason=f"UNOBSERVED:preflight:{exc}",
                    )
                )
        return results

    def get_array(name: str) -> Optional[np.ndarray]:
        if name not in timeseries:
            return None
        return np.asarray(timeseries[name], dtype=float)

    primary_results: List[MetricResult] = []
    secondary_results: List[MetricResult] = []
    freq_results: List[MetricResult] = []

    errors = get_array("error")
    unsafe_mask = get_array("unsafe_mask")
    rocof = get_array("rocof")
    sat_mask = get_array("saturation")
    control = get_array("control")
    shield_mask = get_array("shielded")

    for entry in registry.get("primary", []):
        name = str(entry["name"])
        unit = str(entry.get("unit", ""))
        hib = bool(entry.get("higher_is_better", False))
        reason = None
        value: Optional[float] = None
        try:
            if name == "abs_error_mean":
                if errors is None:
                    raise ValueError("missing error")
                value = _safe_mean(np.abs(errors))
            elif name == "itae_proxy":
                if errors is None:
                    raise ValueError("missing error")
                value = _safe_integral(np.abs(errors) * time, dt)
            elif name == "time_unsafe":
                if unsafe_mask is None:
                    raise ValueError("missing unsafe mask")
                value = float(np.sum(unsafe_mask.astype(float)) * dt)
            elif name == "max_rocof":
                if rocof is None:
                    raise ValueError("missing rocof")
                value = _safe_max_abs(rocof)
            elif name == "sat_incidence":
                if sat_mask is None:
                    raise ValueError("missing saturation mask")
                value = _mask_ratio(sat_mask)
            elif name == "effort":
                if control is None:
                    raise ValueError("missing control")
                value = _safe_integral(np.abs(control), dt)
            elif name == "jerk":
                if control is None:
                    raise ValueError("missing control")
                value = _jerk(control, dt)
            else:
                reason = "UNOBSERVED:unknown-metric"
        except ValueError as exc:
            reason = f"UNOBSERVED:{exc}"
        primary_results.append(_make_result(name, unit, hib, value, reason))

    for entry in registry.get("secondary", []):
        name = str(entry["name"])
        unit = str(entry.get("unit", ""))
        hib = bool(entry.get("higher_is_better", False))
        value = None
        reason = None
        try:
            if name == "overshoot":
                if errors is None:
                    raise ValueError("missing error")
                value = _overshoot(errors)
            elif name == "settling_time":
                if errors is None:
                    raise ValueError("missing error")
                value = _settling_time(time, np.abs(errors))
            elif name == "sse":
                if errors is None:
                    raise ValueError("missing error")
                value = abs(_steady_state_error(errors))
            elif name == "valve_travel":
                if control is None:
                    raise ValueError("missing control")
                value = _valve_travel(control)
            elif name == "shielded_pct":
                if shield_mask is None:
                    raise ValueError("missing shield mask")
                value = _mask_ratio(shield_mask)
            else:
                reason = "UNOBSERVED:unknown-metric"
        except ValueError as exc:
            reason = f"UNOBSERVED:{exc}"
        secondary_results.append(_make_result(name, unit, hib, value, reason))

    frequency_signal = get_array("frequency")
    reference_signal = get_array("reference_frequency")
    freq_entries = list(registry.get("frequency", []))
    if freq_entries:
        if frequency_signal is None or reference_signal is None:
            for entry in freq_entries:
                freq_results.append(
                    _make_result(
                        name=str(entry["name"]),
                        unit=str(entry.get("unit", "")),
                        higher_is_better=bool(entry.get("higher_is_better", False)),
                        value=None,
                        reason="UNOBSERVED:missing-frequency",
                    )
                )
        else:
            band_map = {
                "band_power_low": (0.0, 0.5),
                "band_power_mid": (0.5, 1.0),
                "band_power_high": (1.0, 2.0),
            }
            try:
                summary = summarize_frequency(time, frequency_signal, reference_signal, band_map.values())
            except ObservabilityError as exc:
                for entry in freq_entries:
                    freq_results.append(
                        _make_result(
                            name=str(entry["name"]),
                            unit=str(entry.get("unit", "")),
                            higher_is_better=bool(entry.get("higher_is_better", False)),
                            value=None,
                            reason=f"UNOBSERVED:{exc}",
                        )
                    )
            else:
                coherence_lookup = summary.band_ci
                coherence_means = summary.coherence_means
                mean_lookup = summary.band_means
                for entry in freq_entries:
                    name = str(entry["name"])
                    hib = bool(entry.get("higher_is_better", False))
                    unit = str(entry.get("unit", ""))
                    value: Optional[float] = None
                    reason = None
                    if name in band_map:
                        band = band_map[name]
                        value = mean_lookup.get(band, 0.0)
                    elif name == "log_decrement":
                        value = summary.log_decrement
                        reason = summary.log_decrement_reason
                    elif name == "coherence_mean":
                        band = band_map.get("band_power_mid", (0.5, 1.0))
                        if summary.coherence_reason is not None:
                            reason = summary.coherence_reason
                            value = None
                        else:
                            value = coherence_means.get(band, 0.0)
                    else:
                        reason = "UNOBSERVED:unknown-frequency-metric"
                    freq_results.append(_make_result(name, unit, hib, value, reason))

    return primary_results + secondary_results + freq_results


__all__ = ["MetricResult", "compute_all", "compute_from_timeseries", "emit_combined_metrics_tidy"]


def compute_from_timeseries(timeseries: Dict[str, np.ndarray], registry: Dict[str, Iterable[Dict[str, object]]], strict: bool = False) -> List[MetricResult]:
    """Backward-compatible shim for legacy callers."""

    return compute_all(timeseries, registry, strict)


def emit_combined_metrics_tidy(*args, **kwargs):  # pragma: no cover - compatibility stub
    raise NotImplementedError("emit_combined_metrics_tidy has been superseded by export_pack")
