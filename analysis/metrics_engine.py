from __future__ import annotations

"""Deterministic KPI computation for evaluation time-series."""

import os
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Any, Optional, Tuple

import numpy as np
import pandas as pd

from your_project import logging_setup

logger = logging_setup.get_logger(__name__)


MetricRow = Dict[str, Any]


@dataclass
class MetricsEngine:

    results_root: str
    controllers: Iterable[str]
    scenarios: Iterable[str]
    core_config: Dict[str, Any]
    run_id: str = "run"
    fallback_dt: float = field(default=0.02)

    def __post_init__(self) -> None:
        sim_cfg = self.core_config.get("simulation", {})
        self.fallback_dt = float(sim_cfg.get("dt", self.fallback_dt))
        safety = self.core_config.get("safety_limits", {})
        grid_cfg = self.core_config.get("grid", {})
        analysis_cfg = self.core_config.get("analysis", {})

        self.freq_nominal = float(grid_cfg.get("f_nominal", 60.0))
        self.freq_band = (
            float(safety.get("min_frequency_hz", self.freq_nominal - 1.0)),
            float(safety.get("max_frequency_hz", self.freq_nominal + 1.0)),
        )
        self.temp_limit = float(safety.get("max_fuel_temp_c", 2800.0))
        self.speed_limit = float(safety.get("max_speed_rpm", 2250.0))

        self.spectral_band = tuple(analysis_cfg.get("spectral_band_hz", (0.0, 1.0)))
        if len(self.spectral_band) != 2:
            self.spectral_band = (0.0, 1.0)
        self.coherence_band = tuple(analysis_cfg.get("coherence_band_hz", (0.01, 0.5)))
        if len(self.coherence_band) != 2:
            self.coherence_band = (0.01, 0.5)
        self.cvar_alpha = float(analysis_cfg.get("cvar_alpha", 0.05))
        self.worst_tail_pct = float(analysis_cfg.get("worst_k_tail_pct", 5.0))
        arrhenius_cfg = analysis_cfg.get("arrhenius", {})
        self.arrhenius_Ea = float(arrhenius_cfg.get("activation_energy_j_mol", 2.6e5))
        self.arrhenius_R = float(arrhenius_cfg.get("gas_constant_j_mol_k", 8.314))

    def _timeseries_path(self, controller: str, scenario: str) -> str:
        return os.path.join(self.results_root, controller, scenario, "timeseries.csv")

    @staticmethod
    def _read_csv(path: str) -> Optional[pd.DataFrame]:
        if not os.path.isfile(path):
            logger.error("Missing timeseries CSV: %s", path)
            return None
        try:
            df = pd.read_csv(path)
            return df
        except Exception as exc:
            logger.error("Failed to read %s: %s", path, exc)
            return None

    @staticmethod
    def _nan_to_default(arr: np.ndarray, default: float = 0.0) -> np.ndarray:
        out = np.array(arr, dtype=float, copy=True)
        mask = ~np.isfinite(out)
        if mask.any():
            out[mask] = default
        return out

    def _safe_series(
        self, df: pd.DataFrame, name: str, fallback: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if name in df:
            arr = df[name].astype(float).to_numpy()
            if np.any(np.isfinite(arr)):
                return arr
        if fallback is not None:
            return np.array(fallback, dtype=float, copy=True)
        return np.full(len(df), np.nan, dtype=float)

    def _step_durations(self, time_s: np.ndarray) -> np.ndarray:
        if time_s.size == 0:
            return np.array([], dtype=float)
        diffs = np.diff(time_s, append=time_s[-1] + self.fallback_dt)
        diffs = np.where(diffs <= 0, self.fallback_dt, diffs)
        return diffs

    @staticmethod
    def _integrate(signal: np.ndarray, dt: np.ndarray) -> float:
        if signal.size == 0 or dt.size == 0 or signal.size != dt.size:
            return float("nan")
        mask = np.isfinite(signal) & np.isfinite(dt)
        if not np.any(mask):
            return float("nan")
        return float(np.sum(signal[mask] * dt[mask]))

    def _rmse(self, err: np.ndarray, dt: np.ndarray) -> float:
        mask = np.isfinite(err) & np.isfinite(dt)
        if not np.any(mask):
            return float("nan")
        num = np.sum((err[mask] ** 2) * dt[mask])
        den = np.sum(dt[mask])
        if den <= 0:
            return float("nan")
        return float(math.sqrt(num / den))

    def _log_decrement(self, err: np.ndarray) -> float:
        finite = err[np.isfinite(err)]
        if finite.size < 3:
            return float("nan")
        peaks: List[float] = []
        for idx in range(1, len(finite) - 1):
            if finite[idx] > finite[idx - 1] and finite[idx] > finite[idx + 1]:
                peaks.append(abs(finite[idx]))
        if len(peaks) < 2:
            return float("nan")
        A1 = peaks[0]
        An = peaks[-1]
        n = len(peaks) - 1
        if A1 <= 0 or An <= 0 or n <= 0:
            return float("nan")
        return float((1.0 / n) * math.log(A1 / An))

    def _compute_psd(
        self, signal: np.ndarray, dt: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        mask = np.isfinite(signal) & np.isfinite(dt)
        if not np.any(mask):
            return np.array([]), np.array([])
        sig = signal.copy()
        sig[~np.isfinite(sig)] = 0.0
        mean_dt = float(np.mean(dt[mask])) if np.any(mask) else self.fallback_dt
        if mean_dt <= 0:
            mean_dt = self.fallback_dt
        fs = 1.0 / mean_dt
        sig_centered = sig - np.nanmean(sig[mask])
        fft = np.fft.rfft(sig_centered)
        freq = np.fft.rfftfreq(len(sig), d=mean_dt)
        psd = (np.abs(fft) ** 2) * mean_dt / max(len(sig), 1)
        return freq, psd

    def _band_integral(
        self, freq: np.ndarray, values: np.ndarray, band: Tuple[float, float]
    ) -> float:
        if freq.size == 0 or values.size == 0 or freq.size != values.size:
            return float("nan")
        low, high = band
        mask = (freq >= low) & (freq <= high)
        if not np.any(mask):
            return float("nan")
        return float(np.trapz(values[mask], freq[mask]))

    def _bandwidth_proxy(self, freq: np.ndarray, psd: np.ndarray) -> float:
        if freq.size == 0 or psd.size == 0:
            return float("nan")
        baseline = psd[0]
        if baseline <= 0:
            return float("nan")
        threshold = baseline / 2.0
        for f, val in zip(freq, psd):
            if val <= threshold:
                return float(f)
        return float("nan")

    def _coherence(
        self,
        reference: np.ndarray,
        output: np.ndarray,
        dt: np.ndarray,
        band: Tuple[float, float],
    ) -> float:
        mask_ref = np.isfinite(reference)
        mask_out = np.isfinite(output)
        mask_dt = np.isfinite(dt)
        mask = mask_ref & mask_out & mask_dt
        if not np.any(mask):
            return float("nan")
        ref = reference.copy()
        out = output.copy()
        ref[~np.isfinite(ref)] = 0.0
        out[~np.isfinite(out)] = 0.0
        mean_dt = float(np.mean(dt[mask]))
        if mean_dt <= 0:
            mean_dt = self.fallback_dt
        fs = 1.0 / mean_dt
        ref_centered = ref - np.nanmean(ref[mask])
        out_centered = out - np.nanmean(out[mask])
        fft_ref = np.fft.rfft(ref_centered)
        fft_out = np.fft.rfft(out_centered)
        freq = np.fft.rfftfreq(len(ref), d=mean_dt)
        S_xx = (np.abs(fft_ref) ** 2) * mean_dt / max(len(ref), 1)
        S_yy = (np.abs(fft_out) ** 2) * mean_dt / max(len(out), 1)
        S_xy = fft_ref * np.conj(fft_out) * mean_dt / max(len(ref), 1)
        denominator = S_xx * S_yy
        mask_den = denominator > 0
        if not np.any(mask_den):
            return float("nan")
        coherence = np.zeros_like(freq)
        coherence[mask_den] = (np.abs(S_xy[mask_den]) ** 2) / denominator[mask_den]
        low, high = band
        band_mask = (freq >= low) & (freq <= high)
        if not np.any(band_mask):
            return float("nan")
        return float(np.nanmean(coherence[band_mask]))

    def _cvar(self, values: np.ndarray, alpha: float) -> float:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return float("nan")
        sorted_vals = np.sort(finite)
        tail_count = max(1, int(math.ceil(alpha * len(sorted_vals))))
        tail = sorted_vals[-tail_count:]
        if tail.size == 0:
            return float("nan")
        return float(np.nanmean(tail))

    def _worst_tail(self, values: np.ndarray, pct: float) -> float:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return float("nan")
        sorted_vals = np.sort(finite)
        tail_count = max(1, int(math.ceil(pct / 100.0 * len(sorted_vals))))
        tail = sorted_vals[-tail_count:]
        if tail.size == 0:
            return float("nan")
        return float(np.nanmean(tail))

    def _percentile(self, values: np.ndarray, q: float) -> float:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return float("nan")
        return float(np.percentile(finite, q))

    def _compute_metric_rows(
        self, controller: str, scenario: str, df: pd.DataFrame
    ) -> List[MetricRow]:
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]
        time = df.get("time_s")
        if time is None:
            logger.warning("time_s column missing for %s/%s", controller, scenario)
            return []
        time = time.astype(float).to_numpy()
        if time.size == 0:
            logger.warning("Empty timeseries for %s/%s", controller, scenario)
            return []

        dt = self._safe_series(df, "dt")
        if not np.any(np.isfinite(dt)):
            dt = self._step_durations(time)
        dt = np.where(~np.isfinite(dt) | (dt <= 0), self.fallback_dt, dt)

        y_target = self._safe_series(df, "y_target")
        y_actual = self._safe_series(df, "y_actual")
        err = self._safe_series(df, "err", fallback=y_target - y_actual)
        abs_err = np.abs(err)
        u = self._safe_series(df, "u")
        du = self._safe_series(df, "du")
        du_dt = self._safe_series(df, "du_dt")
        freq = self._safe_series(df, "freq_hz")
        freq_dev = self._safe_series(
            df, "freq_deviation_hz", fallback=freq - self.freq_nominal
        )
        sat_flag = self._safe_series(df, "sat_flag")
        safety_flag_series = self._safe_series(df, "safety_flag")
        safety_margin = self._safe_series(df, "safety_margin_min")
        temp_c = self._safe_series(df, "temp_C")
        temp_limit_series = self._safe_series(
            df, "temp_limit_C", fallback=np.full(len(df), self.temp_limit)
        )
        thermal_margin = self._safe_series(df, "thermal_margin_C")
        rocof = self._safe_series(df, "rocof_hz_s")

        total_time = float(np.sum(dt))

        iae = self._integrate(np.abs(err), dt)
        itae = self._integrate(time * np.abs(err), dt)
        rmse = self._rmse(err, dt)

        finite_target = y_target[np.isfinite(y_target)]
        final_target = finite_target[-1] if finite_target.size else math.nan

        finite_actual = y_actual[np.isfinite(y_actual)]
        peak_actual = float(np.max(finite_actual)) if finite_actual.size else math.nan
        overshoot_pct = float("nan")
        if np.isfinite(peak_actual) and np.isfinite(final_target):
            denom = max(abs(final_target), 1e-6)
            overshoot_pct = max(0.0, (peak_actual - final_target) / denom * 100.0)

        settling_time = float("nan")
        if np.isfinite(final_target):
            tolerance = 0.02 * max(abs(final_target), 1.0)
            abs_err_series = np.abs(err)
            for idx in range(len(abs_err_series)):
                window = abs_err_series[idx:]
                finite_window = window[np.isfinite(window)]
                if finite_window.size == 0:
                    continue
                if np.all(finite_window <= tolerance):
                    settling_time = float(time[idx])
                    break

        log_decrement = self._log_decrement(err)

        freq_psd_freq, freq_psd = self._compute_psd(err, dt)
        spectral_power = self._band_integral(
            freq_psd_freq, freq_psd, self.spectral_band
        )
        bandwidth_proxy = self._bandwidth_proxy(freq_psd_freq, freq_psd)
        coherence = self._coherence(y_target, y_actual, dt, self.coherence_band)

        unsafe_mask = np.zeros(len(dt), dtype=bool)
        if np.any(np.isfinite(safety_flag_series)):
            unsafe_mask |= safety_flag_series > 0.5
        freq_low, freq_high = self.freq_band
        unsafe_mask |= np.where(
            np.isfinite(freq), (freq < freq_low) | (freq > freq_high), False
        )
        if np.any(np.isfinite(thermal_margin)):
            unsafe_mask |= thermal_margin < 0.0
        total_time_unsafe = float(np.sum(dt[unsafe_mask]))

        safety_margin_min = (
            float(np.nanmin(safety_margin))
            if np.any(np.isfinite(safety_margin))
            else float("nan")
        )

        stability_flag = float(1.0)
        if (
            np.any(~np.isfinite(y_actual))
            or np.any(~np.isfinite(u))
            or np.any(np.abs(y_actual[np.isfinite(y_actual)]) > 1e6)
        ):
            stability_flag = float(0.0)

        worst_tail = self._worst_tail(np.abs(err), self.worst_tail_pct)

        effort_l1 = self._integrate(np.abs(u), dt)
        effort_l2 = self._integrate(u**2, dt)
        jerk_l2 = self._integrate(du_dt**2, dt)

        sign_u = np.sign(u)
        sign_u[~np.isfinite(sign_u)] = 0.0
        reversals = 0
        if sign_u.size > 1:
            sign_change = (sign_u[1:] * sign_u[:-1]) < 0
            reversals = int(np.sum(sign_change))

        time_in_saturation = float(np.sum(dt[np.isfinite(sat_flag) & (sat_flag > 0.5)]))
        sat_events = 0
        finite_sat = sat_flag[np.isfinite(sat_flag)]
        if finite_sat.size > 1:
            transitions = np.logical_and(finite_sat[1:] > 0.5, finite_sat[:-1] <= 0.5)
            sat_events = int(np.sum(transitions))

        max_freq_dev = (
            float(np.nanmax(np.abs(freq_dev)))
            if np.any(np.isfinite(freq_dev))
            else float("nan")
        )
        out_of_band = np.where(
            np.isfinite(freq), (freq < freq_low) | (freq > freq_high), False
        )
        time_out_of_band = float(np.sum(dt[out_of_band]))

        time_above_temp = float("nan")
        max_temp_delta = float("nan")
        temp_limit = (
            float(np.nanmean(temp_limit_series[np.isfinite(temp_limit_series)]))
            if np.any(np.isfinite(temp_limit_series))
            else self.temp_limit
        )
        if np.any(np.isfinite(temp_c)):
            temp_delta = temp_c - temp_limit
            time_above_temp = float(
                np.sum(dt[np.isfinite(temp_delta) & (temp_delta > 0)])
            )
            max_temp_delta = float(np.nanmax(np.maximum(temp_delta, 0.0)))

        arrhenius = float("nan")
        temp_k = temp_c + 273.15
        valid_temp = temp_k > 0
        if np.any(valid_temp & np.isfinite(temp_k)):
            expo = np.exp(
                -self.arrhenius_Ea
                / (self.arrhenius_R * temp_k[valid_temp & np.isfinite(temp_k)])
            )
            expo_dt = dt[valid_temp & np.isfinite(temp_k)]
            arrhenius = float(np.sum(expo * expo_dt))

        cvar = self._cvar(np.abs(err), self.cvar_alpha)
        p50 = self._percentile(np.abs(err), 50.0)
        p90 = self._percentile(np.abs(err), 90.0)
        p95 = self._percentile(np.abs(err), 95.0)

        max_rocof = (
            float(np.nanmax(np.abs(rocof)))
            if np.any(np.isfinite(rocof))
            else float("nan")
        )

        metrics: Dict[str, float] = {
            "iae_abs_err": iae,
            "itae_abs_err": itae,
            "rmse_error": rmse,
            "overshoot_pct": overshoot_pct,
            "settling_time_s": settling_time,
            "log_decrement": log_decrement,
            "spectral_error_power": spectral_power,
            "glfi": float("nan"),
            "bandwidth_proxy_hz": bandwidth_proxy,
            "coherence_lowmid": coherence,
            "total_time_unsafe_s": total_time_unsafe,
            "safety_margin_min": safety_margin_min,
            "stability_flag": stability_flag,
            "worst_tail_abs_err": worst_tail,
            "effort_l1": effort_l1,
            "effort_l2": effort_l2,
            "jerk_l2": jerk_l2,
            "reversal_count": float(reversals),
            "time_in_saturation_s": time_in_saturation,
            "saturation_events": float(sat_events),
            "max_freq_deviation_hz": max_freq_dev,
            "time_out_of_band_s": time_out_of_band,
            "time_above_temp_limit_s": time_above_temp,
            "max_delta_temp_over_limit_C": max_temp_delta,
            "arrhenius_exposure": arrhenius,
            "cvar_abs_err": cvar,
            "abs_err_p50": p50,
            "abs_err_p90": p90,
            "abs_err_p95": p95,
            "max_rocof_hz_s": max_rocof,
        }

        if np.isfinite(iae):
            glfi_denom = self._integrate(np.abs(y_target) + 1.0, dt)
            if np.isfinite(glfi_denom) and glfi_denom > 0:
                metrics["glfi"] = float(max(0.0, 1.0 - (iae / glfi_denom)))

        rows: List[MetricRow] = []
        hib_map = {
            "iae_abs_err": False,
            "itae_abs_err": False,
            "rmse_error": False,
            "overshoot_pct": False,
            "settling_time_s": False,
            "log_decrement": False,
            "spectral_error_power": False,
            "glfi": True,
            "bandwidth_proxy_hz": True,
            "coherence_lowmid": True,
            "total_time_unsafe_s": False,
            "safety_margin_min": True,
            "stability_flag": True,
            "worst_tail_abs_err": False,
            "effort_l1": False,
            "effort_l2": False,
            "jerk_l2": False,
            "reversal_count": False,
            "time_in_saturation_s": False,
            "saturation_events": False,
            "max_freq_deviation_hz": False,
            "time_out_of_band_s": False,
            "time_above_temp_limit_s": False,
            "max_delta_temp_over_limit_C": False,
            "arrhenius_exposure": False,
            "cvar_abs_err": False,
            "abs_err_p50": False,
            "abs_err_p90": False,
            "abs_err_p95": False,
            "max_rocof_hz_s": False,
        }

        for metric, value in metrics.items():
            rows.append(
                {
                    "run_id": self.run_id,
                    "controller": controller,
                    "scenario": scenario,
                    "metric": metric,
                    "value": float(value) if np.isfinite(value) else float("nan"),
                    "higher_is_better": bool(hib_map.get(metric, False)),
                }
            )
        return rows

    def compute(self) -> List[MetricRow]:
        all_rows: List[MetricRow] = []
        scen_list = list(self.scenarios)
        ctrl_list = list(self.controllers)
        for ctrl in ctrl_list:
            for scenario in scen_list:
                path = self._timeseries_path(ctrl, scenario)
                df = self._read_csv(path)
                if df is None or df.empty:
                    logger.warning(
                        "Skipping metrics for %s/%s due to missing data", ctrl, scenario
                    )
                    continue
                rows = self._compute_metric_rows(ctrl, scenario, df)
                all_rows.extend(rows)
        return all_rows


def emit_combined_metrics_tidy(rows: Iterable[MetricRow], out_csv: str) -> str:
    df = pd.DataFrame(list(rows))
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df = df.sort_values(["controller", "scenario", "metric"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    logger.info("Combined metrics written to %s (rows=%d)", out_csv, len(df))
    return out_csv


def compute_from_timeseries(
    results_root: str,
    controllers: Iterable[str],
    scenarios: Iterable[str],
    core_config: Dict[str, Any],
    run_id: str = "report",
) -> str:
    engine = MetricsEngine(
        results_root=results_root,
        controllers=controllers,
        scenarios=scenarios,
        core_config=core_config,
        run_id=run_id,
    )
    rows = engine.compute()
    out_csv = os.path.join(
        os.path.dirname(results_root), "metrics", "combined_metrics_tidy.csv"
    )
    return emit_combined_metrics_tidy(rows, out_csv)


def compute_multi_seed_metrics(
    results_roots: Iterable[str],
    controllers: Iterable[str],
    scenarios: Iterable[str],
    core_config: Dict[str, Any],
    seeds: Iterable[int],
) -> pd.DataFrame:

    records: List[pd.DataFrame] = []
    for seed, root in zip(seeds, results_roots):
        run_id = f"seed_{seed}"
        csv_path = compute_from_timeseries(
            root, controllers, scenarios, core_config, run_id=run_id
        )
        df = pd.read_csv(csv_path)
        df["seed"] = seed
        records.append(df)
    if not records:
        return pd.DataFrame()
    combined = pd.concat(records, ignore_index=True)
    grouped = combined.groupby(
        ["controller", "scenario", "metric", "higher_is_better"], as_index=False
    )
    summary = grouped["value"].agg(["mean", "std", "count"]).reset_index()
    summary = summary.rename(
        columns={"mean": "value_mean", "std": "value_std", "count": "n"}
    )
    return summary


__all__ = [
    "MetricsEngine",
    "compute_from_timeseries",
    "emit_combined_metrics_tidy",
    "compute_multi_seed_metrics",
]
