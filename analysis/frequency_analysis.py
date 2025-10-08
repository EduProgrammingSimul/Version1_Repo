from __future__ import annotations

"""Frequency-domain and step-response analysis utilities."""

import os
import math
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from analysis.scenario_definitions import get_scenarios
from your_project import logging_setup

logger = logging_setup.get_logger(__name__)


def _read_timeseries(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        logger.error("Failed to read timeseries %s: %s", path, exc)
        return pd.DataFrame()


def _compute_psd(signal: np.ndarray, dt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(signal) & np.isfinite(dt)
    if not np.any(mask):
        return np.array([]), np.array([])
    sig = signal.copy()
    sig[~np.isfinite(sig)] = 0.0
    mean_dt = float(np.mean(dt[mask]))
    if mean_dt <= 0:
        return np.array([]), np.array([])
    sig_centered = sig - np.nanmean(sig[mask])
    fft = np.fft.rfft(sig_centered)
    freq = np.fft.rfftfreq(len(sig), d=mean_dt)
    psd = (np.abs(fft) ** 2) * mean_dt / max(len(sig), 1)
    return freq, psd


def _bandpower(freq: np.ndarray, psd: np.ndarray, band: Tuple[float, float]) -> float:
    if freq.size == 0 or psd.size == 0:
        return float("nan")
    lo, hi = band
    mask = (freq >= lo) & (freq <= hi)
    if not np.any(mask):
        return float("nan")
    return float(np.trapz(psd[mask], freq[mask]))


def _noise_amplification(error_psd: np.ndarray, control_psd: np.ndarray) -> float:
    if (
        error_psd.size == 0
        or control_psd.size == 0
        or error_psd.size != control_psd.size
    ):
        return float("nan")
    denom = float(np.sum(error_psd))
    if denom <= 0:
        return float("nan")
    return float(np.sum(control_psd) / denom)


def _step_metrics(time: np.ndarray, error: np.ndarray) -> Tuple[float, float]:
    finite = np.isfinite(error)
    if np.sum(finite) < 3:
        return float("nan"), float("nan")
    err = error[finite]
    t = time[finite]

    peaks_idx = []
    for idx in range(1, len(err) - 1):
        if err[idx] > err[idx - 1] and err[idx] > err[idx + 1]:
            peaks_idx.append(idx)
    if len(peaks_idx) < 2:
        return float("nan"), float("nan")
    peak_vals = np.abs(err[peaks_idx])
    delta = math.log(peak_vals[0] / peak_vals[-1]) / (len(peak_vals) - 1)
    zeta = delta / math.sqrt(delta**2 + (2 * math.pi) ** 2)
    if len(peaks_idx) >= 2:
        periods = np.diff(t[peaks_idx])
        omega_d = (
            2 * math.pi / np.mean(periods) if np.mean(periods) > 0 else float("nan")
        )
    else:
        omega_d = float("nan")
    if np.isfinite(zeta) and zeta < 1.0 and np.isfinite(omega_d):
        omega_n = omega_d / math.sqrt(1 - zeta**2)
    else:
        omega_n = float("nan")
    return float(zeta), float(omega_n)


def analyze_frequency_domain(
    results_root: str,
    controllers: Iterable[str],
    scenarios: Iterable[str],
    core_config: Dict[str, Any],
    freq_band: Tuple[float, float] = (0.0, 1.0),
) -> pd.DataFrame:
    scenario_defs = get_scenarios(core_config)
    records: List[Dict[str, Any]] = []
    for controller in controllers:
        for scenario in scenarios:
            path = os.path.join(results_root, controller, scenario, "timeseries.csv")
            df = _read_timeseries(path)
            if df.empty:
                continue
            df.columns = [c.strip() for c in df.columns]
            time = df.get("time_s", pd.Series(dtype=float)).to_numpy(dtype=float)
            dt = df.get("dt", pd.Series(dtype=float)).to_numpy(dtype=float)
            if dt.size == 0:
                dt = np.diff(time, append=time[-1] + 0.02)
            error = df.get("err")
            control = df.get("u")
            if error is None or control is None:
                continue
            error = error.astype(float).to_numpy()
            control = control.astype(float).to_numpy()

            freq, psd_error = _compute_psd(error, dt)
            _, psd_control = _compute_psd(control, dt)
            bandpower_err = _bandpower(freq, psd_error, freq_band)
            bandpower_ctrl = _bandpower(freq, psd_control, freq_band)
            noise_gain = _noise_amplification(psd_error, psd_control)
            zeta, omega_n = _step_metrics(time, error)
            scenario_cfg = scenario_defs.get(scenario, {})
            tags = scenario_cfg.get("analysis_tags", [])
            records.append(
                {
                    "controller": controller,
                    "scenario": scenario,
                    "psd_bandpower_error": bandpower_err,
                    "psd_bandpower_control": bandpower_ctrl,
                    "noise_amplification": noise_gain,
                    "damping_ratio": zeta,
                    "natural_frequency": omega_n,
                    "tags": "|".join(tags) if isinstance(tags, list) else "",
                }
            )
    return pd.DataFrame(records)


def emit_frequency_analysis(
    results_root: str,
    controllers: Iterable[str],
    scenarios: Iterable[str],
    core_config: Dict[str, Any],
    out_csv: str,
    freq_band: Tuple[float, float] = (0.0, 1.0),
) -> str:
    df = analyze_frequency_domain(
        results_root, controllers, scenarios, core_config, freq_band=freq_band
    )
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info("Frequency analysis written to %s (rows=%d)", out_csv, len(df))
    return out_csv


__all__ = ["analyze_frequency_domain", "emit_frequency_analysis"]
