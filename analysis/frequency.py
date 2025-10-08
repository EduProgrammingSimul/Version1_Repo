"""Frequency-domain utilities with strict observability guards."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
from scipy import signal


@dataclass
class FrequencySummary:
    band_means: Dict[Tuple[float, float], float]
    coherence_means: Dict[Tuple[float, float], float]
    band_ci: Dict[Tuple[float, float], Tuple[float, float]]
    log_decrement: float | None
    log_decrement_reason: str | None
    coherence_reason: str | None
    welch_freq: np.ndarray
    welch_psd: np.ndarray


class ObservabilityError(RuntimeError):
    """Raised when frequency metrics cannot be computed."""


def _validate_time(time: np.ndarray) -> tuple[float, float]:
    if time.ndim != 1 or time.size < 8:
        raise ObservabilityError("insufficient time samples")
    diffs = np.diff(time)
    if np.any(diffs <= 0):
        raise ObservabilityError("time must be strictly increasing")
    dt = float(np.median(diffs))
    if not np.allclose(diffs, dt, rtol=1e-3, atol=1e-9):
        raise ObservabilityError("time-step must be uniform")
    fs = 1.0 / dt
    return dt, fs


def _nperseg(fs: float) -> int:
    target = max(fs * 10.0, 8.0)
    power = int(math.floor(math.log2(target)))
    return int(min(8192, 2**max(power, 3)))


def welch_psd(time: np.ndarray, signal_in: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, int]:
    dt, fs = _validate_time(time)
    data = np.asarray(signal_in, dtype=float)
    if data.shape != time.shape:
        raise ObservabilityError("signal length mismatch")
    nperseg = _nperseg(fs)
    if time.size < 4 * nperseg:
        raise ObservabilityError("insufficient samples for Welch")
    freq, psd = signal.welch(
        data,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=int(0.5 * nperseg),
        detrend="constant",
        return_onesided=True,
        scaling="density",
    )
    psd = np.maximum(psd, 1e-18)
    return freq, psd, fs, nperseg


def log_decrement(time: np.ndarray, signal_in: np.ndarray) -> tuple[float | None, str | None]:
    try:
        _, fs = _validate_time(time)
    except ObservabilityError as exc:
        return None, f"UNOBSERVED:{exc}"
    data = np.asarray(signal_in, dtype=float)
    if data.shape != time.shape:
        return None, "UNOBSERVED:length-mismatch"
    normed = data - np.mean(data)
    peaks, _ = signal.find_peaks(np.abs(normed))
    if peaks.size < 3:
        return None, "UNOBSERVED:need>=3peaks"
    first = np.abs(normed[peaks[0]])
    last = np.abs(normed[peaks[-1]])
    cycles = peaks.size - 1
    if first <= 0 or last <= 0 or cycles <= 0:
        return None, "UNOBSERVED:nonpositive-peaks"
    return float((1.0 / cycles) * math.log(first / last)), None


def coherence_band_means(
    time: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    bands: Iterable[Tuple[float, float]],
) -> tuple[Dict[Tuple[float, float], float], Dict[Tuple[float, float], Tuple[float, float]], str | None]:
    try:
        dt, fs = _validate_time(time)
    except ObservabilityError as exc:
        return {}, {}, f"UNOBSERVED:{exc}"
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.shape != time.shape or y_arr.shape != time.shape:
        return {}, {}, "UNOBSERVED:length-mismatch"
    nperseg = _nperseg(fs)
    if time.size < 4 * nperseg:
        return {}, {}, "UNOBSERVED:insufficient-samples"
    freq, coh = signal.coherence(
        x_arr,
        y_arr,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=int(0.5 * nperseg),
    )
    coh = np.clip(coh, 0.0, 1.0)
    means: Dict[Tuple[float, float], float] = {}
    ci: Dict[Tuple[float, float], Tuple[float, float]] = {}
    dof = max(1, int(2 * time.size / nperseg) - 1)
    for band in bands:
        low, high = band
        mask = (freq >= low) & (freq <= high)
        if not np.any(mask):
            means[band] = 0.0
            ci[band] = (0.0, 0.0)
            continue
        vals = coh[mask]
        mean_val = float(np.mean(vals))
        std = float(np.std(vals) / math.sqrt(vals.size)) if vals.size > 1 else 0.0
        half = 1.96 * std
        means[band] = mean_val
        ci[band] = (max(0.0, mean_val - half), min(1.0, mean_val + half))
    return means, ci, None


def summarize_frequency(
    time: np.ndarray,
    signal_in: np.ndarray,
    reference: np.ndarray,
    bands: Iterable[Tuple[float, float]],
) -> FrequencySummary:
    freq, psd, _, _ = welch_psd(time, signal_in)
    band_means: Dict[Tuple[float, float], float] = {}
    for low, high in bands:
        mask = (freq >= low) & (freq <= high)
        band_means[(low, high)] = float(np.mean(psd[mask])) if np.any(mask) else 0.0
    log_dec, reason = log_decrement(time, signal_in)
    coh_means, coh_ci, coh_reason = coherence_band_means(time, signal_in, reference, bands)
    for band in bands:
        if band not in coh_means:
            coh_means[band] = 0.0
            coh_ci[band] = (0.0, 0.0)
    return FrequencySummary(
        band_means=band_means,
        coherence_means=coh_means,
        band_ci=coh_ci,
        log_decrement=log_dec,
        log_decrement_reason=reason,
        coherence_reason=coh_reason,
        welch_freq=freq,
        welch_psd=psd,
    )


__all__ = [
    "FrequencySummary",
    "ObservabilityError",
    "welch_psd",
    "log_decrement",
    "coherence_band_means",
    "summarize_frequency",
]
