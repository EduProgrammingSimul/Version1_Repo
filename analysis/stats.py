"""Statistical utilities with deterministic behaviour."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
from scipy import stats as sci_stats


def bootstrap_ci(x: np.ndarray, level: float = 0.95, n_boot: int = 2000, seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        raise ValueError("no finite samples")
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(finite, size=finite.size, replace=True)
        boot[i] = float(np.mean(sample))
    alpha = (1.0 - level) / 2.0
    lower = float(np.quantile(boot, alpha))
    upper = float(np.quantile(boot, 1 - alpha))
    return lower, upper


def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        raise ValueError("no finite pairs")
    stat, p = sci_stats.wilcoxon(a[mask], b[mask], zero_method="wilcox", correction=False)
    return float(p)


def bh_fdr(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.size == 0:
        raise ValueError("empty p-values")
    order = np.argsort(p)
    ranked = p[order]
    thresholds = q * (np.arange(1, p.size + 1) / p.size)
    passed = ranked <= thresholds
    if not np.any(passed):
        return np.zeros_like(p, dtype=bool)
    cutoff = np.max(np.where(passed)[0])
    mask = np.zeros_like(p, dtype=bool)
    mask[order[: cutoff + 1]] = True
    return mask


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        raise ValueError("no finite pairs")
    diff = a[mask] - b[mask]
    mean_diff = float(np.mean(diff))
    pooled = float(np.std(diff, ddof=1))
    if pooled == 0.0:
        return 0.0
    return mean_diff / pooled


def aggregate_by_scenario(metric: Dict[str, Iterable[float]]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for scenario, values in metric.items():
        arr = np.asarray(list(values), dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue
        summary[scenario] = float(np.mean(finite))
    return summary


__all__ = [
    "bootstrap_ci",
    "wilcoxon_paired",
    "bh_fdr",
    "cohen_d",
    "aggregate_by_scenario",
]
