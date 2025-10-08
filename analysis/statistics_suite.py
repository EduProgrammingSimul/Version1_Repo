from __future__ import annotations

"""Statistical inference utilities for controller comparisons."""

import os
from typing import Iterable, List

import numpy as np
import pandas as pd
from scipy import stats

from your_project import logging_setup

logger = logging_setup.get_logger(__name__)


def _benjamini_hochberg(p_values: List[float]) -> List[float]:
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(m, dtype=float)
    for rank, idx in enumerate(order, start=1):
        adjusted[idx] = min(ranked[rank - 1] * m / rank, 1.0)

    for i in range(m - 2, -1, -1):
        adjusted[order[i]] = min(adjusted[order[i]], adjusted[order[i + 1]])
    return adjusted.tolist()


def run_paired_tests(metrics_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    results: List[dict] = []
    metrics = sorted(metrics_df["metric"].unique())

    for metric in metrics:
        metric_slice = metrics_df[metrics_df["metric"] == metric]
        pivot = metric_slice.pivot_table(
            index="scenario", columns="controller", values="value"
        )
        controllers = list(pivot.columns)
        n_ctrl = len(controllers)
        if n_ctrl < 2:
            continue
        for i in range(n_ctrl):
            for j in range(i + 1, n_ctrl):
                ctrl_a = controllers[i]
                ctrl_b = controllers[j]
                paired = pivot[[ctrl_a, ctrl_b]].dropna()
                if paired.empty or len(paired) < 2:
                    continue
                diff = paired[ctrl_b] - paired[ctrl_a]
                try:
                    stat, p_val = stats.wilcoxon(
                        diff, zero_method="wilcox", mode="auto"
                    )
                except ValueError:
                    stat, p_val = (float("nan"), float("nan"))
                mean_diff = float(diff.mean())
                median_diff = float(diff.median())
                std_diff = float(diff.std(ddof=1)) if len(diff) > 1 else float("nan")
                effect_size = (
                    mean_diff / std_diff
                    if np.isfinite(std_diff) and std_diff > 0
                    else float("nan")
                )
                wins = np.sum(diff > 0)
                losses = np.sum(diff < 0)
                results.append(
                    {
                        "metric": metric,
                        "controller_a": ctrl_a,
                        "controller_b": ctrl_b,
                        "n": int(len(diff)),
                        "wilcoxon_stat": float(stat),
                        "p_value": float(p_val),
                        "mean_diff": mean_diff,
                        "median_diff": median_diff,
                        "effect_size_cohens_d": effect_size,
                        "wins_b_over_a": int(wins),
                        "wins_a_over_b": int(losses),
                    }
                )

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df["p_adj_fdr"] = _benjamini_hochberg(df["p_value"].tolist())
    df["significant"] = df["p_adj_fdr"] < alpha
    return df


def emit_paired_tests(tidy_csv: str, out_csv: str, alpha: float = 0.05) -> str:
    if not os.path.isfile(tidy_csv):
        raise FileNotFoundError(tidy_csv)
    df = pd.read_csv(tidy_csv)
    if df.empty:
        raise ValueError("No data available for statistical tests")
    results = run_paired_tests(df, alpha=alpha)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    results.to_csv(out_csv, index=False)
    logger.info("Statistical tests written to %s (rows=%d)", out_csv, len(results))
    return out_csv


__all__ = ["run_paired_tests", "emit_paired_tests"]
