from __future__ import annotations

import json
import os
from typing import List

import numpy as np
import pandas as pd
from scipy import stats

REQUIRED = {"controller", "scenario", "metric", "value"}


def _load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    rename = {
        "metric_name": "metric",
        "val": "value",
        "score": "value",
        "ctrl": "controller",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    missing = REQUIRED.difference(df.columns)
    if missing:
        raise ValueError(
            f"combined_metrics_tidy.csv missing columns: {sorted(missing)}"
        )
    return df


def compute_metric_correlations(
    out_root: str, tidy_csv: str, method: str = "pearson", eval_seed: int = 42
) -> str:
    df = _load(tidy_csv)
    wide = df.pivot_table(
        index=["scenario", "controller"],
        columns="metric",
        values="value",
        aggfunc="mean",
    )
    cols = [c for c in wide.columns if np.issubdtype(wide[c].dtype, np.number)]
    wide = wide[cols]

    constant_metrics: List[str] = []
    stds = wide.std(skipna=True)
    for metric, std in stds.items():
        if np.isclose(std, 0.0, equal_nan=False):
            constant_metrics.append(metric)

    rows = []
    metrics = list(wide.columns)
    n = len(wide)
    for i in range(len(metrics)):
        for j in range(i + 1, len(metrics)):
            a, b = metrics[i], metrics[j]
            const_flag = a in constant_metrics or b in constant_metrics
            if const_flag or n < 2:
                r_val, p_val = 0.0, 1.0
            else:
                series_a = wide[a].astype(float).to_numpy()
                series_b = wide[b].astype(float).to_numpy()
                mask = np.isfinite(series_a) & np.isfinite(series_b)
                if mask.sum() < 2:
                    r_val, p_val = 0.0, 1.0
                    const_flag = True
                else:
                    r_val, p_val = stats.pearsonr(series_a[mask], series_b[mask])
            rows.append(
                {
                    "metric_i": a,
                    "metric_j": b,
                    "r": float(r_val),
                    "p_value": float(p_val),
                    "n": int(n),
                    "method": method,
                    "constant_pair": bool(const_flag),
                }
            )

    out_dir = os.path.join(out_root, "metrics")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "corr_metrics.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    meta = {
        "eval_seed": int(eval_seed),
        "method": method,
        "constant_metrics": sorted(constant_metrics),
    }
    with open(
        os.path.join(out_dir, "corr_metrics.meta.json"), "w", encoding="utf-8"
    ) as fh:
        json.dump(meta, fh, indent=2)
    return out_csv
