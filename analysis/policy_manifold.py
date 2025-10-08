from __future__ import annotations

import json
import os
from typing import List

import numpy as np
import pandas as pd


TIMESERIES_FILENAME = "timeseries.csv"


def _collect_timeseries(val_root: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for controller in sorted(os.listdir(val_root)):
        ctrl_dir = os.path.join(val_root, controller)
        if not os.path.isdir(ctrl_dir):
            continue
        for scenario in sorted(os.listdir(ctrl_dir)):
            scen_dir = os.path.join(ctrl_dir, scenario)
            path = os.path.join(scen_dir, TIMESERIES_FILENAME)
            if not os.path.isfile(path):
                continue
            df = pd.read_csv(path)
            df["controller"] = controller
            df["scenario"] = scenario
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_policy_manifold(
    out_root: str, val_root: str, eval_seed: int = 42
) -> str | None:
    try:
        from sklearn.decomposition import PCA
    except Exception:
        return None

    df = _collect_timeseries(val_root)
    if df.empty:
        return None

    df.columns = [c.strip().lower() for c in df.columns]
    feature_candidates = [
        ("y_actual", None),
        ("y_target", None),
        ("freq_hz", None),
        ("temp_c", None),
        ("u", None),
    ]
    selected = [name for name, _ in feature_candidates if name in df.columns]
    if len(selected) < 3:
        return None

    features = df[selected].astype(float)
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    if features.empty:
        return None

    meta = df.loc[features.index, ["controller", "scenario"]]
    action = (
        features["u"].to_numpy() if "u" in features.columns else np.zeros(len(features))
    )

    X = features.to_numpy(dtype=float)
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-9)

    pca = PCA(n_components=2, random_state=eval_seed)
    pcs = pca.fit_transform(X)

    out_df = pd.DataFrame(
        {
            "controller": meta["controller"].to_numpy(),
            "scenario": meta["scenario"].to_numpy(),
            "pc1": pcs[:, 0],
            "pc2": pcs[:, 1],
            "action": action,
            "abs_action": np.abs(action),
        }
    )

    metrics_dir = os.path.join(out_root, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    parquet_path = os.path.join(metrics_dir, "policy_manifold_points.parquet")
    out_df.to_parquet(parquet_path, index=False)

    meta_path = os.path.join(metrics_dir, "policy_manifold_points.meta.json")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "explained_variance_ratio": list(
                    map(float, pca.explained_variance_ratio_)
                ),
                "features": selected,
                "eval_seed": int(eval_seed),
                "path": parquet_path,
                "format": "parquet",
            },
            handle,
            indent=2,
        )

    return parquet_path
