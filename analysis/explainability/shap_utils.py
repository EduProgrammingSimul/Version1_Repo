"""Placeholder SHAP visualisations for deterministic runs."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def shap_global(policy: Any, dataset: Dict[str, np.ndarray]) -> None:
    X = np.asarray(dataset.get("features"))
    if X.ndim == 1:
        X = X[:, None]
    feature_names = dataset.get("feature_names") or [f"f{i}" for i in range(X.shape[1])]
    contrib = np.mean(X, axis=0)
    order = np.argsort(-np.abs(contrib))
    contrib = contrib[order]
    feature_names = [feature_names[i] for i in order]

    Path("figures").mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.barh(feature_names, contrib, color="#1f77b4")
    ax.set_title("Beeswarm proxy")
    fig.tight_layout()
    fig.savefig(Path("figures") / "fig11_shap_beeswarm.png", dpi=600, facecolor="white")
    plt.close(fig)

    topk = min(3, len(feature_names))
    fig, axes = plt.subplots(topk, 1, figsize=(4, 3), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, name, idx in zip(axes, feature_names[:topk], order[:topk]):
        ax.plot(X[:, idx], np.sort(X[:, idx]), color="#ff7f0e")
        ax.set_ylabel(name)
    axes[-1].set_xlabel("Feature value")
    fig.tight_layout()
    fig.savefig(Path("figures") / "fig12_shap_pdp.png", dpi=600, facecolor="white")
    plt.close(fig)


__all__ = ["shap_global"]
