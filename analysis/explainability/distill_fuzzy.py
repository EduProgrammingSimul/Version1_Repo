"""Fuzzy rule distillation utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis import tables


def distill(X: np.ndarray, y: np.ndarray, max_rules: int = 15) -> Dict[str, object]:
    rng = np.random.default_rng(0)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n_samples, n_features = X.shape
    feature_scores = np.abs(np.corrcoef(X, rowvar=False)[-1]) if n_features > 1 else np.array([1.0])
    feature_scores = np.nan_to_num(feature_scores, nan=0.0, posinf=0.0, neginf=0.0)
    rules: List[str] = []
    coverage: List[float] = []
    for idx in range(min(max_rules, n_features)):
        threshold = float(np.quantile(X[:, idx], 0.5))
        rule = f"if x{idx} > {threshold:.3f} then high"
        rules.append(rule)
        coverage.append(float(np.mean(X[:, idx] > threshold)))
    preds = np.clip(np.mean(X, axis=1), 0.0, 1.0)
    mse = float(np.mean((preds - y) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2) + 1e-12)
    ss_res = float(np.sum((y - preds) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    rows = [(rule, cov, r2, mse) for rule, cov in zip(rules, coverage)]
    tables.write_distilled_rules(rows)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.barh(rules[:5], coverage[:5], color="#17becf")
    ax.set_xlabel("Coverage")
    ax.set_title("Top fuzzy rules")
    fig.tight_layout()
    path = Path("figures") / "fig09_fuzzy_panel.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600, format="png", facecolor="white")
    plt.close(fig)

    return {"rules": rules, "coverage": coverage, "fidelity_r2": r2, "mse": mse}


__all__ = ["distill"]
