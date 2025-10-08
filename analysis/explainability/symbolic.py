"""Symbolic regression helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis import tables


def _format_equation(coeffs: np.ndarray) -> str:
    terms = []
    degree = len(coeffs) - 1
    for i, coeff in enumerate(coeffs):
        power = degree - i
        if power == 0:
            terms.append(f"{coeff:.3f}")
        elif power == 1:
            terms.append(f"{coeff:.3f}*x")
        else:
            terms.append(f"{coeff:.3f}*x^{power}")
    return " + ".join(terms)


def fit_symbolic(X: np.ndarray, y: np.ndarray, max_complexity: int = 20) -> List[Dict[str, object]]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    x_main = X[:, 0]
    degrees = list(range(1, min(4, max_complexity + 1)))
    models: List[Dict[str, object]] = []
    rows: List[List[object]] = []
    fidelity: List[float] = []
    complexities: List[int] = []
    for degree in degrees:
        coeffs = np.polyfit(x_main, y, degree)
        preds = np.polyval(coeffs, x_main)
        mse = float(np.mean((preds - y) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2) + 1e-12)
        ss_res = float(np.sum((y - preds) ** 2))
        r2 = 1.0 - ss_res / ss_tot
        equation = _format_equation(coeffs)
        models.append({"equation": equation, "complexity": degree, "mse": mse, "r2": r2})
        rows.append([equation, degree, mse, r2])
        fidelity.append(r2)
        complexities.append(degree)
    tables.write_symbolic_models(rows)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(complexities, fidelity, marker="o", color="#7f7f7f")
    ax.set_xlabel("Complexity")
    ax.set_ylabel("R²")
    fig.tight_layout()
    path = Path("figures") / "fig10_symbolic_curve.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600, format="png", facecolor="white")
    plt.close(fig)
    return models


__all__ = ["fit_symbolic"]
