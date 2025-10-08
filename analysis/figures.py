"""Deterministic Matplotlib figure factory."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_DEFAULT_FONT = {"family": "DejaVu Sans", "size": 10}


def _ensure_dir() -> Path:
    path = Path("figures")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save(fig: plt.Figure, name: str, dpi: int, tiff: bool = True) -> Dict[str, str]:
    matplotlib.rc("font", **_DEFAULT_FONT)
    fig.tight_layout()
    base = _ensure_dir() / name
    outputs = {}
    png_path = base.with_suffix(".png")
    fig.savefig(png_path, dpi=dpi, format="png", facecolor="white")
    outputs["png"] = str(png_path)
    if tiff:
        tif_path = base.with_suffix(".tif")
        fig.savefig(tif_path, dpi=dpi, format="tiff", facecolor="white")
        outputs["tif"] = str(tif_path)
    plt.close(fig)
    return outputs


def plot_kpi_bars_ci(labels: Sequence[str], values: Sequence[float], cis: Sequence[Tuple[float, float]], dpi: int) -> Dict[str, str]:
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6, 3))
    heights = np.asarray(values, dtype=float)
    lower = np.asarray([v[0] for v in cis], dtype=float)
    upper = np.asarray([v[1] for v in cis], dtype=float)
    ax.bar(x, heights, color="#1f77b4")
    ax.errorbar(x, heights, yerr=[heights - lower, upper - heights], fmt="none", ecolor="black", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("KPI")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    return _save(fig, "fig01_kpi_bars", dpi)


def plot_ecdf_cvar(samples: Sequence[float], alpha: float, dpi: int) -> Dict[str, str]:
    data = np.sort(np.asarray(samples, dtype=float))
    probs = np.linspace(0, 1, data.size, endpoint=False)
    cvar = float(np.mean(data[probs >= 1 - alpha])) if data.size else 0.0
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.step(data, probs, where="post", color="#ff7f0e")
    ax.axhline(1 - alpha, color="black", linestyle="--")
    ax.axvline(cvar, color="red", linestyle=":", label=f"CVaR@{alpha:.2f}")
    ax.legend()
    ax.set_xlabel("Metric")
    ax.set_ylabel("ECDF")
    return _save(fig, "fig02_ecdf_cvar", dpi)


def plot_pareto(x: Sequence[float], y: Sequence[float], labels: Sequence[str], dpi: int) -> Dict[str, str]:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(x, y, c="#2ca02c")
    for xi, yi, label in zip(x, y, labels):
        ax.annotate(label, (xi, yi), textcoords="offset points", xytext=(3, 3))
    ax.set_xlabel("Tracking error")
    ax.set_ylabel("Effort")
    ax.grid(True, linestyle=":", alpha=0.5)
    return _save(fig, "fig03_pareto", dpi)


def plot_frequency_panel(freq: np.ndarray, psd: np.ndarray, bands: Iterable[Tuple[float, float]], dpi: int) -> Dict[str, str]:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(freq, 10 * np.log10(np.maximum(psd, 1e-18)), color="#9467bd")
    for low, high in bands:
        ax.axvspan(low, high, color="#9467bd", alpha=0.1)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [dB]")
    ax.set_xlim(freq.min(), freq.max())
    ax.grid(True, linestyle="--", alpha=0.4)
    return _save(fig, "fig04_frequency", dpi)


def plot_ablation_deltas(labels: Sequence[str], deltas: Sequence[float], dpi: int) -> Dict[str, str]:
    order = np.argsort(deltas)
    labels_arr = np.asarray(labels)[order]
    deltas_arr = np.asarray(deltas)[order]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(np.arange(len(labels_arr)), deltas_arr, color="#8c564b")
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_yticks(np.arange(len(labels_arr)))
    ax.set_yticklabels(labels_arr)
    ax.set_xlabel("Δ KPI")
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    return _save(fig, "fig05_ablation", dpi)


def plot_explainability_panel(importances: Dict[str, float], dpi: int) -> Dict[str, str]:
    labels = list(importances.keys())
    values = [importances[k] for k in labels]
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.barh(labels, values, color="#e377c2")
    ax.set_xlabel("Importance")
    ax.set_title("Global importance")
    return _save(fig, "fig06_explainability", dpi)


def plot_symbolic_curve(complexity: Sequence[int], fidelity: Sequence[float], dpi: int) -> Dict[str, str]:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(complexity, fidelity, marker="o", color="#7f7f7f")
    ax.set_xlabel("Complexity")
    ax.set_ylabel("Fidelity (R²)")
    ax.grid(True, linestyle=":", alpha=0.5)
    return _save(fig, "fig07_symbolic", dpi)


def plot_uncertainty_vs_violations(uncertainty: Sequence[float], violations: Sequence[float], dpi: int) -> Dict[str, str]:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.scatter(uncertainty, violations, color="#bcbd22")
    ax.set_xlabel("Uncertainty")
    ax.set_ylabel("Violations")
    ax.grid(True, linestyle=":", alpha=0.5)
    return _save(fig, "fig08_uncertainty", dpi)


__all__ = [
    "plot_kpi_bars_ci",
    "plot_ecdf_cvar",
    "plot_pareto",
    "plot_frequency_panel",
    "plot_ablation_deltas",
    "plot_explainability_panel",
    "plot_symbolic_curve",
    "plot_uncertainty_vs_violations",
]
