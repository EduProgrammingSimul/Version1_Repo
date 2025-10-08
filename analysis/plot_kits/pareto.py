from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def _pareto_front(points: np.ndarray) -> np.ndarray:
    idx = np.argsort(points[:, 0])
    points = points[idx]
    pareto = []
    best_y = float("inf")
    for i, (x, y) in enumerate(points):
        if y < best_y:
            pareto.append(idx[i])
            best_y = y
    return np.array(pareto, dtype=int)


def plot_pareto(df, cfg: dict, path_base: str) -> dict:
    x_metric = cfg.get("x_metric")
    y_metric = cfg.get("y_metric")
    label_col = cfg.get("label", "controller")
    title = cfg.get("title", "Pareto Front")
    xlabel = cfg.get("xlabel", x_metric or "Metric X")
    ylabel = cfg.get("ylabel", y_metric or "Metric Y")

    if x_metric is None or y_metric is None:
        raise ValueError("Pareto plot requires 'x_metric' and 'y_metric' in cfg")

    data = df[[x_metric, y_metric, label_col]].dropna()
    if data.empty:
        raise ValueError("No data available for Pareto plot")

    points = data[[x_metric, y_metric]].to_numpy(dtype=float)
    pareto_idx = _pareto_front(points)

    fig, ax = plt.subplots(figsize=cfg.get("figsize", (7, 5)))
    ax.scatter(
        data[x_metric],
        data[y_metric],
        c=cfg.get("color", "#1f77b4"),
        alpha=0.7,
        label="Candidates",
    )
    pareto_points = data.iloc[pareto_idx]
    ax.plot(
        pareto_points[x_metric],
        pareto_points[y_metric],
        color=cfg.get("front_color", "#d62728"),
        linewidth=2.0,
        label="Pareto front",
    )

    if cfg.get("annotate", True):
        for _, row in data.iterrows():
            ax.annotate(
                str(row[label_col]),
                (row[x_metric], row[y_metric]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.2)
    ax.legend()

    outputs = {}
    formats = cfg.get("formats", ["png", "svg"])
    for ext in formats:
        out_path = f"{path_base}.{ext}"
        fig.savefig(out_path, bbox_inches="tight", dpi=cfg.get("dpi", 300))
        outputs[ext] = out_path
    plt.close(fig)
    return outputs
