from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_ecdf(df, cfg: dict, path_base: str) -> dict:
    metric = cfg.get("metric")
    group_col = cfg.get("group", "controller")
    title = cfg.get("title", f"ECDF of {metric}")
    xlabel = cfg.get("xlabel", metric or "Value")

    if metric is None:
        raise ValueError("ECDF plot requires 'metric' in cfg")

    groups = df[group_col].unique()
    fig, ax = plt.subplots(figsize=cfg.get("figsize", (7, 5)))
    cmap = plt.get_cmap(cfg.get("cmap", "tab10"))

    for idx, group in enumerate(sorted(groups)):
        subset = df[df[group_col] == group][metric].dropna().to_numpy(dtype=float)
        if subset.size == 0:
            continue
        sorted_vals = np.sort(subset)
        y = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.step(
            sorted_vals, y, where="post", color=cmap(idx % cmap.N), label=str(group)
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cumulative probability")
    ax.grid(True, alpha=0.2)
    if cfg.get("legend", True):
        ax.legend()

    outputs = {}
    formats = cfg.get("formats", ["png", "svg"])
    for ext in formats:
        out_path = f"{path_base}.{ext}"
        fig.savefig(out_path, bbox_inches="tight", dpi=cfg.get("dpi", 300))
        outputs[ext] = out_path
    plt.close(fig)
    return outputs
