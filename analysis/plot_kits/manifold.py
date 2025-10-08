from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def _cov_ellipse(cov: np.ndarray, mean: np.ndarray, n_std: float = 1.0):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    return width, height, angle


def plot_manifold_confidence(df, cfg: dict, path_base: str) -> dict:
    x_col = cfg.get("x_col", "pc1")
    y_col = cfg.get("y_col", "pc2")
    group_col = cfg.get("group", "controller")
    n_std = float(cfg.get("n_std", 1.0))

    fig, ax = plt.subplots(figsize=cfg.get("figsize", (6, 6)))
    cmap = plt.get_cmap(cfg.get("cmap", "tab10"))

    outputs = {}
    for idx, (group, sub) in enumerate(df.groupby(group_col)):
        x = sub[x_col].astype(float).to_numpy()
        y = sub[y_col].astype(float).to_numpy()
        ax.scatter(x, y, label=str(group), alpha=0.6, color=cmap(idx % cmap.N))
        if len(sub) >= 3:
            cov = np.cov(np.vstack([x, y]))
            mean = np.array([np.nanmean(x), np.nanmean(y)])
            width, height, angle = _cov_ellipse(cov, mean, n_std)
            ellipse = Ellipse(
                mean,
                width,
                height,
                angle=angle,
                edgecolor=cmap(idx % cmap.N),
                facecolor="none",
                linewidth=1.5,
            )
            ax.add_patch(ellipse)

    ax.set_title(cfg.get("title", "Policy manifold"))
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.2)
    if cfg.get("legend", True):
        ax.legend()

    for ext in cfg.get("formats", ["png", "svg"]):
        out_path = f"{path_base}.{ext}"
        fig.savefig(out_path, bbox_inches="tight", dpi=cfg.get("dpi", 300))
        outputs[ext] = out_path
    plt.close(fig)
    return outputs
