import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _save(fig, path_base: str, formats=("png", "svg"), dpi: int = 300):
    out = {}
    if "png" in formats:
        out["png"] = f"{path_base}.png"
        fig.savefig(out["png"], bbox_inches="tight", dpi=dpi)
    if "svg" in formats:
        out["svg"] = f"{path_base}.svg"
        fig.savefig(out["svg"], bbox_inches="tight")
    plt.close(fig)
    return out


def plot_heatmap(
    df: pd.DataFrame, fig_cfg: dict, path_base: str, formats=("png", "svg")
):
    dpi = int(fig_cfg.get("dpi", 300))
    rows = fig_cfg.get("rows", "scenario")
    cols = fig_cfg.get("cols", "controller")
    val = fig_cfg.get("value_col", "score_norm")
    annotate = bool(fig_cfg.get("annotate", False))

    grouped = df.groupby([rows, cols])[val].mean().reset_index()
    pivot = grouped.pivot(index=rows, columns=cols, values=val).sort_index()
    data = pivot.values.astype(float)

    fig, ax = plt.subplots(
        figsize=(max(6, data.shape[1] * 0.6), max(6, data.shape[0] * 0.6))
    )
    im = ax.imshow(data, aspect="auto", cmap=fig_cfg.get("cmap", "viridis"))
    ax.set_xticks(np.arange(data.shape[1]), labels=list(pivot.columns))
    ax.set_yticks(np.arange(data.shape[0]), labels=list(pivot.index))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    if annotate:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{data[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if data[i, j] > np.nanmean(data) else "black",
                )
    fig.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout()
    return _save(fig, path_base, formats=formats, dpi=dpi)


def plot_corr_heatmap(
    df: pd.DataFrame, fig_cfg: dict, path_base: str, formats=("png", "svg")
):
    dpi = int(fig_cfg.get("dpi", 300))
    metrics = sorted(set(df["metric_i"]).union(set(df["metric_j"])))
    if not metrics:
        raise ValueError("No metrics available for correlation heatmap")
    index = {m: i for i, m in enumerate(metrics)}
    size = len(metrics)
    mat = np.eye(size)
    for _, row in df.iterrows():
        i = index[row["metric_i"]]
        j = index[row["metric_j"]]
        val = float(row["r"])
        mat[i, j] = val
        mat[j, i] = val

    mask = np.tril_indices_from(mat, k=-1)
    mat[mask] = np.nan

    cmap_name = fig_cfg.get("cmap", "coolwarm")
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="#f5f5f5")

    fig_size = max(7, size * 0.7)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(mat, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(np.arange(size), labels=metrics)
    ax.set_yticks(np.arange(size), labels=metrics)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(size):
        for j in range(size):
            value = mat[i, j]
            if np.isnan(value):
                continue
            color = "white" if abs(value) > 0.5 else "#1a1a1a"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
    ax.set_title(fig_cfg.get("title", "Metric Correlation"))
    plt.tight_layout()
    return _save(fig, path_base, formats=formats, dpi=dpi)
