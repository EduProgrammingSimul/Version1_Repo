from __future__ import annotations

import matplotlib.pyplot as plt


def plot_timeseries_ci(df, cfg: dict, path_base: str) -> dict:
    time_col = cfg.get("time_col", "time_s")
    mean_col = cfg.get("mean_col", "value_mean")
    low_col = cfg.get("low_col", "value_low")
    high_col = cfg.get("high_col", "value_high")
    group_col = cfg.get("group", "controller")

    if time_col not in df or mean_col not in df:
        raise ValueError("Timeseries CI plot requires columns for time and mean")

    fig, ax = plt.subplots(figsize=cfg.get("figsize", (8, 4.5)))
    cmap = plt.get_cmap(cfg.get("cmap", "tab10"))

    for idx, (group, sub) in enumerate(df.groupby(group_col)):
        sub = sub.sort_values(time_col)
        ax.plot(
            sub[time_col], sub[mean_col], color=cmap(idx % cmap.N), label=str(group)
        )
        if low_col in sub and high_col in sub:
            ax.fill_between(
                sub[time_col],
                sub[low_col],
                sub[high_col],
                color=cmap(idx % cmap.N),
                alpha=0.2,
            )

    ax.set_title(cfg.get("title", "Time-series with confidence interval"))
    ax.set_xlabel(cfg.get("xlabel", time_col))
    ax.set_ylabel(cfg.get("ylabel", mean_col))
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
