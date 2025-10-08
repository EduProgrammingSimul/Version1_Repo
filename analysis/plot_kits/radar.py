import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _resolve_formats(fig_cfg: dict) -> tuple[str, ...]:
    formats = fig_cfg.get("formats")
    if isinstance(formats, (list, tuple)) and formats:
        return tuple(formats)
    return ("png", "svg")


def plot_radar(df: pd.DataFrame, fig_cfg: dict, path_base: str):
    dpi = int(fig_cfg.get("dpi", 300))
    idx_col = fig_cfg.get("index", "controller")
    dims = fig_cfg.get("dimensions", [])
    val_col = fig_cfg.get("value_col", "value_norm")

    controllers = list(df[idx_col].unique())
    if not controllers or not dims:
        raise ValueError(
            "Radar plot requires at least one controller and one dimension"
        )

    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure()
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims)
    ax.set_rlabel_position(0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"])
    ax.set_ylim(0, 1)

    outputs = {}
    for controller in controllers:
        sub = df[df[idx_col] == controller].drop_duplicates(
            subset=["dimension"], keep="last"
        )
        sub = sub.set_index("dimension").reindex(dims)
        values = sub[val_col].astype(float).fillna(0.0).tolist()
        values += values[:1]
        ax.plot(angles, values, label=str(controller))
        ax.fill(angles, values, alpha=0.12)

    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.1))

    for ext in _resolve_formats(fig_cfg):
        out_path = f"{path_base}.{ext}"
        fig.savefig(out_path, bbox_inches="tight", dpi=dpi if ext == "png" else None)
        outputs[ext] = out_path
    plt.close(fig)
    return outputs


def plot_radar_with_ci(df: pd.DataFrame, fig_cfg: dict, path_base: str):
    dpi = int(fig_cfg.get("dpi", 300))
    idx_col = fig_cfg.get("index", "controller")
    dims = fig_cfg.get("dimensions", [])
    mean_col = fig_cfg.get("mean_col", "value_mean")
    low_col = fig_cfg.get("low_col", "value_low")
    high_col = fig_cfg.get("high_col", "value_high")

    controllers = list(df[idx_col].unique())
    if not controllers or not dims:
        raise ValueError("Radar CI plot requires controllers and dimensions")

    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure()
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims)
    ax.set_rlabel_position(0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"])
    ax.set_ylim(0, 1)

    outputs = {}
    for controller in controllers:
        sub = df[df[idx_col] == controller]
        sub = (
            sub.drop_duplicates(subset=["dimension"], keep="last")
            .set_index("dimension")
            .reindex(dims)
        )
        mean_values = sub[mean_col].astype(float).fillna(0.0).tolist()
        low_values = sub[low_col].astype(float).fillna(0.0).tolist()
        high_values = sub[high_col].astype(float).fillna(0.0).tolist()

        mean_values += mean_values[:1]
        low_values += low_values[:1]
        high_values += high_values[:1]

        ax.plot(angles, mean_values, label=str(controller))
        ax.fill_between(angles, low_values, high_values, alpha=0.15)

    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.1))

    for ext in _resolve_formats(fig_cfg):
        out_path = f"{path_base}.{ext}"
        fig.savefig(out_path, bbox_inches="tight", dpi=dpi if ext == "png" else None)
        outputs[ext] = out_path
    plt.close(fig)
    return outputs
