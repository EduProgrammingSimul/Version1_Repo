import pandas as pd
import matplotlib.pyplot as plt


def plot_scatter(
    df: pd.DataFrame, fig_cfg: dict, path_base: str, formats=("png", "svg")
):
    dpi = int(fig_cfg.get("dpi", 300))
    x = fig_cfg.get("x", "pc1")
    y = fig_cfg.get("y", "pc2")
    hue = fig_cfg.get("hue", "controller")

    fig, ax = plt.subplots()
    for key, grp in df.groupby(hue):
        ax.scatter(
            grp[x], grp[y], label=str(key), s=fig_cfg.get("point_size", 16), alpha=0.8
        )
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend()
    ax.grid(True, alpha=0.2)

    out = {}
    if "png" in formats:
        out["png"] = f"{path_base}.png"
        plt.savefig(out["png"], bbox_inches="tight", dpi=dpi)
    if "svg" in formats:
        out["svg"] = f"{path_base}.svg"
        plt.savefig(out["svg"], bbox_inches="tight")
    plt.close(fig)
    return out
