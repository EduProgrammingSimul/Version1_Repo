from __future__ import annotations
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_DEFAULTS = {
    "figure.dpi": 300,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "lines.linewidth": 1.8,
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def apply_style(presets_path: str | None = None) -> None:

    for k, v in _DEFAULTS.items():
        try:
            matplotlib.rcParams[k] = v
        except Exception:
            pass

    if presets_path and os.path.isfile(presets_path):
        try:
            import yaml

            with open(presets_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            if isinstance(cfg, dict):
                dpi = int(cfg.get("dpi", _DEFAULTS["figure.dpi"]))
                matplotlib.rcParams["figure.dpi"] = dpi
                font = cfg.get("font", {})
                if isinstance(font, dict):
                    if "family" in font:
                        matplotlib.rcParams["font.family"] = font["family"]
                    if "size" in font:
                        matplotlib.rcParams["font.size"] = font["size"]
                lines = cfg.get("lines", {})
                if isinstance(lines, dict):
                    if "linewidth" in lines:
                        matplotlib.rcParams["lines.linewidth"] = lines["linewidth"]
        except Exception:

            pass


def savefig(path_wo_ext: str) -> list[str]:

    png = path_wo_ext + ".png"
    svg = path_wo_ext + ".svg"
    plt.savefig(png, bbox_inches="tight", dpi=300)
    plt.savefig(svg, bbox_inches="tight")
    plt.close()
    return [png, svg]


class VisualizationEngine:

    def __init__(self, out_dir: str | None = None, presets_path: str | None = None):
        self.out_dir = out_dir or "results/figures"
        self.presets_path = presets_path
        ensure_dir(self.out_dir)
        apply_style(presets_path)

    def set_output(self, out_dir: str) -> None:
        self.out_dir = out_dir
        ensure_dir(self.out_dir)

    def style(self, presets_path: str | None = None) -> None:
        apply_style(presets_path or self.presets_path)

    def close_all(self) -> None:
        plt.close("all")

    def save(self, name_wo_ext: str, fig=None, subdir: str | None = None) -> list[str]:

        target = self.out_dir if not subdir else os.path.join(self.out_dir, subdir)
        ensure_dir(target)
        path_wo = os.path.join(target, name_wo_ext)
        if fig is not None:
            fig.savefig(path_wo + ".png", bbox_inches="tight", dpi=300)
            fig.savefig(path_wo + ".svg", bbox_inches="tight")
            plt.close(fig)
        else:
            savefig(path_wo)
        return [path_wo + ".png", path_wo + ".svg"]

    def savefig(self, name_wo_ext: str, subdir: str | None = None) -> list[str]:

        target = self.out_dir if not subdir else os.path.join(self.out_dir, subdir)
        ensure_dir(target)
        return savefig(os.path.join(target, name_wo_ext))

    def new_figure(self):
        return plt.figure()

    def subplot(self, *args, **kwargs):
        return plt.subplot(*args, **kwargs)

    def legend(self, *args, **kwargs):
        return plt.legend(*args, **kwargs)

    def title(self, *args, **kwargs):
        return plt.title(*args, **kwargs)

    def xlabel(self, *args, **kwargs):
        return plt.xlabel(*args, **kwargs)

    def ylabel(self, *args, **kwargs):
        return plt.ylabel(*args, **kwargs)
