from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import pandas as pd
import yaml

from analysis.plot_kits.radar import plot_radar, plot_radar_with_ci
from analysis.plot_kits.heatmap import plot_heatmap, plot_corr_heatmap
from analysis.plot_kits.scatter import plot_scatter
from analysis.plot_kits.pareto import plot_pareto
from analysis.plot_kits.ecdf import plot_ecdf
from analysis.plot_kits.timeseries import plot_timeseries_ci
from analysis.plot_kits.manifold import plot_manifold_confidence
import matplotlib.pyplot as plt


@dataclass
class FigureTask:
    id: str
    kind: str
    cfg: Dict[str, Any]


def _resolve_path(base_out: str, path_entry: str) -> str:
    if os.path.isabs(path_entry):
        return path_entry
    if path_entry.startswith("results/"):
        rel = path_entry.split("/", 1)[1]
        return os.path.join(base_out, rel)
    return os.path.join(base_out, path_entry)


def _load_source(task: FigureTask, out_root: str, validation_root: str) -> pd.DataFrame:
    source_type = task.cfg.get("source", "csv").lower()
    path = task.cfg.get("path")
    if source_type == "csv":
        if not path:
            raise ValueError(f"Figure {task.id} missing CSV path")
        data_path = _resolve_path(out_root, path)
        return pd.read_csv(data_path)
    if source_type == "parquet":
        if not path:
            raise ValueError(f"Figure {task.id} missing Parquet path")
        data_path = _resolve_path(out_root, path)
        try:
            return pd.read_parquet(data_path)
        except ImportError as exc:
            raise ImportError(
                "Parquet support requires the 'pyarrow' dependency. Install pyarrow to render figure "
                f"{task.id}."
            ) from exc
    if source_type == "timeseries":
        frames: List[pd.DataFrame] = []
        for ctrl in sorted(os.listdir(validation_root)):
            ctrl_dir = os.path.join(validation_root, ctrl)
            if not os.path.isdir(ctrl_dir):
                continue
            for scen in sorted(os.listdir(ctrl_dir)):
                csv_path = os.path.join(ctrl_dir, scen, "timeseries.csv")
                if not os.path.isfile(csv_path):
                    continue
                df = pd.read_csv(csv_path)
                df["controller"] = ctrl
                df["scenario"] = scen
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    raise ValueError(f"Unsupported source type '{source_type}' for figure {task.id}")


def _ensure_figure_dir(base_out: str) -> str:
    out_dir = os.path.join(base_out, "figures")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _plot_metric_summary(
    df: pd.DataFrame, task: FigureTask, path_base: str
) -> Dict[str, str]:
    metric_name = task.cfg.get("metric")
    if metric_name is None:
        raise ValueError(f"Figure {task.id} requires 'metric' in config")
    subset = df[df["metric"] == metric_name]
    if subset.empty:
        raise ValueError(f"Metric {metric_name} not present for figure {task.id}")
    summary = subset.groupby("controller")["value"].mean().sort_values()
    dpi = int(task.cfg.get("dpi", 300))
    fig, ax = plt.subplots()
    summary.plot(kind="bar", ax=ax, color=task.cfg.get("color", "#1f77b4"))
    ax.set_ylabel(task.cfg.get("ylabel", metric_name))
    ax.set_xlabel("Controller")
    ax.set_title(task.cfg.get("title", metric_name))
    ax.grid(True, axis="y", alpha=0.2)
    outputs = {}
    formats = task.cfg.get("formats", ["png", "svg"])
    if "png" in formats:
        outputs["png"] = f"{path_base}.png"
        fig.savefig(outputs["png"], bbox_inches="tight", dpi=dpi)
    if "svg" in formats:
        outputs["svg"] = f"{path_base}.svg"
        fig.savefig(outputs["svg"], bbox_inches="tight")
    plt.close(fig)
    return outputs


def _plot_action_stats(
    df: pd.DataFrame, task: FigureTask, path_base: str
) -> Dict[str, str]:
    if "u" not in df.columns:
        raise ValueError("Timeseries missing 'u' column for action stats figure")
    dpi = int(task.cfg.get("dpi", 300))
    fig, ax = plt.subplots()
    controllers = sorted(df["controller"].unique())
    data = [
        df[df["controller"] == ctrl]["u"].astype(float).dropna().to_numpy()
        for ctrl in controllers
    ]
    ax.boxplot(data, labels=controllers, showmeans=True)
    ax.set_ylabel("Action (u)")
    ax.set_title(task.cfg.get("title", "Action distribution"))
    ax.grid(True, axis="y", alpha=0.2)
    outputs = {}
    formats = task.cfg.get("formats", ["png", "svg"])
    if "png" in formats:
        outputs["png"] = f"{path_base}.png"
        fig.savefig(outputs["png"], bbox_inches="tight", dpi=dpi)
    if "svg" in formats:
        outputs["svg"] = f"{path_base}.svg"
        fig.savefig(outputs["svg"], bbox_inches="tight")
    plt.close(fig)
    return outputs


def _plot_tornado(df: pd.DataFrame, task: FigureTask, path_base: str) -> Dict[str, str]:
    baseline = task.cfg.get("baseline")
    challenger = task.cfg.get("challenger")
    if baseline is None or challenger is None:
        raise ValueError(f"Figure {task.id} requires 'baseline' and 'challenger'")
    dpi = int(task.cfg.get("dpi", 300))
    pivot = df.pivot_table(
        index="metric", columns="controller", values="value", aggfunc="mean"
    )
    if baseline not in pivot.columns or challenger not in pivot.columns:
        raise ValueError(
            f"Controllers {baseline}/{challenger} missing for tornado figure {task.id}"
        )
    diff = (pivot[challenger] - pivot[baseline]).sort_values()
    fig, ax = plt.subplots()
    colors = ["#d62728" if val < 0 else "#2ca02c" for val in diff]
    ax.barh(diff.index, diff.values, color=colors)
    ax.set_xlabel(f"{challenger} - {baseline}")
    ax.set_title(task.cfg.get("title", "Controller delta by metric"))
    ax.axvline(0.0, color="black", linewidth=0.8)
    plt.tight_layout()
    outputs = {}
    formats = task.cfg.get("formats", ["png", "svg"])
    if "png" in formats:
        outputs["png"] = f"{path_base}.png"
        fig.savefig(outputs["png"], bbox_inches="tight", dpi=dpi)
    if "svg" in formats:
        outputs["svg"] = f"{path_base}.svg"
        fig.savefig(outputs["svg"], bbox_inches="tight")
    plt.close(fig)
    return outputs


def render_all(
    registry_path: str,
    out_root: str,
    eval_seed: int,
    validation_root: str | None = None,
) -> List[Dict[str, Any]]:
    if validation_root is None:
        validation_root = os.path.join(out_root, "validation")
    if not os.path.isfile(registry_path):
        raise FileNotFoundError(f"Figure registry not found: {registry_path}")

    with open(registry_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    figures = [
        FigureTask(id=item["id"], kind=item["kind"], cfg=item)
        for item in payload.get("figures", [])
    ]
    results: List[Dict[str, Any]] = []
    figure_dir = _ensure_figure_dir(out_root)

    for task in figures:
        data = _load_source(task, out_root, validation_root)
        if data.empty:
            raise ValueError(f"Figure {task.id} has no data to plot")
        path_base = os.path.join(figure_dir, task.id)
        kind = task.kind.lower()
        if kind == "radar":
            outputs = plot_radar(data, task.cfg, path_base)
        elif kind == "radar_ci":
            outputs = plot_radar_with_ci(data, task.cfg, path_base)
        elif kind == "heatmap":
            outputs = plot_heatmap(data, task.cfg, path_base)
        elif kind == "corr_heatmap":
            outputs = plot_corr_heatmap(data, task.cfg, path_base)
        elif kind == "scatter":
            outputs = plot_scatter(data, task.cfg, path_base)
        elif kind == "metric_summary":
            outputs = _plot_metric_summary(data, task, path_base)
        elif kind == "action_stats":
            outputs = _plot_action_stats(data, task, path_base)
        elif kind == "tornado":
            outputs = _plot_tornado(data, task, path_base)
        elif kind == "pareto":
            outputs = plot_pareto(data, task.cfg, path_base)
        elif kind == "ecdf":
            outputs = plot_ecdf(data, task.cfg, path_base)
        elif kind == "timeseries_ci":
            outputs = plot_timeseries_ci(data, task.cfg, path_base)
        elif kind == "policy_manifold":
            outputs = plot_scatter(data, task.cfg, path_base)
        elif kind == "manifold_confidence":
            outputs = plot_manifold_confidence(data, task.cfg, path_base)
        else:
            raise ValueError(f"Unsupported figure kind '{task.kind}'")

        meta_path = os.path.join(figure_dir, f"{task.id}.meta.json")
        meta = {
            "figure_id": task.id,
            "kind": task.kind,
            "inputs": task.cfg,
            "outputs": outputs,
            "eval_seed": int(eval_seed),
        }
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
        meta["meta_path"] = meta_path
        results.append(meta)
    return results


__all__ = ["render_all"]
