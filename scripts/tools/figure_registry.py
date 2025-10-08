from __future__ import annotations
import os, glob, json, math
from typing import List, Optional, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from analysis.visualization_engine import apply_style, ensure_dir


def _read_tidy_csv(path: str) -> Optional[pd.DataFrame]:
    if not path or not os.path.isfile(path):
        return None
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        try:
            df = pd.read_csv(path)
        except Exception:
            return None

    def _norm(s: str) -> str:
        return s.replace("\ufeff", "").strip().lower()

    df.rename(columns={c: _norm(c) for c in df.columns}, inplace=True)
    mapping: Dict[str, str] = {}
    for c in list(df.columns):
        if c in ("controller", "ctrl", "agent"):
            mapping[c] = "controller"
        elif c in ("scenario", "case"):
            mapping[c] = "scenario"
        elif c in ("metric", "metric_name", "name"):
            mapping[c] = "metric"
        elif c in ("value", "val", "score"):
            mapping[c] = "value"
        elif c in ("higher_is_better", "hib"):
            mapping[c] = "higher_is_better"
    if mapping:
        df.rename(columns=mapping, inplace=True)
    return df


def _safe_read_csv(p: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(p, sep=None, engine="python")
    except Exception:
        try:
            return pd.read_csv(p)
        except Exception:
            return None


def _first_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in cols:
            return cols[c]
    for k, v in cols.items():
        for c in candidates:
            if c in k:
                return v
    return None


def _metric_orientation(df: pd.DataFrame) -> Dict[str, bool]:
    orient: Dict[str, bool] = {}
    if "metric" not in df.columns:
        return orient
    if "higher_is_better" in df.columns:
        tmp = df.dropna(subset=["metric", "higher_is_better"]).drop_duplicates(
            subset=["metric"]
        )
        for _, row in tmp.iterrows():
            try:
                orient[row["metric"]] = bool(row["higher_is_better"])
            except Exception:
                pass
    return orient


def _normalize_scores(series: pd.Series, higher_is_better: bool) -> pd.Series:
    x = series.astype(float)
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    if math.isclose(xmin, xmax):
        return pd.Series(np.full_like(x, 0.5, dtype=float), index=series.index)
    return (
        (x - xmin) / (xmax - xmin) if higher_is_better else (xmax - x) / (xmax - xmin)
    )


def _plot_metric_bars_all(df: pd.DataFrame, out_dir: str) -> List[str]:
    ensure_dir(out_dir)
    figs: List[str] = []
    if (
        df is None
        or df.empty
        or not {"controller", "metric", "value"}.issubset(df.columns)
    ):
        return figs
    for m in sorted(df["metric"].unique()):
        sub = df[df["metric"] == m].groupby("controller")["value"].mean().sort_values()
        if len(sub) == 0:
            continue
        apply_style()
        ax = sub.plot(kind="bar")
        hib = None
        if "higher_is_better" in df.columns:
            tmp = df[df["metric"] == m]
            if not tmp.empty and tmp["higher_is_better"].notna().any():
                hib = bool(tmp["higher_is_better"].iloc[0])
        suffix = (
            " (higher is better)"
            if hib
            else " (lower is better)" if hib is not None else ""
        )
        ax.set_title(f"Mean {m} by controller{suffix}")
        ax.set_xlabel("controller")
        ax.set_ylabel(m)
        outp = os.path.join(out_dir, f"{m}_by_controller.png")
        plt.savefig(outp)
        plt.close()
        figs.append(outp)
    return figs


def _plot_metric_bars_per_scenario(df: pd.DataFrame, out_dir: str) -> List[str]:
    ensure_dir(out_dir)
    figs: List[str] = []
    if (
        df is None
        or df.empty
        or not {"controller", "metric", "value", "scenario"}.issubset(df.columns)
    ):
        return figs
    for sc in sorted(df["scenario"].unique()):
        df_sc = df[df["scenario"] == sc]
        for m in sorted(df_sc["metric"].unique()):
            sub = (
                df_sc[df_sc["metric"] == m]
                .groupby("controller")["value"]
                .mean()
                .sort_values()
            )
            if len(sub) == 0:
                continue
            apply_style()
            ax = sub.plot(kind="bar")
            ax.set_title(f"{sc}: {m} by controller")
            ax.set_xlabel("controller")
            ax.set_ylabel(m)
            outp = os.path.join(out_dir, f"{sc}_{m}_by_controller.png")
            plt.savefig(outp)
            plt.close()
            figs.append(outp)
    return figs


def _plot_metric_distributions(df: pd.DataFrame, out_dir: str) -> List[str]:
    ensure_dir(out_dir)
    figs: List[str] = []
    if (
        df is None
        or df.empty
        or not {"controller", "metric", "value"}.issubset(df.columns)
    ):
        return figs
    for m in sorted(df["metric"].unique()):
        sub = df[df["metric"] == m]
        groups = [g["value"].dropna().to_numpy() for _, g in sub.groupby("controller")]
        labels = [str(k) for k, _ in sub.groupby("controller")]
        if not groups or sum(len(g) for g in groups) == 0:
            continue
        apply_style()
        fig, ax = plt.subplots()
        ax.boxplot(groups, labels=labels, showfliers=False)
        ax.set_title(f"{m} distribution by controller")
        ax.set_ylabel(m)
        outp = os.path.join(out_dir, f"{m}_distribution_by_controller.png")
        plt.savefig(outp)
        plt.close(fig)
        figs.append(outp)
    return figs


def _plot_cross_metric_scatter(
    df: pd.DataFrame,
    out_dir: str,
    x_metric: str = "total_time_unsafe_s",
    y_metric: str = "grid_load_following_index",
) -> List[str]:
    ensure_dir(out_dir)
    figs: List[str] = []
    req = {"controller", "metric", "value", "scenario"}
    if df is None or df.empty or not req.issubset(df.columns):
        return figs
    have = set(df["metric"].unique())
    if x_metric not in have or y_metric not in have:
        return figs
    sub = df[df["metric"].isin([x_metric, y_metric])]
    wide = sub.pivot_table(
        index=["scenario", "controller"],
        columns="metric",
        values="value",
        aggfunc="mean",
    ).reset_index()
    if wide.empty or x_metric not in wide.columns or y_metric not in wide.columns:
        return figs
    apply_style()
    fig, ax = plt.subplots()
    for ctrl, g in wide.groupby("controller"):
        ax.scatter(g[x_metric], g[y_metric], label=str(ctrl), alpha=0.9)
    ax.set_xlabel(x_metric + " (lower is better)")
    ax.set_ylabel(y_metric + " (higher is better)")
    ax.set_title(f"{y_metric} vs {x_metric} per scenario")
    ax.legend()
    outp = os.path.join(out_dir, f"scatter_{y_metric}_vs_{x_metric}.png")
    plt.savefig(outp)
    plt.close(fig)
    figs.append(outp)
    return figs


def _plot_frequency_deviation_hist(df: pd.DataFrame, out_dir: str) -> List[str]:
    ensure_dir(out_dir)
    figs: List[str] = []
    req = {"controller", "metric", "value"}
    if df is None or df.empty or not req.issubset(df.columns):
        return figs
    if "max_freq_deviation_hz" not in set(df["metric"].unique()):
        return figs
    sub = df[df["metric"] == "max_freq_deviation_hz"]
    apply_style()
    fig, ax = plt.subplots()
    for ctrl, g in sub.groupby("controller"):
        ax.hist(g["value"].dropna().to_numpy(), bins=20, alpha=0.5, label=str(ctrl))
    ax.set_title("Max frequency deviation histogram")
    ax.set_xlabel("max_freq_deviation_hz")
    ax.set_ylabel("count")
    ax.legend()
    outp = os.path.join(out_dir, "hist_max_freq_deviation_hz.png")
    plt.savefig(outp)
    plt.close(fig)
    figs.append(outp)
    return figs


def _plot_radar(
    df: pd.DataFrame, out_dir: str, per_scenario: bool = False, max_metrics: int = 10
) -> List[str]:
    ensure_dir(out_dir)
    figs: List[str] = []
    req = {"controller", "metric", "value"}
    if df is None or df.empty or not req.issubset(df.columns):
        return figs
    orient = _metric_orientation(df)
    metrics = sorted(df["metric"].unique())
    if len(metrics) < 3:
        return figs
    metrics = metrics[:max_metrics]

    def _make_radar(data: pd.DataFrame, title: str, fname: str):
        mat = []
        labels = []
        for ctrl, g in data.groupby("controller"):
            row = []
            for m in metrics:
                vals = g[g["metric"] == m]["value"]
                if vals.empty:
                    row.append(np.nan)
                else:
                    hib = orient.get(m, False)
                    pivot = (
                        data[data["metric"] == m].groupby("controller")["value"].mean()
                    )
                    norm = _normalize_scores(pivot, hib)
                    row.append(float(norm.get(ctrl, np.nan)))
            if all(np.isnan(row)):
                return
            mat.append(row)
            labels.append(str(ctrl))
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        apply_style()
        fig = plt.figure()
        ax = plt.subplot(111, polar=True)
        for i, ctrl in enumerate(labels):
            vals = mat[i]
            vals = [0.5 if (v is None or np.isnan(v)) else v for v in vals]
            vals += vals[:1]
            ax.plot(angles, vals, label=ctrl, alpha=0.9)
            ax.fill(angles, vals, alpha=0.1)
        ax.set_xticks(np.linspace(0, 2 * np.pi, len(metrics), endpoint=False))
        ax.set_xticklabels(metrics, fontsize=8)
        ax.set_yticklabels([])
        ax.set_title(title)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        outp = os.path.join(out_dir, fname)
        plt.savefig(outp, bbox_inches="tight")
        plt.close(fig)
        figs.append(outp)

    if per_scenario and "scenario" in df.columns:
        for sc in sorted(df["scenario"].unique()):
            sub = df[df["scenario"] == sc]
            _make_radar(
                sub, title=f"{sc}: normalized metrics radar", fname=f"radar_{sc}.png"
            )
    else:
        agg = df.groupby(["controller", "metric"])["value"].mean().reset_index()
        _make_radar(
            agg, title="Overall normalized metrics radar", fname="radar_overall.png"
        )
    return figs


def _plot_tornado(
    df: pd.DataFrame, out_dir: str, baseline: str = "PID", challenger: str = "RL"
) -> List[str]:
    ensure_dir(out_dir)
    figs: List[str] = []
    req = {"controller", "metric", "value"}
    if df is None or df.empty or not req.issubset(df.columns):
        return figs
    orient = _metric_orientation(df)
    agg = df.groupby(["controller", "metric"])["value"].mean().reset_index()
    metrics = sorted(agg["metric"].unique())
    if not metrics:
        return figs
    diffs = []
    for m in metrics:
        sub = agg[agg["metric"] == m].set_index("controller")["value"]
        hib = orient.get(m, False)
        norm = _normalize_scores(sub, hib)
        b = float(norm.get(baseline, np.nan))
        c = float(norm.get(challenger, np.nan))
        if not (np.isnan(b) or np.isnan(c)):
            diffs.append((m, c - b))
    if not diffs:
        return figs
    diffs.sort(key=lambda x: abs(x[1]), reverse=True)
    labels = [m for m, _ in diffs]
    values = [v for _, v in diffs]
    apply_style()
    fig, ax = plt.subplots()
    y = np.arange(len(labels))
    ax.barh(y, values)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.axvline(0.0, linestyle="--")
    ax.set_xlabel(f"Normalized improvement ({challenger} - {baseline})")
    ax.set_title(f"Tornado: {challenger} vs {baseline} (mean over scenarios)")
    plt.tight_layout()
    outp = os.path.join(out_dir, f"tornado_{challenger}_vs_{baseline}.png")
    plt.savefig(outp)
    plt.close(fig)
    figs.append(outp)
    return figs


def _plot_timeseries_overlays(
    timeseries_root: str, out_dir: str, coverage_log: Dict
) -> List[str]:
    ensure_dir(out_dir)
    figures: List[str] = []
    pattern = os.path.join(timeseries_root, "*", "*", "timeseries.csv")
    files = glob.glob(pattern)
    if not files:
        coverage_log["timeseries_found"] = 0
        return figures
    index: dict[str, dict[str, str]] = {}
    for p in files:
        parts = p.split(os.sep)
        try:
            scenario = parts[-2]
            controller = parts[-3]
        except Exception:
            continue
        index.setdefault(scenario, {})[controller] = p
    coverage_log["timeseries_found"] = len(files)
    coverage_log["timeseries_skips"] = {}

    def _pick(df, names):
        cols = {c.lower(): c for c in df.columns}
        for n in names:
            if n in cols:
                return cols[n]
        for k, v in cols.items():
            for n in names:
                if n in k:
                    return v
        return None

    for scenario, ctrl_map in sorted(index.items()):
        apply_style()
        have_any = False
        skips = []
        for controller, csvp in sorted(ctrl_map.items()):
            df = _safe_read_csv(csvp)
            if df is None:
                skips.append({"controller": controller, "reason": "csv_unreadable"})
                continue
            df.columns = [c.lower() for c in df.columns]
            t = _pick(df, ["time_s", "t", "time"])
            y = _pick(
                df,
                [
                    "power_mw",
                    "core_power_mw",
                    "electric_power_mw",
                    "p_mw",
                    "mech_power_mw",
                    "mechanical_power_mw",
                    "power",
                ],
            )
            if not t or not y:
                skips.append(
                    {
                        "controller": controller,
                        "reason": f"missing_columns (t={bool(t)}, y={bool(y)})",
                    }
                )
                continue
            plt.plot(df[t], df[y], label=controller, alpha=0.95)
            have_any = True
        if have_any:
            plt.title(f"{scenario}: Power trajectory")
            plt.xlabel("time [s]")
            plt.ylabel("power [MW]")
            plt.legend()
            outp = os.path.join(out_dir, f"{scenario}_power_overlay.png")
            plt.savefig(outp)
            plt.close()
            figures.append(outp)
        if skips:
            coverage_log.setdefault("timeseries_skips", {}).setdefault(scenario, {})[
                "power"
            ] = skips
        apply_style()
        have_any = False
        skips = []
        for controller, csvp in sorted(ctrl_map.items()):
            df = _safe_read_csv(csvp)
            if df is None:
                skips.append({"controller": controller, "reason": "csv_unreadable"})
                continue
            df.columns = [c.lower() for c in df.columns]
            t = _pick(df, ["time_s", "t", "time"])
            y = _pick(
                df, ["freq_hz", "frequency_hz", "grid_freq_hz", "frequency", "freq"]
            )
            if not t or not y:
                skips.append(
                    {
                        "controller": controller,
                        "reason": f"missing_columns (t={bool(t)}, y={bool(y)})",
                    }
                )
                continue
            plt.plot(df[t], df[y], label=controller, alpha=0.95)
            have_any = True
        if have_any:
            plt.title(f"{scenario}: Frequency trajectory")
            plt.xlabel("time [s]")
            plt.ylabel("frequency [Hz]")
            plt.legend()
            outp = os.path.join(out_dir, f"{scenario}_frequency_overlay.png")
            plt.savefig(outp)
            plt.close()
            figures.append(outp)
        if skips:
            coverage_log.setdefault("timeseries_skips", {}).setdefault(scenario, {})[
                "frequency"
            ] = skips
        apply_style()
        have_any = False
        skips = []
        for controller, csvp in sorted(ctrl_map.items()):
            df = _safe_read_csv(csvp)
            if df is None:
                skips.append({"controller": controller, "reason": "csv_unreadable"})
                continue
            df.columns = [c.lower() for c in df.columns]
            t = _pick(df, ["time_s", "t", "time"])
            y = _pick(
                df,
                [
                    "control_valve",
                    "valve",
                    "u_valve",
                    "u",
                    "action_cmd",
                    "cmd",
                    "actuation",
                ],
            )
            if not t or not y:
                skips.append(
                    {
                        "controller": controller,
                        "reason": f"missing_columns (t={bool(t)}, y={bool(y)})",
                    }
                )
                continue
            plt.plot(df[t], df[y], label=controller, alpha=0.95)
            have_any = True
        if have_any:
            plt.title(f"{scenario}: Control effort overlay")
            plt.xlabel("time [s]")
            plt.ylabel("control signal [arb]")
            plt.legend()
            outp = os.path.join(out_dir, f"{scenario}_control_overlay.png")
            plt.savefig(outp)
            plt.close()
            figures.append(outp)
        if skips:
            coverage_log.setdefault("timeseries_skips", {}).setdefault(scenario, {})[
                "control"
            ] = skips
    return figures


def render_all(
    out_dir: str,
    combined_csv: str,
    timeseries_root: str,
    render_all_metric_bars: bool = True,
    render_per_scenario_metric_bars: bool = True,
    render_distributions: bool = True,
    render_scatter: bool = True,
    render_histograms: bool = True,
    render_radar: bool = True,
    render_radar_per_scenario: bool = False,
    render_tornado: bool = True,
    tornado_baseline: str = "PID",
    tornado_challenger: str = "RL",
    seed: int | None = None,
) -> list[str]:
    ensure_dir(out_dir)
    figs: list[str] = []
    coverage_log: Dict = {}
    try:
        import random as _random

        if seed is not None:
            _random.seed(int(seed))
            np.random.seed(int(seed))
    except Exception:
        pass
    tidy = _read_tidy_csv(combined_csv) if combined_csv else None
    if tidy is not None and not tidy.empty:
        if render_all_metric_bars:
            figs += _plot_metric_bars_all(
                tidy, out_dir=os.path.join(out_dir, "metrics_all")
            )
        if render_per_scenario_metric_bars:
            figs += _plot_metric_bars_per_scenario(
                tidy, out_dir=os.path.join(out_dir, "metrics_per_scenario")
            )
        if render_distributions:
            figs += _plot_metric_distributions(
                tidy, out_dir=os.path.join(out_dir, "metrics_distributions")
            )
        if render_scatter:
            figs += _plot_cross_metric_scatter(
                tidy, out_dir=os.path.join(out_dir, "metrics_scatter")
            )
        if render_histograms:
            figs += _plot_frequency_deviation_hist(
                tidy, out_dir=os.path.join(out_dir, "metrics_hist")
            )
        if render_radar:
            figs += _plot_radar(tidy, out_dir=os.path.join(out_dir, "metrics_radar"))
        if render_radar_per_scenario:
            figs += _plot_radar(
                tidy,
                out_dir=os.path.join(out_dir, "metrics_radar_per_scenario"),
                per_scenario=True,
            )
        if render_tornado:
            figs += _plot_tornado(
                tidy,
                out_dir=os.path.join(out_dir, "metrics_tornado"),
                baseline=tornado_baseline,
                challenger=tornado_challenger,
            )
    figs += _plot_timeseries_overlays(
        timeseries_root=timeseries_root,
        out_dir=os.path.join(out_dir, "timeseries"),
        coverage_log=coverage_log,
    )
    with open(os.path.join(out_dir, "_coverage.json"), "w", encoding="utf-8") as f:
        json.dump({"eval_seed": seed, **coverage_log}, f, indent=2)
    return figs
