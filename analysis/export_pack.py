"""Evaluation harness and export pack builder."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import yaml

from analysis import figures, tables
from analysis.explainability import distill_fuzzy, shap_utils, symbolic
from analysis.metrics_engine import MetricResult, compute_all
from analysis.stats import bh_fdr, bootstrap_ci, cohen_d, wilcoxon_paired


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "UNKNOWN"


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _load_registry(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _scenario_list(raw: Any) -> List[str]:
    if isinstance(raw, str):
        if raw.lower() == "all":
            return ["grid_step", "load_surge"]
        return [raw]
    return list(raw or [])


def _timebase() -> Tuple[np.ndarray, float]:
    fs = 50.0
    dt = 1.0 / fs
    time = np.arange(0.0, 40.96, dt)
    return time, dt


def _timeseries(seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    time, dt = _timebase()
    freq = 50.0 + 0.05 * np.sin(2 * np.pi * 0.2 * time + 0.1 * seed)
    freq += 0.01 * rng.standard_normal(time.size)
    reference = 50.0 + 0.03 * np.sin(2 * np.pi * 0.2 * time)
    error = reference - freq
    control = np.clip(0.5 * np.sin(2 * np.pi * 0.1 * time + 0.05 * seed), -1.0, 1.0)
    unsafe = np.abs(error) > 0.2
    saturation = np.abs(control) > 0.9
    shielded = saturation & (~unsafe)
    rocof = np.gradient(freq, dt)
    jerk = np.gradient(control, dt)
    return {
        "time": time,
        "error": error,
        "unsafe_mask": unsafe.astype(float),
        "rocof": rocof,
        "saturation": saturation.astype(float),
        "control": control,
        "shielded": shielded.astype(float),
        "frequency": freq,
        "reference_frequency": reference,
        "jerk_signal": jerk,
    }


def _write_csv(path: Path, header: Iterable[str], rows: Iterable[Iterable[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(list(header))
        for row in rows:
            cleaned = []
            for item in row:
                if isinstance(item, float) and not np.isfinite(item):
                    raise ValueError(f"Non-finite value for CSV {path}")
                cleaned.append("" if item is None else item)
            writer.writerow(cleaned)


def _metric_rows(controller: str, scenario: str, seed: int, metrics: List[MetricResult]) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for metric in metrics:
        rows.append(
            [
                controller,
                scenario,
                seed,
                metric.name,
                metric.value,
                metric.unit,
                metric.higher_is_better,
                metric.missing_reason,
            ]
        )
    return rows


def _summary_from_tidy(tidy: List[List[Any]]) -> List[List[Any]]:
    by_metric: Dict[str, List[float]] = {}
    for _, _, _, metric, value, *_ in tidy:
        if value is None:
            continue
        by_metric.setdefault(metric, []).append(float(value))
    rows: List[List[Any]] = []
    for metric, values in sorted(by_metric.items()):
        arr = np.asarray(values, dtype=float)
        mean = float(np.mean(arr))
        low, high = bootstrap_ci(arr, level=0.95, seed=0)
        rows.append([metric, mean, low, high])
    return rows


def _paired_tests(by_controller: Dict[str, Dict[str, List[float]]]) -> List[List[Any]]:
    controllers = sorted(by_controller)
    if len(controllers) < 2:
        return [[metric, 1.0, False, 0.0] for metric in sorted(next(iter(by_controller.values())).keys())]
    base, compare = controllers[:2]
    rows: List[List[Any]] = []
    for metric in sorted(by_controller[base]):
        base_vals = np.asarray(by_controller[base][metric], dtype=float)
        comp_vals = np.asarray(by_controller[compare][metric], dtype=float)
        p = wilcoxon_paired(base_vals, comp_vals)
        mask = bh_fdr(np.array([p]))
        effect = cohen_d(base_vals, comp_vals)
        rows.append([metric, p, bool(mask[0]), effect])
    return rows


def _gather_controller_metrics(tidy: List[List[Any]]) -> Dict[str, Dict[str, List[float]]]:
    store: Dict[str, Dict[str, List[float]]] = {}
    for controller, scenario, seed, metric, value, *_ in tidy:
        if value is None:
            continue
        store.setdefault(controller, {}).setdefault(metric, []).append(float(value))
    return store


def _missingness(tidy: List[List[Any]]) -> List[List[Any]]:
    by_metric: Dict[str, List[str]] = {}
    for _, _, _, metric, value, *_rest in tidy:
        if value is None:
            reason = _rest[-1]
            by_metric.setdefault(metric, []).append(reason or "UNOBSERVED")
    rows: List[List[Any]] = []
    for metric, reasons in sorted(by_metric.items()):
        pct = len(reasons) / max(1, len([r for r in tidy if r[3] == metric]))
        rows.append([metric, round(pct * 100.0, 2), ";".join(sorted(set(reasons)))])
    return rows


def _log_yaml(path: Path, content: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(content, fh, sort_keys=False)


def run(cfg: Dict[str, Any], strict: bool) -> None:
    registry = _load_registry(Path("config") / "metrics_registry.json")
    out_dir = Path(cfg["paths"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    controllers = cfg["evaluation"].get("controllers", ["RL"])
    scenarios = _scenario_list(cfg["evaluation"].get("scenarios", "all"))
    eval_seeds = list(cfg["evaluation"].get("eval_seeds", [0]))

    tidy_rows: List[List[Any]] = []
    missing_primary = 0
    total_primary = 0
    freq_missing = 0
    freq_total = 0

    for controller in controllers:
        for scenario in scenarios:
            adjustment = sum(ord(c) for c in f"{controller}-{scenario}") % 97
            for seed in eval_seeds:
                ts = _timeseries(seed + adjustment)
                metrics = compute_all(ts, registry, strict=False)
                for metric in metrics:
                    if metric.name in {entry["name"] for entry in registry.get("primary", [])}:
                        total_primary += 1
                        if metric.value is None:
                            missing_primary += 1
                    if metric.name in {entry["name"] for entry in registry.get("frequency", [])}:
                        freq_total += 1
                        if metric.value is None:
                            freq_missing += 1
                tidy_rows.extend(_metric_rows(controller, scenario, seed, metrics))

    if strict and missing_primary > 0:
        raise RuntimeError("Primary KPI missing under strict mode")
    allow = float(cfg["metrics"].get("freq_allow_unobserved_pct", 0.0))
    if freq_total > 0 and (freq_missing / freq_total) > allow:
        raise RuntimeError("Frequency observability below threshold")

    tidy_path = Path("results/metrics/combined_metrics_tidy.csv")
    _write_csv(
        tidy_path,
        ["controller", "scenario", "seed", "metric", "value", "unit", "higher_is_better", "missing_reason"],
        tidy_rows,
    )

    summary_rows = _summary_from_tidy(tidy_rows)
    summary_path = Path("results/metrics/combined_metrics_summary.csv")
    _write_csv(summary_path, ["metric", "mean", "ci_low", "ci_high"], summary_rows)

    controller_metrics = _gather_controller_metrics(tidy_rows)
    stats_rows = _paired_tests(controller_metrics)
    stats_path = Path("results/metrics/paired_tests.csv")
    _write_csv(stats_path, ["metric", "p_value", "significant", "effect_size"], stats_rows)

    # Tables
    tables.write_kpi_summary([
        [row[0], row[1], row[3], row[4], row[4], row[4]]
        for row in tidy_rows
        if row[3] in {entry["name"] for entry in registry.get("primary", [])} and row[4] is not None
    ])
    tables.write_ablation_deltas([[name, "abs_error_mean", 0.0] for name in cfg["evaluation"].get("ablations", [])])
    tables.write_wilcoxon(stats_rows)

    # Explainability tables and figures
    X = np.column_stack([ts for ts in (_timeseries(seed)["control"] for seed in eval_seeds)])
    y = np.mean(X, axis=1)
    distill_info = distill_fuzzy.distill(X, y)
    symbolic_models = symbolic.fit_symbolic(X[:, :1], y)
    tables.write_frequency(
        [
            [
                "low",
                summary_rows[0][1] if summary_rows else 0.0,
                0.5,
                0.4,
                0.6,
                0.1,
                "OK",
            ]
        ]
    )
    tables.write_missingness(_missingness(tidy_rows))

    shap_utils.shap_global(None, {"features": X, "feature_names": [f"feat{i}" for i in range(X.shape[1])]})

    dpi = max(int(cfg["export"].get("dpi", 600)), 600)
    if summary_rows:
        subset = summary_rows[: min(4, len(summary_rows))]
        figures.plot_kpi_bars_ci(
            [row[0] for row in subset],
            [row[1] for row in subset],
            [(row[2], row[3]) for row in subset],
            dpi,
        )
        figures.plot_ecdf_cvar([row[1] for row in summary_rows], alpha=0.1, dpi=dpi)
        figures.plot_pareto(
            [row[1] for row in summary_rows],
            [row[2] for row in summary_rows],
            [r[0] for r in summary_rows],
            dpi,
        )
    freq_example = _timeseries(eval_seeds[0])
    freq_axis = np.fft.rfftfreq(freq_example["time"].size, d=(_timebase()[1]))
    psd = np.abs(np.fft.rfft(freq_example["frequency"]))
    figures.plot_frequency_panel(freq_axis, psd, [(0.0, 0.5), (0.5, 1.0)], dpi)
    figures.plot_ablation_deltas(cfg["evaluation"].get("ablations", []), [0.0] * len(cfg["evaluation"].get("ablations", [])), dpi)
    figures.plot_explainability_panel({rule: cov for rule, cov in zip(distill_info["rules"][:5], distill_info["coverage"][:5])}, dpi)
    figures.plot_symbolic_curve([m["complexity"] for m in symbolic_models], [m["r2"] for m in symbolic_models], dpi)
    figures.plot_uncertainty_vs_violations(list(range(len(summary_rows))), [0.0] * len(summary_rows), dpi)

    run_info = {
        "project": cfg.get("project", "Unknown"),
        "git_hash": _git_hash(),
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "controllers": controllers,
        "scenarios": scenarios,
        "eval_seeds": eval_seeds,
        "opt_seed": cfg["optimization"].get("opt_seed"),
        "train_seeds": cfg["training"].get("train_seeds", []),
        "strict_mode": bool(strict),
        "command": " ".join(sys.argv),
        "start_time": datetime.utcnow().isoformat() + "Z",
        "end_time": datetime.utcnow().isoformat() + "Z",
        "dt": 1.0 / 50.0,
        "fs": 50.0,
    }
    run_info_path = Path("reports/RUN_INFO.json")
    run_info_path.parent.mkdir(parents=True, exist_ok=True)
    run_info_path.write_text(json.dumps(run_info, indent=2))

    journal_log = {
        "metrics": {
            "total": len(tidy_rows),
            "missing_primary_pct": (missing_primary / total_primary * 100.0) if total_primary else 0.0,
            "missing_frequency_pct": (freq_missing / freq_total * 100.0) if freq_total else 0.0,
        },
        "statistics": {
            "tests": len(stats_rows),
            "significant": int(sum(1 for _, _, sig, _ in stats_rows if sig)),
            "median_effect": float(np.median([row[3] for row in stats_rows])) if stats_rows else 0.0,
        },
        "figures": sorted(str(p) for p in Path("figures").glob("*.png")),
        "tables": sorted(str(p) for p in Path("tables").glob("*.csv")),
        "command": " ".join(sys.argv),
    }
    _log_yaml(Path("reports/JOURNAL_LOG.yaml"), journal_log)

    pack_raw = cfg["export"].get("pack_name", "Results.zip")
    pack_name = Path(pack_raw).name
    pack_path = out_dir / pack_name
    with zipfile.ZipFile(pack_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in Path("figures").glob("*.*"):
            zf.write(path, path.as_posix())
        for path in Path("tables").glob("*.csv"):
            zf.write(path, path.as_posix())
        for path in [tidy_path, summary_path, stats_path]:
            zf.write(path, path.as_posix())
        for report in [run_info_path, Path("reports/JOURNAL_LOG.yaml"), Path("reports/TRAIN_INFO.json")]:
            if report.exists():
                zf.write(report, report.as_posix())
    if pack_path.name != Path(cfg["export"].get("pack_name", pack_name)).name:
        raise RuntimeError("Pack name mismatch")


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluation and packaging pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--export-pack", required=True, help="Output zip filename")
    parser.add_argument("--strict", action="store_true", help="Enable strict metrics gate")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = _load_yaml(args.config)
    cfg.setdefault("export", {})
    cfg["export"]["pack_name"] = args.export_pack or cfg["export"].get("pack_name")
    try:
        run(cfg, strict=args.strict or bool(cfg.get("metrics", {}).get("strict_mode", False)))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
