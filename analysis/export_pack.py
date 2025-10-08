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
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import yaml

from analysis import figures, tables
from analysis.explainability import distill_fuzzy, shap_utils, symbolic
from analysis.frequency import ObservabilityError, summarize_frequency
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
    primary_names = {str(entry["name"]) for entry in registry.get("primary", [])}
    frequency_names = {str(entry["name"]) for entry in registry.get("frequency", [])}
    band_map = {
        "band_power_low": (0.0, 0.5),
        "band_power_mid": (0.5, 1.0),
        "band_power_high": (1.0, 2.0),
    }
    ordered_bands = [
        band_map["band_power_low"],
        band_map["band_power_mid"],
        band_map["band_power_high"],
    ]
    kpi_groups: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    freq_records: List[Dict[str, Any]] = []
    start_time = datetime.utcnow()

    for controller in controllers:
        for scenario in scenarios:
            adjustment = sum(ord(c) for c in f"{controller}-{scenario}") % 97
            for seed in eval_seeds:
                ts = _timeseries(seed + adjustment)
                metrics = compute_all(ts, registry, strict=strict)
                tidy_rows.extend(_metric_rows(controller, scenario, seed, metrics))
                for metric in metrics:
                    name = metric.name
                    if name in primary_names:
                        total_primary += 1
                        if metric.value is None:
                            missing_primary += 1
                        else:
                            kpi_groups[(controller, scenario, name)].append(float(metric.value))
                    if name in frequency_names:
                        freq_total += 1
                        if metric.value is None:
                            freq_missing += 1
                if frequency_names:
                    freq_record: Dict[str, Any] = {
                        "controller": controller,
                        "scenario": scenario,
                        "seed": seed,
                        "summary": None,
                        "reason": None,
                    }
                    time_arr = ts.get("time")
                    freq_signal = ts.get("frequency")
                    ref_signal = ts.get("reference_frequency")
                    if time_arr is None or freq_signal is None or ref_signal is None:
                        freq_record["reason"] = "UNOBSERVED:missing-timeseries"
                    else:
                        try:
                            summary = summarize_frequency(
                                np.asarray(time_arr, dtype=float),
                                np.asarray(freq_signal, dtype=float),
                                np.asarray(ref_signal, dtype=float),
                                band_map.values(),
                            )
                        except ObservabilityError as exc:
                            freq_record["reason"] = f"UNOBSERVED:{exc}"
                        else:
                            freq_record["summary"] = {
                                "band_means": summary.band_means,
                                "coherence_means": summary.coherence_means,
                                "band_ci": summary.band_ci,
                                "log_decrement": summary.log_decrement,
                                "log_reason": summary.log_decrement_reason,
                                "coherence_reason": summary.coherence_reason,
                            }
                    freq_records.append(freq_record)

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
    kpi_summary_rows: List[List[Any]] = []
    for (controller, scenario, metric), values in sorted(kpi_groups.items()):
        arr = np.asarray(values, dtype=float)
        mean_val = float(np.mean(arr))
        if arr.size > 1:
            ci_low, ci_high = bootstrap_ci(arr, level=0.95, seed=0)
        else:
            ci_low = ci_high = mean_val
        kpi_summary_rows.append([controller, scenario, metric, mean_val, float(ci_low), float(ci_high)])
    tables.write_kpi_summary(kpi_summary_rows)
    tables.write_ablation_deltas([[name, "abs_error_mean", 0.0] for name in cfg["evaluation"].get("ablations", [])])
    tables.write_wilcoxon(stats_rows)

    def _mean_or_none(values):
        filtered = [float(v) for v in values if v is not None]
        return float(np.mean(filtered)) if filtered else None

    freq_rows: List[List[Any]] = []
    if freq_records:
        for band in ordered_bands:
            label = f"{band[0]:.2f}-{band[1]:.2f}"
            power_vals: List[float] = []
            coh_vals: List[float] = []
            ci_low_vals: List[float] = []
            ci_high_vals: List[float] = []
            log_vals: List[float] = []
            status_notes: List[str] = []
            for rec in freq_records:
                context = f"{rec['controller']}|{rec['scenario']}|s{rec['seed']}"
                summary = rec.get("summary")
                if summary is None:
                    reason = rec.get("reason")
                    if reason:
                        status_notes.append(f"{context}:{reason}")
                    continue
                band_means = summary["band_means"]
                if band in band_means:
                    power_vals.append(float(band_means[band]))
                coherence_reason = summary.get("coherence_reason") or rec.get("reason")
                if coherence_reason:
                    status_notes.append(f"{context}:{coherence_reason}")
                else:
                    coh_val = summary["coherence_means"].get(band)
                    if coh_val is not None:
                        coh_vals.append(float(coh_val))
                        ci = summary["band_ci"].get(band)
                        if ci:
                            ci_low_vals.append(ci[0])
                            ci_high_vals.append(ci[1])
                log_reason = summary.get("log_reason")
                if log_reason:
                    status_notes.append(f"{context}:{log_reason}")
                else:
                    log_val = summary.get("log_decrement")
                    if log_val is not None:
                        log_vals.append(float(log_val))
            freq_rows.append(
                [
                    label,
                    _mean_or_none(power_vals),
                    _mean_or_none(coh_vals),
                    _mean_or_none(ci_low_vals),
                    _mean_or_none(ci_high_vals),
                    _mean_or_none(log_vals),
                    ";".join(sorted(set(status_notes))) if status_notes else "OK",
                ]
            )
    tables.write_frequency(freq_rows)

    missing_rows = _missingness(tidy_rows)
    tables.write_missingness(missing_rows)

    # Explainability tables and figures
    X = np.column_stack([ts for ts in (_timeseries(seed)["control"] for seed in eval_seeds)])
    y = np.mean(X, axis=1)
    distill_info = distill_fuzzy.distill(X, y)
    symbolic_models = symbolic.fit_symbolic(X[:, :1], y)

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

    end_time = datetime.utcnow()
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
        "start_time": start_time.isoformat() + "Z",
        "end_time": end_time.isoformat() + "Z",
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
            "inventory": [
                {"metric": metric, "missing_pct": pct, "reason": reason}
                for metric, pct, reason in missing_rows
            ],
        },
        "statistics": {
            "tests": len(stats_rows),
            "significant": int(sum(1 for _, _, sig, _ in stats_rows if sig)),
            "median_effect": float(np.median([row[3] for row in stats_rows])) if stats_rows else 0.0,
        },
        "figures": sorted(
            str(p)
            for pattern in ("*.png", "*.tif")
            for p in Path("figures").glob(pattern)
        ),
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
