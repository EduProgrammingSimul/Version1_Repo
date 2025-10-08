from __future__ import annotations

"""Validate deterministic evaluation outputs and build aggregate artefacts."""

import argparse
import hashlib
import json
import os
import platform
import random
import sys
from typing import Dict, Iterable, List, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analysis.metrics_engine import compute_from_timeseries
from analysis.performance_profiles import build_perf_profiles, build_scenario_scores
from analysis.correlation_engine import compute_metric_correlations
from analysis.policy_manifold import build_policy_manifold
from analysis.figure_runner import render_all
from analysis.statistics_suite import emit_paired_tests
from analysis.frequency_analysis import emit_frequency_analysis
from analysis.scenario_definitions import get_scenarios
from analysis.parameter_manager import ParameterManager
from your_project import logging_setup

logger = logging_setup.get_logger(__name__)

SCHEMA_MAP = {
    "metrics/combined_metrics_tidy.csv": "results_schema/combined_metrics_schema.json",
    "metrics/perf_profile_7d.csv": "results_schema/perf_profile_schema.json",
    "metrics/corr_metrics.csv": "results_schema/corr_schema.json",
}


def _expand_controllers(values: Iterable[str]) -> List[str]:
    if len(values) == 1 and "," in values[0]:
        return [v.strip().upper() for v in values[0].split(",") if v.strip()]
    return [v.strip().upper() for v in values if v.strip()]


def _expand_scenarios(
    arg: str, suite_file: str | None, core_config: Dict[str, any]
) -> List[str]:
    if arg.strip().lower() != "all":
        return [s.strip() for s in arg.split(",") if s.strip()]
    library = list(get_scenarios(core_config).keys())
    if library:
        return sorted(library)
    path = suite_file or os.path.join("config", "scenario_suite.txt")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8-sig") as handle:
            scenarios = [
                line.strip()
                for line in handle
                if line.strip() and not line.startswith("#")
            ]
            if scenarios:
                return sorted(scenarios)
    raise ValueError(
        "Unable to expand scenarios; provide --suite-file or explicit --scenarios list"
    )


def _hash_timeseries(df: pd.DataFrame) -> str:
    cols = [
        c for c in ["time_s", "y_actual", "u", "freq_hz", "temp_C"] if c in df.columns
    ]
    if not cols:
        raise ValueError(
            "Timeseries missing required numeric columns for hash computation"
        )
    arr = df[cols].sort_values(cols[0]).to_numpy(dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _collect_timeseries(
    validation_root: str, controllers: Iterable[str], scenarios: Iterable[str]
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, List[str]]]]:
    missing: List[Tuple[str, str, str]] = []
    identity: List[Tuple[str, List[str]]] = []
    fingerprint_map: Dict[str, Dict[str, str]] = {}
    for scenario in scenarios:
        fingerprint_map[scenario] = {}
        for controller in controllers:
            path = os.path.join(validation_root, controller, scenario, "timeseries.csv")
            if not os.path.isfile(path):
                missing.append((controller, scenario, "timeseries.csv missing"))
                continue
            df = pd.read_csv(path)
            if df.empty:
                missing.append((controller, scenario, "timeseries empty"))
                continue
            fingerprint_map[scenario][controller] = _hash_timeseries(df)
    for scenario, ctrl_map in fingerprint_map.items():
        reverse: Dict[str, List[str]] = {}
        for ctrl, fp in ctrl_map.items():
            reverse.setdefault(fp, []).append(ctrl)
        for fp, ctrl_list in reverse.items():
            if len(ctrl_list) > 1:
                identity.append((scenario, ctrl_list))
    return missing, identity


def _validate_csv(path: str, schema_path: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV missing for schema validation: {path}")
    with open(schema_path, "r", encoding="utf-8-sig") as schema_file:
        schema = json.load(schema_file)
    df = pd.read_csv(path)

    if "required_columns" in schema:
        required = schema["required_columns"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"{os.path.basename(path)} missing columns: {missing}")
        return

    try:
        import jsonschema
    except ImportError as exc:
        raise RuntimeError("jsonschema package required for schema validation") from exc

    records = df.to_dict(orient="records")
    array_schema = {"type": "array", "items": schema}
    validator = jsonschema.Draft7Validator(array_schema)
    errors = sorted(validator.iter_errors(records), key=lambda err: err.path)
    if errors:
        messages = [f"row {list(err.path)}: {err.message}" for err in errors]
        raise ValueError(f"Schema validation failed for {path}: {'; '.join(messages)}")


def _write_csv(
    rows: List[Tuple[str, str, str]], header: List[str], out_path: str
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(rows, columns=header).to_csv(
        out_path, index=False, encoding="utf-8-sig"
    )


def _write_run_info(
    out_root: str,
    eval_seed: int,
    controllers: List[str],
    scenarios: List[str],
    *,
    eval_seeds: List[int] | None = None,
) -> str:
    reports_dir = os.path.join(out_root, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    info = {
        "eval_seed": eval_seed,
        "eval_seeds": eval_seeds,
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "controllers": controllers,
        "scenarios": scenarios,
    }
    path = os.path.join(reports_dir, "RUN_INFO.json")
    with open(path, "w", encoding="utf-8-sig") as handle:
        json.dump(info, handle, indent=2)
    return path


def _ensure_utf8_csv(path: str) -> None:
    if not path or not os.path.isfile(path):
        return
    df = pd.read_csv(path, encoding="utf-8-sig")
    tmp_path = f"{path}.tmp"
    df.to_csv(tmp_path, index=False, encoding="utf-8-sig", lineterminator="\n")
    os.replace(tmp_path, path)


def _export_csv_to_pdf(
    csv_path: str, df: pd.DataFrame | None = None, max_rows: int = 60
) -> None:
    try:
        data = (
            df.copy() if df is not None else pd.read_csv(csv_path, encoding="utf-8-sig")
        )
        preview = data.head(max_rows)
        if preview.empty:
            return
        fig_height = max(2.5, 0.35 * len(preview) + 1)
        fig, ax = plt.subplots(figsize=(11, fig_height))
        ax.axis("off")
        table = ax.table(
            cellText=preview.values, colLabels=preview.columns, loc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(6)
        table.scale(1, 1.2)
        ax.set_title(os.path.basename(csv_path), fontsize=9, pad=12)
        pdf_path = os.path.splitext(csv_path)[0] + ".pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        logger.warning("Failed to create PDF companion for %s: %s", csv_path, exc)


def _export_parquet_to_csv(parquet_path: str) -> str | None:
    if not parquet_path or not os.path.isfile(parquet_path):
        return None
    if not parquet_path.lower().endswith(".parquet"):
        return None
    df = pd.read_parquet(parquet_path)
    csv_path = os.path.splitext(parquet_path)[0] + ".csv"
    tmp_path = f"{csv_path}.tmp"
    df.to_csv(tmp_path, index=False, encoding="utf-8-sig", lineterminator="\n")
    os.replace(tmp_path, csv_path)
    return csv_path


def _bootstrap_dimension_ci(
    perf_csv: str, out_csv: str, bootstrap_samples: int, seed: int
) -> str:
    if not os.path.isfile(perf_csv):
        raise FileNotFoundError(perf_csv)
    df = pd.read_csv(perf_csv)
    rng = np.random.default_rng(seed)
    rows = []
    for (controller, dimension), group in df.groupby(["controller", "dimension"]):
        values = group["value_norm"].dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        mean = float(values.mean())
        if values.size > 1 and bootstrap_samples > 0:
            samples = rng.choice(
                values, size=(bootstrap_samples, values.size), replace=True
            )
            means = samples.mean(axis=1)
            lower = float(np.percentile(means, 2.5))
            upper = float(np.percentile(means, 97.5))
        else:
            lower = upper = mean
        rows.append(
            {
                "controller": controller,
                "dimension": dimension,
                "value_mean": mean,
                "value_low": lower,
                "value_high": upper,
            }
        )
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation artefacts and perform audits."
    )
    parser.add_argument(
        "--controllers",
        nargs="+",
        required=True,
        help="Controllers to include (PID FLC RL)",
    )
    parser.add_argument("--scenarios", default="all", help="Scenario list or 'all'")
    parser.add_argument(
        "--suite-file", default=None, help="Scenario suite file for --scenarios all"
    )
    parser.add_argument("--out", default="results", help="Output root directory")
    parser.add_argument(
        "--eval-seed", type=int, required=True, help="Primary evaluation seed"
    )
    parser.add_argument(
        "--eval-seeds",
        nargs="*",
        default=None,
        help="Optional additional seeds for multi-seed aggregation",
    )
    parser.add_argument(
        "--allow-identical",
        action="store_true",
        help="Do not fail when identical traces detected",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Bootstrap samples for uncertainty bands",
    )
    args = parser.parse_args()

    random.seed(args.eval_seed)
    np.random.seed(args.eval_seed)

    controllers = _expand_controllers(args.controllers)
    seed_values = [int(args.eval_seed)]
    if args.eval_seeds:
        seed_values.extend(int(s) for s in args.eval_seeds)
    seed_values = sorted(set(seed_values))
    multi_seed = len(seed_values) > 1

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path = os.path.join(project_root, "config", "parameters.py")
    param_manager = ParameterManager(config_filepath=config_path)
    full_config = param_manager.get_all_parameters()
    core_config = full_config.get("CORE_PARAMETERS", {})
    scenarios = _expand_scenarios(args.scenarios, args.suite_file, core_config)

    def _validation_root(seed: int) -> str:
        if multi_seed:
            return os.path.join(args.out, f"seed_{seed}", "validation")
        return os.path.join(args.out, "validation")

    validation_roots = []
    for seed in seed_values:
        vroot = _validation_root(seed)
        if not os.path.isdir(vroot):
            raise FileNotFoundError(f"Validation directory not found: {vroot}")
        validation_roots.append(vroot)

    reports_dir = os.path.join(args.out, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    missing_rows: List[Tuple[str, str, str]] = []
    identity_rows: List[Tuple[str, List[str]]] = []
    for seed, vroot in zip(seed_values, validation_roots):
        missing, identity = _collect_timeseries(vroot, controllers, scenarios)
        missing_rows.extend(
            [(ctrl, scen, f"seed={seed}: {issue}") for ctrl, scen, issue in missing]
        )
        identity_rows.extend(
            [(f"seed={seed}:{scen}", ctrls) for scen, ctrls in identity]
        )

    missing_csv = os.path.join(reports_dir, "missing_timeseries.csv")
    _write_csv(missing_rows, ["controller", "scenario", "issue"], missing_csv)
    if missing_rows:
        logger.error("Timeseries missing or invalid. See %s", missing_csv)
        sys.exit(1)

    identical_csv = os.path.join(reports_dir, "controller_identity_issues.csv")
    identity_table = [(scenario, ",".join(ctrls)) for scenario, ctrls in identity_rows]
    _write_csv(identity_table, ["scenario", "controllers"], identical_csv)
    if identity_rows and not args.allow_identical:
        logger.error("Identical controller traces detected. See %s", identical_csv)
        sys.exit(1)

    freq_band = tuple(
        core_config.get("analysis", {}).get("spectral_band_hz", (0.0, 1.0))
    )
    metrics_frames: List[pd.DataFrame] = []
    metrics_paths: List[str] = []
    freq_frames: List[pd.DataFrame] = []
    freq_paths: List[str] = []

    for seed, vroot in zip(seed_values, validation_roots):
        run_id = f"seed_{seed}" if multi_seed else "report"
        metrics_path_seed = compute_from_timeseries(
            results_root=vroot,
            controllers=controllers,
            scenarios=scenarios,
            core_config=core_config,
            run_id=run_id,
        )
        _ensure_utf8_csv(metrics_path_seed)
        metrics_paths.append(metrics_path_seed)
        df_metrics = pd.read_csv(metrics_path_seed)
        df_metrics["seed"] = seed
        metrics_frames.append(df_metrics)

        freq_out_seed = emit_frequency_analysis(
            vroot,
            controllers,
            scenarios,
            core_config,
            os.path.join(
                os.path.dirname(vroot), "metrics", "frequency_domain_metrics.csv"
            ),
            freq_band=freq_band,
        )
        if freq_out_seed and os.path.isfile(freq_out_seed):
            _ensure_utf8_csv(freq_out_seed)
            freq_paths.append(freq_out_seed)
            freq_df = pd.read_csv(freq_out_seed)
            freq_df["seed"] = seed
            freq_frames.append(freq_df)

    metrics_dir = os.path.join(args.out, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    if not metrics_frames:
        raise ValueError("No metrics computed; aborting")

    if multi_seed:
        metrics_all = pd.concat(metrics_frames, ignore_index=True)
        metrics_all_path = os.path.join(metrics_dir, "combined_metrics_by_seed.csv")
        metrics_all.to_csv(metrics_all_path, index=False, encoding="utf-8-sig")
        summary = metrics_all.groupby(
            ["controller", "scenario", "metric", "higher_is_better"], as_index=False
        ).agg(
            value_mean=("value", "mean"),
            value_std=("value", "std"),
            n=("value", "count"),
        )
        summary_path = os.path.join(metrics_dir, "combined_metrics_summary.csv")
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        metrics_mean = summary.rename(columns={"value_mean": "value"})
        metrics_mean["run_id"] = "multi_seed"
        metrics_path = os.path.join(metrics_dir, "combined_metrics_tidy.csv")
        metrics_mean.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    else:
        metrics_path = metrics_paths[0]

    _ensure_utf8_csv(metrics_path)

    if freq_frames:
        freq_all = pd.concat(freq_frames, ignore_index=True)
        if multi_seed:
            freq_all_path = os.path.join(metrics_dir, "frequency_domain_by_seed.csv")
            freq_all.to_csv(freq_all_path, index=False, encoding="utf-8-sig")
            freq_summary = freq_all.groupby(
                ["controller", "scenario"], as_index=False
            ).agg(
                psd_bandpower_error=("psd_bandpower_error", "mean"),
                psd_bandpower_control=("psd_bandpower_control", "mean"),
                noise_amplification=("noise_amplification", "mean"),
                damping_ratio=("damping_ratio", "mean"),
                natural_frequency=("natural_frequency", "mean"),
            )
            freq_metrics_path = os.path.join(
                metrics_dir, "frequency_domain_metrics.csv"
            )
            freq_summary.to_csv(freq_metrics_path, index=False, encoding="utf-8-sig")
        else:
            freq_metrics_path = freq_paths[0]
    else:
        freq_metrics_path = None

    if freq_metrics_path:
        _ensure_utf8_csv(freq_metrics_path)

    perf_path = build_perf_profiles(args.out, metrics_path, args.eval_seed)
    _ensure_utf8_csv(perf_path)
    scenario_scores_path = build_scenario_scores(args.out, metrics_path, args.eval_seed)
    _ensure_utf8_csv(scenario_scores_path)
    corr_path = compute_metric_correlations(
        args.out, metrics_path, eval_seed=args.eval_seed
    )
    _ensure_utf8_csv(corr_path)

    stats_path = emit_paired_tests(
        metrics_path, os.path.join(args.out, "metrics", "paired_tests.csv")
    )
    _ensure_utf8_csv(stats_path)
    ci_path = _bootstrap_dimension_ci(
        perf_path,
        os.path.join(args.out, "metrics", "perf_profile_7d_ci.csv"),
        args.bootstrap_samples,
        args.eval_seed,
    )
    _ensure_utf8_csv(ci_path)

    validation_root_primary = validation_roots[0]
    manifold_path = build_policy_manifold(
        args.out, validation_root_primary, eval_seed=args.eval_seed
    )
    manifold_csv = None
    if manifold_path:
        manifold_csv = _export_parquet_to_csv(manifold_path)
        if manifold_csv:
            logger.info("Policy manifold CSV exported to %s", manifold_csv)

    registry_path = os.path.join(project_root, "config", "figure_registry.yaml")
    render_all(
        registry_path=registry_path,
        out_root=args.out,
        eval_seed=args.eval_seed,
        validation_root=validation_root_primary,
    )

    for rel_path, schema in SCHEMA_MAP.items():
        csv_path = os.path.join(args.out, rel_path)
        if not os.path.isfile(csv_path):
            continue
        schema_path = os.path.join(project_root, schema)
        _validate_csv(csv_path, schema_path)
        logger.info("Validated %s against %s", csv_path, schema)

    run_info_path = _write_run_info(
        args.out, args.eval_seed, controllers, scenarios, eval_seeds=seed_values
    )
    logger.info("Run info written to %s", run_info_path)
    logger.info("Validation and reporting complete.")


if __name__ == "__main__":
    main()
