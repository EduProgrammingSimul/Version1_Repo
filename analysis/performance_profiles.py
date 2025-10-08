from __future__ import annotations

import json
import os
from typing import Optional, Dict

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"controller", "scenario", "metric", "value", "higher_is_better"}


KNOWN_BOUNDS = {
    "stability_flag": (0.0, 1.0),
    "glfi": (0.0, 1.0),
    "coherence_lowmid": (0.0, 1.0),
}

ROBUSTNESS_WEIGHTS = {"unsafe_time": 0.6, "cvar": 0.4}


def _load_tidy(tidy_csv: str) -> pd.DataFrame:
    df = pd.read_csv(tidy_csv)
    df.columns = [c.strip().lower() for c in df.columns]
    rename = {
        "metric_name": "metric",
        "metricid": "metric",
        "val": "value",
        "score": "value",
        "ctrl": "controller",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            f"combined_metrics_tidy.csv missing required columns: {sorted(missing)}"
        )
    return df


def _normalize_metric(
    series: pd.Series, higher_is_better: bool, metric: str
) -> pd.Series:
    values = series.astype(float)
    mask = values.replace([np.inf, -np.inf], np.nan).notna()
    if not mask.any():
        return pd.Series(np.nan, index=series.index)

    bounded = KNOWN_BOUNDS.get(metric)
    if bounded:
        lo, hi = bounded
        denom = hi - lo
        if denom <= 1e-12:
            return pd.Series(0.5, index=series.index)
        if higher_is_better:
            norm = (values - lo) / denom
        else:
            norm = (hi - values) / denom
        norm = norm.clip(0.0, 1.0)
        return norm

    finite_values = values[mask]
    p5 = np.percentile(finite_values, 5)
    p95 = np.percentile(finite_values, 95)
    if np.isclose(p95, p5):
        return pd.Series(0.5, index=series.index)
    clipped = values.clip(p5, p95)
    if higher_is_better:
        norm = (clipped - p5) / (p95 - p5 + 1e-12)
    else:
        norm = (p95 - clipped) / (p95 - p5 + 1e-12)
    norm = norm.clip(0.0, 1.0)
    return norm


def _blend(values: list[float]) -> float:
    finite = [v for v in values if v is not None and np.isfinite(v)]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def _penalty(norm_value: Optional[float]) -> Optional[float]:
    if norm_value is None or not np.isfinite(norm_value):
        return None
    return 1.0 - norm_value


def build_perf_profiles(out_root: str, tidy_csv: str, eval_seed: int) -> str:
    df = _load_tidy(tidy_csv)
    rows = []

    for (scenario, metric), group in df.groupby(["scenario", "metric"]):
        hib = bool(group["higher_is_better"].iloc[0])
        series = group.set_index("controller")["value"]
        norm = _normalize_metric(series, hib, metric)
        for ctrl, value in series.items():
            rows.append(
                {
                    "controller": ctrl,
                    "scenario": scenario,
                    "metric": metric,
                    "value": float(value) if np.isfinite(value) else float("nan"),
                    "value_norm": float(norm.get(ctrl, np.nan)),
                }
            )

    if not rows:
        raise ValueError("No metrics available to build performance profiles")

    norm_df = pd.DataFrame(rows)

    dims_records = []
    for (scenario, controller), sub in norm_df.groupby(["scenario", "controller"]):
        metric_map = {row.metric: row.value_norm for row in sub.itertuples()}

        unsafe_norm = metric_map.get("total_time_unsafe_s")
        cvar_norm = metric_map.get("cvar_abs_err")
        robustness_penalties = []
        if unsafe_norm is not None:
            robustness_penalties.append(
                _penalty(unsafe_norm) * ROBUSTNESS_WEIGHTS["unsafe_time"]
            )
        if cvar_norm is not None:
            robustness_penalties.append(
                _penalty(cvar_norm) * ROBUSTNESS_WEIGHTS["cvar"]
            )
        robustness_penalty_sum = _blend(robustness_penalties)
        robustness = (
            1.0 - robustness_penalty_sum
            if robustness_penalty_sum is not None
            and np.isfinite(robustness_penalty_sum)
            else float("nan")
        )

        control_efficiency = 1.0 - _blend(
            [
                _penalty(metric_map.get("effort_l1")),
                _penalty(metric_map.get("effort_l2")),
            ]
        )

        oscillation = 1.0 - _blend(
            [
                _penalty(metric_map.get("overshoot_pct")),
                _penalty(metric_map.get("log_decrement")),
                _penalty(metric_map.get("settling_time_s")),
                _penalty(metric_map.get("spectral_error_power")),
            ]
        )

        thermal = 1.0 - _blend(
            [
                _penalty(metric_map.get("time_above_temp_limit_s")),
                _penalty(metric_map.get("max_delta_temp_over_limit_C")),
                _penalty(metric_map.get("arrhenius_exposure")),
            ]
        )

        stability = metric_map.get("stability_flag")
        load_following = _blend(
            [
                metric_map.get("glfi"),
                metric_map.get("iae_abs_err"),
            ]
        )
        actuator = 1.0 - _blend(
            [
                _penalty(metric_map.get("time_in_saturation_s")),
                _penalty(metric_map.get("jerk_l2")),
                _penalty(metric_map.get("reversal_count")),
            ]
        )

        dims_records.extend(
            [
                {
                    "controller": controller,
                    "scenario": scenario,
                    "dimension": "Robustness",
                    "value_norm": robustness,
                },
                {
                    "controller": controller,
                    "scenario": scenario,
                    "dimension": "Control Efficiency",
                    "value_norm": control_efficiency,
                },
                {
                    "controller": controller,
                    "scenario": scenario,
                    "dimension": "Oscillation Damping",
                    "value_norm": oscillation,
                },
                {
                    "controller": controller,
                    "scenario": scenario,
                    "dimension": "Thermal Resilience",
                    "value_norm": thermal,
                },
                {
                    "controller": controller,
                    "scenario": scenario,
                    "dimension": "Stability",
                    "value_norm": stability,
                },
                {
                    "controller": controller,
                    "scenario": scenario,
                    "dimension": "Load Following",
                    "value_norm": load_following,
                },
                {
                    "controller": controller,
                    "scenario": scenario,
                    "dimension": "Actuator Preservation",
                    "value_norm": actuator,
                },
            ]
        )

    dims_df = pd.DataFrame(dims_records)
    out_dir = os.path.join(out_root, "metrics")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "perf_profile_7d.csv")
    dims_df.to_csv(out_csv, index=False)

    meta = {
        "note": "Values normalized per scenario across controllers with 5-95 clipping.",
        "dimensions": [
            "Robustness",
            "Control Efficiency",
            "Oscillation Damping",
            "Thermal Resilience",
            "Stability",
            "Load Following",
            "Actuator Preservation",
        ],
        "eval_seed": int(eval_seed),
    }
    with open(
        os.path.join(out_dir, "perf_profile_7d.meta.json"), "w", encoding="utf-8"
    ) as fh:
        json.dump(meta, fh, indent=2)
    return out_csv


def build_scenario_scores(out_root: str, tidy_csv: str, eval_seed: int) -> str:
    prof_path = os.path.join(out_root, "metrics", "perf_profile_7d.csv")
    if not os.path.isfile(prof_path):
        raise FileNotFoundError(f"Missing performance profile CSV: {prof_path}")
    prof = pd.read_csv(prof_path)
    if prof.empty:
        out_csv = os.path.join(out_root, "metrics", "scenario_controller_score.csv")
        pd.DataFrame(
            columns=["scenario", "controller", "score_norm", "raw_score"]
        ).to_csv(out_csv, index=False)
        return out_csv

    pivot = prof.pivot_table(
        index=["scenario", "controller"],
        columns="dimension",
        values="value_norm",
        aggfunc="mean",
    )

    weights = {
        "Load Following": 0.35,
        "Robustness": 0.35,
        "Control Efficiency": 0.15,
        "Actuator Preservation": 0.15,
    }

    for dim, weight in weights.items():
        if dim not in pivot.columns:
            pivot[dim] = 0.5
        else:
            pivot[dim] = pivot[dim].fillna(0.5)
    pivot["raw_score"] = 0.0
    for dim, weight in weights.items():
        pivot["raw_score"] += weight * pivot[dim]

    rows = []
    for scenario, group in pivot.reset_index().groupby("scenario"):
        values = group.set_index("controller")["raw_score"]
        if values.max() > values.min():
            norm = (values - values.min()) / (values.max() - values.min())
        else:
            norm = values * 0 + 0.5
        for controller, score in norm.items():
            rows.append(
                {
                    "scenario": scenario,
                    "controller": controller,
                    "score_norm": float(score),
                    "raw_score": float(values.loc[controller]),
                }
            )

    out_dir = os.path.join(out_root, "metrics")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "scenario_controller_score.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    with open(
        os.path.join(out_dir, "scenario_controller_score.meta.json"),
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump({"weights": weights, "eval_seed": int(eval_seed)}, fh, indent=2)
    return out_csv
