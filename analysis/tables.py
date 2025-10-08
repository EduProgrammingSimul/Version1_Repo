"""Structured CSV writers for journal tables."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Sequence


def _ensure_dir() -> Path:
    path = Path("tables")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_csv(name: str, header: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    path = _ensure_dir() / name
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for row in rows:
            clean = ["" if val is None else val for val in row]
            if any(isinstance(val, float) and not (val == val) for val in row):
                raise ValueError(f"NaN detected in table {name}")
            writer.writerow(clean)
    return str(path)


def write_kpi_summary(rows: Iterable[Sequence[object]]) -> str:
    header = ["controller", "scenario", "metric", "mean", "ci_low", "ci_high"]
    return _write_csv("T1_kpi_summary_ci.csv", header, rows)


def write_ablation_deltas(rows: Iterable[Sequence[object]]) -> str:
    header = ["ablation", "metric", "delta"]
    return _write_csv("T2_ablation_deltas.csv", header, rows)


def write_wilcoxon(results: Iterable[Sequence[object]]) -> str:
    header = ["metric", "p_value", "significant", "effect_size"]
    return _write_csv("T3_wilcoxon_fdr_effectsize.csv", header, results)


def write_distilled_rules(rows: Iterable[Sequence[object]]) -> str:
    header = ["rule", "coverage", "fidelity_r2", "mse"]
    return _write_csv("T4_distilled_rules.csv", header, rows)


def write_symbolic_models(rows: Iterable[Sequence[object]]) -> str:
    header = ["equation", "complexity", "mse", "r2"]
    return _write_csv("T5_symbolic_models.csv", header, rows)


def write_frequency(rows: Iterable[Sequence[object]]) -> str:
    header = ["band", "power", "coherence", "ci_low", "ci_high", "log_decrement", "status"]
    return _write_csv("T6_frequency_bandwidth_damping.csv", header, rows)


def write_missingness(rows: Iterable[Sequence[object]]) -> str:
    header = ["metric", "missing_pct", "reason"]
    return _write_csv("T7_missingness_audit.csv", header, rows)


__all__ = [
    "write_kpi_summary",
    "write_ablation_deltas",
    "write_wilcoxon",
    "write_distilled_rules",
    "write_symbolic_models",
    "write_frequency",
    "write_missingness",
]
