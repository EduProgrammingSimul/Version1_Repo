"""analysis package public API (compat-safe)
Exports: MetricsEngine, compute_from_timeseries, emit_combined_metrics_tidy,
apply_style, ensure_dir, VisualizationEngine (shim).
"""

from .metrics_engine import (
    MetricsEngine,
    compute_from_timeseries,
    emit_combined_metrics_tidy,
)
from .visualization_engine import (
    apply_style,
    ensure_dir,
    VisualizationEngine,
)
from .figure_runner import render_all

__all__ = [
    "MetricsEngine",
    "compute_from_timeseries",
    "emit_combined_metrics_tidy",
    "apply_style",
    "ensure_dir",
    "VisualizationEngine",
    "render_all",
]
