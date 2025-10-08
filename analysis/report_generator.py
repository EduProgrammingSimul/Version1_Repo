import logging
import pandas as pd
import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape
import os
import datetime
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class ReportGenerator:

    def __init__(self, report_config: Dict[str, Any], base_config: Dict[str, Any]):
        self.config = report_config
        self.base_config = base_config
        self.template_dir = self.config.get(
            "template_dir", os.path.join(os.path.dirname(__file__), "templates")
        )
        self.output_dir = self.config.get("report_output_dir", "results/reports")
        self.plot_dir = self.base_config.get("reporting", {}).get(
            "plot_output_dir", "results/plots"
        )
        self.comparison_criteria = self.config.get("comparison_criteria", {})
        self.crs_metrics_config = self.comparison_criteria.get("crs_metrics", {})

        try:
            os.makedirs(self.output_dir, exist_ok=True)
            self.jinja_env = Environment(
                loader=FileSystemLoader(self.template_dir),
                autoescape=select_autoescape(["html", "xml", "md"]),
            )
            logger.info(
                f"Report Generator v3.3 initialized with CRS config: {self.crs_metrics_config}"
            )
        except Exception as e:
            logger.error(
                f"Failed to initialize ReportGenerator Jinja2 env: {e}", exc_info=True
            )
            self.jinja_env = None

    def _format_metric(self, value: Any) -> str:

        if pd.isna(value) or value is None:
            return "**NaN**"
        if np.isinf(value):
            return "Inf" if value > 0 else "-Inf"
        if isinstance(value, (int, float, np.number)):
            if value == 0:
                return "0"
            if abs(value) < 1e-4:
                return f"{value:.3e}"
            if abs(value) >= 1000:
                return f"{value:,.2f}"
            return f"{value:.4g}"
        return str(value)

    def _calculate_composite_robustness_score(
        self, metrics_df: pd.DataFrame
    ) -> Optional[pd.Series]:

        if metrics_df.empty or not self.crs_metrics_config:
            return None

        weighted_ranks = {}

        for metric, weight in self.crs_metrics_config.get(
            "higher_is_better", {}
        ).items():
            if metric in metrics_df.columns and metrics_df[metric].notna().any():
                ranks = metrics_df[metric].rank(
                    method="min", ascending=False, na_option="bottom"
                )
                weighted_ranks[metric] = ranks * weight

        for metric, weight in self.crs_metrics_config.get(
            "lower_is_better", {}
        ).items():
            if metric in metrics_df.columns and metrics_df[metric].notna().any():
                ranks = metrics_df[metric].rank(
                    method="min", ascending=True, na_option="bottom"
                )
                weighted_ranks[metric] = ranks * weight

        if not weighted_ranks:
            return None

        total_weighted_rank = pd.DataFrame(weighted_ranks).sum(axis=1)

        min_rank, max_rank = total_weighted_rank.min(), total_weighted_rank.max()
        if max_rank == min_rank:
            return pd.Series(1.0, index=total_weighted_rank.index)

        return 1 - ((total_weighted_rank - min_rank) / (max_rank - min_rank))

    def _perform_comparative_analysis(self, metrics_df: pd.DataFrame) -> Dict[str, Any]:

        analysis = {"winners": {}, "recommendation": "N/A"}
        if metrics_df.empty:
            return analysis

        higher_is_better = list(
            self.crs_metrics_config.get("higher_is_better", {}).keys()
        )
        for metric in metrics_df.columns:
            if not metrics_df[metric].notna().any():
                continue

            clean_series = metrics_df[metric].dropna()
            if clean_series.empty:
                continue

            ascending = metric not in higher_is_better
            try:
                best_value = clean_series.min() if ascending else clean_series.max()
                best_controller = (
                    clean_series.idxmin() if ascending else clean_series.idxmax()
                )
                analysis["winners"][
                    metric
                ] = f"**{best_controller}** ({self._format_metric(best_value)})"
            except TypeError:
                analysis["winners"][metric] = "N/A (mixed types)"

        primary_metric = self.comparison_criteria.get(
            "primary_metric", "composite_robustness_score"
        )
        if primary_metric in analysis["winners"]:
            analysis["recommendation"] = (
                f"Based on the primary metric '{primary_metric}', the recommended controller is {analysis['winners'][primary_metric]}."
            )

        return analysis

    def generate_report(
        self,
        all_scenario_metrics: Dict[str, Dict[str, Dict[str, float]]],
        scenarios_config: Dict[str, Dict[str, Any]],
        controller_details: Optional[Dict[str, Any]] = None,
        report_filename: Optional[str] = None,
    ) -> Optional[str]:

        if not self.jinja_env:
            logger.error("Cannot generate report: Jinja2 environment not initialized.")
            return None

        if not report_filename:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"report_{timestamp}.md"

        report_path = os.path.join(self.output_dir, report_filename)
        logger.info(f"Generating analysis report: {report_path}")

        template = self.jinja_env.get_template(
            self.config.get("template_name", "report_template_v2.md")
        )

        processed_scenarios = {}
        for scenario_name, metrics_by_controller in all_scenario_metrics.items():
            if not metrics_by_controller:
                continue

            metrics_df = pd.DataFrame.from_dict(metrics_by_controller, orient="index")
            crs_scores = self._calculate_composite_robustness_score(metrics_df)
            if crs_scores is not None:
                metrics_df["composite_robustness_score"] = crs_scores

            analysis_summary = self._perform_comparative_analysis(metrics_df)
            plot_paths = self._find_plots_for_scenario(scenario_name)

            cols = sorted(
                [col for col in metrics_df.columns if metrics_df[col].notna().any()]
            )
            if "composite_robustness_score" in cols:
                cols.insert(0, cols.pop(cols.index("composite_robustness_score")))

            metrics_table = [["Controller"] + cols]
            for controller, row in metrics_df.reindex(columns=cols).iterrows():
                formatted_row = [controller] + [
                    self._format_metric(row.get(col)) for col in cols
                ]
                metrics_table.append(formatted_row)

            processed_scenarios[scenario_name] = {
                "config": scenarios_config.get(scenario_name, {}),
                "metrics_table": metrics_table,
                "analysis": analysis_summary,
                "plots": plot_paths,
                "controller_names": list(metrics_by_controller.keys()),
            }

        try:
            report_content = template.render(
                report_title="PWR Control Strategy Analysis Report",
                generation_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                scenarios_data=processed_scenarios,
                controller_details=controller_details,
                report_config=self.config,
            )
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            logger.info(f"Successfully generated report: {report_path}")
            return report_path
        except Exception as e:
            logger.error(f"Failed to render or write report: {e}", exc_info=True)
            return None

    def _find_plots_for_scenario(
        self, scenario_name: str
    ) -> Dict[str, List[Dict[str, str]]]:

        plots = {"time_series": [], "metric_comparison": []}
        if not os.path.exists(self.plot_dir):
            return plots

        safe_scenario_name = scenario_name.replace(" ", "_").replace("/", "_")
        for filename in os.listdir(self.plot_dir):
            if filename.startswith(safe_scenario_name) and any(
                filename.endswith(ext) for ext in [".png", ".svg"]
            ):
                plot_info = {
                    "title": filename.replace("_", " ").rsplit(".", 1)[0].title(),
                    "path": os.path.relpath(
                        os.path.join(self.plot_dir, filename), self.output_dir
                    ),
                }
                if "timeseries" in filename:
                    plots["time_series"].append(plot_info)
                elif "metric_comparison" in filename:
                    plots["metric_comparison"].append(plot_info)
        return plots
