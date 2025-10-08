import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


from analysis.scenario_executor import ScenarioExecutor
from analysis.metrics_engine import MetricsEngine

logger = logging.getLogger(__name__)


def run_single_sim_and_extract_detailed_metrics(
    base_core_config: Dict[str, Any],
    scenario_config: Dict[str, Any],
    controller_name: str,
    controller_instance: Any,
) -> Dict[str, Any]:

    scenario_display_name = scenario_config.get("name", "UnnamedScenario")
    logger.info(
        f"--- Running Sim for Metrics: {scenario_display_name} / {controller_name} ---"
    )

    output_results = {
        "completed_successfully": False,
        "termination_reason": "Unknown",
        "metrics": {},
        "raw_results_df": pd.DataFrame(),
        "error_message": None,
    }

    try:

        full_config_for_executor = {"CORE_PARAMETERS": base_core_config}
        executor = ScenarioExecutor(base_env_config_full=full_config_for_executor)
        metrics_engine = MetricsEngine(base_core_config.get("metrics_config", {}))

        if hasattr(controller_instance, "reset"):
            controller_instance.reset()

        results_df = executor.execute(
            scenario_name=scenario_display_name,
            scenario_config_from_caller=scenario_config,
            controller_name=controller_name,
            controller_instance=controller_instance,
        )
        output_results["raw_results_df"] = results_df

        if results_df.empty or len(results_df) < 2:
            logger.warning(
                f"Sim for '{scenario_display_name}/{controller_name}' returned empty/insufficient data."
            )
            output_results["error_message"] = "Simulation returned no valid data."

            output_results["termination_reason"] = "Insufficient Data / Early Failure"
            return output_results

        logger.debug(
            f"Sim for '{scenario_display_name}/{controller_name}' successful. Results length: {len(results_df)}"
        )

        safety_limits = base_core_config.get("safety_limits", {})
        metrics_dict = metrics_engine.calculate(
            results_df, safety_limits, scenario_config
        )
        output_results["metrics"] = metrics_dict

        if metrics_dict.get("total_time_unsafe_s", 0.0) > 1e-6:
            output_results["completed_successfully"] = False
            output_results["termination_reason"] = "Safety Limit Violation"
            logger.warning(
                f"Scenario '{scenario_display_name}' for '{controller_name}' deemed failed due to safety violations."
            )
        else:
            output_results["completed_successfully"] = True
            output_results["termination_reason"] = "Completed Nominally"

    except Exception as e:
        logger.error(
            f"Unhandled exception in run_single_sim_and_extract_detailed_metrics for '{scenario_display_name}/{controller_name}': {e}",
            exc_info=True,
        )
        output_results["error_message"] = str(e)
        output_results["completed_successfully"] = False
        output_results["termination_reason"] = "Exception during execution"

    finally:
        logger.info(
            f"--- Finished Sim for Metrics: {scenario_display_name} / {controller_name}. Success: {output_results['completed_successfully']} ---"
        )
        return output_results
