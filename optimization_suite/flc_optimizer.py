import logging
import os
import time
import yaml
import numpy as np
from scipy.optimize import differential_evolution
from typing import Dict, Any, Optional
import datetime

from optimization_suite.auto_validator import auto_validate_and_report
from optimization_suite.optimization_utils import (
    run_single_sim_and_extract_detailed_metrics,
)
from controllers import FLCController

logger = logging.getLogger(__name__)


def _flc_objective_function(
    params_vector: np.ndarray,
    full_base_config: Dict[str, Any],
    scenarios_to_run: Dict[str, Any],
    param_names: list,
) -> float:

    start_time_objective_eval = time.time()
    core_params = full_base_config["CORE_PARAMETERS"]
    cost_weights = (
        core_params.get("reporting", {})
        .get("comparison_criteria", {})
        .get("crs_metrics", {})
    )

    current_flc_params = dict(zip(param_names, map(float, params_vector)))
    params_str = ", ".join([f"{k}={v:.4f}" for k, v in current_flc_params.items()])
    logger.info(f"--- Evaluating FLC Params: {params_str} ---")

    total_cost = 0.0
    num_failures = 0

    for scenario_name, scenario_config in scenarios_to_run.items():
        try:
            base_flc_config = core_params.get("controllers", {}).get("FLC", {}).copy()
            flc_instance_config = {**base_flc_config, **current_flc_params}
            sim_dt = core_params.get("simulation", {}).get("dt", 0.02)
            controller_instance = FLCController(config=flc_instance_config, dt=sim_dt)

            sim_results = run_single_sim_and_extract_detailed_metrics(
                base_core_config=core_params,
                scenario_name=scenario_name,
                scenario_config=scenario_config,
                controller_name=f"FLC_opt_{scenario_name}",
                controller_instance=controller_instance,
            )

            if not sim_results["completed_successfully"] or not sim_results["metrics"]:
                total_cost += 1_000_000
                num_failures += 1
                continue

            metrics = sim_results["metrics"]
            cost = (
                metrics.get("transient_severity_score", 10)
                * cost_weights.get("lower_is_better", {}).get(
                    "transient_severity_score", 0.25
                )
                + metrics.get("thermal_transient_burden", 100)
                * cost_weights.get("lower_is_better", {}).get(
                    "thermal_transient_burden", 0.20
                )
                + metrics.get("control_effort_valve_sq_sum", 1)
                * cost_weights.get("lower_is_better", {}).get(
                    "control_effort_valve_sq_sum", 0.15
                )
            )
            cost -= metrics.get("grid_load_following_index", 0) * cost_weights.get(
                "higher_is_better", {}
            ).get("grid_load_following_index", 0.30)

            total_cost += cost

        except Exception as e:
            logger.error(
                f"Exception during FLC objective for '{scenario_name}': {e}",
                exc_info=True,
            )
            total_cost += 1_000_000
            num_failures += 1

    total_cost += num_failures * 500_000

    eval_duration = time.time() - start_time_objective_eval
    logger.info(
        f"--- FLC Objective Evaluated. Final Cost: {total_cost:.4f}. Duration: {eval_duration:.2f}s ---"
    )

    return total_cost if np.isfinite(total_cost) else 1e12


def optimize_flc_scaling_de(
    base_config: Dict[str, Any],
    config_file_path_for_validation: str,
    **cli_overrides: Any,
) -> Optional[Dict[str, float]]:

    from analysis.scenario_definitions import get_scenarios

    logger.info(
        "--- Starting FLC Scaling Factor Optimization (Differential Evolution) ---"
    )
    start_time = time.time()

    core_params = base_config.get("CORE_PARAMETERS", {})
    opt_settings = {
        **core_params.get("optimization", {}).get("FLC", {}),
        **cli_overrides,
    }
    validation_scenarios = get_scenarios(core_params)

    param_names = opt_settings.get(
        "param_names", ["error_scaling", "derror_scaling", "output_scaling"]
    )
    bounds = [
        tuple(opt_settings.get("bounds", {}).get(p, (0.1, 10.0))) for p in param_names
    ]

    de_params = {"maxiter": 30, "popsize": 15, "tol": 0.01, "workers": 1, "disp": True}
    logger.info(f"DE Params: {de_params}, Bounds: {bounds}")

    try:
        result = differential_evolution(
            _flc_objective_function,
            bounds,
            args=(base_config, validation_scenarios, param_names),
            **de_params,
        )

        logger.info(
            f"DE finished in {time.time() - start_time:.2f}s. Success: {result.success}"
        )

        if result.success and np.isfinite(result.fun):
            optimized_factors = dict(zip(param_names, map(float, result.x)))
            logger.info(f"Optimized FLC Factors: {optimized_factors}")

            project_root = os.path.abspath(
                os.path.join(os.path.dirname(config_file_path_for_validation), "..")
            )
            save_flc_params(optimized_factors, project_root)

            auto_validate_and_report(
                controller_type="FLC",
                controller_params=optimized_factors,
                config_path=config_file_path_for_validation,
            )
            return optimized_factors
        else:
            logger.error(f"FLC Optimization failed. Message: {result.message}")
            return None

    except Exception as e:
        logger.error(f"Exception during FLC differential_evolution: {e}", exc_info=True)
        return None


def save_flc_params(params: Dict[str, float], project_root: str):

    filepath = os.path.join(
        project_root, "config", "optimized_controllers", "FLC_optimized.yaml"
    )
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data_to_save = {"FLC_optimized": params}
        with open(filepath, "w") as f:
            yaml.dump(data_to_save, f, default_flow_style=False, sort_keys=False)
        logger.info(f"FLC parameters saved successfully to: {filepath}")
    except Exception as e:
        logger.error(f"Failed to save FLC parameters to {filepath}: {e}", exc_info=True)
