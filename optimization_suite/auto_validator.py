import logging
import os
import time
from typing import Any, Dict, Optional

from analysis.parameter_manager import ParameterManager
from analysis.scenario_definitions import get_scenarios
from analysis.scenario_executor import ScenarioExecutor
from analysis.metrics_engine import MetricsEngine
from analysis.report_generator import ReportGenerator

from controllers import load_controller

logger = logging.getLogger(__name__)


def auto_validate_and_report(
    controller_identifier: str, config_path: str, save_tag: Optional[str] = None
) -> Optional[str]:

    timestamp = save_tag or time.strftime("%Y%m%d_%H%M%S")

    report_controller_name = os.path.splitext(os.path.basename(controller_identifier))[
        0
    ]

    logger.info(
        f"--- Auto-Validation Started for Controller: {report_controller_name} ---"
    )

    try:
        param_manager = ParameterManager(config_filepath=config_path)
        full_config = param_manager.get_all_parameters()
        core_config = full_config.get("CORE_PARAMETERS")
        if not core_config:
            raise ValueError("'CORE_PARAMETERS' not found in configuration.")

        scenarios = get_scenarios(core_config)
        if not scenarios:
            raise ValueError("No scenarios found for validation.")

        executor = ScenarioExecutor(full_config)
        metrics_engine = MetricsEngine(core_config)
        reporter = ReportGenerator(core_config.get("reporting", {}), core_config)

        dt_sim = core_config.get("simulation", {}).get("dt", 0.02)

        controller_instance, loaded_name = load_controller(
            controller_name_or_path=controller_identifier,
            base_config=full_config,
            dt=dt_sim,
        )

        if not controller_instance:
            logger.error(
                f"Failed to load controller for auto-validation using identifier: {controller_identifier}"
            )
            return None

        all_scenario_metrics: Dict[str, Dict[str, Dict[str, float]]] = {
            s_name: {} for s_name in scenarios
        }
        for scenario_name, scenario_conf in scenarios.items():
            logger.info(
                f"--- Validating '{loaded_name}' on Scenario: {scenario_name} ---"
            )
            controller_instance.reset()
            results_df = executor.execute(
                scenario_name, scenario_conf, loaded_name, controller_instance
            )

            metrics = metrics_engine.calculate(results_df, core_config)
            all_scenario_metrics[scenario_name][loaded_name] = metrics

        report_filename = f"validation_report_{report_controller_name}_{timestamp}.md"
        final_report_path = reporter.generate_report(
            all_scenario_metrics, scenarios, report_filename=report_filename
        )
        logger.info(f"--- Auto-Validation Finished. Report at: {final_report_path} ---")
        return final_report_path

    except Exception as e:
        logger.error(f"Auto-validation failed critically: {e}", exc_info=True)
        return None
