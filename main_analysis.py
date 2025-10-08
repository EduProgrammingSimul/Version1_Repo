import logging
import os
import sys
import argparse
import pandas as pd
from typing import Dict, Any, Optional, List


import matplotlib

matplotlib.use("Agg")


project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from analysis.parameter_manager import ParameterManager
from analysis.scenario_definitions import get_scenarios
from analysis.scenario_executor import ScenarioExecutor
from analysis.metrics_engine import MetricsEngine
from analysis.report_generator import ReportGenerator
from controllers import load_controller


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def run_full_analysis(
    config_path: str, controllers_to_run: List[str], generate_report: bool = True
) -> Optional[Dict[str, Dict[str, Dict[str, float]]]]:

    logger.info(
        "=" * 60
        + f"\n DTAF Comparative Analysis v3.5 Starting ".center(60)
        + "\n"
        + "=" * 60
    )

    try:
        param_manager = ParameterManager(config_filepath=config_path)
        full_config = param_manager.get_all_parameters()
        core_config = full_config.get("CORE_PARAMETERS")
        if not core_config:
            raise ValueError(
                "'CORE_PARAMETERS' key missing from the configuration file."
            )
        scenarios = get_scenarios(core_config)
    except Exception as e:
        logger.critical(
            f"Failed to load configuration or scenarios: {e}", exc_info=True
        )
        return None

    executor = ScenarioExecutor(full_config)
    metrics_engine = MetricsEngine(core_config)

    logger.info(f"Controllers to be tested: {controllers_to_run}")

    controllers_to_test = {}
    dt_sim = core_config.get("simulation", {}).get("dt", 0.02)
    for name_or_path in controllers_to_run:
        instance, report_name = load_controller(name_or_path, full_config, dt_sim)
        if instance:
            controllers_to_test[report_name] = instance
        else:
            logger.error(f"Could not load controller: {name_or_path}")

    if not controllers_to_test:
        logger.critical("No valid controllers could be loaded for analysis. Exiting.")
        return None

    all_scenario_metrics: Dict[str, Dict[str, Dict[str, float]]] = {
        s_name: {} for s_name in scenarios
    }

    for scenario_name, scenario_conf in scenarios.items():
        logger.info(f"\n===== Starting Scenario: {scenario_name} =====")
        for ctrl_name, ctrl_instance in controllers_to_test.items():
            logger.info(f"  --- Running Controller: {ctrl_name} ---")
            ctrl_instance.reset()

            results_df = executor.execute(
                scenario_name, scenario_conf, ctrl_name, ctrl_instance
            )

            metrics = metrics_engine.calculate(results_df, scenario_conf)
            all_scenario_metrics[scenario_name][ctrl_name] = metrics

    if generate_report:
        logger.info("\nGenerating full report and visualizations...")
        reporter = ReportGenerator(core_config.get("reporting", {}), core_config)
        reporter.generate_report(all_scenario_metrics, scenarios)

    logger.info(
        "=" * 60
        + f"\n DTAF Comparative Analysis Finished ".center(60)
        + "\n"
        + "=" * 60
    )
    return all_scenario_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DTAF Comparative Analysis")
    parser.add_argument(
        "--controllers",
        nargs="*",
        help="Specify controller names or direct paths to .zip models.",
    )
    args = parser.parse_args()

    controllers = args.controllers
    if not controllers:

        project_root = os.path.abspath(os.path.dirname(__file__))
        controllers = ["PID_optimized", "FLC_optimized"]

        search_dirs = [
            os.path.join(project_root, "results", "tuning_cache"),
            os.path.join(project_root, "config", "optimized_controllers"),
            os.path.join(project_root, "results", "rl_models", "champion_agent_tuned"),
        ]

        latest_agent_path = ""
        latest_mtime = 0

        for d in search_dirs:
            if not os.path.exists(d):
                continue
            for f in os.listdir(d):
                if f.endswith(".zip"):
                    path = os.path.join(d, f)
                    try:
                        mtime = os.path.getmtime(path)
                        if mtime > latest_mtime:
                            latest_mtime = mtime
                            latest_agent_path = path
                    except FileNotFoundError:
                        continue

        if latest_agent_path:
            logger.info(f"Found latest RL agent: {latest_agent_path}")
            controllers.append(latest_agent_path)

    config_file = os.path.join(project_root, "config", "parameters.py")
    run_full_analysis(config_path=config_file, controllers_to_run=controllers)
