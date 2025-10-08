import logging
import os
from typing import Any, Dict
import importlib

from analysis.parameter_manager import ParameterManager
from analysis.scenario_definitions import get_scenarios

logger = logging.getLogger(__name__)


class OptimizationManager:

    def __init__(self, config_path: str):

        logger.info(f"Initializing OptimizationManager with config path: {config_path}")
        self.config_path = config_path
        self.project_root = os.path.abspath(
            os.path.join(os.path.dirname(config_path), "..")
        )
        try:
            self.param_manager = ParameterManager(config_filepath=self.config_path)
        except Exception as e:
            logger.error(f"Failed to create ParameterManager: {e}", exc_info=True)
            raise RuntimeError(f"Could not initialize ParameterManager: {e}") from e

        self.full_config = self.param_manager.get_all_parameters()
        if not self.full_config or "CORE_PARAMETERS" not in self.full_config:
            raise ValueError(
                "Parameters not loaded correctly or 'CORE_PARAMETERS' key is missing."
            )

        logger.info("OptimizationManager initialized successfully.")

    def run_optimization(self, controller_type: str, **kwargs: Any) -> bool:

        logger.info(
            f"--- Received optimization request for Controller: {controller_type} ---"
        )

        if kwargs:
            self.param_manager.update_with_cli_args(kwargs)
            self.full_config = self.param_manager.get_all_parameters()

        try:
            if controller_type.upper() == "PID":
                pid_optimizer_module = importlib.import_module(
                    "optimization_suite.pid_global_optimizer"
                )
                logger.info("Dynamically loaded PID optimizer.")
                return pid_optimizer_module.tune_pid_global_de(
                    base_config=self.full_config,
                    config_file_path_for_validation=self.config_path,
                    **kwargs,
                )

            elif controller_type.upper() == "FLC":
                flc_optimizer_module = importlib.import_module(
                    "optimization_suite.flc_optimizer"
                )
                logger.info("Dynamically loaded FLC optimizer.")
                return flc_optimizer_module.optimize_flc_scaling_de(
                    base_config=self.full_config,
                    config_file_path_for_validation=self.config_path,
                    **kwargs,
                )

            elif controller_type.upper() == "RL_AGENT":
                rl_trainer_module = importlib.import_module(
                    "optimization_suite.rl_trainer"
                )
                logger.info("Dynamically loaded RL trainer.")

                trainer = rl_trainer_module.RLTrainer(
                    base_config_full=self.full_config, config_path=self.config_path
                )
                return trainer.train()

            else:
                logger.error(
                    f"Unknown controller type for optimization: '{controller_type}'"
                )
                return False

        except ImportError as e:
            logger.critical(
                f"Failed to import the necessary module for '{controller_type}': {e}",
                exc_info=True,
            )
            return False
        except Exception as e:
            logger.critical(
                f"An unexpected critical error occurred during the '{controller_type}' optimization task: {e}",
                exc_info=True,
            )
            return False
