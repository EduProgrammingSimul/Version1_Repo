import logging
import os
import yaml
import copy
from typing import Dict, Any, Optional, Tuple

from .base_controller import BaseController
from .pid_controller import PIDController
from .flc_controller import FLCController
from .rl_interface import RLAgentWrapper, load_rl_agent_from_file

__all__ = [
    "BaseController",
    "PIDController",
    "FLCController",
    "RLAgentWrapper",
    "load_rl_agent_from_file",
    "load_controller",
    "create_controller_with_custom_config",
]

logger = logging.getLogger(__name__)


def create_controller_with_custom_config(
    controller_type: str,
    custom_params: Dict[str, Any],
    base_config: Dict[str, Any],
    dt: float,
) -> Optional[BaseController]:

    core_params = base_config.get("CORE_PARAMETERS", {})
    base_name = controller_type.split("_")[0].upper()

    final_config = copy.deepcopy(core_params.get("controllers", {}).get(base_name, {}))
    final_config.update(custom_params)

    logger.info(f"Creating '{controller_type}' with custom configuration.")

    try:
        if base_name == "PID":
            return PIDController(config=final_config, dt=dt)
        elif base_name == "FLC":
            return FLCController(config=final_config, dt=dt)
        else:
            logger.error(
                f"Cannot create controller of type '{base_name}' with custom config."
            )
            return None
    except Exception as e:
        logger.error(
            f"Failed to instantiate controller '{controller_type}' with custom config: {e}",
            exc_info=True,
        )
        return None


def load_controller(
    controller_name_or_path: str, base_config: Dict[str, Any], dt: float
) -> Tuple[Optional[BaseController], str]:

    core_params = base_config.get("CORE_PARAMETERS", {})

    if os.path.exists(controller_name_or_path) and controller_name_or_path.endswith(
        ".zip"
    ):
        logger.info(
            f"Loading controller from direct path: '{controller_name_or_path}'..."
        )
        model_name = os.path.splitext(os.path.basename(controller_name_or_path))[0]
        loaded_model = load_rl_agent_from_file(controller_name_or_path, algorithm="SAC")
        if loaded_model:
            return (
                RLAgentWrapper(model=loaded_model, config=core_params, dt=dt),
                model_name,
            )
        return None, model_name

    logger.info(f"Loading controller by name: '{controller_name_or_path}'...")
    controller_name = controller_name_or_path
    base_name = controller_name.split("_optimized")[0].upper()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    optimized_config_dir = os.path.join(project_root, "config", "optimized_controllers")

    final_config = copy.deepcopy(core_params.get("controllers", {}).get(base_name, {}))

    if "_optimized" in controller_name.lower() and base_name in ["PID", "FLC"]:
        opt_path = os.path.join(optimized_config_dir, f"{controller_name}.yaml")
        if os.path.exists(opt_path):
            try:
                with open(opt_path, "r") as f:
                    loaded_opt = yaml.safe_load(f)
                yaml_key = next(
                    (k for k in loaded_opt if k.lower() == controller_name.lower()),
                    None,
                )
                if yaml_key:
                    final_config.update(loaded_opt[yaml_key])
            except Exception as e:
                logger.error(f"Could not parse YAML for {controller_name}: {e}")
        else:
            logger.warning(
                f"Optimized .yaml not found for '{controller_name}'. Using base config."
            )

    try:
        if base_name == "PID":
            return PIDController(config=final_config, dt=dt), controller_name
        elif base_name == "FLC":
            return FLCController(config=final_config, dt=dt), controller_name
        elif base_name.startswith("RL_AGENT"):
            model_path_abs = os.path.join(
                optimized_config_dir, f"{controller_name}.zip"
            )
            if not os.path.exists(model_path_abs):
                return None, controller_name
            loaded_model = load_rl_agent_from_file(model_path_abs, algorithm="SAC")
            if loaded_model:
                return (
                    RLAgentWrapper(model=loaded_model, config=core_params, dt=dt),
                    controller_name,
                )
            return None, controller_name
        return None, controller_name
    except Exception as e:
        logger.error(
            f"Failed to instantiate controller '{controller_name}': {e}", exc_info=True
        )
        return None, controller_name
