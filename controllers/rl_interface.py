import os
import logging
import numpy as np
from typing import Dict, Any, Optional

from .base_controller import BaseController
from stable_baselines3.common.base_class import BaseAlgorithm

logger = logging.getLogger(__name__)


class RLAgentWrapper(BaseController):

    def __init__(self, model: BaseAlgorithm, config: Dict[str, Any], dt: float):

        super().__init__(config, dt)
        self.model = model
        logger.info(
            f"RLAgentWrapper initialized with model: {model.__class__.__name__}"
        )

    def step(self, observation: np.ndarray) -> float:

        action, _ = self.model.predict(observation, deterministic=True)
        return float(action[0])

    def reset(self):

        super().reset()

    def update_parameters(self, new_params: Dict[str, Any]):

        super().update_parameters(new_params)
        logger.warning(
            "update_parameters called on RLAgentWrapper, but this is not supported."
        )

    def get_parameters(self) -> Dict[str, Any]:

        super().get_parameters()
        try:
            return {
                "model_class": self.model.__class__.__name__,
                "policy": self.model.policy.__class__.__name__,
            }
        except AttributeError:
            return {"model_class": "Unknown"}


def load_rl_agent_from_file(
    path: str, algorithm: str = "SAC"
) -> Optional[BaseAlgorithm]:

    logger.info(f"Utility function loading RL agent from: {path}")
    if not os.path.exists(path):
        logger.error(f"RL model file not found at: {path}")
        return None

    try:
        if algorithm.upper() == "SAC":
            from stable_baselines3 import SAC as AgentClass
        elif algorithm.upper() == "PPO":
            from stable_baselines3 import PPO as AgentClass
        else:
            logger.error(f"Unsupported algorithm '{algorithm}' for loading.")
            return None

        model = AgentClass.load(path)
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load RL model from {path}: {e}", exc_info=True)
        return None
