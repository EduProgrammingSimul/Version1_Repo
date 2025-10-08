import numpy as np
from abc import ABC, abstractmethod
import logging
from typing import Dict, Any


logger = logging.getLogger(__name__)


class BaseController(ABC):

    @abstractmethod
    def __init__(self, config: Dict[str, Any], dt: float):

        if not isinstance(config, dict):
            logger.error(
                "Initialization failed: Controller config must be a dictionary."
            )
            raise TypeError("Controller config must be a dictionary.")
        if not isinstance(dt, (float, int)) or dt <= 0:
            logger.error(
                f"Initialization failed: Time step dt must be a positive number, got {dt}."
            )
            raise ValueError("Time step dt must be a positive number.")

        self.config = config
        self.dt = dt

        logger.info(
            f"Initializing BaseController implementation: {self.__class__.__name__}"
        )
        logger.debug(f"  > Config provided: {config}")
        logger.debug(f"  > Time step (dt): {dt}")

    @abstractmethod
    def step(self, observation: np.ndarray) -> float:

        pass

    @abstractmethod
    def reset(self):

        logger.info(f"Resetting controller state for: {self.__class__.__name__}")

        pass

    @abstractmethod
    def update_parameters(self, new_params: Dict[str, Any]):

        logger.info(
            f"Attempting to update parameters for {self.__class__.__name__} with: {new_params}"
        )

        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:

        logger.info(f"Getting parameters for {self.__class__.__name__}")

        pass
