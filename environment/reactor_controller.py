import numpy as np
import logging

logger = logging.getLogger(__name__)


class ReactorController:

    def __init__(self, dt: float):

        self.kp = 0.00008
        self.ki = 0.00002

        self.dt = dt
        self.setpoint = 306.5

        self._integral = 0.0

        self._reactivity_limits = (-0.005, 0.005)

    def reset(self, setpoint: float = 306.5):

        self._integral = 0.0
        self.setpoint = setpoint
        logger.debug(f"ReactorController reset with setpoint: {self.setpoint:.2f} C")

    def step(self, current_moderator_temp: float) -> float:

        if self.dt <= 0:
            return 0.0

        error = self.setpoint - current_moderator_temp

        self._integral += error * self.dt
        self._integral = np.clip(self._integral, -5.0, 5.0)

        reactivity_output = (self.kp * error) + (self.ki * self._integral)

        return float(
            np.clip(
                reactivity_output,
                self._reactivity_limits[0],
                self._reactivity_limits[1],
            )
        )
