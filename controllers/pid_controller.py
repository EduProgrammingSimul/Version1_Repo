import numpy as np
import logging
from .base_controller import BaseController
from typing import Dict, Any

logger = logging.getLogger(__name__)


class PIDController(BaseController):

    MEASUREMENT_OBS_INDEX = 4

    def __init__(self, config: Dict[str, Any], dt: float):

        super().__init__(config, dt)
        try:

            self.kp = float(config.get("kp", 0.20))
            self.ki = float(config.get("ki", 0.04))
            self.kd = float(config.get("kd", 0.05))
            self.setpoint = float(config.get("setpoint", 1800.0))

            output_limits = config.get("output_limits", (0.0, 1.0))
            self.output_min, self.output_max = float(output_limits[0]), float(
                output_limits[1]
            )

            if self.output_min >= self.output_max:
                raise ValueError(
                    f"PID output_min ({self.output_min}) must be less than output_max ({self.output_max})."
                )

            self.deriv_filter_tau = float(config.get("deriv_filter_tau", 0.05))
            self.use_filter = (
                self.deriv_filter_tau > (self.dt * 1.5) if self.dt > 0 else False
            )

            self._integral = 0.0
            self._previous_error = 0.0
            self._derivative_state = 0.0

            logger.info(
                f"PID Controller ready: Kp={self.kp:.4f}, Ki={self.ki:.4f}, Kd={self.kd:.4f}"
            )

        except (ValueError, TypeError) as e:
            logger.error(
                f"Failed to initialize PIDController due to invalid config: {e}",
                exc_info=True,
            )
            raise

    def step(self, observation: np.ndarray) -> float:

        if self.dt <= 0:
            return 0.5 * (self.output_max + self.output_min)

        try:
            measurement = observation[self.MEASUREMENT_OBS_INDEX]
        except IndexError:
            logger.error(
                f"Observation vector length {len(observation)} is too short. Using setpoint as measurement."
            )
            measurement = self.setpoint

        error = self.setpoint - measurement
        p_term = self.kp * error
        raw_derivative = (error - self._previous_error) / self.dt

        if self.use_filter:
            d_deriv_state_dt = (
                raw_derivative - self._derivative_state
            ) / self.deriv_filter_tau
            self._derivative_state += d_deriv_state_dt * self.dt
            effective_derivative = self._derivative_state
        else:
            effective_derivative = raw_derivative

        d_term = self.kd * effective_derivative
        output_before_i = p_term + d_term

        is_at_max_and_saturating = (output_before_i >= self.output_max) and (error > 0)
        is_at_min_and_saturating = (output_before_i <= self.output_min) and (error < 0)

        if not is_at_max_and_saturating and not is_at_min_and_saturating:
            self._integral += error * self.dt

        i_term = self.ki * self._integral
        output_unclamped = p_term + i_term + d_term
        output_final = np.clip(output_unclamped, self.output_min, self.output_max)
        self._previous_error = error

        return float(output_final)

    def reset(self):

        super().reset()
        self._integral = 0.0
        self._previous_error = 0.0
        self._derivative_state = 0.0
        logger.info("PID internal states reset.")

    def update_parameters(self, new_params: Dict[str, Any]):

        super().update_parameters(new_params)
        self.kp = float(new_params.get("kp", self.kp))
        self.ki = float(new_params.get("ki", self.ki))
        self.kd = float(new_params.get("kd", self.kd))
        logger.info(
            f"PID Parameters Updated Live: Kp={self.kp:.4f}, Ki={self.ki:.4f}, Kd={self.kd:.4f}"
        )

    def get_parameters(self) -> Dict[str, Any]:

        super().get_parameters()
        return {"kp": self.kp, "ki": self.ki, "kd": self.kd, "setpoint": self.setpoint}
