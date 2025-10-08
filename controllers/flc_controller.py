import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import logging
from .base_controller import BaseController
from typing import Dict, Any

logger = logging.getLogger(__name__)


class FLCController(BaseController):

    SPEED_RPM_OBS_INDEX = 4

    def __init__(self, config: Dict[str, Any], dt: float):

        super().__init__(config, dt)
        logger.info("Initializing FLCController v2.2...")
        try:

            self.setpoint = float(config.get("setpoint", 1800.0))
            self.output_min, self.output_max = map(
                float, config.get("output_limits", (0.0, 1.0))
            )
            self.dvalve_limit_per_step = float(config.get("dvalve_limit", 0.1))
            self.max_speed_error = float(config.get("max_speed_error", 100.0))
            self.max_speed_error_change = float(
                config.get("max_speed_error_change", 50.0)
            )
            self.max_dvalve_abs = float(config.get("max_dvalve_abs", 0.05))
            self.error_scaling = float(config.get("error_scaling", 1.0))
            self.derror_scaling = float(config.get("derror_scaling", 1.0))
            self.output_scaling = float(config.get("output_scaling", 1.0))

            self._build_fuzzy_system()

            self._last_error = 0.0
            self.initial_valve_pos = float(config.get("initial_valve_pos", 0.8))
            self._current_valve_position = np.clip(
                self.initial_valve_pos, self.output_min, self.output_max
            )

            logger.info("FLC Controller ready.")

        except (ValueError, TypeError) as e:
            logger.error(
                f"Failed to initialize FLCController due to invalid config: {e}",
                exc_info=True,
            )
            raise

    def _build_fuzzy_system(self):

        error_universe = np.linspace(-self.max_speed_error, self.max_speed_error, 31)
        derror_universe = np.linspace(
            -self.max_speed_error_change, self.max_speed_error_change, 31
        )
        dvalve_universe = np.linspace(-self.max_dvalve_abs, self.max_dvalve_abs, 31)
        self.error_var = ctrl.Antecedent(error_universe, "error")
        self.derror_var = ctrl.Antecedent(derror_universe, "derror")
        self.dvalve_var = ctrl.Consequent(dvalve_universe, "dvalve")
        self.error_var.automf(names=["NB", "NS", "ZE", "PS", "PB"])
        self.derror_var.automf(names=["NB", "NS", "ZE", "PS", "PB"])
        self.dvalve_var.automf(
            names=["NB", "NS", "ZE", "PS", "PB"], variable_type="trimf"
        )
        rules = [
            ctrl.Rule(
                self.error_var["NB"] | self.derror_var["PB"], self.dvalve_var["NB"]
            ),
            ctrl.Rule(
                self.error_var["NS"] & self.derror_var["PS"], self.dvalve_var["NS"]
            ),
            ctrl.Rule(
                self.error_var["ZE"] & self.derror_var["ZE"], self.dvalve_var["ZE"]
            ),
            ctrl.Rule(
                self.error_var["PS"] & self.derror_var["NS"], self.dvalve_var["PS"]
            ),
            ctrl.Rule(
                self.error_var["PB"] | self.derror_var["NB"], self.dvalve_var["PB"]
            ),
        ]
        self.valve_ctrl_sys = ctrl.ControlSystem(rules)
        self.valve_simulation = ctrl.ControlSystemSimulation(self.valve_ctrl_sys)

    def step(self, observation: np.ndarray) -> float:

        if self.dt <= 0:
            return self._current_valve_position

        try:
            measurement = observation[self.SPEED_RPM_OBS_INDEX]
        except IndexError:
            logger.error(f"Observation vector length is too short. Using setpoint.")
            measurement = self.setpoint

        current_error = self.setpoint - measurement
        delta_error = (current_error - self._last_error) / self.dt
        scaled_error = np.clip(
            current_error * self.error_scaling,
            -self.max_speed_error,
            self.max_speed_error,
        )
        scaled_derror = np.clip(
            delta_error * self.derror_scaling,
            -self.max_speed_error_change,
            self.max_speed_error_change,
        )

        try:
            self.valve_simulation.input["error"] = scaled_error
            self.valve_simulation.input["derror"] = scaled_derror
            self.valve_simulation.compute()
            delta_valve_fuzzy = self.valve_simulation.output["dvalve"]
            if np.isnan(delta_valve_fuzzy):
                delta_valve_fuzzy = 0.0
        except Exception:
            delta_valve_fuzzy = 0.0

        delta_valve_scaled = delta_valve_fuzzy * self.output_scaling
        delta_valve_limited = np.clip(
            delta_valve_scaled, -self.dvalve_limit_per_step, self.dvalve_limit_per_step
        )

        self._current_valve_position += delta_valve_limited
        output_final = np.clip(
            self._current_valve_position, self.output_min, self.output_max
        )
        self._current_valve_position = output_final
        self._last_error = current_error

        return float(output_final)

    def reset(self):

        super().reset()
        self._last_error = 0.0
        self._current_valve_position = np.clip(
            self.initial_valve_pos, self.output_min, self.output_max
        )
        if self.valve_simulation:
            self.valve_simulation.reset()
        logger.info(f"FLC internal states reset.")

    def update_parameters(self, new_params: Dict[str, Any]):

        super().update_parameters(new_params)

    def get_parameters(self) -> Dict[str, Any]:

        super().get_parameters()

        return {}
