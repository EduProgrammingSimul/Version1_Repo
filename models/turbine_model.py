import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class TurbineModel:

    def __init__(self, turbine_params: Dict[str, Any], coupling_params: Dict[str, Any]):

        logger.info("Initializing robust TurbineModel.")
        try:

            self.eta_transfer = coupling_params["eta_transfer"]
            self.tau_delay = coupling_params.get("tau_delay", 2.0)

            self.tau_t = turbine_params["tau_t"]
            self.tau_v = turbine_params["tau_v"]
            self.omega_nominal_rpm = turbine_params["omega_nominal_rpm"]

            self.mechanical_power: float = 0.0
            self.valve_position: float = 0.8
            self.speed_rpm: float = self.omega_nominal_rpm

            logger.info("TurbineModel initialized successfully.")

        except KeyError as e:
            logger.error(
                f"FATAL: Missing required key in turbine/coupling params: {e}",
                exc_info=True,
            )
            raise

    def reset(self, initial_mech_power: float = 2800.0, initial_valve_pos: float = 0.8):

        logger.info(f"Resetting TurbineModel.")
        self.mechanical_power = initial_mech_power
        self.valve_position = initial_valve_pos
        self.speed_rpm = self.omega_nominal_rpm
        logger.debug(
            f"Reset state: MechPower={self.mechanical_power:.2f} MW, ValvePos={self.valve_position:.3f}"
        )

    def step(self, dt: float, thermal_power_mw: float, valve_command: float) -> float:

        dv_dt = (1 / self.tau_v) * (valve_command - self.valve_position)
        self.valve_position += dv_dt * dt
        self.valve_position = np.clip(self.valve_position, 0.0, 1.0)

        effective_steam_power = self.valve_position * (
            self.eta_transfer * thermal_power_mw
        )

        dp_mech_dt = (1 / self.tau_t) * (effective_steam_power - self.mechanical_power)
        self.mechanical_power += dp_mech_dt * dt

        logger.debug(
            f"Turbine step: V_cmd={valve_command:.3f}, V_act={self.valve_position:.3f}, P_mech={self.mechanical_power:.2f} MW"
        )

        return self.mechanical_power
