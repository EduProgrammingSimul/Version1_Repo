import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ReactorModel:

    def __init__(self, params: Dict[str, Any]):

        logger.info("Initializing robust ReactorModel.")
        try:

            self.beta_i = np.array(params["beta_i"])
            self.lambda_i = np.array(params["lambda_i"])
            self.Lambda = params["Lambda"]
            self.beta_total = params["beta_total"]

            self.alpha_f = params["alpha_f"]
            self.alpha_c = params["alpha_c"]

            self.C_f = params["C_f"]
            self.C_c = params["C_c"]
            self.Omega = params["Omega"]

            self.P0 = params["P0"]
            self.T_inlet = params["T_inlet"]
            self.T_coolant0 = params["T_coolant0"]
            self.T_fuel0 = params["T_fuel0"]

            self.power_level: float = 0.0
            self.precursor_concentrations: np.ndarray = np.zeros_like(self.beta_i)
            self.T_fuel: float = 0.0
            self.T_moderator: float = 0.0

            logger.info("ReactorModel initialized successfully.")

        except KeyError as e:
            logger.error(
                f"FATAL: Missing required key in reactor_params: {e}", exc_info=True
            )
            raise

    def reset(self, initial_power_fraction: float = 1.0):

        logger.info(
            f"Resetting ReactorModel to {initial_power_fraction*100:.1f}% power."
        )

        self.power_level = initial_power_fraction

        self.T_fuel = self.T_fuel0 * initial_power_fraction
        self.T_moderator = self.T_coolant0 * initial_power_fraction

        if self.Lambda > 1e-9:
            self.precursor_concentrations = (
                self.beta_i / (self.lambda_i * self.Lambda)
            ) * self.power_level
        else:
            self.precursor_concentrations.fill(0.0)

        logger.debug(
            f"Reset state: Power={self.power_level:.3f}, T_fuel={self.T_fuel:.2f}C"
        )

    def step(self, dt: float, rod_reactivity: float) -> float:

        delta_T_f = self.T_fuel - self.T_fuel0
        delta_T_c = self.T_moderator - self.T_coolant0
        rho_feedback = self.alpha_f * delta_T_f + self.alpha_c * delta_T_c

        total_reactivity = rho_feedback + rod_reactivity

        lambda_c_sum = np.sum(self.lambda_i * self.precursor_concentrations)

        dp_dt = (
            (total_reactivity - self.beta_total) / self.Lambda
        ) * self.power_level + lambda_c_sum
        self.power_level += dp_dt * dt

        dc_dt = (
            self.beta_i / self.Lambda
        ) * self.power_level - self.lambda_i * self.precursor_concentrations
        self.precursor_concentrations += dc_dt * dt

        generated_power_mw = self.power_level * self.P0

        dtf_dt = (1 / self.C_f) * (
            generated_power_mw - self.Omega * (self.T_fuel - self.T_moderator)
        )
        self.T_fuel += dtf_dt * dt

        dtc_dt = (1 / self.C_c) * (self.Omega * (self.T_fuel - self.T_moderator))
        self.T_moderator += dtc_dt * dt

        self.power_level = max(0.0, self.power_level)

        logger.debug(
            f"Reactor step: P={self.power_level * self.P0:.2f} MW, Rho={total_reactivity*1e5:.2f} pcm"
        )

        return self.power_level * self.P0
