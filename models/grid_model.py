import numpy as np
import logging
from typing import Dict, Any, Callable

logger = logging.getLogger(__name__)


class GridModel:

    def __init__(self, grid_params: Dict[str, Any], sim_params: Dict[str, Any]):

        logger.info("Initializing robust GridModel.")
        try:

            self.H = grid_params["H"]
            self.D = grid_params["D"]
            self.f_nominal = grid_params["f_nominal"]
            self.S_base = grid_params["S_base"]

            self.frequency: float = self.f_nominal
            self.omega_pu: float = 1.0
            self.delta: float = 0.0
            self.current_demand: float = 0.0
            self.load_profile_func: Optional[Callable[[float, int], float]] = None

            logger.info("GridModel initialized successfully.")

        except KeyError as e:
            logger.error(
                f"FATAL: Missing required key in grid_params: {e}", exc_info=True
            )
            raise

    def reset(self, initial_load_mw: float):

        logger.info(f"Resetting GridModel with initial load: {initial_load_mw:.2f} MW")
        self.frequency = self.f_nominal
        self.omega_pu = 1.0
        self.delta = 0.0
        self.current_demand = initial_load_mw

        self.load_profile_func = lambda time_s, step: initial_load_mw

    def set_load_profile(self, load_func: Callable[[float, int], float]):

        self.load_profile_func = load_func
        logger.info("Dynamic load profile has been set for the GridModel.")

    def step(self, dt: float, mechanical_power_mw: float, time_s: float, step_num: int):

        self.current_demand = self.load_profile_func(time_s, step_num)

        p_m_pu = mechanical_power_mw / self.S_base
        p_e_pu = self.current_demand / self.S_base

        d_omega_pu_dt = (1 / (2 * self.H)) * (
            p_m_pu - p_e_pu - self.D * (self.omega_pu - 1.0)
        )

        self.omega_pu += d_omega_pu_dt * dt

        self.frequency = self.omega_pu * self.f_nominal

        d_delta_dt = (self.omega_pu - 1.0) * 2 * np.pi * self.f_nominal
        self.delta += d_delta_dt * dt

        logger.debug(
            f"Grid step: Freq={self.frequency:.4f} Hz, P_mech={mechanical_power_mw:.2f} MW, P_elec={self.current_demand:.2f} MW"
        )
