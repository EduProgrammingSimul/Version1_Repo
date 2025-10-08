import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging
from typing import Dict, Any, Optional, List
import random
import copy


from models.reactor_model import ReactorModel
from models.turbine_model import TurbineModel
from models.grid_model import GridModel


from .reactor_controller import ReactorController

logger = logging.getLogger(__name__)


class PWRGymEnvUnified(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        reactor_params: Dict[str, Any],
        turbine_params: Dict[str, Any],
        grid_params: Dict[str, Any],
        coupling_params: Dict[str, Any],
        sim_params: Dict[str, Any],
        safety_limits: Dict[str, Any],
        rl_normalization_factors: Dict[str, float],
        all_scenarios_definitions: Dict[str, Any],
        initial_scenario_name: Any,
        is_training_env: bool = False,
        rl_training_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.reactor_base_params = reactor_params
        self.turbine_base_params = turbine_params
        self.grid_base_params = grid_params
        self.coupling_base_params = coupling_params
        self.sim_params = sim_params
        self.safety_limits = safety_limits
        self.norm_factors = rl_normalization_factors
        self.all_scenarios = all_scenarios_definitions
        self.is_training_env = is_training_env
        self.rl_training_config = rl_training_config or {}

        self.dt = self.sim_params.get("dt", 0.02)
        self.max_steps = self.sim_params.get("max_steps", 5000)
        self.active_scenario_names: List[str] = (
            initial_scenario_name
            if isinstance(initial_scenario_name, list)
            else [initial_scenario_name]
        )

        self.base_reward_weights = self.rl_training_config.get("reward_weights", {})
        self.active_reward_weights = self.base_reward_weights.copy()

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

        self.reactor_controller = ReactorController(dt=self.dt)

        self._initialize_internal_state()

        self.reset()

    def _initialize_internal_state(self):

        self.current_step = 0
        self.last_valve_pos = 0.8
        self.last_action = 0.5
        self.last_freq_error = 0.0

    def set_active_scenario(self, scenario_names: List[str]):

        self.active_scenario_names = scenario_names
        logger.info(
            f"Environment active scenarios set to: {self.active_scenario_names} for next reset."
        )

    def update_reward_weights(self, new_weights: Dict[str, float]):

        self.active_reward_weights = self.base_reward_weights.copy()
        self.active_reward_weights.update(new_weights)
        logger.warning(
            f"Reward weights updated by curriculum: {self.active_reward_weights}"
        )

    def _normalize_obs(self, raw_obs: np.ndarray) -> np.ndarray:

        norm_obs = np.zeros_like(raw_obs, dtype=np.float32)
        p_mw, t_fuel, v_pos, freq, speed, p_err = raw_obs

        norm_obs[0] = (
            p_mw - self.norm_factors.get("reactor_power_mw", 3411.0) * 0.9
        ) / (self.norm_factors.get("reactor_power_mw", 3411.0) * 0.5)
        norm_obs[1] = (t_fuel - self.reactor_base_params.get("T_fuel0", 829.0)) / (
            self.norm_factors.get("T_fuel", 2800.0) * 0.5
        )
        norm_obs[2] = (v_pos - 0.5) * 2.0
        norm_obs[3] = (
            freq - self.norm_factors.get("grid_frequency_hz", 60.0)
        ) / self.safety_limits.get("freq_deviation_limit_hz", 1.0)
        norm_obs[4] = (speed - self.norm_factors.get("speed_rpm", 1800.0)) / (
            self.norm_factors.get("speed_rpm", 1800.0) * 0.25
        )
        norm_obs[5] = p_err / self.norm_factors.get("power_error", 500.0)

        return np.clip(norm_obs, -1.0, 1.0)

    def _get_raw_obs_and_info(self) -> (np.ndarray, Dict[str, Any]):

        power_error = self.turbine.mechanical_power - self.grid.current_demand

        raw_obs = np.array(
            [
                self.reactor.power_level * self.reactor.P0,
                self.reactor.T_fuel,
                self.turbine.valve_position,
                self.grid.frequency,
                self.turbine.speed_rpm,
                power_error,
            ],
            dtype=np.float32,
        )

        info = {
            "time_s": self.current_step * self.dt,
            "reactor_power_mw": raw_obs[0],
            "T_fuel": raw_obs[1],
            "T_moderator": self.reactor.T_moderator,
            "v_pos_actual": raw_obs[2],
            "grid_frequency_hz": raw_obs[3],
            "speed_rpm": raw_obs[4],
            "power_error": raw_obs[5],
            "load_demand_mw": self.grid.current_demand,
            "mechanical_power_mw": self.turbine.mechanical_power,
            "rotor_angle_rad": self.grid.delta,
        }
        return raw_obs, info

    def _calculate_reward(self, info: Dict[str, Any], terminated: bool) -> float:

        weights = self.active_reward_weights
        if terminated:
            return weights.get("w_unsafe_penalty", -10000.0)

        freq_error = info["grid_frequency_hz"] - self.grid_base_params.get(
            "f_nominal", 60.0
        )
        stability_penalty = np.square(freq_error) * weights.get("w_freq_error_sq", 50.0)

        valve_movement = abs(info["v_pos_actual"] - self.last_valve_pos)
        jerk_penalty = np.square(
            info["v_pos_actual"] - 2 * self.last_valve_pos + self.last_action
        ) * weights.get("w_jerk_penalty", 200.0)

        is_calm = abs(freq_error) < weights.get("calm_threshold_hz", 0.05) and abs(
            info["power_error"]
        ) < weights.get("calm_threshold_mw", 50.0)

        cost_of_control_multiplier = (
            weights.get("cost_multiplier_calm", 20.0) if is_calm else 1.0
        )

        efficiency_penalty = cost_of_control_multiplier * (
            valve_movement * weights.get("w_valve_movement", 100.0) + jerk_penalty
        )

        robustness_bonus = 0.0
        if self.current_scenario_config.get("is_adversarial_drill", False):

            robustness_bonus = weights.get("w_robustness_bonus", 10.0)

        total_reward = robustness_bonus - stability_penalty - efficiency_penalty

        self.last_freq_error = freq_error
        self.last_action = self.last_valve_pos
        self.last_valve_pos = info["v_pos_actual"]

        return total_reward

    def step(self, action: np.ndarray):

        self.current_step += 1
        valve_command = float(action[0])

        rod_reactivity = self.reactor_controller.step(
            current_moderator_temp=self.reactor.T_moderator
        )
        thermal_power = self.reactor.step(self.dt, rod_reactivity)
        mech_power = self.turbine.step(self.dt, thermal_power, valve_command)
        self.grid.step(
            self.dt, mech_power, self.current_step * self.dt, self.current_step
        )
        self.turbine.speed_rpm = self.grid.omega_pu * self.turbine_base_params.get(
            "omega_nominal_rpm", 1800.0
        )

        raw_obs, info = self._get_raw_obs_and_info()
        info["rod_reactivity"] = rod_reactivity

        terminated = self._check_termination_conditions(raw_obs)
        truncated = self.current_step >= self.max_steps
        normalized_obs = self._normalize_obs(raw_obs)
        reward = (
            self._calculate_reward(info, terminated) if self.is_training_env else 0.0
        )

        return normalized_obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):

        super().reset(seed=seed)
        self._initialize_internal_state()

        scenario_name = random.choice(self.active_scenario_names)
        self.current_scenario_config = self.all_scenarios.get(scenario_name, {})

        temp_grid_params = copy.deepcopy(self.grid_base_params)
        if self.current_scenario_config.get("is_domain_randomization_drill", False):
            temp_grid_params["H"] *= np.random.uniform(0.85, 1.15)
            temp_grid_params["D"] *= np.random.uniform(0.85, 1.15)

        self.reactor = ReactorModel(self.reactor_base_params)
        self.turbine = TurbineModel(self.turbine_base_params, self.coupling_base_params)
        self.grid = GridModel(temp_grid_params, self.sim_params)

        reset_opts = self.current_scenario_config.get("reset_options", {})
        initial_power_fraction = reset_opts.get("initial_power_level", 0.9)
        initial_thermal_power = self.reactor.P0 * initial_power_fraction
        initial_mech_power = initial_thermal_power * self.turbine.eta_transfer
        initial_load_mw = reset_opts.get("initial_load_MW", initial_mech_power)

        self.reactor.reset(initial_power_fraction)
        self.turbine.reset(initial_mech_power, initial_valve_pos=0.8)
        self.grid.reset(initial_load_mw)
        self.reactor.T_moderator = self.reactor_base_params.get("T_coolant0", 306.5)
        self.reactor.T_fuel = (
            initial_thermal_power / self.reactor.Omega
        ) + self.reactor.T_moderator
        self.reactor_controller.reset(setpoint=self.reactor.T_moderator)

        if "load_profile_func" in self.current_scenario_config:
            self.grid.set_load_profile(
                self.current_scenario_config["load_profile_func"]
            )

        raw_obs, info = self._get_raw_obs_and_info()
        info["rod_reactivity"] = 0.0

        logger.debug(
            f"Environment reset to stable equilibrium for scenario: '{scenario_name}'"
        )
        return self._normalize_obs(raw_obs), info

    def _check_termination_conditions(self, raw_obs: np.ndarray) -> bool:

        if not np.all(np.isfinite(raw_obs)):
            logger.warning("Terminating due to non-finite observation (NaN/Inf).")
            return True

        _, t_fuel, _, freq, speed, _ = raw_obs
        if t_fuel > self.safety_limits.get("max_fuel_temp_c", 2800.0):
            return True
        if speed > self.safety_limits.get("max_speed_rpm", 2250.0):
            return True
        if (
            not self.safety_limits.get("min_frequency_hz", 59.0)
            < freq
            < self.safety_limits.get("max_frequency_hz", 61.0)
        ):
            return True
        return False
