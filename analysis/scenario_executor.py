from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.scenario_definitions import get_scenarios
from environment.pwr_gym_env import PWRGymEnvUnified
from controllers.base_controller import BaseController
from your_project import logging_setup

logger = logging_setup.get_logger(__name__)


@dataclass(frozen=True)
class EvalConfig:

    dt: float
    horizon_s: float
    eval_seed: int


class ScenarioExecutor:

    REQUIRED_COLUMNS = [
        "time_s",
        "dt",
        "y_target",
        "y_actual",
        "err",
        "abs_err",
        "u",
        "u_min",
        "u_max",
        "action_cmd",
        "du",
        "du_dt",
        "abs_du",
        "sat_flag",
        "actuator_saturated",
        "freq_hz",
        "freq_deviation_hz",
        "abs_freq_deviation_hz",
        "rocof_hz_s",
        "freq_margin_low_hz",
        "freq_margin_high_hz",
        "safety_flag",
        "safety_margin_min",
        "event_marker",
        "disturbance_magnitude",
        "temp_C",
        "temp_limit_C",
        "thermal_margin_C",
        "speed_rpm",
        "speed_margin_rpm",
        "load_demand_mw",
        "mechanical_power_mw",
        "power_error_mw",
        "reward",
        "rod_reactivity",
    ]
    STRING_COLUMNS = {"event_marker"}
    SAT_EPS = 1e-9
    EVENT_TIME_TOL = 1e-6

    def __init__(
        self, core_config: Dict[str, Any], out_root: str, eval_seed: int
    ) -> None:
        if not core_config:
            raise ValueError("core_config is empty; cannot build scenario executor")

        self.core_config = core_config
        sim_cfg = core_config.get("simulation", {})
        self.dt = float(sim_cfg.get("dt", 0.02))
        self.max_steps_default = int(sim_cfg.get("max_steps", 5000))
        self.out_root = out_root
        self.eval_seed = int(eval_seed)
        self.scenario_defs = get_scenarios(core_config)
        if not self.scenario_defs:
            raise RuntimeError(
                "No scenarios available from analysis.scenario_definitions"
            )

        grid_cfg = core_config.get("grid", {})
        safety_cfg = core_config.get("safety_limits", {})
        self.freq_nominal = float(grid_cfg.get("f_nominal", 60.0))
        self.freq_limits = (
            float(safety_cfg.get("min_frequency_hz", self.freq_nominal - 1.0)),
            float(safety_cfg.get("max_frequency_hz", self.freq_nominal + 1.0)),
        )
        self.temp_limit = float(safety_cfg.get("max_fuel_temp_c", 2800.0))
        self.speed_limit = float(safety_cfg.get("max_speed_rpm", 2250.0))

        self.results_root = os.path.join(out_root, "validation")
        os.makedirs(self.results_root, exist_ok=True)

        self._event_cache: Dict[str, List[Dict[str, Any]]] = {}

    def _build_env(self, scenario_name: str) -> PWRGymEnvUnified:
        cfg = self.core_config
        env = PWRGymEnvUnified(
            reactor_params=cfg.get("reactor", {}),
            turbine_params=cfg.get("turbine", {}),
            grid_params=cfg.get("grid", {}),
            coupling_params=cfg.get("coupling", {}),
            sim_params=cfg.get("simulation", {}),
            safety_limits=cfg.get("safety_limits", {}),
            rl_normalization_factors=cfg.get("rl_normalization_factors", {}),
            all_scenarios_definitions=self.scenario_defs,
            initial_scenario_name=[scenario_name],
            is_training_env=False,
            rl_training_config=cfg.get("rl_training_adv", {}),
        )
        env.set_active_scenario([scenario_name])
        env.action_space.seed(self.eval_seed)
        return env

    @staticmethod
    def _controller_observation(
        ctrl: BaseController, info: Dict[str, Any], obs: np.ndarray
    ) -> np.ndarray:

        if ctrl.__class__.__name__.lower().startswith("rl") or hasattr(ctrl, "model"):
            return obs
        measurement = float(info.get("speed_rpm", info.get("mechanical_power_mw", 0.0)))
        obs_vec = np.zeros(5, dtype=float)
        obs_vec[4] = measurement
        return obs_vec

    def _controller_bounds(
        self, controller: BaseController, fallback: Tuple[float, float]
    ) -> Tuple[float, float]:

        lo: Optional[float] = None
        hi: Optional[float] = None
        for attr in ("output_min", "umin", "u_min"):
            if hasattr(controller, attr):
                try:
                    lo = float(getattr(controller, attr))
                    break
                except Exception:
                    continue
        if lo is None and hasattr(controller, "config"):
            cfg = getattr(controller, "config")
            if isinstance(cfg, dict):
                limits = cfg.get("output_limits") or cfg.get("limits")
                if isinstance(limits, (tuple, list)) and len(limits) >= 2:
                    try:
                        lo = float(limits[0])
                        hi = float(limits[1])
                    except Exception:
                        lo = None
                        hi = None
        for attr in ("output_max", "umax", "u_max"):
            if hasattr(controller, attr):
                try:
                    hi = float(getattr(controller, attr))
                    break
                except Exception:
                    continue
        if lo is None:
            lo = float(fallback[0])
        if hi is None:
            hi = float(fallback[1])
        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
            lo, hi = float(fallback[0]), float(fallback[1])
        return lo, hi

    def _normalize_event_definitions(self, scenario_name: str) -> List[Dict[str, Any]]:
        if scenario_name in self._event_cache:
            return self._event_cache[scenario_name]
        cfg = self.scenario_defs.get(scenario_name, {})
        schedule: List[Dict[str, Any]] = []

        raw_schedule = cfg.get("event_schedule")
        if isinstance(raw_schedule, list):
            for idx, entry in enumerate(raw_schedule):
                if not isinstance(entry, dict):
                    continue
                time_s = float(entry.get("time_s", 0.0))
                end_time = entry.get("end_time")
                end_time = float(end_time) if end_time is not None else time_s
                schedule.append(
                    {
                        "time_s": time_s,
                        "end_time": end_time,
                        "label": str(entry.get("label", f"event_{idx}")),
                        "magnitude": float(entry.get("magnitude", 0.0)),
                    }
                )

        for mod in cfg.get("env_modifications", []) or []:
            if not isinstance(mod, dict):
                continue
            start_time = float(mod.get("start_time", 0.0))
            end_time = float(mod.get("end_time", start_time))
            label = f"env_{mod.get('type', 'mod')}"
            magnitude = float(abs(mod.get("imbalance_mw", mod.get("magnitude", 0.0))))
            schedule.append(
                {
                    "time_s": start_time,
                    "end_time": end_time,
                    "label": label,
                    "magnitude": magnitude,
                }
            )

        if cfg.get("is_adversarial_drill"):
            horizon = float(cfg.get("max_steps", self.max_steps_default)) * self.dt
            schedule.append(
                {
                    "time_s": 0.0,
                    "end_time": horizon,
                    "label": "adversarial_drill",
                    "magnitude": 0.0,
                }
            )

        schedule.sort(key=lambda ev: ev["time_s"])
        self._event_cache[scenario_name] = schedule
        return schedule

    def _active_events(
        self, events: List[Dict[str, Any]], time_s: float
    ) -> List[Dict[str, Any]]:
        active: List[Dict[str, Any]] = []
        for event in events:
            start = event["time_s"] - self.EVENT_TIME_TOL
            end = event["end_time"] + self.EVENT_TIME_TOL
            if start <= time_s <= end:
                active.append(event)
        return active

    def _safety_flag(
        self, info: Dict[str, Any], temp_limit: float, freq_limits: tuple[float, float]
    ) -> int:
        freq = float(info.get("grid_frequency_hz", math.nan))
        temp = float(info.get("T_fuel", math.nan))
        low, high = freq_limits
        breached = False
        if not math.isnan(freq):
            breached = breached or (freq < low or freq > high)
        if not math.isnan(temp):
            breached = breached or (temp > temp_limit)
        speed = float(info.get("speed_rpm", math.nan))
        if not math.isnan(speed) and math.isfinite(self.speed_limit):
            breached = breached or (speed > self.speed_limit)
        return int(breached)

    def _scenario_steps(self, scenario_name: str) -> int:
        scen = self.scenario_defs.get(scenario_name, {})
        raw_steps = int(scen.get("max_steps", 0))
        if raw_steps <= 0:
            return int(self.max_steps_default)
        return raw_steps

    def simulate(
        self, controller_name: str, controller: BaseController, scenario_name: str
    ) -> pd.DataFrame:

        scenario_key = scenario_name
        if scenario_key not in self.scenario_defs:
            raise KeyError(f"Scenario '{scenario_name}' not found in definitions")

        env = self._build_env(scenario_key)
        try:
            action_low = np.atleast_1d(env.action_space.low)
            action_high = np.atleast_1d(env.action_space.high)
            env_u_min = float(action_low[0])
            env_u_max = float(action_high[0])
        except Exception:
            env_u_min, env_u_max = 0.0, 1.0

        controller_u_min, controller_u_max = self._controller_bounds(
            controller, (env_u_min, env_u_max)
        )

        temp_limit = self.temp_limit
        freq_min, freq_max = self.freq_limits
        freq_nominal = self.freq_nominal
        speed_limit = self.speed_limit

        event_defs = self._normalize_event_definitions(scenario_key)

        try:
            if hasattr(controller, "reset"):
                controller.reset()
            obs, info = env.reset(seed=self.eval_seed)

            rows: List[Dict[str, Any]] = []
            max_steps = self._scenario_steps(scenario_key)

            prev_u = math.nan
            prev_freq = math.nan
            prev_target = math.nan
            prev_time = math.nan

            for step_idx in range(max_steps):
                ctrl_obs = self._controller_observation(controller, info, obs)
                try:
                    raw_action = float(controller.step(ctrl_obs))
                except Exception as exc:
                    logger.error(
                        "Controller %s failed at step %d: %s",
                        controller_name,
                        step_idx,
                        exc,
                    )
                    break

                action_cmd = raw_action
                controller_clipped = float(
                    np.clip(action_cmd, controller_u_min, controller_u_max)
                )
                env_clipped = float(np.clip(controller_clipped, env_u_min, env_u_max))
                sat_flag = int(
                    (abs(controller_clipped - action_cmd) > self.SAT_EPS)
                    or (abs(env_clipped - controller_clipped) > self.SAT_EPS)
                )

                obs, reward, terminated, truncated, info = env.step(
                    np.array([env_clipped], dtype=np.float32)
                )

                time_s = float(info.get("time_s", (step_idx + 1) * self.dt))
                y_target = float(info.get("load_demand_mw", math.nan))
                y_actual = float(info.get("mechanical_power_mw", math.nan))
                freq_hz = float(info.get("grid_frequency_hz", math.nan))
                temp_c = float(info.get("T_fuel", math.nan))
                speed_rpm = float(info.get("speed_rpm", math.nan))
                power_error = float(info.get("power_error", y_actual - y_target))
                rod_reactivity = float(info.get("rod_reactivity", math.nan))

                dt_step = self.dt
                if math.isfinite(prev_time):
                    candidate = time_s - prev_time
                    if math.isfinite(candidate) and candidate > 0:
                        dt_step = candidate
                prev_time = time_s

                err = y_target - y_actual
                abs_err = abs(err) if math.isfinite(err) else math.nan

                du = env_clipped - prev_u if math.isfinite(prev_u) else math.nan
                du_dt = du / dt_step if math.isfinite(du) and dt_step > 0 else math.nan
                abs_du = abs(du) if math.isfinite(du) else math.nan

                if math.isfinite(freq_hz) and math.isfinite(prev_freq) and dt_step > 0:
                    rocof = (freq_hz - prev_freq) / dt_step
                else:
                    rocof = math.nan
                prev_freq = freq_hz

                freq_dev = (
                    freq_hz - freq_nominal if math.isfinite(freq_hz) else math.nan
                )
                abs_freq_dev = abs(freq_dev) if math.isfinite(freq_dev) else math.nan

                freq_margin_low = (
                    freq_hz - freq_min
                    if math.isfinite(freq_hz) and math.isfinite(freq_min)
                    else math.nan
                )
                freq_margin_high = (
                    freq_max - freq_hz
                    if math.isfinite(freq_hz) and math.isfinite(freq_max)
                    else math.nan
                )
                thermal_margin = (
                    temp_limit - temp_c
                    if math.isfinite(temp_limit) and math.isfinite(temp_c)
                    else math.nan
                )
                speed_margin = (
                    speed_limit - speed_rpm
                    if math.isfinite(speed_limit) and math.isfinite(speed_rpm)
                    else math.nan
                )

                margins = [
                    m
                    for m in (
                        freq_margin_low,
                        freq_margin_high,
                        thermal_margin,
                        speed_margin,
                    )
                    if math.isfinite(m)
                ]
                safety_margin_min = min(margins) if margins else math.nan

                active_events = self._active_events(event_defs, time_s)
                event_labels = [ev.get("label", "event") for ev in active_events]
                event_magnitude = sum(
                    abs(float(ev.get("magnitude", 0.0)))
                    for ev in active_events
                    if math.isfinite(float(ev.get("magnitude", 0.0)))
                )

                if math.isfinite(y_target) and math.isfinite(prev_target):
                    load_delta = abs(y_target - prev_target)
                else:
                    load_delta = 0.0
                prev_target = y_target
                disturbance_magnitude = float(load_delta + event_magnitude)

                safety_flag = self._safety_flag(
                    info, temp_limit=temp_limit, freq_limits=(freq_min, freq_max)
                )

                row = {
                    "time_s": time_s,
                    "dt": dt_step,
                    "y_target": y_target,
                    "y_actual": y_actual,
                    "err": err,
                    "abs_err": abs_err,
                    "u": env_clipped,
                    "u_min": controller_u_min,
                    "u_max": controller_u_max,
                    "action_cmd": action_cmd,
                    "du": du,
                    "du_dt": du_dt,
                    "abs_du": abs_du,
                    "sat_flag": float(sat_flag),
                    "actuator_saturated": float(sat_flag),
                    "freq_hz": freq_hz,
                    "freq_deviation_hz": freq_dev,
                    "abs_freq_deviation_hz": abs_freq_dev,
                    "rocof_hz_s": rocof,
                    "freq_margin_low_hz": freq_margin_low,
                    "freq_margin_high_hz": freq_margin_high,
                    "safety_flag": float(safety_flag),
                    "safety_margin_min": safety_margin_min,
                    "event_marker": "|".join(event_labels) if event_labels else "",
                    "disturbance_magnitude": disturbance_magnitude,
                    "temp_C": temp_c,
                    "temp_limit_C": temp_limit,
                    "thermal_margin_C": thermal_margin,
                    "speed_rpm": speed_rpm,
                    "speed_margin_rpm": speed_margin,
                    "load_demand_mw": y_target,
                    "mechanical_power_mw": y_actual,
                    "power_error_mw": power_error,
                    "reward": reward,
                    "rod_reactivity": rod_reactivity,
                }

                rows.append(row)
                prev_u = env_clipped

                if terminated or truncated:
                    break

            df = pd.DataFrame(rows)
        finally:
            env.close()

        missing_cols = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        for col in missing_cols:
            if col in self.STRING_COLUMNS:
                df[col] = ""
            else:
                df[col] = np.nan
        ordered_cols = self.REQUIRED_COLUMNS + [
            c for c in df.columns if c not in self.REQUIRED_COLUMNS
        ]
        df = df[ordered_cols]
        return df

    def _export_pdf_preview(
        self, csv_path: str, df: pd.DataFrame, max_rows: int = 60
    ) -> None:
        try:
            preview = df.head(max_rows)
            if preview.empty:
                return
            fig_height = max(2.5, 0.35 * len(preview) + 1)
            fig, ax = plt.subplots(figsize=(11, fig_height))
            ax.axis("off")
            table = ax.table(
                cellText=preview.values, colLabels=preview.columns, loc="center"
            )
            table.auto_set_font_size(False)
            table.set_fontsize(6)
            table.scale(1, 1.2)
            ax.set_title(os.path.basename(csv_path), fontsize=9, pad=12)
            pdf_path = os.path.splitext(csv_path)[0] + ".pdf"
            fig.savefig(pdf_path, bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:
            logger.warning("Failed to create PDF companion for %s: %s", csv_path, exc)

    def write_timeseries(
        self, controller_name: str, scenario_name: str, df: pd.DataFrame
    ) -> str:
        ctrl_dir = os.path.join(self.results_root, controller_name)
        scen_dir = os.path.join(ctrl_dir, scenario_name)
        os.makedirs(scen_dir, exist_ok=True)
        path = os.path.join(scen_dir, "timeseries.csv")
        tmp_path = f"{path}.tmp"
        df.to_csv(tmp_path, index=False, encoding="utf-8-sig", lineterminator="\n")
        os.replace(tmp_path, path)
        meta = {
            "controller": controller_name,
            "scenario": scenario_name,
            "eval_seed": self.eval_seed,
            "rows": len(df),
            "columns": list(df.columns),
        }
        with open(
            os.path.join(scen_dir, "timeseries.meta.json"), "w", encoding="utf-8"
        ) as fh:
            json.dump(meta, fh, indent=2)
        logger.info("Timeseries written: %s (rows=%d)", path, len(df))
        return path

    def run_matrix(
        self, controllers: Dict[str, BaseController], scenarios: Iterable[str]
    ) -> Dict[tuple[str, str], str]:
        outputs: Dict[tuple[str, str], str] = {}
        for scenario in scenarios:
            for ctrl_name, ctrl_instance in controllers.items():
                logger.info("Simulating controller=%s scenario=%s", ctrl_name, scenario)
                df = self.simulate(ctrl_name, ctrl_instance, scenario)
                path = self.write_timeseries(ctrl_name, scenario, df)
                outputs[(ctrl_name, scenario)] = path
        return outputs


def run(
    controllers: List[str],
    scenarios: List[str],
    cfg: EvalConfig,
    out_root: str,
    controller_builder: Optional[Dict[str, BaseController]] = None,
    seed: Optional[int] = None,
) -> Dict[tuple[str, str], str]:

    raise RuntimeError(
        "The functional run() helper is no longer supported. Instantiate "
        "ScenarioExecutor and call run_matrix() with explicit controller instances."
    )


__all__ = ["ScenarioExecutor", "EvalConfig"]
