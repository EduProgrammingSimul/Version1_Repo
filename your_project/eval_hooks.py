from __future__ import annotations

"""
Evaluation hooks for validation runs with enriched deterministic telemetry.

This module guarantees per-step rows with the canonical keys:
    time_s, dt, y_target, y_actual, err, abs_err, u, u_min, u_max,
    action_cmd, du, du_dt, abs_du, sat_flag, freq_hz, freq_deviation_hz,
    abs_freq_deviation_hz, rocof_hz_s, freq_margin_low_hz, freq_margin_high_hz,
    safety_flag, safety_margin_min, event_marker, disturbance_magnitude,
    temp_C, temp_limit_C, thermal_margin_C, speed_rpm, speed_margin_rpm,
    load_demand_mw, mechanical_power_mw, power_error_mw, reward, rod_reactivity

Controllers:
- PIDController(config: dict, dt: float).step(observation)  # expects obs[4] as measurement
- FLCController(config: dict, dt: float).step(observation)  # expects obs[4] as measurement
- RL: keeps a safe stub unless a compatible policy is available

Behavior:
- Builds a per-(controller, scenario) context on reset.
- Updates controller setpoint each step from a scenario reference profile.
- Uses a simple first-order plant as a fallback "environment" so outputs are valid even
  without a real env. (The real env path can be plugged in later if desired.)
- Populates all canonical keys; the schema guard becomes a safety net rather than the norm.
- ASCII-only to avoid Unicode decode issues.
"""

from typing import Any, Callable, Dict, Optional, Tuple, List
from dataclasses import dataclass, field
import hashlib
import math
import os
import importlib
import numpy as np
import yaml

try:
    from analysis import scenario_definitions as _scenario_defs
except Exception:
    _scenario_defs = None

from your_project import logging_setup

logger = logging_setup.get_logger(__name__)


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
LEGACY_COMPAT_COLUMNS = ["t", "actuator_saturated"]
STRING_COLUMNS = {"event_marker"}
SAT_EPS = 1e-9


def ensure_row_schema(row: Dict[str, Any]) -> Dict[str, Any]:

    for key in REQUIRED_COLUMNS:
        if key not in row:
            if key in STRING_COLUMNS:
                row[key] = ""
            else:
                row[key] = float("nan")
    for key in LEGACY_COMPAT_COLUMNS:
        if key not in row:
            row[key] = float("nan")
    return row


_DEFAULT_CORE_CONFIG: Dict[str, Any] = {
    "simulation": {"dt": 0.02},
    "initial_conditions": {"electrical_load_mw": 3008.5},
    "coupling": {"eta_transfer": 0.98},
    "grid": {"f_nominal": 60.0},
    "safety_limits": {
        "min_frequency_hz": 59.0,
        "max_frequency_hz": 61.0,
        "max_fuel_temp_c": 2800.0,
        "max_speed_rpm": 2250.0,
    },
}


def _load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error("Failed to read YAML: %s err=%s", path, e)
        return {}


def _load_controllers_cfg() -> Dict[str, Any]:

    here = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(here, "controllers.yaml")
    raw = _load_yaml(cfg_path)
    if not isinstance(raw, dict):
        return {}
    lowered: Dict[str, Any] = {}
    for key, value in raw.items():
        lowered[str(key).lower()] = value
    return lowered


def _load_scenario_configs() -> Dict[str, Dict[str, Any]]:

    scenarios: Dict[str, Dict[str, Any]] = {}
    if _scenario_defs and hasattr(_scenario_defs, "get_scenarios"):
        try:
            raw = _scenario_defs.get_scenarios(_DEFAULT_CORE_CONFIG)
            if isinstance(raw, dict):
                scenarios = {
                    str(name).strip().lower(): cfg
                    for name, cfg in raw.items()
                    if isinstance(cfg, dict)
                }
        except Exception as exc:
            logger.error(
                "Failed to load scenarios via analysis.scenario_definitions: %s", exc
            )
    if not scenarios:
        logger.warning(
            "Scenario metadata unavailable; falling back to default scenario list"
        )
        scenarios = {
            "baseline_steady_state": {},
            "sudden_load_increase_5pct": {},
            "gradual_load_increase_10pct": {},
            "cascading_grid_fault_and_recovery": {},
            "steady_state_efficiency_probe": {},
            "parameter_randomization_drills": {},
            "deceptive_sensor_noise": {},
            "combined_challenge_final_exam": {},
        }
    return scenarios


_SCENARIO_CONFIGS = _load_scenario_configs()


@dataclass
class PIDStub:
    kp: float
    ki: float
    kd: float
    umin: float
    umax: float
    dt: float
    integral: float = 0.0
    prev_error: float = 0.0

    def step(self, observation: np.ndarray) -> float:
        measurement = float(observation[4]) if observation.size > 4 else 0.0
        error = getattr(self, "setpoint", 0.0) - measurement
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt if self.dt > 0 else 0.0
        self.prev_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return max(self.umin, min(self.umax, output))


@dataclass
class FLCStub:
    k_small: float
    k_large: float
    ti: float
    umin: float
    umax: float
    dt: float
    last_output: float = 0.5

    def step(self, observation: np.ndarray) -> float:
        measurement = float(observation[4]) if observation.size > 4 else 0.0
        error = getattr(self, "setpoint", 0.0) - measurement
        gain = (
            self.k_large
            if abs(error) > 0.05 * getattr(self, "setpoint", 1.0)
            else self.k_small
        )
        delta = gain * error * self.dt / max(self.ti, 1e-6)
        self.last_output = max(self.umin, min(self.umax, self.last_output + delta))
        return self.last_output


@dataclass
class RLStub:
    gain: float
    umin: float
    umax: float
    dt: float

    def step(self, observation: np.ndarray) -> float:
        measurement = float(observation[4]) if observation.size > 4 else 0.0
        error = getattr(self, "setpoint", 0.0) - measurement
        output = getattr(self, "bias", 0.0) + self.gain * error
        return max(self.umin, min(self.umax, output))


@dataclass
class FirstOrderPlant:
    Kp: float = 1.0
    tau: float = 2.0
    y: float = 1.0

    def step(self, u: float, dt: float, disturbance: float = 0.0) -> float:
        dy = (self.Kp * u - self.y) / max(self.tau, 1e-6)
        self.y += (dy + disturbance) * dt
        return self.y


@dataclass
class ScenarioProfile:
    plant_factory: Callable[[], FirstOrderPlant]
    disturbance: Callable[[float], float]
    meas_noise_std: float
    actuator_bias: float
    setpoint_base: float
    freq_coupling: float


def _zero_disturbance(_t: float) -> float:
    return 0.0


def _sinusoidal_disturbance(
    amplitude: float, freq_hz: float
) -> Callable[[float], float]:
    omega = 2 * math.pi * freq_hz
    return lambda t: amplitude * math.sin(omega * t)


def _build_scenario_profiles() -> Dict[str, ScenarioProfile]:
    profiles: Dict[str, ScenarioProfile] = {}
    profiles["baseline_steady_state"] = ScenarioProfile(
        plant_factory=lambda: FirstOrderPlant(Kp=1.0, tau=2.0, y=1.0),
        disturbance=_zero_disturbance,
        meas_noise_std=0.002,
        actuator_bias=0.0,
        setpoint_base=1.0,
        freq_coupling=0.25,
    )
    profiles["sudden_load_increase_5pct"] = ScenarioProfile(
        plant_factory=lambda: FirstOrderPlant(Kp=1.05, tau=1.8, y=1.0),
        disturbance=_zero_disturbance,
        meas_noise_std=0.003,
        actuator_bias=0.0,
        setpoint_base=1.0,
        freq_coupling=0.3,
    )
    profiles["gradual_load_increase_10pct"] = ScenarioProfile(
        plant_factory=lambda: FirstOrderPlant(Kp=1.0, tau=2.3, y=1.0),
        disturbance=_zero_disturbance,
        meas_noise_std=0.003,
        actuator_bias=0.0,
        setpoint_base=1.0,
        freq_coupling=0.28,
    )
    profiles["steady_state_efficiency_probe"] = ScenarioProfile(
        plant_factory=lambda: FirstOrderPlant(Kp=0.95, tau=3.0, y=1.0),
        disturbance=_zero_disturbance,
        meas_noise_std=0.0015,
        actuator_bias=-0.01,
        setpoint_base=1.0,
        freq_coupling=0.22,
    )
    profiles["deceptive_sensor_noise"] = ScenarioProfile(
        plant_factory=lambda: FirstOrderPlant(Kp=1.1, tau=1.6, y=1.0),
        disturbance=_sinusoidal_disturbance(0.015, 0.15),
        meas_noise_std=0.01,
        actuator_bias=0.0,
        setpoint_base=1.0,
        freq_coupling=0.33,
    )
    profiles["parameter_randomization_drills"] = ScenarioProfile(
        plant_factory=lambda: FirstOrderPlant(Kp=1.0, tau=2.0, y=1.0),
        disturbance=_zero_disturbance,
        meas_noise_std=0.004,
        actuator_bias=0.0,
        setpoint_base=1.0,
        freq_coupling=0.3,
    )
    profiles["cascading_grid_fault_and_recovery"] = ScenarioProfile(
        plant_factory=lambda: FirstOrderPlant(Kp=1.05, tau=1.7, y=1.0),
        disturbance=_sinusoidal_disturbance(0.02, 0.08),
        meas_noise_std=0.004,
        actuator_bias=0.0,
        setpoint_base=1.0,
        freq_coupling=0.35,
    )
    profiles["combined_challenge_final_exam"] = ScenarioProfile(
        plant_factory=lambda: FirstOrderPlant(Kp=1.08, tau=1.5, y=1.0),
        disturbance=_sinusoidal_disturbance(0.018, 0.05),
        meas_noise_std=0.006,
        actuator_bias=0.0,
        setpoint_base=1.0,
        freq_coupling=0.38,
    )
    return profiles


_SCENARIO_PROFILES = _build_scenario_profiles()
DEFAULT_PROFILE = _SCENARIO_PROFILES.get(
    "baseline_steady_state",
    ScenarioProfile(
        plant_factory=lambda: FirstOrderPlant(Kp=1.0, tau=2.0, y=1.0),
        disturbance=_zero_disturbance,
        meas_noise_std=0.003,
        actuator_bias=0.0,
        setpoint_base=1.0,
        freq_coupling=0.28,
    ),
)


def _scenario_profile(name: str) -> ScenarioProfile:
    return _SCENARIO_PROFILES.get(name.strip().lower(), DEFAULT_PROFILE)


def _target_profile(scenario: str, t: float, base: float = 1.0) -> float:
    s = scenario.strip().lower()
    if s == "baseline_steady_state":
        return base
    if s == "sudden_load_increase_5pct":
        return base + (0.05 * base if t >= 50.0 else 0.0)
    if s == "gradual_load_increase_10pct":
        return base + 0.10 * base * min(max(t / 200.0, 0.0), 1.0)
    if s == "cascading_grid_fault_and_recovery":
        return base * (0.8 if 40.0 <= t <= 60.0 else 1.0)
    if s == "steady_state_efficiency_probe":
        return base + 0.02 * math.sin(2 * math.pi * 0.02 * t)
    if s == "parameter_randomization_drills":
        return base + 0.03 * math.sin(2 * math.pi * 0.01 * t + 0.5)
    if s == "deceptive_sensor_noise":
        return base + 0.02 * math.sin(2 * math.pi * 0.04 * t)
    if s == "combined_challenge_final_exam":
        return (
            base
            + 0.05 * math.sin(2 * math.pi * 0.005 * t)
            + (0.05 * base if t >= 120.0 else 0.0)
        )
    return base


def _scenario_event_schedule(name: str) -> List[Dict[str, Any]]:
    cfg = _SCENARIO_CONFIGS.get(name.strip().lower(), {})
    schedule_raw = cfg.get("event_schedule") or []
    schedule: List[Dict[str, Any]] = []
    if isinstance(schedule_raw, list):
        for idx, entry in enumerate(schedule_raw):
            if not isinstance(entry, dict):
                continue
            time_s = float(entry.get("time_s", 0.0))
            end_time = entry.get("end_time", time_s)
            try:
                end_time = float(end_time)
            except Exception:
                end_time = time_s
            schedule.append(
                {
                    "time_s": time_s,
                    "end_time": end_time,
                    "label": str(entry.get("label", f"event_{idx}")),
                    "magnitude": float(entry.get("magnitude", 0.0)),
                }
            )
    schedule.sort(key=lambda ev: ev["time_s"])
    return schedule


def _active_events(events: List[Dict[str, Any]], time_s: float) -> List[Dict[str, Any]]:
    active: List[Dict[str, Any]] = []
    for event in events:
        start = event["time_s"] - 1e-6
        end = event["end_time"] + 1e-6
        if start <= time_s <= end:
            active.append(event)
    return active


@dataclass
class HookContext:
    controller: str
    scenario: str
    dt: float
    policy: Any
    umin: float
    umax: float
    env: Any = None
    mapping: Dict[str, str] = field(default_factory=dict)
    plant: Optional[FirstOrderPlant] = None
    banner_done: bool = False
    profile: ScenarioProfile = DEFAULT_PROFILE
    rng_seed: int = 0
    rng: Optional[np.random.Generator] = None
    prev_u: float = math.nan
    prev_freq: float = math.nan
    prev_target: float = math.nan
    prev_time: float = math.nan
    freq_limits: Tuple[float, float] = (59.0, 61.0)
    freq_nominal: float = 60.0
    temp_limit: float = 2800.0
    speed_limit: float = 2250.0
    event_schedule: List[Dict[str, Any]] = field(default_factory=list)


_REGISTRY: Dict[Tuple[str, str], HookContext] = {}


def _try_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def _make_pid_policy(cfg: Dict[str, Any], dt: float, umin: float, umax: float) -> Any:
    pid_mod = _try_import("controllers.pid_controller")
    pid_cfg = {
        "kp": float(cfg.get("kp", 0.20)),
        "ki": float(cfg.get("ki", 0.04)),
        "kd": float(cfg.get("kd", 0.05)),
        "setpoint": float(cfg.get("setpoint", 1800.0)),
        "output_limits": tuple(cfg.get("output_limits", (umin, umax))),
    }
    if pid_mod and hasattr(pid_mod, "PIDController"):
        try:
            policy = pid_mod.PIDController(pid_cfg, dt)
            logger.info("PIDController constructed: (config, dt)")
            return policy
        except Exception as e:
            logger.error("PIDController(config, dt) failed, using PIDStub. err=%s", e)
    return PIDStub(
        kp=pid_cfg["kp"],
        ki=pid_cfg["ki"],
        kd=pid_cfg["kd"],
        umin=umin,
        umax=umax,
        dt=dt,
    )


def _make_flc_policy(cfg: Dict[str, Any], dt: float, umin: float, umax: float) -> Any:
    flc_mod = _try_import("controllers.flc_controller")
    flc_cfg = {
        "setpoint": float(cfg.get("setpoint", 1800.0)),
        "output_limits": tuple(cfg.get("output_limits", (umin, umax))),
        "dvalve_limit": float(cfg.get("dvalve_limit", 0.1)),
        "max_speed_error": float(cfg.get("max_speed_error", 100.0)),
        "max_speed_error_change": float(cfg.get("max_speed_error_change", 50.0)),
        "max_dvalve_abs": float(cfg.get("max_dvalve_abs", 0.05)),
        "error_scaling": float(cfg.get("error_scaling", 1.0)),
        "derror_scaling": float(cfg.get("derror_scaling", 1.0)),
        "output_scaling": float(cfg.get("output_scaling", 1.0)),
    }
    if flc_mod and hasattr(flc_mod, "FLCController"):
        try:
            policy = flc_mod.FLCController(flc_cfg, dt)
            logger.info("FLCController constructed: (config, dt)")
            return policy
        except Exception as e:
            logger.error("FLCController(config, dt) failed, using FLCStub. err=%s", e)
    return FLCStub(
        k_small=cfg.get("k_small", cfg.get("k", 0.8)),
        k_large=(
            cfg.get("k_large", 1.1 * cfg.get("k", 0.8))
            if "k_large" in cfg or "k" in cfg
            else 0.9
        ),
        ti=float(cfg.get("ti", 3.0)),
        umin=umin,
        umax=umax,
        dt=dt,
    )


def _make_rl_policy(cfg: Dict[str, Any], dt: float, umin: float, umax: float) -> Any:
    rl_mod = _try_import("controllers.rl_interface")
    model_path = cfg.get("model_path")
    if rl_mod and hasattr(rl_mod, "RLPolicy"):
        try:
            try:
                policy = rl_mod.RLPolicy(model_path=model_path, umin=umin, umax=umax)
            except TypeError:
                policy = rl_mod.RLPolicy(model_path=model_path)
            logger.info("RLPolicy constructed")
            for n, v in (("umin", umin), ("umax", umax)):
                if hasattr(policy, n):
                    try:
                        setattr(policy, n, v)
                    except Exception:
                        pass
            if not hasattr(policy, "step") and hasattr(policy, "act"):

                def _step_wrap(observation: np.ndarray) -> float:
                    meas = float(observation[4]) if observation.size > 4 else 0.0
                    sp = getattr(policy, "setpoint", 1.0)
                    e = sp - meas
                    cmd = float(policy.act(e, dt))
                    return max(umin, min(umax, cmd))

                policy.step = _step_wrap
            return policy
        except Exception as e:
            logger.error("RLPolicy failed, using RLStub. err=%s", e)
    gain = float(cfg.get("gain", 0.6))
    return RLStub(gain=gain, umin=umin, umax=umax, dt=dt)


def _scenario_seed(controller: str, scenario: str) -> int:
    key = f"{controller.strip().upper()}::{scenario.strip().lower()}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return int(digest, 16) % (2**32)


def _make_context(controller: str, scenario: str, dt: float) -> HookContext:
    all_cfg = _load_controllers_cfg()
    ckey = controller.strip().upper()
    sub = all_cfg.get(ckey.lower(), {})

    if "output_limits" in sub:
        limits = tuple(sub.get("output_limits", (0.0, 1.0)))
        umin, umax = float(limits[0]), float(limits[1])
    else:
        umin = float(sub.get("umin", 0.0))
        umax = float(sub.get("umax", 1.0))

    if ckey == "PID":
        policy = _make_pid_policy(sub, dt, umin, umax)
    elif ckey == "FLC":
        policy = _make_flc_policy(sub, dt, umin, umax)
    else:
        policy = _make_rl_policy(sub, dt, umin, umax)

    profile = _scenario_profile(scenario)
    seed = _scenario_seed(ckey, scenario)
    rng = np.random.default_rng(seed)

    scenario_key = scenario.strip().lower()
    scen_cfg = _SCENARIO_CONFIGS.get(scenario_key, {})
    safety_cfg = scen_cfg.get(
        "safety_limits", _DEFAULT_CORE_CONFIG.get("safety_limits", {})
    )
    grid_cfg = _DEFAULT_CORE_CONFIG.get("grid", {})

    ctx = HookContext(
        controller=ckey,
        scenario=scenario,
        dt=dt,
        policy=policy,
        umin=umin,
        umax=umax,
        env=None,
        mapping={},
        plant=profile.plant_factory(),
        banner_done=False,
        profile=profile,
        rng_seed=seed,
        rng=rng,
        freq_limits=(
            float(safety_cfg.get("min_frequency_hz", 59.0)),
            float(safety_cfg.get("max_frequency_hz", 61.0)),
        ),
        freq_nominal=float(grid_cfg.get("f_nominal", 60.0)),
        temp_limit=float(safety_cfg.get("max_fuel_temp_c", 2800.0)),
        speed_limit=float(safety_cfg.get("max_speed_rpm", 2250.0)),
        event_schedule=_scenario_event_schedule(scenario),
    )
    return ctx


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _safety_flag(
    freq: float,
    temp: float,
    freq_limits: Tuple[float, float],
    temp_limit: float,
    speed: float,
    speed_limit: float,
) -> int:
    low, high = freq_limits
    breached = False
    if math.isfinite(freq):
        breached |= freq < low or freq > high
    if math.isfinite(temp):
        breached |= temp > temp_limit
    if math.isfinite(speed) and math.isfinite(speed_limit):
        breached |= speed > speed_limit
    return int(breached)


def reset_fn(*args, **kwargs) -> None:
    cfg = kwargs.get("cfg")
    controller = kwargs.get("controller", args[0] if len(args) > 0 else "<unknown>")
    scenario = kwargs.get("scenario", args[1] if len(args) > 1 else "<unknown>")

    dt = getattr(cfg, "dt", kwargs.get("dt", 0.02))
    ctx = _make_context(controller, scenario, dt)

    ctx.prev_u = math.nan
    ctx.prev_freq = math.nan
    ctx.prev_target = math.nan
    ctx.prev_time = math.nan

    key = (ctx.controller, ctx.scenario)
    _REGISTRY[key] = ctx

    if not ctx.banner_done:
        mode = "real" if ctx.env is not None else "fallback"
        logger.info(
            "reset_fn: controller=%s scenario=%s dt=%s umin=%s umax=%s mode=%s",
            ctx.controller,
            ctx.scenario,
            ctx.dt,
            ctx.umin,
            ctx.umax,
            mode,
        )
        if ctx.mapping:
            logger.info("reset_fn mapping: %s", ctx.mapping)
        ctx.banner_done = True


def step_fn(*args, **kwargs) -> Dict[str, Any]:
    controller = kwargs.get("controller", args[0] if len(args) > 0 else "<unknown>")
    scenario = kwargs.get("scenario", args[1] if len(args) > 1 else "<unknown>")
    time_s = float(kwargs.get("t", args[2] if len(args) > 2 else float("nan")))
    dt = float(kwargs.get("dt", args[3] if len(args) > 3 else 0.02))

    key = (controller.strip().upper(), scenario)
    ctx = _REGISTRY.get(key)
    if ctx is None or not math.isclose(ctx.dt, dt, rel_tol=1e-6, abs_tol=1e-9):
        ctx = _make_context(controller, scenario, dt)
        _REGISTRY[key] = ctx
        logger.debug("step_fn: created context lazily for %s", key)

    profile = ctx.profile
    rng = ctx.rng or np.random.default_rng(ctx.rng_seed)
    ctx.rng = rng

    y_target = _target_profile(scenario, time_s, base=profile.setpoint_base)
    measurement = ctx.plant.y if ctx.plant is not None else profile.setpoint_base
    meas_noise = (
        rng.normal(0.0, profile.meas_noise_std) if profile.meas_noise_std > 0.0 else 0.0
    )
    y_meas = measurement + meas_noise

    try:
        setattr(ctx.policy, "setpoint", y_target)
    except Exception:
        pass

    obs = np.zeros(5, dtype=float)
    obs[4] = y_meas

    if hasattr(ctx.policy, "step"):
        raw_cmd = float(ctx.policy.step(obs))
    elif hasattr(ctx.policy, "act"):
        error = y_target - y_meas
        raw_cmd = float(ctx.policy.act(error, dt))
    else:
        raw_cmd = 0.5 * (y_target - y_meas)

    command = raw_cmd + profile.actuator_bias
    u_clamped = _clamp(command, ctx.umin, ctx.umax)
    sat_flag = float(abs(u_clamped - command) > SAT_EPS)

    if ctx.plant is None:
        ctx.plant = profile.plant_factory()
    disturbance = profile.disturbance(time_s)
    y_actual = ctx.plant.step(u_clamped, dt, disturbance)

    freq_hz = ctx.freq_nominal - profile.freq_coupling * (y_target - y_actual)

    dt_step = ctx.dt
    if math.isfinite(ctx.prev_time):
        candidate = time_s - ctx.prev_time
        if math.isfinite(candidate) and candidate > 0:
            dt_step = candidate
    ctx.prev_time = time_s

    err = y_target - y_actual
    abs_err = abs(err) if math.isfinite(err) else math.nan

    du = u_clamped - ctx.prev_u if math.isfinite(ctx.prev_u) else math.nan
    du_dt = du / dt_step if math.isfinite(du) and dt_step > 0 else math.nan
    abs_du = abs(du) if math.isfinite(du) else math.nan
    ctx.prev_u = u_clamped

    if math.isfinite(freq_hz) and math.isfinite(ctx.prev_freq) and dt_step > 0:
        rocof = (freq_hz - ctx.prev_freq) / dt_step
    else:
        rocof = math.nan
    ctx.prev_freq = freq_hz

    freq_dev = freq_hz - ctx.freq_nominal if math.isfinite(freq_hz) else math.nan
    abs_freq_dev = abs(freq_dev) if math.isfinite(freq_dev) else math.nan

    freq_margin_low = (
        freq_hz - ctx.freq_limits[0]
        if math.isfinite(freq_hz) and math.isfinite(ctx.freq_limits[0])
        else math.nan
    )
    freq_margin_high = (
        ctx.freq_limits[1] - freq_hz
        if math.isfinite(freq_hz) and math.isfinite(ctx.freq_limits[1])
        else math.nan
    )
    temp_c = float("nan")
    thermal_margin = math.nan
    speed_rpm = float("nan")
    speed_margin = math.nan

    margins = [
        m
        for m in (freq_margin_low, freq_margin_high, thermal_margin, speed_margin)
        if math.isfinite(m)
    ]
    safety_margin_min = min(margins) if margins else math.nan

    if math.isfinite(y_target) and math.isfinite(ctx.prev_target):
        load_delta = abs(y_target - ctx.prev_target)
    else:
        load_delta = 0.0
    ctx.prev_target = y_target

    active_events = _active_events(ctx.event_schedule, time_s)
    event_labels = [ev.get("label", "event") for ev in active_events]
    event_mag = sum(
        abs(float(ev.get("magnitude", 0.0)))
        for ev in active_events
        if math.isfinite(float(ev.get("magnitude", 0.0)))
    )
    disturbance_magnitude = float(load_delta + event_mag)

    safety_flag = float(
        _safety_flag(
            freq_hz, temp_c, ctx.freq_limits, ctx.temp_limit, speed_rpm, ctx.speed_limit
        )
    )

    power_error = y_actual - y_target

    row = {
        "time_s": time_s,
        "dt": dt_step,
        "y_target": y_target,
        "y_actual": y_actual,
        "err": err,
        "abs_err": abs_err,
        "u": u_clamped,
        "u_min": ctx.umin,
        "u_max": ctx.umax,
        "action_cmd": command,
        "du": du,
        "du_dt": du_dt,
        "abs_du": abs_du,
        "sat_flag": sat_flag,
        "freq_hz": freq_hz,
        "freq_deviation_hz": freq_dev,
        "abs_freq_deviation_hz": abs_freq_dev,
        "rocof_hz_s": rocof,
        "freq_margin_low_hz": freq_margin_low,
        "freq_margin_high_hz": freq_margin_high,
        "safety_flag": safety_flag,
        "safety_margin_min": safety_margin_min,
        "event_marker": "|".join(event_labels) if event_labels else "",
        "disturbance_magnitude": disturbance_magnitude,
        "temp_C": temp_c,
        "temp_limit_C": ctx.temp_limit,
        "thermal_margin_C": thermal_margin,
        "speed_rpm": speed_rpm,
        "speed_margin_rpm": speed_margin,
        "load_demand_mw": y_target,
        "mechanical_power_mw": y_actual,
        "power_error_mw": power_error,
        "reward": 0.0,
        "rod_reactivity": float("nan"),
        "t": time_s,
        "actuator_saturated": sat_flag,
    }

    logger.debug(
        "step_fn: ctrl=%s sc=%s t=%.4f y_ref=%.4f y=%.4f u=%.4f raw=%.4f dist=%.4f",
        controller,
        scenario,
        time_s,
        y_target,
        y_actual,
        u_clamped,
        command,
        disturbance,
    )
    return ensure_row_schema(row)


__all__ = ["reset_fn", "step_fn"]
