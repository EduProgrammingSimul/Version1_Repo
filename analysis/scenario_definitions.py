import numpy as np
import logging
from typing import Dict, Any, Callable

logger = logging.getLogger(__name__)


def constant_load(load_mw: float) -> Callable[[float, int], float]:

    return lambda time_s, step: float(load_mw)


def gradual_load_change(
    initial: float, final: float, start_t: float, duration: float
) -> Callable[[float, int], float]:

    if duration <= 1e-6:
        duration = 1e-6

    def profile(time_s: float, step: int) -> float:
        if time_s < start_t:
            return float(initial)
        elif time_s < start_t + duration:
            fraction = (time_s - start_t) / duration
            return float(initial) + (float(final) - float(initial)) * fraction
        else:
            return float(final)

    return profile


def step_load_change(
    initial: float, final: float, step_t: float
) -> Callable[[float, int], float]:

    return lambda time_s, step: (
        float(final) if time_s >= float(step_t) else float(initial)
    )


def multi_step_load_profile(steps: list) -> Callable[[float, int], float]:

    def profile(time_s: float, step: int) -> float:
        current_load = steps[0][0]
        for load, start_time in steps:
            if time_s >= start_time:
                current_load = load
            else:
                break
        return float(current_load)

    return profile


def prbs_load_profile(
    base: float, amplitude: float, total_time: float, dt: float, seed: int = 0
) -> Callable[[float, int], float]:

    steps = max(1, int(total_time / dt) + 1)
    rng = np.random.default_rng(seed)
    sequence = rng.choice([-1.0, 1.0], size=steps)
    values = base + amplitude * sequence

    def profile(time_s: float, step: int) -> float:
        idx = min(step, len(values) - 1)
        return float(values[idx])

    return profile


def multisine_load_profile(
    base: float, amplitudes: list[float], freqs_hz: list[float]
) -> Callable[[float, int], float]:

    amplitudes = list(amplitudes)
    freqs_hz = list(freqs_hz)

    def profile(time_s: float, step: int) -> float:
        total = base
        for amp, freq in zip(amplitudes, freqs_hz):
            total += amp * np.sin(2 * np.pi * freq * time_s)
        return float(total)

    return profile


def get_scenarios(core_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:

    scenarios: Dict[str, Dict[str, Any]] = {}
    try:
        sim_dt = core_config["simulation"]["dt"]
        initial_load_base = core_config.get("initial_conditions", {}).get(
            "electrical_load_mw", 3008.5
        )
        eta_base = core_config.get("coupling", {}).get("eta_transfer", 0.98)
    except KeyError as e:
        logger.error(
            f"Failed to get required value from core_config: {e}", exc_info=True
        )
        return {}

    total_time_default = 400.0

    scenarios["baseline_steady_state"] = {
        "description": "Baseline steady-state operation at 90% power.",
        "load_profile_func": constant_load(initial_load_base),
        "max_steps": int(500 / sim_dt),
        "reset_options": {"initial_power_level": 0.9},
        "analysis_tags": ["baseline"],
        "event_schedule": [
            {
                "time_s": 0.0,
                "end_time": 0.0,
                "label": "steady_state_start",
                "magnitude": 0.0,
            }
        ],
    }
    scenarios["gradual_load_increase_10pct"] = {
        "description": "Gradual load ramp from 90% to 100%.",
        "load_profile_func": gradual_load_change(
            initial_load_base * 0.9, initial_load_base, 20.0, 300.0
        ),
        "max_steps": int(400 / sim_dt),
        "reset_options": {"initial_power_level": 0.9},
        "analysis_tags": ["ramp", "validation"],
        "event_schedule": [
            {
                "time_s": 20.0,
                "end_time": 320.0,
                "label": "load_ramp",
                "magnitude": abs(initial_load_base * 0.1),
            }
        ],
    }
    scenarios["sudden_load_increase_5pct"] = {
        "description": "Sudden +5% load increase from nominal.",
        "load_profile_func": step_load_change(
            initial_load_base, initial_load_base * 1.05, 20.0
        ),
        "max_steps": int(300 / sim_dt),
        "analysis_tags": ["step"],
        "event_schedule": [
            {
                "time_s": 20.0,
                "end_time": 20.0,
                "label": "load_step",
                "magnitude": abs(initial_load_base * 0.05),
            }
        ],
    }

    scenarios["steady_state_efficiency_probe"] = {
        "description": "RL Drill: A long, quiet hold to enforce control efficiency.",
        "load_profile_func": constant_load(initial_load_base),
        "max_steps": int(1800 / sim_dt),
        "is_efficiency_probe": True,
        "analysis_tags": ["efficiency", "validation"],
        "event_schedule": [
            {
                "time_s": 0.0,
                "end_time": 1800.0,
                "label": "efficiency_hold",
                "magnitude": 0.0,
            }
        ],
    }

    scenarios["deceptive_sensor_noise"] = {
        "description": "Adversarial Test: High, dynamic sensor noise during a load ramp.",
        "load_profile_func": gradual_load_change(
            initial_load_base * 0.9, initial_load_base, 20.0, 300.0
        ),
        "max_steps": int(400 / sim_dt),
        "reset_options": {"initial_power_level": 0.9},
        "adversarial_noise": {
            "active": True,
            "initial_magnitude": 0.05,
            "final_magnitude": 0.15,
            "bias_magnitude": 8.0,
        },
        "is_adversarial_drill": True,
        "analysis_tags": ["adversarial", "noise"],
        "event_schedule": [
            {
                "time_s": 0.0,
                "end_time": 20.0,
                "label": "noise_ramp_in",
                "magnitude": 0.0,
            },
            {
                "time_s": 20.0,
                "end_time": 320.0,
                "label": "load_ramp",
                "magnitude": abs(initial_load_base * 0.1),
            },
            {
                "time_s": 320.0,
                "end_time": 400.0,
                "label": "noise_high",
                "magnitude": 0.0,
            },
        ],
    }
    scenarios["parameter_randomization_drills"] = {
        "description": "RL Drill: Train against randomized physics for generalization.",
        "load_profile_func": constant_load(initial_load_base),
        "max_steps": int(400 / sim_dt),
        "is_domain_randomization_drill": True,
        "is_adversarial_drill": True,
        "analysis_tags": ["adversarial", "randomization"],
        "event_schedule": [
            {
                "time_s": 0.0,
                "end_time": 400.0,
                "label": "parameter_randomization",
                "magnitude": 0.0,
            }
        ],
    }

    scenarios["cascading_grid_fault_and_recovery"] = {
        "description": "Adversarial Drill: A cascading grid fault followed by recovery demand.",
        "load_profile_func": multi_step_load_profile(
            [
                (initial_load_base, 0.0),
                (initial_load_base * 0.8, 20.0),
                (initial_load_base * 0.85, 120.0),
                (initial_load_base * 1.05, 150.0),
            ]
        ),
        "max_steps": int(500 / sim_dt),
        "env_modifications": [
            {
                "type": "grid_power_imbalance",
                "imbalance_mw": 50.0,
                "start_time": 20.0,
                "end_time": 25.0,
            }
        ],
        "is_adversarial_drill": True,
        "analysis_tags": ["adversarial", "fault"],
        "event_schedule": [
            {
                "time_s": 0.0,
                "end_time": 0.0,
                "label": "nominal_start",
                "magnitude": 0.0,
            },
            {
                "time_s": 20.0,
                "end_time": 25.0,
                "label": "load_drop",
                "magnitude": abs(initial_load_base * 0.2),
            },
            {
                "time_s": 120.0,
                "end_time": 120.0,
                "label": "load_partial_recover",
                "magnitude": abs(initial_load_base * 0.05),
            },
            {
                "time_s": 150.0,
                "end_time": 150.0,
                "label": "load_recovery_push",
                "magnitude": abs(initial_load_base * 0.2),
            },
        ],
    }

    scenarios["combined_challenge_final_exam"] = {
        "description": "Final Exam: Compound failure with grid fault, component degradation, and noise.",
        "load_profile_func": step_load_change(
            initial_load_base, initial_load_base * 1.1, 20.0
        ),
        "max_steps": int(400 / sim_dt),
        "env_modifications": [
            {
                "type": "parameter_ramp",
                "parameter_path": ["coupling", "eta_transfer"],
                "start_value": eta_base,
                "end_value": eta_base * 0.90,
                "start_time": 50.0,
                "duration": 150.0,
            }
        ],
        "adversarial_noise": {
            "active": True,
            "initial_magnitude": 0.02,
            "final_magnitude": 0.05,
            "bias_magnitude": 2.0,
        },
        "is_adversarial_drill": True,
        "analysis_tags": ["adversarial", "compound"],
        "event_schedule": [
            {
                "time_s": 20.0,
                "end_time": 20.0,
                "label": "load_step",
                "magnitude": abs(initial_load_base * 0.1),
            },
            {
                "time_s": 50.0,
                "end_time": 200.0,
                "label": "coupling_degradation",
                "magnitude": abs(eta_base * 0.1),
            },
            {
                "time_s": 0.0,
                "end_time": 400.0,
                "label": "sensor_noise",
                "magnitude": 0.0,
            },
        ],
    }

    total_time_prbs = total_time_default
    prbs_profile = prbs_load_profile(
        initial_load_base,
        amplitude=initial_load_base * 0.05,
        total_time=total_time_prbs,
        dt=sim_dt,
        seed=123,
    )
    scenarios["prbs_excitation_probe"] = {
        "description": "Frequency-domain probe using a PRBS load sequence.",
        "load_profile_func": prbs_profile,
        "max_steps": int(total_time_prbs / sim_dt),
        "analysis_tags": ["prbs", "frequency_probe"],
        "is_frequency_probe": True,
        "event_schedule": [
            {
                "time_s": 0.0,
                "end_time": total_time_prbs,
                "label": "prbs_excitation",
                "magnitude": initial_load_base * 0.05,
            }
        ],
    }

    multisine_profile = multisine_load_profile(
        base=initial_load_base,
        amplitudes=[
            initial_load_base * 0.02,
            initial_load_base * 0.015,
            initial_load_base * 0.01,
        ],
        freqs_hz=[0.01, 0.05, 0.12],
    )
    scenarios["multisine_frequency_probe"] = {
        "description": "Multi-sine excitation to estimate closed-loop bandwidth.",
        "load_profile_func": multisine_profile,
        "max_steps": int(total_time_default / sim_dt),
        "analysis_tags": ["multisine", "frequency_probe"],
        "is_frequency_probe": True,
        "event_schedule": [
            {
                "time_s": 0.0,
                "end_time": total_time_default,
                "label": "multisine_excitation",
                "magnitude": initial_load_base * 0.02,
            }
        ],
    }

    scenarios["actuator_saturation_challenge"] = {
        "description": "Stress test emphasizing actuator limits with alternating overload demands.",
        "load_profile_func": multi_step_load_profile(
            [
                (initial_load_base * 1.05, 0.0),
                (initial_load_base * 0.95, 40.0),
                (initial_load_base * 1.1, 80.0),
                (initial_load_base * 0.9, 120.0),
                (initial_load_base * 1.12, 160.0),
            ]
        ),
        "max_steps": int(220 / sim_dt),
        "analysis_tags": ["saturation", "stress_test"],
        "event_schedule": [
            {
                "time_s": 0.0,
                "end_time": 0.0,
                "label": "high_demand",
                "magnitude": abs(initial_load_base * 0.05),
            },
            {
                "time_s": 40.0,
                "end_time": 40.0,
                "label": "demand_drop",
                "magnitude": abs(initial_load_base * 0.1),
            },
            {
                "time_s": 80.0,
                "end_time": 80.0,
                "label": "overload_push",
                "magnitude": abs(initial_load_base * 0.15),
            },
            {
                "time_s": 160.0,
                "end_time": 160.0,
                "label": "final_overload",
                "magnitude": abs(initial_load_base * 0.12),
            },
        ],
    }

    for name, config_dict in scenarios.items():
        if "reset_options" not in config_dict:
            config_dict["reset_options"] = {}

    logger.info(
        f"Defined {len(scenarios)} total scenarios, including excitation probes and adversarial drills."
    )
    return scenarios
