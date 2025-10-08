import numpy as np

CORE_PARAMETERS = {
    "simulation": {"dt": 0.02, "max_steps": 5000},
    "reactor": {
        "beta_i": np.array(
            [0.000215, 0.001424, 0.001274, 0.002568, 0.000748, 0.000273]
        ),
        "lambda_i": np.array([0.0124, 0.0305, 0.1110, 0.3010, 1.1400, 3.0100]),
        "Lambda": 1.0e-4,
        "beta_total": 0.006502,
        "P0": 3411.0,
        "T_inlet": 290.0,
        "T_coolant0": 306.5,
        "T_fuel0": 829.0,
        "C_f": 70.5,
        "C_c": 26.3,
        "Omega": 6.53,
        "alpha_f": -4.5e-5,
        "alpha_c": -1.2e-4,
        "n_initial": 1.0,
    },
    "coupling": {"eta_transfer": 0.98},
    "turbine": {"tau_t": 0.3, "tau_v": 0.1, "omega_nominal_rpm": 1800.0, "H": 4.0},
    "grid": {"H": 5.0, "D": 1.0, "f_nominal": 60.0, "S_base": 1000.0},
    "safety_limits": {
        "max_fuel_temp_c": 2800.0,
        "fuel_temp_warning_fraction": 0.95,
        "max_speed_rpm": 2250.0,
        "min_frequency_hz": 59.0,
        "max_frequency_hz": 61.0,
        "freq_deviation_limit_hz": 1.0,
    },
    "reporting": {
        "template_dir": "analysis/templates",
        "report_output_dir": "results/reports",
        "plot_output_dir": "results/plots",
        "template_name": "report_template_v2.md",
        "comparison_criteria": {
            "primary_metric": "composite_robustness_score",
            "crs_metrics": {
                "higher_is_better": {
                    "grid_load_following_index": 0.30,
                    "robustness_score": 0.40,
                },
                "lower_is_better": {
                    "transient_severity_score": 0.15,
                    "control_effort_valve_sq_sum": 0.15,
                },
            },
        },
    },
    "rl_normalization_factors": {
        "reactor_power_mw": 3411.0,
        "T_fuel": 2800.0,
        "v_pos_actual": 1.0,
        "grid_frequency_hz": 60.0,
        "speed_rpm": 1800.0,
        "power_error": 500.0,
    },
    "controllers": {
        "PID": {
            "kp": 0.05,
            "ki": 0.01,
            "kd": 0.005,
            "setpoint": 1800.0,
            "output_limits": (0.0, 1.0),
        },
        "FLC": {
            "setpoint": 1800.0,
            "error_scaling": 1.0,
            "derror_scaling": 1.0,
            "output_scaling": 1.0,
        },
        "RL_AGENT": {},
    },
    "rl_training_adv": {
        "total_timesteps": 15_000_000,
        "model_save_path": "results/rl_models/robust_expert_agent_v3",
        "tensorboard_log_dir": "results/rl_logs/robust_expert_agent_v3_tb/",
        "algorithm": "SAC",
        "policy": "MlpPolicy",
        "learning_starts": 120000,
        "eval_freq": 80000,
        "learning_rate": 0.0003,
        "batch_size": 1024,
        "policy_kwargs": {"net_arch": [512, 512]},
        "buffer_size": 1_200_000,
        "reward_weights": {
            "w_unsafe_penalty": -10000.0,
            "w_freq_error_sq": 50.0,
            "w_valve_movement": 100.0,
            "w_jerk_penalty": 200.0,
            "calm_threshold_hz": 0.05,
            "calm_threshold_mw": 50.0,
            "cost_multiplier_calm": 25.0,
            "w_robustness_bonus": 15.0,
        },
        "use_curriculum": True,
        "curriculum_config": {
            "phases": {
                1: {
                    "name": "Phase 1: Foundational Stability",
                    "scenarios": [
                        "baseline_steady_state",
                        "gradual_load_increase_10pct",
                    ],
                    "thresholds": {"min_avg_reward": -7000},
                },
                2: {
                    "name": "Phase 2: Efficiency & Preservation Drills",
                    "scenarios": ["steady_state_efficiency_probe"],
                    "thresholds": {"min_avg_reward": -2000},
                    "reward_override": {
                        "cost_multiplier_calm": 50.0,
                        "w_valve_movement": 200.0,
                    },
                },
                3: {
                    "name": "Phase 3: Adversarial Hardening",
                    "scenarios": [
                        "deceptive_sensor_noise",
                        "parameter_randomization_drills",
                    ],
                    "thresholds": {"min_avg_reward": 500},
                },
                4: {
                    "name": "Phase 4: Cascading Failure Resilience",
                    "scenarios": ["cascading_grid_fault_and_recovery"],
                    "thresholds": {"min_avg_reward": 1000},
                    "reward_override": {"w_robustness_bonus": 30.0},
                },
                5: {
                    "name": "Phase 5: Final Comprehensive Exam",
                    "scenarios": ["combined_challenge_final_exam"],
                    "thresholds": {"min_avg_reward": 2000},
                },
            }
        },
    },
}


def get_config(set_name="default"):

    return CORE_PARAMETERS
