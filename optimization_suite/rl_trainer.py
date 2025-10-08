import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from environment.pwr_gym_env import PWRGymEnvUnified
from analysis.scenario_definitions import get_scenarios
from analysis.metrics_engine import MetricsEngine
from optimization_suite.auto_validator import auto_validate_and_report

logger = logging.getLogger(__name__)


class MultiObjectiveCurriculumCallback(EvalCallback):

    def __init__(self, *args, curriculum_config: Dict, core_config: Dict, **kwargs):
        super(MultiObjectiveCurriculumCallback, self).__init__(*args, **kwargs)

        self.curriculum_config = curriculum_config
        self.core_config = core_config
        self.metrics_engine = MetricsEngine(core_config)
        self.phases = self.curriculum_config.get("phases", {})
        self.sorted_phase_keys = sorted(self.phases.keys())
        self.current_phase_index = 0
        logger.info(
            "MultiObjectiveCurriculumCallback initialized. Starting at Phase 1."
        )

    def _on_evaluation_end(self) -> bool:

        self.logger.info(
            f"Evaluation finished. Mean reward: {self.last_mean_reward:.2f}"
        )

        all_metrics_data = []
        for _ in range(self.n_eval_episodes):
            obs, done, episode_data = self.eval_env.reset(), [False], []
            while not all(done):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, info = self.eval_env.step(action)
                episode_data.append(info[0])

            metrics = self.metrics_engine.calculate(
                pd.DataFrame(episode_data), self.core_config
            )
            all_metrics_data.append(metrics)

        avg_metrics = pd.DataFrame(all_metrics_data).mean().to_dict()
        avg_metrics["min_avg_reward"] = self.last_mean_reward

        self._check_and_promote(avg_metrics)

        return super()._on_evaluation_end()

    def _check_and_promote(self, avg_metrics: Dict[str, float]):
        if self.current_phase_index >= len(self.sorted_phase_keys) - 1:
            return

        current_phase = self.phases[self.sorted_phase_keys[self.current_phase_index]]
        thresholds = current_phase.get("thresholds", {})

        passed_all_checks = True
        for metric, target in thresholds.items():
            current_value = avg_metrics.get(metric)
            if current_value is None or pd.isna(current_value):
                passed_all_checks = False
                break

            if ("min_" in metric and current_value < target) or (
                "max_" in metric and current_value > target
            ):
                passed_all_checks = False
                break

        self.logger.info(
            f"[Curriculum] Phase '{current_phase.get('name')}': Passed All Checks = {passed_all_checks}"
        )

        if passed_all_checks:
            self.current_phase_index += 1
            next_phase = self.phases[self.sorted_phase_keys[self.current_phase_index]]
            logger.warning(
                f"[Curriculum] PROMOTION! Graduated to Phase '{next_phase.get('name')}'."
            )

            self.training_env.env_method("set_active_scenario", next_phase["scenarios"])

            if "reward_override" in next_phase:
                self.training_env.env_method(
                    "update_reward_weights", next_phase["reward_override"]
                )


class RLTrainer:

    def __init__(self, base_config_full: Dict[str, Any], config_path: str):
        self.logger = logging.getLogger(__name__)
        self.full_config = base_config_full
        self.config_path = config_path
        self.core_config = self.full_config.get("CORE_PARAMETERS", {})
        self.rl_config = self.core_config.get("rl_training_adv", {})

        phases = self.rl_config.get("curriculum_config", {}).get("phases", {})
        first_phase_scenarios = phases.get(1, {}).get(
            "scenarios", ["baseline_steady_state"]
        )

        def make_train_env():
            env = PWRGymEnvUnified(
                **self._get_env_params(first_phase_scenarios, is_training=True)
            )
            return Monitor(env)

        self.env = DummyVecEnv([make_train_env])

        self.model = self._setup_agent()

    def _get_env_params(self, scenarios, is_training):

        return {
            "reactor_params": self.core_config.get("reactor", {}),
            "turbine_params": self.core_config.get("turbine", {}),
            "grid_params": self.core_config.get("grid", {}),
            "coupling_params": self.core_config.get("coupling", {}),
            "sim_params": self.core_config.get("simulation", {}),
            "safety_limits": self.core_config.get("safety_limits", {}),
            "rl_normalization_factors": self.core_config.get(
                "rl_normalization_factors", {}
            ),
            "all_scenarios_definitions": get_scenarios(self.core_config),
            "initial_scenario_name": scenarios,
            "is_training_env": is_training,
            "rl_training_config": self.rl_config if is_training else {},
        }

    def _setup_agent(self):

        algo_params = {
            k: v
            for k, v in self.rl_config.items()
            if k
            in [
                "learning_rate",
                "buffer_size",
                "learning_starts",
                "batch_size",
                "tau",
                "gamma",
                "train_freq",
                "gradient_steps",
                "ent_coef",
                "target_update_interval",
                "target_entropy",
                "use_sde",
                "sde_sample_freq",
                "policy_kwargs",
            ]
        }
        return SAC(
            policy=self.rl_config.get("policy", "MlpPolicy"),
            env=self.env,
            verbose=0,
            tensorboard_log=self.rl_config.get("tensorboard_log_dir"),
            **algo_params,
        )

    def train(self) -> bool:

        total_timesteps = self.rl_config.get("total_timesteps", 1_000_000)
        save_path = self.rl_config.get(
            "model_save_path", "results/rl_models/default_run"
        )
        self.logger.info(
            f"RL training started. Artifacts will be saved to: {save_path}"
        )

        os.makedirs(save_path, exist_ok=True)

        def make_eval_env():
            eval_scenarios = ["combined_challenge", "sudden_load_increase_5pct"]
            env = PWRGymEnvUnified(
                **self._get_env_params(eval_scenarios, is_training=True)
            )
            return Monitor(env)

        eval_env = DummyVecEnv([make_eval_env])

        best_model_path = os.path.join(save_path, "best_model.zip")

        eval_callback = MultiObjectiveCurriculumCallback(
            eval_env=eval_env,
            curriculum_config=self.rl_config.get("curriculum_config", {}),
            core_config=self.core_config,
            n_eval_episodes=5,
            eval_freq=self.rl_config.get("eval_freq", 50000),
            log_path=save_path,
            best_model_save_path=save_path,
            deterministic=True,
        )

        callbacks = [
            CheckpointCallback(
                save_freq=100_000,
                save_path=save_path,
                name_prefix="rl_model_checkpoint",
            ),
            eval_callback,
        ]

        try:
            self.model.learn(total_timesteps=total_timesteps, callback=callbacks)

            final_model_path = os.path.join(save_path, "RL_Agent_optimized.zip")
            self.model.save(final_model_path)
            self.logger.info(
                f"Training complete. Final champion model saved to {final_model_path}"
            )

            best_performing_model_path = os.path.join(save_path, "best_model.zip")
            if not os.path.exists(best_performing_model_path):
                best_performing_model_path = final_model_path

            self.logger.info(
                f"--- Triggering Post-Training Validation on Best Model: {best_performing_model_path} ---"
            )
            auto_validate_and_report(
                controller_identifier=best_performing_model_path,
                config_path=self.config_path,
                save_tag="post_training_best",
            )
            return True

        except Exception as e:
            self.logger.error(
                f"A critical error occurred during RL training: {e}", exc_info=True
            )
            return False
