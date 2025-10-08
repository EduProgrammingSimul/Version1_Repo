from __future__ import annotations

"""Deterministic controller factory that always loads optimized artifacts.

This module centralizes instantiation of the evaluation controllers so that
all downstream tooling (scenario executor, pipelines, UI) can request fresh
instances without worrying about file paths or configuration plumbing.

The factory enforces the use of the optimized controller artifacts located in
``config/optimized_controllers`` and records the metadata necessary for
reproducibility.
"""

from dataclasses import dataclass
import os
from typing import Dict, Any, Tuple

from analysis.parameter_manager import ParameterManager
from controllers import load_controller
from controllers.base_controller import BaseController


_OPTIMIZED_NAME_MAP: Dict[str, str] = {
    "PID": "PID_optimized",
    "FLC": "FLC_optimized",
    "RL": "RL_Agent_Optimized",
}


@dataclass(frozen=True)
class ControllerSpec:
    """Descriptor for a controller available to the evaluation pipeline."""

    name: str
    optimized_key: str
    artifact_path: str


class ControllerFactory:
    """Factory responsible for loading controllers backed by optimized artifacts."""

    def __init__(self, config_path: str) -> None:
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        self._param_manager = ParameterManager(config_filepath=config_path)
        self._full_config = self._param_manager.get_all_parameters()
        self._core = self._full_config.get("CORE_PARAMETERS", {})
        if not self._core:
            raise ValueError("Configuration missing 'CORE_PARAMETERS'.")

        sim_cfg = self._core.get("simulation", {})
        self._dt = float(sim_cfg.get("dt", 0.02))
        self._specs = self._build_specs(config_path)

    @staticmethod
    def _build_specs(config_path: str) -> Dict[str, ControllerSpec]:
        project_root = os.path.abspath(os.path.join(os.path.dirname(config_path), ".."))
        optimized_dir = os.path.join(project_root, "config", "optimized_controllers")
        specs: Dict[str, ControllerSpec] = {}
        for external_name, optimized_key in _OPTIMIZED_NAME_MAP.items():
            if optimized_key.endswith(".zip"):
                artifact_candidate = os.path.join(optimized_dir, optimized_key)
            elif optimized_key.lower().startswith("rl_agent"):
                artifact_candidate = os.path.join(optimized_dir, f"{optimized_key}.zip")
            else:
                artifact_candidate = os.path.join(
                    optimized_dir, f"{optimized_key}.yaml"
                )
            specs[external_name] = ControllerSpec(
                name=external_name,
                optimized_key=optimized_key,
                artifact_path=artifact_candidate,
            )
        return specs

    @property
    def timestep(self) -> float:
        """Return the simulation step (seconds)."""

        return self._dt

    @property
    def core_config(self) -> Dict[str, Any]:
        """Expose the immutable CORE_PARAMETERS dictionary."""

        return self._core

    def list_available(self) -> Tuple[str, ...]:
        """Return the tuple of controller identifiers supported by the factory."""

        return tuple(sorted(self._specs.keys()))

    def build(self, controller_name: str) -> BaseController:
        """Instantiate a fresh controller backed by the optimized artifact.

        Args:
            controller_name: Logical controller identifier (PID, FLC, RL).
        """

        key = controller_name.strip().upper()
        if key not in self._specs:
            raise KeyError(
                f"Unsupported controller '{controller_name}'. Available: {self.list_available()}"
            )

        spec = self._specs[key]
        if not os.path.exists(spec.artifact_path):
            raise FileNotFoundError(
                f"Optimized artifact missing for {controller_name}: {spec.artifact_path}"
            )

        instance, resolved_name = load_controller(
            controller_name_or_path=spec.optimized_key,
            base_config=self._full_config,
            dt=self._dt,
        )
        if instance is None:
            raise RuntimeError(
                f"Failed to instantiate controller '{controller_name}' from optimized artifact."
            )

        return instance


__all__ = ["ControllerFactory", "ControllerSpec"]
