from __future__ import annotations

import argparse
import os
import sys
import random
from typing import List, Dict

import numpy as np

from analysis.controller_factory import ControllerFactory
from analysis.scenario_executor import ScenarioExecutor
from your_project import logging_setup

logger = logging_setup.get_logger(__name__)


def parse_suite(path: str) -> List[str]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Scenario suite file not found: {path}")
    scenarios: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            entry = line.strip()
            if not entry or entry.startswith("#"):
                continue
            scenarios.append(entry)
    if not scenarios:
        raise ValueError(f"No scenarios found in suite file: {path}")
    return scenarios


def _resolve_config_path() -> str:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(project_root, "config", "parameters.py")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministic seeded evaluation across controllers and scenarios."
    )
    parser.add_argument("--out", required=True, help="Output root directory (results)")
    parser.add_argument(
        "--suite-file", required=True, help="Path to scenario suite file"
    )
    parser.add_argument(
        "--controllers",
        nargs="+",
        default=["PID", "FLC", "RL"],
        help="Controllers to evaluate",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        required=True,
        help="Primary deterministic evaluation seed",
    )
    parser.add_argument(
        "--eval-seeds",
        nargs="*",
        default=None,
        help="Optional additional seeds for multi-seed evaluation",
    )
    args = parser.parse_args()

    seed_values = [int(args.eval_seed)]
    if args.eval_seeds:
        seed_values.extend(int(s) for s in args.eval_seeds)
    seed_values = sorted(set(seed_values))
    multi_seed = len(seed_values) > 1

    suite_scenarios = parse_suite(args.suite_file)
    config_path = _resolve_config_path()
    factory = ControllerFactory(config_path=config_path)

    available = set(factory.list_available())
    requested = [ctrl.strip().upper() for ctrl in args.controllers]
    invalid = [c for c in requested if c not in available]
    if invalid:
        raise ValueError(
            f"Unsupported controllers requested: {invalid}. Available: {sorted(available)}"
        )

    logger.info(
        "Starting deterministic evaluation: controllers=%s, scenarios=%s, seeds=%s",
        requested,
        suite_scenarios,
        seed_values,
    )

    for seed in seed_values:
        out_root_seed = (
            os.path.join(args.out, f"seed_{seed}") if multi_seed else args.out
        )
        os.makedirs(out_root_seed, exist_ok=True)

        os.environ["EVAL_SEED"] = str(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        random.seed(seed)

        scenario_executor = ScenarioExecutor(
            core_config=factory.core_config, out_root=out_root_seed, eval_seed=seed
        )

        for scenario_name in suite_scenarios:
            controllers: Dict[str, object] = {
                ctrl: factory.build(ctrl) for ctrl in requested
            }
            scenario_executor.run_matrix(controllers, [scenario_name])

        logger.info(
            "Seed %s completed. Outputs in %s",
            seed,
            os.path.join(out_root_seed, "validation"),
        )

    logger.info("Deterministic evaluation completed for all seeds.")


if __name__ == "__main__":
    main()
