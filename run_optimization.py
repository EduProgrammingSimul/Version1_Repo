"""
run_optimization.py

Seeded validation helper that leverages the deterministic ScenarioExecutor and
optimized controller factory. Optimisation flows remain unimplemented.
"""

import os
import sys
import argparse
from typing import List

from your_project import logging_setup
from analysis.controller_factory import ControllerFactory
from analysis.scenario_executor import ScenarioExecutor
from analysis.scenario_definitions import get_scenarios

logger = logging_setup.get_logger(__name__)

_DEFAULT_CORE_CONFIG = {
    "simulation": {"dt": 0.02},
    "initial_conditions": {"electrical_load_mw": 3008.5},
    "coupling": {"eta_transfer": 0.98},
}

DEFAULT_SUITE_PATH = os.path.join("config", "scenario_suite.txt")
DEFAULT_FALLBACK = ["baseline_steady_state", "sudden_load_increase_5pct"]


def _library_scenarios(factory: ControllerFactory) -> List[str]:
    try:
        core = factory.core_config
    except Exception:
        core = _DEFAULT_CORE_CONFIG
    scenarios = get_scenarios(core)
    return sorted(scenarios.keys()) if scenarios else []


def _read_suite_file(path: str) -> List[str]:
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Suite file not found: {path}")
    scenarios: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            scenarios.append(s)
    return scenarios


def _expand_scenarios(
    factory: ControllerFactory, arg_value: str, suite_file: str | None
) -> List[str]:
    if not arg_value:
        return []
    if arg_value.lower() != "all":
        return [s.strip() for s in arg_value.split(",") if s.strip()]
    library = _library_scenarios(factory)
    if library:
        return library
    if suite_file:
        try:
            sc = _read_suite_file(suite_file)
            logger.info("Expanded scenarios via --suite-file=%s", suite_file)
            return sc
        except Exception as exc:
            logger.error("Failed to read suite file %s: %s", suite_file, exc)
    try:
        sc = _read_suite_file(DEFAULT_SUITE_PATH)
        logger.info("Expanded scenarios via default suite file")
        return sc
    except Exception:
        logger.warning("Falling back to minimal scenario list: %s", DEFAULT_FALLBACK)
        return DEFAULT_FALLBACK[:]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run controller validation using deterministic executor"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Skip optimization; run validation only",
    )
    parser.add_argument(
        "--controller",
        type=str,
        required=True,
        help="Controller name, e.g., PID or FLC",
    )
    parser.add_argument(
        "--scenarios", type=str, default="all", help="Comma list or 'all'"
    )
    parser.add_argument(
        "--suite-file",
        type=str,
        default=None,
        help="Scenario suite file for --scenarios all",
    )
    parser.add_argument("--out", type=str, default="results", help="Root output folder")
    parser.add_argument("--eval-seed", type=int, default=42, help="Evaluation seed")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path = os.path.join(project_root, "config", "parameters.py")

    factory = ControllerFactory(config_path=config_path)
    scenarios = _expand_scenarios(factory, args.scenarios, suite_file=args.suite_file)
    if not scenarios:
        logger.error("No scenarios to run after expansion. Exiting.")
        sys.exit(2)

    if args.validate_only:
        executor = ScenarioExecutor(
            factory.core_config, out_root=args.out, eval_seed=args.eval_seed
        )
        ctrl_id = args.controller.strip().upper()
        try:
            controller_instance = factory.build(ctrl_id)
        except KeyError:
            logger.error(
                "Unknown controller '%s'. Available: %s",
                ctrl_id,
                factory.list_available(),
            )
            sys.exit(2)
        executor.run_matrix({ctrl_id: controller_instance}, scenarios)
        logger.info("Validation complete for controller %s", ctrl_id)
        return

    logger.warning("Optimization flow is not implemented in this runner.")
    sys.exit(0)


if __name__ == "__main__":
    main()
