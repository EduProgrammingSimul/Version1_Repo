"""
run_training.py

Validation harness for RL (and other) controllers using the deterministic
ScenarioExecutor. Training flows remain unimplemented.
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

DEFAULT_SUITE_PATH = os.path.join("config", "scenario_suite.txt")
DEFAULT_FALLBACK = ["baseline_steady_state", "sudden_load_increase_5pct"]


def _library_scenarios(factory: ControllerFactory) -> List[str]:
    scenarios = get_scenarios(factory.core_config)
    return sorted(scenarios.keys()) if scenarios else []


def _read_suite_file(path: str) -> List[str]:
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Suite file not found: {path}")
    entries: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            entry = line.strip()
            if entry and not entry.startswith("#"):
                entries.append(entry)
    return entries


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
            logger.info("Expanded scenarios via suite file %s", suite_file)
            return sc
        except Exception as exc:
            logger.error("Failed to read suite file %s: %s", suite_file, exc)
    try:
        return _read_suite_file(DEFAULT_SUITE_PATH)
    except Exception:
        logger.warning("Falling back to minimal scenario list: %s", DEFAULT_FALLBACK)
        return DEFAULT_FALLBACK[:]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run RL validation using deterministic executor"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Skip training; run validation only",
    )
    parser.add_argument(
        "--controllers",
        type=str,
        default="RL",
        help="Comma list of controllers (default RL)",
    )
    parser.add_argument(
        "--scenarios", type=str, default="all", help="Comma list or 'all'"
    )
    parser.add_argument(
        "--suite-file",
        type=str,
        default=None,
        help="Scenario suite file when --scenarios all",
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

    controllers = [c.strip().upper() for c in args.controllers.split(",") if c.strip()]
    if not controllers:
        logger.error("No controllers specified. Exiting.")
        sys.exit(2)

    if args.validate_only:
        executor = ScenarioExecutor(
            factory.core_config, out_root=args.out, eval_seed=args.eval_seed
        )
        controller_instances = {}
        for ctrl in controllers:
            try:
                controller_instances[ctrl] = factory.build(ctrl)
            except KeyError:
                logger.error(
                    "Unknown controller '%s'. Available: %s",
                    ctrl,
                    factory.list_available(),
                )
                sys.exit(2)
        executor.run_matrix(controller_instances, scenarios)
        logger.info("Validation complete for controllers %s", controllers)
        return

    logger.warning("Training flow is not implemented in this runner.")
    sys.exit(0)


if __name__ == "__main__":
    main()
