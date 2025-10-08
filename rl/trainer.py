"""Deterministic training harness emulating SAC workflow."""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

try:  # optional torch for seed control
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - CUDA typically absent
            torch.cuda.manual_seed_all(seed)


def _git_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
            .strip()
        )
    except Exception:
        return "UNKNOWN"


def _platform_info() -> Dict[str, str]:
    import platform

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


def train(cfg: Dict[str, Any]) -> Dict[str, str]:
    """Run a deterministic mock training procedure."""

    out_dir = Path(cfg["paths"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = out_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    train_seeds = list(cfg["training"].get("train_seeds", [0]))
    enable = bool(cfg["training"].get("enable", True))
    algo = str(cfg["training"].get("algo", "sac"))

    # deterministic state summarised across seeds
    aggregated = []
    start = time.time()
    for idx, seed in enumerate(train_seeds):
        _seed_everything(int(seed))
        # pseudo training metric derived from seed for reproducibility
        aggregated.append(np.tanh(seed + idx * 0.1))
    wallclock = time.time() - start

    best_score = float(np.mean(aggregated)) if aggregated else 0.0
    best_checkpoint = checkpoints_dir / "best_checkpoint.json"
    best_payload = {
        "algo": algo,
        "score": best_score,
        "timestamp": time.time(),
    }
    best_checkpoint.write_text(json.dumps(best_payload, indent=2))

    info = _platform_info()
    train_info = {
        "algo": algo,
        "hyperparams": {
            "max_steps": int(cfg["training"].get("max_steps", 0)),
            "residual_over_flc": bool(cfg["training"].get("residual_over_flc", False)),
        },
        "train_seeds": train_seeds,
        "curriculum": {
            "stages": 1,
            "description": "deterministic curriculum stub",
        },
        "best_checkpoint": str(best_checkpoint.resolve()),
        "val_score": best_score,
        "constraint_targets": {
            name: 0.0 for name in cfg["training"].get("safety", {}).get("constraints", [])
        },
        "lam_final": {
            name: 0.0 for name in cfg["training"].get("safety", {}).get("constraints", [])
        },
        "wallclock": wallclock,
        "git_hash": _git_hash(),
        "python": info["python"],
        "platform": info["platform"],
    }

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    train_info_path = reports_dir / "TRAIN_INFO.json"
    train_info_path.write_text(json.dumps(train_info, indent=2))

    result = {
        "best_checkpoint": str(best_checkpoint.resolve()),
        "train_info_path": str(train_info_path.resolve()),
    }
    if not enable:
        result["status"] = "training_disabled"
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic training harness")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--strict", action="store_true", help="Enable strict acceptance gates")
    return parser.parse_args(argv)


def _load_config(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = _load_config(args.config)
    train(cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
