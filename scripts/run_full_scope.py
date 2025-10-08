from __future__ import annotations
import argparse, os, sys, subprocess, random
import numpy as np


def sh(cmd, env=None):
    print(">>", " ".join(cmd))
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def main():
    ap = argparse.ArgumentParser(
        description="Run validation->metrics->figures in one go (seeded, ALL scenarios)."
    )
    ap.add_argument("--out", default="results", help="Output root")
    ap.add_argument(
        "--suite-file",
        default="config/scenario_suite.txt",
        help="Scenario suite (ALL scenarios)",
    )
    ap.add_argument("--controllers", nargs="+", default=["PID", "FLC", "RL"])
    ap.add_argument(
        "--rl-model", default="config/optimized_controllers/RL_Agent_Optimized.zip"
    )
    ap.add_argument("--eval-seed", type=int, default=int(os.getenv("EVAL_SEED", "42")))
    ap.add_argument(
        "--skip-figures", action="store_true", help="Run validation & metrics only"
    )
    args = ap.parse_args()

    os.environ["EVAL_SEED"] = str(args.eval_seed)
    os.environ["PYTHONHASHSEED"] = str(args.eval_seed)
    random.seed(args.eval_seed)
    np.random.seed(args.eval_seed)
    child_env = os.environ.copy()

    sh(
        [
            sys.executable,
            "scripts/full_eval.py",
            "--out",
            args.out,
            "--suite-file",
            args.suite_file,
            "--controllers",
            *args.controllers,
            "--rl-model",
            args.rl_model,
            "--eval-seed",
            str(args.eval_seed),
        ],
        env=child_env,
    )

    if not args.skip - figures:
        sh(
            [
                sys.executable,
                "scripts/validate_and_report.py",
                "--controllers",
                *args.controllers,
                "--scenarios",
                "all",
                "--suite-file",
                args.suite_file,
                "--out",
                args.out,
                "--render-plots",
                "--eval-seed",
                str(args.eval_seed),
            ],
            env=child_env,
        )

    print("✅ Full-scope pipeline finished.")


if __name__ == "__main__":
    main()
