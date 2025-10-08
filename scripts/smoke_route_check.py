from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd


def _hash_timeseries(df: pd.DataFrame) -> str:
    cols = [
        c for c in ["time_s", "y_actual", "u", "freq_hz", "temp_C"] if c in df.columns
    ]
    if not cols:
        raise ValueError(
            "Timeseries missing required numeric columns for hash computation"
        )
    data = df[cols].sort_values(by=cols[0]).to_numpy(dtype=float)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    payload = data.tobytes()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-check deterministic evaluation outputs."
    )
    parser.add_argument(
        "results_root", help="Path to results directory produced by full_eval"
    )
    parser.add_argument(
        "--controllers",
        nargs="+",
        default=["PID", "FLC", "RL"],
        help="Expected controllers",
    )
    args = parser.parse_args()

    validation_root = os.path.join(args.results_root, "validation")
    if not os.path.isdir(validation_root):
        print(f"Validation directory missing: {validation_root}", file=sys.stderr)
        sys.exit(1)

    controllers = [c.strip() for c in args.controllers]
    missing = []
    identity_issues = []
    timeseries_counter = 0

    scenario_hashes = defaultdict(dict)

    for controller in controllers:
        ctrl_dir = os.path.join(validation_root, controller)
        if not os.path.isdir(ctrl_dir):
            missing.append((controller, "<all>", "controller directory missing"))
            continue
        for scenario in sorted(os.listdir(ctrl_dir)):
            scen_dir = os.path.join(ctrl_dir, scenario)
            ts_path = os.path.join(scen_dir, "timeseries.csv")
            if not os.path.isfile(ts_path):
                missing.append((controller, scenario, "timeseries.csv missing"))
                continue
            df = pd.read_csv(ts_path)
            if df.empty:
                missing.append((controller, scenario, "timeseries empty"))
                continue
            try:
                fingerprint = _hash_timeseries(df)
            except Exception as exc:
                print(f"Failed to hash {ts_path}: {exc}", file=sys.stderr)
                sys.exit(1)
            scenario_hashes[scenario][controller] = fingerprint
            timeseries_counter += 1

    for scenario, ctrl_map in scenario_hashes.items():
        fingerprints = defaultdict(list)
        for ctrl, fp in ctrl_map.items():
            fingerprints[fp].append(ctrl)
        for fp, ctrl_list in fingerprints.items():
            if len(ctrl_list) > 1:
                identity_issues.append((scenario, ctrl_list))

    if missing:
        print("Missing or invalid timeseries detected:")
        for ctrl, scenario, reason in missing:
            print(f"  - controller={ctrl} scenario={scenario}: {reason}")
        sys.exit(1)

    if identity_issues:
        print("Identical traces detected across controllers:")
        for scenario, ctrl_list in identity_issues:
            print(f"  - scenario={scenario} controllers={ctrl_list}")
        sys.exit(1)

    expected = len(scenario_hashes) * len(controllers)
    if timeseries_counter != expected:
        print(
            f"Count mismatch: expected {expected} timeseries, found {timeseries_counter}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"Smoke check passed: {len(scenario_hashes)} scenarios × {len(controllers)} controllers."
    )


if __name__ == "__main__":
    main()
