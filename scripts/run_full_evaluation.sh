#!/usr/bin/env bash
# Full-scope runner: evaluates ALL controllers across ALL scenarios from a suite file,
# then aggregates metrics & renders ALL figures.
# Usage:
#   ./scripts/run_full_evaluation.sh [OUT_DIR] [SUITE_FILE] [RL_MODEL_PATH]
# Examples:
#   ./scripts/run_full_evaluation.sh results
#   ./scripts/run_full_evaluation.sh results config/scenario_suite.txt config/optimized_controllers/RL_Agent_Optimized.zip
set -euo pipefail

OUT="${1:-results}"
SUITE="${2:-config/scenario_suite.txt}"
RL_MODEL="${3:-config/optimized_controllers/RL_Agent_Optimized.zip}"
PY="${PYTHON:-python}"

if [[ ! -f "$SUITE" ]]; then
  echo "[FATAL] Missing suite file: $SUITE"
  echo "Create it with one scenario per line. See config/scenario_suite.txt"
  exit 1
fi

if [[ ! -f "$RL_MODEL" ]]; then
  echo "[FATAL] Missing RL model zip at: $RL_MODEL"
  exit 1
fi

echo "[INFO] Using OUT=$OUT"
echo "[INFO] Using SUITE=$SUITE"
echo "[INFO] Using RL_MODEL=$RL_MODEL"
mkdir -p "$OUT"

while IFS= read -r SC || [[ -n "$SC" ]]; do
  # skip empty lines and comments
  [[ -z "$SC" || "$SC" =~ ^# ]] && continue
  echo "===================="
  echo "[SCENARIO] $SC"
  echo "--------------------"
  echo "[RUN] PID"
  "$PY" run_optimization.py --validate-only --controller PID --scenarios "$SC" --out "$OUT"

  echo "[RUN] FLC"
  "$PY" run_optimization.py --validate-only --controller FLC --scenarios "$SC" --out "$OUT"

  echo "[RUN] RL"
  "$PY" run_training.py --validate-only --model "$RL_MODEL" --controllers RL --scenarios "$SC" --out "$OUT" --deterministic
done < "$SUITE"

echo "[INFO] Aggregating metrics, pivots, and rendering ALL figures..."
"$PY" scripts/validate_and_report.py \
  --controllers PID FLC RL \
  --scenarios all \
  --suite-file "$SUITE" \
  --out "$OUT" \
  --render-plots

echo "[DONE] Full evaluation complete."
echo "Artifacts under: $OUT"
