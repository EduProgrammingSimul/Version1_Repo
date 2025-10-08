#!/usr/bin/env bash
set -euo pipefail

CONFIG="config/config.yaml"
OUT="smoke_results.zip"

bash scripts/journal_train.sh --config "$CONFIG"
bash scripts/journal_eval.sh --config "$CONFIG" --export-pack "$OUT"

echo "Smoke run complete: $OUT"
