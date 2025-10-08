#!/usr/bin/env bash
set -euo pipefail

CONFIG=""
STRICT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --strict)
      STRICT="--strict"
      shift 1
      ;;
    *)
      echo "Usage: $0 --config <path> [--strict]" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$CONFIG" ]]; then
  echo "--config is required" >&2
  exit 1
fi

PACK_NAME=$(python - "$CONFIG" <<'PY'
import sys, yaml
from pathlib import Path

if len(sys.argv) < 2:
    raise SystemExit("config path required")

cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
pack = cfg.get('export', {}).get('pack_name', 'Results.zip')
print(pack)
PY
)

bash scripts/journal_train.sh --config "$CONFIG" ${STRICT}

bash scripts/journal_eval.sh --config "$CONFIG" ${STRICT} --export-pack "$PACK_NAME"
