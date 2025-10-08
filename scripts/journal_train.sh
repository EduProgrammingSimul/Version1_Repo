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

python -m rl.trainer --config "$CONFIG" $STRICT
