#!/usr/bin/env bash
set -euo pipefail

CONFIG=""
STRICT=""
EXPORT_PACK=""

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
    --export-pack)
      EXPORT_PACK="$2"
      shift 2
      ;;
    *)
      echo "Usage: $0 --config <path> [--strict] --export-pack <zip>" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$CONFIG" || -z "$EXPORT_PACK" ]]; then
  echo "--config and --export-pack are required" >&2
  exit 1
fi

python -m analysis.export_pack --config "$CONFIG" --export-pack "$EXPORT_PACK" $STRICT
