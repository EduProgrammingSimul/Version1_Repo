#!/usr/bin/env bash
set -euo pipefail
# Auto-format and lint-fix
python -m pip install --upgrade pip >/dev/null 2>&1 || true
pip install black ruff >/dev/null 2>&1 || true
black .
ruff check --fix .
echo "Formatting & linting complete."