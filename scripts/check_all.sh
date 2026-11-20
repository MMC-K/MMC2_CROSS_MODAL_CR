#!/usr/bin/env bash
set -euo pipefail
CHECK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CHECK_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" scripts/check_data.py
"$PYTHON_BIN" scripts/prepare_data.py
"$PYTHON_BIN" -m unittest discover -s tests -v
