#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export RUN_DIR MODEL PYTHON_BIN
"$PYTHON_BIN" scripts/check_data.py
"$PYTHON_BIN" scripts/download_model.py
bash scripts/train.sh
bash scripts/evaluate.sh base
bash scripts/evaluate.sh lora
"$PYTHON_BIN" scripts/compare.py --run-dir "$RUN_DIR"
