#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
condition="${1:-lora}"
if [[ $# -gt 0 ]]; then shift; fi
ADAPTER_ARGS=()
if [[ "$condition" == lora ]]; then
  ADAPTER_ARGS=(--adapter "$RUN_DIR/best_adapter")
  NAME=lora_test
elif [[ "$condition" == base ]]; then
  NAME=base_test
else
  echo "Usage: bash scripts/evaluate.sh [base|lora] [runner arguments]" >&2
  exit 1
fi
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  -m kommerce_coref.runner --mode evaluate --model "$MODEL" --output-dir "$RUN_DIR" \
  --eval-batch "${EVAL_BATCH:-16}" --name "$NAME" "${ADAPTER_ARGS[@]}" "$@"
