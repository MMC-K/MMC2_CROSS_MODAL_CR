#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  -m kommerce_coref.runner --mode train --model "$MODEL" --output-dir "$RUN_DIR" \
  --micro-batch "${MICRO_BATCH:-16}" --eval-batch "${EVAL_BATCH:-16}" "$@"
