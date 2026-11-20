#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  -m kommerce_coref.runner --mode infer --model "$MODEL" --adapter "$RUN_DIR/best_adapter" \
  --input examples/inference_sample.jsonl --output-dir "$RUN_DIR/inference" --name predictions "$@"
