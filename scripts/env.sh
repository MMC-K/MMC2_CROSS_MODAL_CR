#!/usr/bin/env bash
# Source this file from another script; all paths resolve from the package root.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${GPUS:-1,2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_DIR="${RUN_DIR:-outputs/run1}"
MODEL="${MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
IFS=',' read -r -a GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
NPROC="${#GPU_LIST[@]}"
if [[ "$NPROC" != 1 && "$NPROC" != 2 ]]; then
  echo "Set GPUS to one or two physical GPU IDs, for example GPUS=0 or GPUS=0,1." >&2
  exit 1
fi
