#!/bin/bash
# Single-GPU ESM pre-computation. For 8-GPU fan-out see precompute_esm_8gpu.sh.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/precompute_esm.sh

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

DATA_DIR=${DATA_DIR:-data/rcsb_boltz_official_full}
OUT_DIR=${OUT_DIR:-data/rcsb_esmc6b_official_full}
ESM_MODEL=${ESM_MODEL:-esmc-6b}
MAX_LENGTH=${MAX_LENGTH:-1024}

mkdir -p "$OUT_DIR"

echo "=== ESM Pre-computation ==="
echo "data_dir : $DATA_DIR"
echo "out_dir  : $OUT_DIR"
echo "esm_model: $ESM_MODEL"
echo "max_length: $MAX_LENGTH"
echo "file_list: ${FILE_LIST:-all}"
echo "==========================="

PYTHONPATH=src PYTHONUNBUFFERED=1 exec uv run --no-sync python -u scripts/precompute_esm.py \
    --data_dir "$DATA_DIR" \
    --out_dir  "$OUT_DIR" \
    --esm_model "$ESM_MODEL" \
    --max_length "$MAX_LENGTH" \
    --device cuda \
    ${FILE_LIST:+--file_list "$FILE_LIST"} \
    "$@"
