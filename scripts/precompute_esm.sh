#!/bin/bash
# Single-GPU ESM pre-computation. For 8-GPU fan-out see precompute_esm_8gpu.sh.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/precompute_esm.sh

set -euo pipefail
cd "$(dirname "$0")/.."

export NETRC=${NETRC:-/NHNHOME/WORKSPACE/0526040024_A/jaemin/.netrc}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}

DATA_DIR=${DATA_DIR:-data/rcsb}
OUT_DIR=${OUT_DIR:-data/rcsb_esm}
ESM_MODEL=${ESM_MODEL:-esm3-open}

mkdir -p "$OUT_DIR"

echo "=== ESM Pre-computation ==="
echo "data_dir : $DATA_DIR"
echo "out_dir  : $OUT_DIR"
echo "esm_model: $ESM_MODEL"
echo "file_list: ${FILE_LIST:-all}"
echo "==========================="

PYTHONPATH=src PYTHONUNBUFFERED=1 exec uv run python -u scripts/precompute_esm.py \
    --data_dir "$DATA_DIR" \
    --out_dir  "$OUT_DIR" \
    --esm_model "$ESM_MODEL" \
    --device cuda \
    ${FILE_LIST:+--file_list "$FILE_LIST"} \
    "$@"
