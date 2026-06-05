#!/bin/bash
# Fan out precompute_esm.py across 8 B200s. Each shard owns 1/8 of unique
# sequences. All share the same scan (re-done per shard but cheap).
#
# Usage:
#   bash scripts/precompute_esm_8gpu.sh data/rcsb data/rcsb_esm [shard_count=8]
set -uo pipefail

DATA_DIR="${1:-data/rcsb}"
OUT_DIR="${2:-data/rcsb_esm}"
SHARDS="${3:-8}"

mkdir -p "$OUT_DIR" outputs
for i in $(seq 0 $((SHARDS-1))); do
  CUDA_VISIBLE_DEVICES="$i" \
  PYTHONPATH=src \
    uv run python scripts/precompute_esm.py \
      --data_dir "$DATA_DIR" \
      --out_dir  "$OUT_DIR" \
      --esm_model esm3-open \
      --device cuda \
      --dtype float16 \
      --shard_idx "$i" \
      --shard_count "$SHARDS" \
      > "outputs/esm_shard_$i.log" 2>&1 &
  echo "launched shard $i pid=$!"
done
echo "All $SHARDS shards launched."
