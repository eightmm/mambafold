#!/usr/bin/env bash
# Fan out precompute_esm.py across eight visible GPUs. Each shard owns one
# eighth of the unique sequences.
#
# Usage:
#   bash scripts/precompute_esm_8gpu.sh \
#     data/rcsb_boltz_official_full data/rcsb_esmc6b_official_full [shard_count=8]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_DIR="${1:-data/rcsb_boltz_official_full}"
OUT_DIR="${2:-data/rcsb_esmc6b_official_full}"
SHARDS="${3:-8}"

mkdir -p "$OUT_DIR" outputs
pids=()
for i in $(seq 0 $((SHARDS-1))); do
  CUDA_VISIBLE_DEVICES="$i" \
  PYTHONPATH=src \
    uv run --no-sync python scripts/precompute_esm.py \
      --data_dir "$DATA_DIR" \
      --out_dir  "$OUT_DIR" \
      --esm_model esmc-6b \
      --device cuda \
      --dtype float16 \
      --max_length 1024 \
      --cache_layout sequence \
      --shard_idx "$i" \
      --shard_count "$SHARDS" \
      > "outputs/esm_shard_$i.log" 2>&1 &
  pids+=("$!")
  echo "launched shard $i pid=${pids[-1]}"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
