#!/usr/bin/env bash
# Precompute ESM embeddings for AFDB SwissProt NPZ files.
# Safe to rerun: precompute_esm.py skips existing sequence embeddings by default.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_DIR="${DATA_DIR:-data/afdb_swissprot/npz}"
OUT_DIR="${OUT_DIR:-data/afdb_swissprot_esmc6b}"
GPUS="${GPUS:-0,1,2,3}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
ESM_MODEL="${ESM_MODEL:-esmc-6b}"
LIMIT="${LIMIT:-0}"
LOG_DIR="${LOG_DIR:-outputs/esm_$(basename "$OUT_DIR")}"
SHARD_FILES="${SHARD_FILES:-0}"
CACHE_LAYOUT="${CACHE_LAYOUT:-sequence}"
SINGLE_CHAIN_FASTA="${SINGLE_CHAIN_FASTA:-}"

mkdir -p "$OUT_DIR" "$LOG_DIR" .cache/tmp
export TMPDIR="${TMPDIR:-$ROOT/.cache/tmp}"
export PYTHONPATH="${PYTHONPATH:-src}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
SHARDS="${SHARDS:-${#GPU_ARR[@]}}"

echo "data_dir=$DATA_DIR"
echo "out_dir=$OUT_DIR"
echo "gpus=$GPUS shards=$SHARDS max_length=$MAX_LENGTH model=$ESM_MODEL cache_layout=$CACHE_LAYOUT"

pids=()
extra_args=()
if [[ "$LIMIT" != "0" ]]; then
  extra_args+=(--limit "$LIMIT")
fi
if [[ "$SHARD_FILES" == "1" ]]; then
  extra_args+=(--shard_files)
fi
if [[ "${FAIL_ON_ERROR:-0}" == "1" ]]; then
  extra_args+=(--fail_on_error)
fi
if [[ -n "$SINGLE_CHAIN_FASTA" ]]; then
  extra_args+=(--single_chain_fasta "$SINGLE_CHAIN_FASTA")
fi
for idx in "${!GPU_ARR[@]}"; do
  gpu="${GPU_ARR[$idx]}"
  log="$LOG_DIR/shard_${idx}.log"
  CUDA_VISIBLE_DEVICES="$gpu" \
    uv run --no-sync python -u scripts/precompute_esm.py \
      --data_dir "$DATA_DIR" \
      --out_dir "$OUT_DIR" \
      --esm_model "$ESM_MODEL" \
      --cache_layout "$CACHE_LAYOUT" \
      --device cuda \
      --dtype float16 \
      --max_length "$MAX_LENGTH" \
      --shard_idx "$idx" \
      --shard_count "$SHARDS" \
      "${extra_args[@]}" \
      > "$log" 2>&1 &
  pids+=("$!")
  echo "launched shard=$idx gpu=$gpu pid=${pids[-1]} log=$log"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

done_count="$(find "$OUT_DIR" -type f -name '*.npy' 2>/dev/null | wc -l)"
echo "done_count=$done_count"
exit "$status"
