#!/bin/bash
# End-to-end evaluation: predict + score for a tier on a checkpoint.
#
# Usage:
#   bash benchmarks/run_eval.sh <CKPT> <TIER> [GPU_ID]
#     CKPT  = path to ckpt_*.pt
#     TIER  = t0_smoke | t1_quick | t2_full
#     GPU_ID = optional, defaults to 0
#
# Example (after Phase 3 finishes):
#   bash benchmarks/run_eval.sh outputs/train/<run>/ckpt_latest.pt t1_quick 0
#
# Outputs:
#   benchmarks/results/<run_label>/{<id>_pred.pdb, <id>_gt.pdb, manifest.json, scores.json, summary.txt}
#
# Notes:
#   - inference uses the project venv (.venv/) via uv run
#   - scoring uses the isolated tools/scoring_venv/ (numpy<2 + DockQ)
#   - the two venvs do not need to know about each other
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT="${1:?Usage: $0 <CKPT> <TIER> [GPU_ID]}"
TIER="${2:?Usage: $0 <CKPT> <TIER> [GPU_ID]}"
GPU_ID="${3:-0}"
# Optional env-var hooks for ablations: SUFFIX appends to RUN_DIR, EXTRA_INF_ARGS
# is splatted into run_inference.py invocation (deliberately unquoted to allow
# multi-word args like "--n_steps 200 --n_recycle 1").
SUFFIX="${SUFFIX:-}"
EXTRA_INF_ARGS="${EXTRA_INF_ARGS:-}"

IDS_FILE="benchmarks/sets/${TIER}.txt"
[[ -f "$IDS_FILE" ]] || { echo "no such tier: $IDS_FILE  (run benchmarks/select_eval_set.py first)"; exit 1; }
[[ -f "$CKPT" ]] || { echo "no such ckpt: $CKPT"; exit 1; }

CKPT_TAG="$(basename "$(dirname "$CKPT")")_$(basename "$CKPT" .pt)"
RUN_DIR="benchmarks/results/${CKPT_TAG}_${TIER}${SUFFIX}"
mkdir -p "$RUN_DIR"

case "$TIER" in
    t0_smoke)        MAX_L=512  ;;
    t1_quick)        MAX_L=1024 ;;
    t2_full)         MAX_L=2048 ;;
    len_0_512)       MAX_L=512  ;;
    len_512_1024)    MAX_L=1024 ;;
    len_1024_2048)   MAX_L=2048 ;;
    *) echo "unknown tier $TIER"; exit 1 ;;
esac

echo "===== MambaFold benchmark =================================="
echo "  ckpt    : $CKPT"
echo "  tier    : $TIER  (max_L=$MAX_L, $(wc -l < $IDS_FILE) ids)"
echo "  out_dir : $RUN_DIR"
echo "  gpu     : $GPU_ID"
echo "============================================================"

# Step 1: inference (project venv)
echo "[1/2] inference... extra_args=[$EXTRA_INF_ARGS]"
# shellcheck disable=SC2086  # $EXTRA_INF_ARGS is intentionally word-split
CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH=src PYTHONUNBUFFERED=1 \
    uv run python -u benchmarks/run_inference.py \
        --ckpt "$CKPT" \
        --ids "$IDS_FILE" \
        --out "$RUN_DIR" \
        --max_length "$MAX_L" \
        $EXTRA_INF_ARGS \
        2>&1 | tee "$RUN_DIR/inference.log"

# Step 2: scoring (isolated venv)
echo "[2/2] scoring..."
tools/scoring_venv/bin/python -u benchmarks/score.py \
    --in_dir "$RUN_DIR" \
    --out "$RUN_DIR/scores.json" \
    2>&1 | tee "$RUN_DIR/score.log"

# Compact summary file (last block of score.log)
grep -A 20 "=== SUMMARY ===" "$RUN_DIR/score.log" > "$RUN_DIR/summary.txt" || true
echo
echo "===== done. summary at $RUN_DIR/summary.txt ====="
