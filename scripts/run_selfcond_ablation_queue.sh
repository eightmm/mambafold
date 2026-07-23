#!/bin/bash
# Sequential direct all-atom architecture ablations.
#
# Run when GPUs are free:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_selfcond_ablation_queue.sh
#
# Hypotheses:
#   1. Self-conditioning improves CASP14 TM/GDT/lDDT at matched steps.
#   2. Pair-free distogram/contact auxiliaries add topology signal without a
#      full pair stack.

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=${CONFIG:-configs/direct_allatom_puremamba_attn6_geo_adaln_sf360.yaml}
BASE_PORT=${BASE_PORT:-29710}
STEPS=${STEPS:-50000}
GPUS=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

run_train() {
    local name="$1"
    local port="$2"
    shift 2
    echo "=== $name ==="
    CUDA_VISIBLE_DEVICES="$GPUS" \
    CONFIG="$CONFIG" \
    OUT_DIR="outputs/train/${name}" \
    MASTER_PORT="$port" \
    bash scripts/train.sh \
        --total_steps "$STEPS" \
        --ckpt_interval 10000 \
        --eval_interval 5000 \
        --wandb_name "$name" \
        "$@"
}

run_train direct_puremamba_sf360_selfcond_50k "$BASE_PORT" \
    --self_conditioning \
    --self_condition_prob 0.5 \
    --wandb_tags direct_allatom pure_mamba attn_every_6 adaln_zero self_conditioning ablation

run_train direct_puremamba_sf360_selfcond_pairaux_50k "$((BASE_PORT + 1))" \
    --self_conditioning \
    --self_condition_prob 0.5 \
    --pairfree_aux_heads \
    --w_distogram 0.2 \
    --w_contact 0.1 \
    --wandb_tags direct_allatom pure_mamba attn_every_6 adaln_zero self_conditioning pairfree_aux ablation
