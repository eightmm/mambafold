#!/usr/bin/env bash
# Eight-rank real-data, production-memory, and checkpoint/resume gates.

set -euo pipefail
cd "$(dirname "$0")/.."

config="configs/direct_allatom_puremamba_attn6_geo_adaln_sf360_esmc6b_gpu8.yaml"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
export MAMBA_SKIP_CUDA_BUILD=${MAMBA_SKIP_CUDA_BUILD:-TRUE}
start_gate="${START_GATE:-1}"

if [[ -z "${CUDA_HOME:-}" ]]; then
    CUDA_HOME="$(.venv/bin/python -c 'from tilelang.env import CUDA_HOME; print(CUDA_HOME)')"
fi
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export CPLUS_INCLUDE_PATH="$CUDA_HOME/include/cccl:$CUDA_HOME/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"

if (( start_gate <= 1 )); then
    echo "[gate 1/4] 8-rank exact-index real-data loader, workers=0"
    uv run --no-sync torchrun --nproc_per_node=8 --master_port=29614 \
        scripts/smoke_real_data_ddp.py --config "$config" --batches 8 --workers 0
fi

if (( start_gate <= 2 )); then
    echo "[gate 2/4] 8-rank automatic worker cap and prefetch"
    uv run --no-sync torchrun --nproc_per_node=8 --master_port=29615 \
        scripts/smoke_real_data_ddp.py --config "$config" --batches 8
fi

if (( start_gate <= 3 )); then
    echo "[gate 3/4] 8-rank variable-length allocator and L=1024 accumulation"
    uv run --no-sync torchrun --nproc_per_node=8 --master_port=29616 \
        scripts/smoke_esmc6b_ddp.py --config "$config" \
        --length-sequence 128,256,384,512,640,768,896,1024
fi

echo "[gate 4/4] production train checkpoint and 8-rank resume"
smoke_out="/tmp/mambafold_esmc6b_gpu8_ckpt_${SLURM_JOB_ID:-manual}"
rm -rf "$smoke_out"
CONFIG="$config" OUT_DIR="$smoke_out" \
    bash scripts/train.sh \
    --total_steps 1 --ckpt_interval 1 --eval_interval 0 \
    --log_interval 1 --no_wandb
CONFIG="$config" OUT_DIR="$smoke_out" RESUME="$smoke_out/ckpt_latest.pt" \
    bash scripts/train.sh \
    --total_steps 2 --ckpt_interval 1 --eval_interval 0 \
    --log_interval 1 --no_wandb
.venv/bin/python -c \
    "import torch; p='$smoke_out/ckpt_latest.pt'; c=torch.load(p, map_location='cpu', weights_only=False); assert c['step']==2; assert len(c['rng_states'])==8; assert c['data_state']['world_size']==8; print('checkpoint_resume_ok', p, c['step'], len(c['rng_states']))"
rm -rf "$smoke_out"

echo "8-GPU preflight passed."
