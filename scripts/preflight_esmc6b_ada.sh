#!/usr/bin/env bash
# Run the real-data and production-shape gates inside a 4-GPU allocation.

set -euo pipefail
cd "$(dirname "$0")/.."

config="configs/direct_allatom_puremamba_attn6_geo_adaln_sf360_esmc6b.yaml"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
export MAMBA_SKIP_CUDA_BUILD=${MAMBA_SKIP_CUDA_BUILD:-TRUE}

if [[ -z "${CUDA_HOME:-}" ]]; then
    CUDA_HOME="$(.venv/bin/python -c 'from tilelang.env import CUDA_HOME; print(CUDA_HOME)')"
fi
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export CPLUS_INCLUDE_PATH="$CUDA_HOME/include/cccl:$CUDA_HOME/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"

echo "[gate 1/3] exact-index real-data loader, workers=0"
uv run --no-sync torchrun --nproc_per_node=4 --master_port=29514 \
    scripts/smoke_real_data_ddp.py --config "$config" --batches 8 --workers 0

echo "[gate 2/3] production automatic worker cap and prefetch"
uv run --no-sync torchrun --nproc_per_node=4 --master_port=29515 \
    scripts/smoke_real_data_ddp.py --config "$config" --batches 16

echo "[gate 3/3] variable-length allocator, then production accumulation at 1024"
uv run --no-sync torchrun --nproc_per_node=4 --master_port=29516 \
    scripts/smoke_esmc6b_ddp.py --config "$config" \
    --length-sequence 128,256,384,512,640,768,896,1024
