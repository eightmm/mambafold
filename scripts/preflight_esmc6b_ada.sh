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

uv run --no-sync torchrun --nproc_per_node=4 --master_port=29515 \
    scripts/smoke_real_data_ddp.py --config "$config" --batches 4

uv run --no-sync torchrun --nproc_per_node=4 --master_port=29516 \
    scripts/smoke_esmc6b_ddp.py --config "$config" \
    --batch-size 10 --length 1024 --grad-accum 1
