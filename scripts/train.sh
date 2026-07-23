#!/bin/bash
# MambaFold direct-launch trainer (no SLURM).
#
# Usage:
#   bash scripts/train.sh [extra train.py args ...]
#
# Environment:
#   CUDA_VISIBLE_DEVICES   GPU selection (default: all visible). e.g. "0,1,2,3"
#   CONFIG                 YAML config (default: configs/direct_allatom_360m.yaml)
#   RESUME                 Checkpoint path to resume from (optional)
#   OUT_DIR                Output dir (default: outputs/train/<timestamp>)
#   MASTER_PORT            DDP rendezvous port (default: 29500)
#
# Examples:
#   # 4-GPU pretrain on GPUs 0-3
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train.sh
#
#   # Resume current run
#   RESUME=outputs/train/prev/ckpt_latest.pt \
#     CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train.sh
#
#   # Single-GPU smoke test
#   CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh --total_steps 100

set -euo pipefail
cd "$(dirname "$0")/.."

# Mamba-3 uses TileLang kernels and does not require the legacy Mamba-1
# selective-scan extension.  Skip that extension when uv has to synchronize a
# fresh environment, then point TileLang at the pinned CUDA 13 compiler wheels.
export MAMBA_SKIP_CUDA_BUILD="${MAMBA_SKIP_CUDA_BUILD:-TRUE}"
if [[ ! -x .venv/bin/python ]]; then
    echo "Missing .venv; run MAMBA_SKIP_CUDA_BUILD=TRUE uv sync --extra dev first." >&2
    exit 2
fi
if [[ -z "${CUDA_HOME:-}" ]]; then
    CUDA_HOME="$(.venv/bin/python -c 'from tilelang.env import CUDA_HOME; print(CUDA_HOME)')"
fi
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export CPLUS_INCLUDE_PATH="$CUDA_HOME/include/cccl:$CUDA_HOME/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"

# ── W&B auth ─────────────────────────────────────────────────────────────────
# Use the submitting user's current home instead of a machine-specific legacy
# workspace path. Callers can still override NETRC explicitly.
export NETRC="${NETRC:-$HOME/.netrc}"

# ── GPU selection ─────────────────────────────────────────────────────────────
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    N_GPU=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
else
    N_GPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
fi

# ── NCCL (single-node multi-GPU) ──────────────────────────────────────────────
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_BUFFSIZE=${NCCL_BUFFSIZE:-16777216}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-1800000}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# ── Config / output ───────────────────────────────────────────────────────────
CONFIG="${CONFIG:-configs/direct_allatom_360m.yaml}"
OUT_DIR="${OUT_DIR:-outputs/train/$(date +%Y%m%d_%H%M%S)}"
MASTER_PORT="${MASTER_PORT:-29500}"
mkdir -p "$OUT_DIR"

echo "=== MambaFold Training ==="
echo "Config   : $CONFIG"
echo "Out dir  : $OUT_DIR"
echo "GPUs     : ${CUDA_VISIBLE_DEVICES:-all (${N_GPU})}"
echo "Resume   : ${RESUME:-none}"
uv run --no-sync python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}, n_gpu={torch.cuda.device_count()}')"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
echo "=========================="

PYTHONPATH=src PYTHONUNBUFFERED=1 exec uv run --no-sync torchrun \
    --nproc_per_node="$N_GPU" \
    --master_port="$MASTER_PORT" \
    scripts/train.py \
    --config "$CONFIG" \
    --out_dir "$OUT_DIR" \
    ${RESUME:+--resume "$RESUME"} \
    "$@"
