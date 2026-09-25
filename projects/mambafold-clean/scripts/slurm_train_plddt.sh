#!/usr/bin/env bash
#SBATCH --job-name=mf-plddt-head
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --qos=long
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm/mf-plddt-head-%j.out
#SBATCH --error=logs/slurm/mf-plddt-head-%j.out

set -euo pipefail
ROOT="${MFCLEAN_ROOT:-${SLURM_SUBMIT_DIR:?}}"
cd "$ROOT"
set -a; source config/paths.env; set +a

CONFIG="${MF_PLDDT_CONFIG:-configs/plddt_selfcond_sde500.yaml}"
ROOT_OUT="${MF_PLDDT_OUT_ROOT:-out/plddt-selfcond-sde500}"
HEAD_OUT="$ROOT_OUT/head"
NPROC="${SLURM_GPUS_ON_NODE:-4}"

if [ -e "$HEAD_OUT" ]; then
  echo "refusing to overwrite confidence-head run: $HEAD_OUT" >&2
  exit 1
fi

# These Ada cards have no NVLink and their PCIe P2P path hangs NCCL. Confidence
# training is much smaller than folding, but still uses DDP for the 4-layer head.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

exec "$MFCLEAN_PYTHON" -m torch.distributed.run \
  --standalone --nproc_per_node="$NPROC" \
  scripts/train_plddt.py \
  --config "$CONFIG" \
  --train-manifest "$ROOT_OUT/rollouts/train/manifest-*.json" \
  --val-manifest "$ROOT_OUT/rollouts/val/manifest-*.json" \
  --out-dir "$HEAD_OUT"
