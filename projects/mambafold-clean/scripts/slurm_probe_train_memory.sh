#!/usr/bin/env bash
#SBATCH --job-name=mf-mem
#SBATCH --qos=veryshort
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/mf-mem-%j.out
#SBATCH --error=logs/slurm/mf-mem-%j.out
#
# Measure the real training-step footprint of candidate model sizes.
# Partition and --gres are supplied at submission so one script covers every
# device type:
#
#   sbatch -p 6000ada --gres=gpu:1              scripts/slurm_probe_train_memory.sh ada
#   sbatch -p heavy   --gres=gpu:h100:1         scripts/slurm_probe_train_memory.sh h100
#   sbatch -p heavy   --gres=gpu:6000pro_maxq:1 scripts/slurm_probe_train_memory.sh pro6000

set -euo pipefail
LABEL="${1:?pass a short device label}"
shift || true
ROOT="${MFCLEAN_ROOT:-${SLURM_SUBMIT_DIR:?}}"
cd "$ROOT"
source config/paths.env
mkdir -p logs/slurm data/audit/train_memory
export PYTHONUNBUFFERED=1

echo "node=$(hostname) job=${SLURM_JOB_ID:-none} label=$LABEL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$MFCLEAN_PYTHON" benchmarks/probe_train_memory.py \
  --out "data/audit/train_memory/${LABEL}.json" "$@"
