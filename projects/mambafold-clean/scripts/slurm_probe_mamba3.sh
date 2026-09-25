#!/usr/bin/env bash
#SBATCH --job-name=mf-dstate
#SBATCH --qos=veryshort
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/mf-dstate-%j.out
#SBATCH --error=logs/slurm/mf-dstate-%j.out
#
# Sweep (d_state, mimo_rank) on one GPU and record what the kernels accept.
# Partition and --gres are supplied at submission so one script covers every
# device type:
#
#   sbatch -p 6000ada --gres=gpu:1                 scripts/slurm_probe_mamba3.sh ada
#   sbatch -p heavy   --gres=gpu:h100:1            scripts/slurm_probe_mamba3.sh h100
#   sbatch -p heavy   --gres=gpu:6000pro_maxq:1    scripts/slurm_probe_mamba3.sh pro6000

set -euo pipefail
LABEL="${1:?pass a short device label}"
ROOT="${MFCLEAN_ROOT:-${SLURM_SUBMIT_DIR:?}}"
cd "$ROOT"
source config/paths.env
mkdir -p logs/slurm data/audit/mamba3_probe
export PYTHONUNBUFFERED=1

echo "node=$(hostname) job=${SLURM_JOB_ID:-none} label=$LABEL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$MFCLEAN_PYTHON" benchmarks/probe_mamba3_dstate.py \
  --out "data/audit/mamba3_probe/${LABEL}.json" \
  --d_states 64,128,256 --mimo_ranks 1,2,4,8
