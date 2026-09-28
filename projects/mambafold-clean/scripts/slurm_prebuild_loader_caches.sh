#!/usr/bin/env bash
#SBATCH --job-name=mf-cache
#SBATCH --partition=cpu_only
#SBATCH --qos=short
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --output=logs/slurm/mf-cache-%j.out
#SBATCH --error=logs/slurm/mf-cache-%j.out
#
# Build the loader's chain index once, on a compute node, so the training job
# does not pay ~1.5 h of probing at startup on every rank and every restart.
#
#   sbatch scripts/slurm_prebuild_loader_caches.sh configs/run_a_mamba.yaml

set -euo pipefail
CONFIG="${1:-configs/run_a_mamba.yaml}"
shift || true
ROOT="${MFCLEAN_ROOT:-${SLURM_SUBMIT_DIR:?}}"
cd "$ROOT"
source config/paths.env
mkdir -p logs/slurm .cache/length_cache
export PYTHONUNBUFFERED=1

echo "node=$(hostname) job=${SLURM_JOB_ID:-none} cpus=${SLURM_CPUS_PER_TASK} config=$CONFIG"

"$MFCLEAN_PYTHON" pipeline/13_prebuild_loader_caches.py --config "$CONFIG" "$@"
