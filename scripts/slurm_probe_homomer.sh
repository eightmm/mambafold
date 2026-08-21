#!/usr/bin/env bash
#SBATCH --job-name=mf-homomer
#SBATCH --partition=cpu_only
#SBATCH --qos=veryshort
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=logs/slurm/mf-homomer-%j.out
#SBATCH --error=logs/slurm/mf-homomer-%j.out
#
#   sbatch scripts/slurm_probe_homomer.sh --sample 20000

set -euo pipefail
ROOT="${MFCLEAN_ROOT:-${SLURM_SUBMIT_DIR:?}}"
cd "$ROOT"
source config/paths.env
mkdir -p logs/slurm
export PYTHONUNBUFFERED=1
echo "node=$(hostname) job=${SLURM_JOB_ID:-none} cpus=${SLURM_CPUS_PER_TASK}"
"$MFCLEAN_PYTHON" benchmarks/probe_homomer_redundancy.py --workers "${SLURM_CPUS_PER_TASK}" "$@"
