#!/usr/bin/env bash
#SBATCH --job-name=mfclean-data
#SBATCH --partition=cpu_only
#SBATCH --qos=long
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=logs/slurm/mfclean-data-%j.out
#SBATCH --error=logs/slurm/mfclean-data-%j.out
#
# Run the whole data pipeline on a compute node. CPU-only; requests no GPU.
#
# Every stage is idempotent, so requeueing or resubmitting costs only the time
# already spent, never the work already done.
#
#   sbatch scripts/slurm_build_data.sh          # all stages
#   sbatch scripts/slurm_build_data.sh 06 07    # only these
#
# Watch with:  tail -f logs/slurm/mfclean-data-<jobid>.out

set -euo pipefail

# Slurm copies the batch script into its spool directory, so BASH_SOURCE does
# not locate the repository under sbatch. SLURM_SUBMIT_DIR is where sbatch was
# invoked; the BASH_SOURCE form is the fallback for running this directly.
ROOT="${MFCLEAN_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
if [ ! -f "$ROOT/pipeline/run_all.sh" ]; then
  echo "FATAL: \$ROOT=$ROOT is not the mambafold-clean root." >&2
  echo "       Submit from the repository root, or set MFCLEAN_ROOT." >&2
  exit 1
fi
cd "$ROOT"
mkdir -p logs/slurm

echo "node=$(hostname) job=${SLURM_JOB_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-?}"
echo "started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

bash pipeline/run_all.sh "$@"

echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
