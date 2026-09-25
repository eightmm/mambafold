#!/usr/bin/env bash
#SBATCH --job-name=mfclean-esmc
#SBATCH --partition=6000ada
#SBATCH --qos=short
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --output=logs/slurm/mfclean-esmc-%j.out
#SBATCH --error=logs/slurm/mfclean-esmc-%j.out
#
# Fill every ESMC-6B embedding the pipeline could not reuse: the RCSB sequences
# stage 05 found uncached, and the AFDB records stage 07b disqualified because
# their UniProt sequence changed between AlphaFold DB v4 and v6.
#
# Needs one GPU and the pinned ESMC-6B revision. Idempotent: precompute_esm
# skips outputs that already exist, so a rerun costs only the scan.
#
#   sbatch scripts/slurm_compute_missing_embeddings.sh

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
# shellcheck source=../config/paths.env
source config/paths.env
mkdir -p logs/slurm
export PYTHONUNBUFFERED=1

echo "node=$(hostname) job=${SLURM_JOB_ID:-none} gpu=${CUDA_VISIBLE_DEVICES:-?}"
"$MFCLEAN_PYTHON" pipeline/12_compute_missing_embeddings.py --source all --device cuda
