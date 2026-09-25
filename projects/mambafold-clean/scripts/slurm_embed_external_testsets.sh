#!/usr/bin/env bash
#SBATCH --job-name=mf-casp-esmc
#SBATCH --partition=6000ada
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=logs/slurm/mf-casp-esmc-%j.out
#SBATCH --error=logs/slurm/mf-casp-esmc-%j.out
#
# ESMC-6B embeddings for the 113 external benchmark targets. Without these the
# rollout has nothing to run on, and they are absent by construction: the
# training caches are sequence-addressed and the benchmark targets are exactly
# the sequences that corpus must not contain. Measured against the two training
# caches the hit rate is 0/70, 1/22 and 2/21.
#
# Writes to data/casp_esmc6b, kept apart from the training caches so a benchmark
# sequence can never be picked up as a training record.
#
#   sbatch scripts/slurm_embed_external_testsets.sh

set -euo pipefail
ROOT="${MFCLEAN_ROOT:-${SLURM_SUBMIT_DIR:?}}"
cd "$ROOT"
set -a; source config/paths.env; set +a
mkdir -p logs/slurm data/casp_esmc6b data/audit
export PYTHONUNBUFFERED=1

echo "node=$(hostname) job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

exec "$MFCLEAN_PYTHON" pipeline/14_embed_external_testsets.py "$@"
