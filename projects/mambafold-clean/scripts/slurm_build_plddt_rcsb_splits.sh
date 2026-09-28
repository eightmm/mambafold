#!/usr/bin/env bash
#SBATCH --job-name=mf-plddt-rcsb-split
#SBATCH --partition=cpu_only
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --qos=long
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/mf-plddt-rcsb-split-%j.out
#SBATCH --error=logs/slurm/mf-plddt-rcsb-split-%j.out

set -euo pipefail
ROOT="${MFCLEAN_ROOT:-${SLURM_SUBMIT_DIR:?}}"
cd "$ROOT"
set -a; source config/paths.env; set +a

ROOT_OUT="${MF_PLDDT_OUT_ROOT:-out/plddt-selfcond-sde200-tau03-rcsb}"
CHECKPOINT="${MF_PLDDT_CHECKPOINT:-out/run-a-selfcond-geo-v1/ckpt_0010000.pt}"
HASH_FILE="$ROOT_OUT/folding-checkpoint.sha256"

test -s "$CHECKPOINT"
mkdir -p logs/slurm "$ROOT_OUT"
PYTHONPATH=src "$MFCLEAN_PYTHON" scripts/build_plddt_rcsb_chain_splits.py \
  --chain-index-workers "${SLURM_CPUS_PER_TASK:-16}"
sha256sum "$CHECKPOINT" > "$HASH_FILE"
