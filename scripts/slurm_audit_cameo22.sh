#!/usr/bin/env bash
#SBATCH --job-name=mf-cameo-audit
#SBATCH --partition=cpu_only
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/mf-cameo-audit-%j.out
#SBATCH --error=logs/slurm/mf-cameo-audit-%j.out

set -euo pipefail
ROOT="${MFCLEAN_ROOT:-${SLURM_SUBMIT_DIR:?}}"
cd "$ROOT"
set -a; source config/paths.env; set +a
mkdir -p logs/slurm data/audit
export PYTHONUNBUFFERED=1

FASTA="benchmarks/external_testsets/cameo22_183.fasta"
EXACT="data/audit/cameo22_183-exact-overlap.json"
EXACT_FASTA="data/audit/cameo22_183-exact-clean.fasta"
EXACT_IDS="data/audit/cameo22_183-exact-clean.ids"

if [ ! -s "$EXACT" ]; then
  "$MFCLEAN_PYTHON" benchmarks/audit_sequence_overlap.py \
    --targets "$FASTA" \
    --training data/audit/rcsb-admitted.fasta \
    --training data/audit/afdb-v4-training.fasta \
    --out "$EXACT" \
    --write-exact-clean-fasta "$EXACT_FASTA" \
    --write-exact-clean-ids "$EXACT_IDS"
else
  echo "[audit] reusing $EXACT"
fi

exec "$MFCLEAN_PYTHON" pipeline/10_homology_gate.py \
  --mmseqs "$MMSEQS_BIN" \
  --threads "${SLURM_CPUS_PER_TASK:-16}" \
  --benchmark cameo22_183 \
  --out data/audit/cameo22_homology_gate.json
