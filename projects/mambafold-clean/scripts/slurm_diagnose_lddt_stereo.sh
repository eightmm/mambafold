#!/usr/bin/env bash
#SBATCH --job-name=mf-stereo-diag
#SBATCH --partition=cpu_only
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/mf-stereo-diag-%j.out
#SBATCH --error=logs/slurm/mf-stereo-diag-%j.out

set -euo pipefail
ROOT="${MFCLEAN_ROOT:-${SLURM_SUBMIT_DIR:?}}"
cd "$ROOT"
set -a; source config/paths.env; set +a

DATASET="${MF_DIAG_DATASET:?set MF_DIAG_DATASET to casp15 or casp16}"
BASELINE_ROOT="${MF_DIAG_BASELINE_ROOT:?set MF_DIAG_BASELINE_ROOT to the existing external comparison root}"
case "$DATASET" in
  casp15|casp16) ;;
  *) echo "unsupported dataset: $DATASET" >&2; exit 2 ;;
esac
MODEL="outputs/benchmarks/run-a-final/$DATASET"
BASELINE="$BASELINE_ROOT/scores/external_accuracy_v2/$DATASET/simplefold_360m"
OUT="outputs/diagnostics/lddt-stereo-$DATASET"
test -s "$MODEL/pairs-admitted/manifest.json"
test -d "$BASELINE/inputs"
test -x "$OPENSTRUCTURE_OST"
test ! -e "$OUT"
mkdir -p logs/slurm

exec "$MFCLEAN_PYTHON" benchmarks/diagnose_lddt_stereo.py \
  --dataset "$DATASET" \
  --manifest "$MODEL/pairs-admitted/manifest.json" \
  --mamba-pairs "$MODEL/pairs-admitted" \
  --simplefold-pairs "$BASELINE/inputs" \
  --mamba-raw "$MODEL/scores-admitted/raw" \
  --simplefold-raw "$BASELINE/raw" \
  --ost "$OPENSTRUCTURE_OST" \
  --out-dir "$OUT"
