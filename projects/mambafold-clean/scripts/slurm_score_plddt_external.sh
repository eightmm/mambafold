#!/usr/bin/env bash
#SBATCH --job-name=mf-plddt-external
#SBATCH --partition=cpu_only
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/mf-plddt-external-%j.out
#SBATCH --error=logs/slurm/mf-plddt-external-%j.out

set -euo pipefail
ROOT="${MFCLEAN_ROOT:-${SLURM_SUBMIT_DIR:?}}"
cd "$ROOT"
set -a; source config/paths.env; set +a
export PYTHONUNBUFFERED=1

DATASET="${MF_BENCHMARK_SET:?set MF_BENCHMARK_SET}"
RUN="${MF_BENCHMARK_RUN:?set MF_BENCHMARK_RUN}"
BASE="outputs/benchmarks/${RUN}/${DATASET}"
for subset in full admitted; do
  "$MFCLEAN_PYTHON" benchmarks/score_plddt_external.py \
    --pairs-dir "$BASE/pairs-$subset" \
    --scores-dir "$BASE/scores-$subset" \
    --out "$BASE/plddt-$subset.json"
done
