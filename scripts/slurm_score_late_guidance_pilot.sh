#!/usr/bin/env bash
#SBATCH --job-name=mf-guide-pilot-score
#SBATCH --partition=cpu_only
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/mf-guide-pilot-score-%j.out
#SBATCH --error=logs/slurm/mf-guide-pilot-score-%j.out

set -euo pipefail
ROOT="${MFCLEAN_ROOT:-${SLURM_SUBMIT_DIR:?}}"
cd "$ROOT"
set -a; source config/paths.env; set +a

BASE=outputs/benchmarks/run-a-late-guide-pilot/casp14
IDS=outputs/diagnostics/late-guide-casp14-pilot/target_ids.txt
test -s "$BASE/rollout/rollout.json"
test -s "$IDS"
test -x "$OPENSTRUCTURE_OST"
test ! -e "$BASE/pairs-admitted"

"$MFCLEAN_PYTHON" benchmarks/stage_rollout_references.py \
  --rollout-dir "$BASE/rollout" \
  --out-dir "$BASE/pairs-admitted" \
  --target-ids "$IDS" \
  --reference-dir "$CASP14_REFERENCE_DIR"

"$MFCLEAN_PYTHON" benchmarks/score_openstructure.py \
  --in-dir "$BASE/pairs-admitted" \
  --out-dir "$BASE/scores-admitted" \
  --ost "$OPENSTRUCTURE_OST" \
  --expected 4
