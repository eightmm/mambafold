#!/usr/bin/env bash
#SBATCH --job-name=mf-guided-score
#SBATCH --partition=cpu_only
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/mf-guided-score-%j.out
#SBATCH --error=logs/slurm/mf-guided-score-%j.out

set -euo pipefail
ROOT="${MFCLEAN_ROOT:-${SLURM_SUBMIT_DIR:?}}"
cd "$ROOT"
set -a; source config/paths.env; set +a

DATASET="${MF_BENCHMARK_SET:?dataset required}"
IDS="${MF_BENCHMARK_TARGET_IDS:?target ID file required}"
OUT_ROOT="${MF_BENCHMARK_OUT_ROOT:?output root required}"
case "$DATASET" in
  casp14|casp15|casp16) ;;
  *) echo "unsupported guided score dataset: $DATASET" >&2; exit 2 ;;
esac
BASE="$OUT_ROOT/$DATASET"
TEMPLATE="outputs/benchmarks/run-a-final/$DATASET/pairs-admitted"
test -s "$BASE/rollout/rollout.json"
test -s "$IDS"
test -s "$TEMPLATE/manifest.json"
test -x "$OPENSTRUCTURE_OST"
test ! -e "$BASE/pairs-admitted"

"$MFCLEAN_PYTHON" benchmarks/stage_guided_pairs.py \
  --rollout-dir "$BASE/rollout" \
  --template-pairs "$TEMPLATE" \
  --target-ids "$IDS" \
  --out-dir "$BASE/pairs-admitted"

EXPECTED="$("$MFCLEAN_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["reference_pair_count"])' "$BASE/pairs-admitted/manifest.json")"
"$MFCLEAN_PYTHON" benchmarks/score_openstructure.py \
  --in-dir "$BASE/pairs-admitted" \
  --out-dir "$BASE/scores-admitted" \
  --ost "$OPENSTRUCTURE_OST" \
  --expected "$EXPECTED"

if [[ "$DATASET" == casp15 ]]; then
  "$MFCLEAN_PYTHON" benchmarks/aggregate_casp_domains.py \
    --manifest "$BASE/pairs-admitted/manifest.json" \
    --pair-summary "$BASE/scores-admitted/summary.json" \
    --out "$BASE/scores-admitted/target-summary.json"
fi
