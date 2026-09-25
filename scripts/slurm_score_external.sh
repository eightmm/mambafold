#!/usr/bin/env bash
#SBATCH --job-name=mf-score
#SBATCH --partition=cpu_only
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/mf-score-%j.out
#SBATCH --error=logs/slurm/mf-score-%j.out

set -euo pipefail
ROOT="${MFCLEAN_ROOT:-${SLURM_SUBMIT_DIR:?}}"
cd "$ROOT"
set -a; source config/paths.env; set +a
mkdir -p logs/slurm
export PYTHONUNBUFFERED=1

DATASET="${MF_BENCHMARK_SET:?set MF_BENCHMARK_SET to casp14, casp15, casp16, or cameo22}"
RUN="${MF_BENCHMARK_RUN:-run-a-final}"
BASE="outputs/benchmarks/${RUN}/${DATASET}"
ROLLOUT="$BASE/rollout"
case "$DATASET" in
  casp14)
    EXPECTED=70
    ADMITTED="data/audit/casp14_70-admitted.ids"
    REFERENCE_ARGS=(--reference-dir "$CASP14_REFERENCE_DIR")
    ;;
  casp15)
    EXPECTED=22
    ADMITTED="data/audit/casp15_single_chain_22-admitted.ids"
    : "${MF_BENCHMARK_REFERENCE_ROOT:?set MF_BENCHMARK_REFERENCE_ROOT to the official CASP15 reference directory}"
    ;;
  casp16)
    EXPECTED=21
    ADMITTED="data/audit/casp16_single_chain_21-admitted.ids"
    : "${MF_BENCHMARK_REFERENCE_ROOT:?set MF_BENCHMARK_REFERENCE_ROOT to the official CASP16 reference directory}"
    ;;
  cameo22)
    EXPECTED=183
    ADMITTED="data/audit/cameo22_183-admitted.ids"
    REFERENCE_ARGS=(
      --reference-manifest "$CAMEO22_REFERENCE_ROOT/reference_manifest.tsv"
      --reference-state state1
    )
    ;;
  *) echo "unsupported benchmark: $DATASET" >&2; exit 2 ;;
esac

test -s "$ROLLOUT/rollout.json"
test -s "$ADMITTED"
test -x "$OPENSTRUCTURE_OST"
ADMITTED_N="$(wc -l < "$ADMITTED")"

if [[ "$DATASET" == casp15 || "$DATASET" == casp16 ]]; then
  FULL_IDS="$BASE/all-targets.ids"
  "$MFCLEAN_PYTHON" - "$ROLLOUT/rollout.json" "$FULL_IDS" <<'PY'
import json, sys
from pathlib import Path
targets = json.loads(Path(sys.argv[1]).read_text())["per_target"]
Path(sys.argv[2]).write_text("".join(f"{target}\n" for target in targets))
PY
  for SUBSET in full admitted; do
    IDS="$FULL_IDS"
    if [[ "$SUBSET" == admitted ]]; then IDS="$ADMITTED"; fi
    "$MFCLEAN_PYTHON" benchmarks/stage_casp_references.py \
      --dataset "$DATASET" --rollout-dir "$ROLLOUT" \
      --reference-root "$MF_BENCHMARK_REFERENCE_ROOT" \
      --target-ids "$IDS" --out-dir "$BASE/pairs-$SUBSET"
  done
else
  "$MFCLEAN_PYTHON" benchmarks/stage_rollout_references.py \
    --rollout-dir "$ROLLOUT" \
    --out-dir "$BASE/pairs-full" \
    "${REFERENCE_ARGS[@]}"
  "$MFCLEAN_PYTHON" benchmarks/stage_rollout_references.py \
    --rollout-dir "$ROLLOUT" \
    --out-dir "$BASE/pairs-admitted" \
    --target-ids "$ADMITTED" \
    "${REFERENCE_ARGS[@]}"
fi

if [[ "$DATASET" == casp15 ]]; then
  FULL_PAIRS="$("$MFCLEAN_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["reference_pair_count"])' "$BASE/pairs-full/manifest.json")"
  ADMITTED_PAIRS="$("$MFCLEAN_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["reference_pair_count"])' "$BASE/pairs-admitted/manifest.json")"
else
  FULL_PAIRS="$EXPECTED"
  ADMITTED_PAIRS="$ADMITTED_N"
fi

"$MFCLEAN_PYTHON" benchmarks/score_openstructure.py \
  --in-dir "$BASE/pairs-full" \
  --out-dir "$BASE/scores-full" \
  --ost "$OPENSTRUCTURE_OST" \
  --expected "$FULL_PAIRS"
"$MFCLEAN_PYTHON" benchmarks/score_openstructure.py \
  --in-dir "$BASE/pairs-admitted" \
  --out-dir "$BASE/scores-admitted" \
  --ost "$OPENSTRUCTURE_OST" \
  --expected "$ADMITTED_PAIRS"

if [[ "$DATASET" == casp15 ]]; then
  for SUBSET in full admitted; do
    "$MFCLEAN_PYTHON" benchmarks/aggregate_casp_domains.py \
      --manifest "$BASE/pairs-$SUBSET/manifest.json" \
      --pair-summary "$BASE/scores-$SUBSET/summary.json" \
      --out "$BASE/scores-$SUBSET/target-summary.json"
  done
fi
