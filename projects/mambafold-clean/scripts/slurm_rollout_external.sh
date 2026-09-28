#!/usr/bin/env bash
#SBATCH --job-name=mf-rollout
#SBATCH --partition=test
#SBATCH --qos=short
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/mf-rollout-%j.out
#SBATCH --error=logs/slurm/mf-rollout-%j.out

set -euo pipefail
ROOT="${MFCLEAN_ROOT:-${SLURM_SUBMIT_DIR:?}}"
cd "$ROOT"
set -a; source config/paths.env; set +a
mkdir -p logs/slurm outputs/benchmarks
export PYTHONUNBUFFERED=1

DATASET="${MF_BENCHMARK_SET:?set MF_BENCHMARK_SET to casp14, casp15, casp16, or cameo22}"
CHECKPOINT="${MF_BENCHMARK_CHECKPOINT:-out/run-a-mamba3-1024-atom14/ckpt_0300000.pt}"
N_STEPS="${MF_BENCHMARK_STEPS:-500}"
SDE_TAU="${MF_BENCHMARK_SDE_TAU:-0.01}"
SEED="${MF_BENCHMARK_SEED:-42}"
LIMIT="${MF_BENCHMARK_LIMIT:-0}"
TARGET_IDS_ARGS=()
if [[ -n "${MF_BENCHMARK_TARGET_IDS:-}" ]]; then
  test -s "$MF_BENCHMARK_TARGET_IDS"
  TARGET_IDS_ARGS=(--target_ids "$MF_BENCHMARK_TARGET_IDS")
fi
PLDDT_CHECKPOINT="${MF_PLDDT_CHECKPOINT:-}"
OUT_ROOT="${MF_BENCHMARK_OUT_ROOT:-outputs/benchmarks/run-a-final}"
GUIDE_MAX_STEP_A="${MF_GEOMETRY_GUIDE_MAX_STEP_A:-0}"
GUIDE_ARGS=()
if [[ "$GUIDE_MAX_STEP_A" != "0" ]]; then
  OST_ROOT="$(dirname "$(dirname "$OPENSTRUCTURE_OST")")"
  GUIDE_ARGS=(
    --geometry-guide-props "$OST_ROOT/share/openstructure/stereo_chemical_props.txt"
    --geometry-guide-start "${MF_GEOMETRY_GUIDE_START:-0.90}"
    --geometry-guide-every "${MF_GEOMETRY_GUIDE_EVERY:-10}"
    --geometry-guide-max-step-A "$GUIDE_MAX_STEP_A"
    --geometry-guide-bond-weight "${MF_GEOMETRY_GUIDE_BOND_WEIGHT:-1.0}"
    --geometry-guide-angle-weight "${MF_GEOMETRY_GUIDE_ANGLE_WEIGHT:-1.0}"
    --geometry-guide-clash-weight "${MF_GEOMETRY_GUIDE_CLASH_WEIGHT:-1.0}"
    --geometry-guide-backbone-scale "${MF_GEOMETRY_GUIDE_BACKBONE_SCALE:-0.1}"
  )
  test -s "${GUIDE_ARGS[1]}"
fi
case "$DATASET" in
  casp14) FASTA="benchmarks/external_testsets/casp14_70.fasta" ;;
  casp15) FASTA="benchmarks/external_testsets/casp15_single_chain_22.fasta" ;;
  casp16) FASTA="benchmarks/external_testsets/casp16_single_chain_21.fasta" ;;
  cameo22) FASTA="benchmarks/external_testsets/cameo22_183.fasta" ;;
  *) echo "unsupported benchmark: $DATASET" >&2; exit 2 ;;
esac

OUT="${OUT_ROOT}/${DATASET}/rollout"
test -s "$CHECKPOINT"
test -s "$FASTA"
PLDDT_ARGS=()
if [ -n "$PLDDT_CHECKPOINT" ]; then
  test -s "$PLDDT_CHECKPOINT"
  PLDDT_ARGS=(--plddt_checkpoint "$PLDDT_CHECKPOINT")
fi
test ! -e "$OUT"

echo "node=$(hostname) job=${SLURM_JOB_ID:-none} dataset=$DATASET"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

exec "$MFCLEAN_PYTHON" scripts/rollout.py \
  --config configs/run_a_mamba.yaml \
  --checkpoint "$CHECKPOINT" \
  "${PLDDT_ARGS[@]}" \
  --fasta "$FASTA" \
  --esm_dir data/casp_esmc6b \
  --out_dir "$OUT" \
  --method sde \
  --n_steps "$N_STEPS" \
  --seed "$SEED" \
  --sde_tau "$SDE_TAU" \
  --sde_eps 0.01 \
  --sde_w_cutoff 0.99 \
  --sde_log_timesteps \
  "${GUIDE_ARGS[@]}" \
  --max_batch_residues 2048 \
  --limit "$LIMIT" \
  "${TARGET_IDS_ARGS[@]}" \
  --use_ema
