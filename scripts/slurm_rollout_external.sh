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
PLDDT_CHECKPOINT="${MF_PLDDT_CHECKPOINT:-}"
OUT_ROOT="${MF_BENCHMARK_OUT_ROOT:-outputs/benchmarks/run-a-final}"
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
  --max_batch_residues 2048 \
  --limit "$LIMIT" \
  --use_ema
