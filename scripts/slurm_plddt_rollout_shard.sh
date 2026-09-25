#!/usr/bin/env bash
#SBATCH --job-name=mf-plddt500-shard
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --qos=long
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm/mf-plddt500-shard-%A_%a.out
#SBATCH --error=logs/slurm/mf-plddt500-shard-%A_%a.out

set -euo pipefail
ROOT="${MFCLEAN_ROOT:-${SLURM_SUBMIT_DIR:?}}"
cd "$ROOT"
set -a; source config/paths.env; set +a

CONFIG="${MF_PLDDT_CONFIG:-configs/plddt_selfcond_sde500.yaml}"
CHECKPOINT="${MF_PLDDT_CHECKPOINT:-out/run-a-selfcond-geo-v1/ckpt_0010000.pt}"
ROOT_OUT="${MF_PLDDT_OUT_ROOT:-out/plddt-selfcond-sde500}"
TRAIN_OUT="$ROOT_OUT/rollouts/train"
VAL_OUT="$ROOT_OUT/rollouts/val"
HASH_FILE="$ROOT_OUT/folding-checkpoint.sha256"
RANK="${SLURM_ARRAY_TASK_ID:?submit as a zero-based array}"
WORLD_SIZE="${MF_PLDDT_WORLD_SIZE:-8}"
N_STEPS="${MF_PLDDT_STEPS:-500}"
N_SEEDS="${MF_PLDDT_N_SEEDS:-2}"
SEED_BATCH_SIZE="${MF_PLDDT_SEED_BATCH_SIZE:-2}"
TRAIN_FILE_LIST="${MF_PLDDT_TRAIN_FILE_LIST:-data/splits/plddt_pilot_train.txt}"
VAL_FILE_LIST="${MF_PLDDT_VAL_FILE_LIST:-data/splits/plddt_pilot_val.txt}"
TARGET_MODE="${MF_PLDDT_TARGET_MODE:-file}"

case "$TARGET_MODE" in
  file) TARGET_FLAG="--file-list" ;;
  chain) TARGET_FLAG="--chain-list" ;;
  *) echo "MF_PLDDT_TARGET_MODE must be file or chain, found: $TARGET_MODE" >&2; exit 2 ;;
esac

if [ "$RANK" -ge "$WORLD_SIZE" ]; then
  echo "array rank $RANK is outside world size $WORLD_SIZE" >&2
  exit 2
fi
test -s "$CHECKPOINT"
test -s "$HASH_FILE"
mkdir -p logs/slurm "$TRAIN_OUT" "$VAL_OUT"

COMMON=(
  --config "$CONFIG"
  --checkpoint "$CHECKPOINT"
  --checkpoint-sha256-file "$HASH_FILE"
  --expected-checkpoint-step 10000
  --n-steps "$N_STEPS"
  --n-seeds "$N_SEEDS"
  --seed-batch-size "$SEED_BATCH_SIZE"
  --rank "$RANK"
  --world-size "$WORLD_SIZE"
)

"$MFCLEAN_PYTHON" scripts/generate_plddt_rollouts.py "${COMMON[@]}" \
  "$TARGET_FLAG" "$TRAIN_FILE_LIST" \
  --out-dir "$TRAIN_OUT"

"$MFCLEAN_PYTHON" scripts/generate_plddt_rollouts.py "${COMMON[@]}" \
  "$TARGET_FLAG" "$VAL_FILE_LIST" \
  --out-dir "$VAL_OUT"
