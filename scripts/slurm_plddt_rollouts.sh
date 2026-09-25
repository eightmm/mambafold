#!/usr/bin/env bash
#SBATCH --job-name=mf-plddt-rollout
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --qos=long
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm/mf-plddt-rollout-%j.out
#SBATCH --error=logs/slurm/mf-plddt-rollout-%j.out

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
NPROC="${SLURM_GPUS_ON_NODE:-8}"
N_STEPS="${MF_PLDDT_STEPS:-500}"

if [ ! -f "$CHECKPOINT" ]; then
  echo "missing checkpoint: $CHECKPOINT" >&2
  exit 1
fi
if [ -e "$TRAIN_OUT" ] || [ -e "$VAL_OUT" ]; then
  echo "refusing to overwrite rollout directories under $ROOT_OUT" >&2
  exit 1
fi
mkdir -p logs/slurm "$ROOT_OUT"
sha256sum "$CHECKPOINT" > "$HASH_FILE"

COMMON=(
  --config "$CONFIG"
  --checkpoint "$CHECKPOINT"
  --checkpoint-sha256-file "$HASH_FILE"
  --expected-checkpoint-step 10000
  --n-steps "$N_STEPS"
  --n-seeds 2
  --seed-batch-size 2
)

"$MFCLEAN_PYTHON" -m torch.distributed.run \
  --standalone --nproc_per_node="$NPROC" \
  scripts/generate_plddt_rollouts.py "${COMMON[@]}" \
  --file-list data/splits/plddt_pilot_train.txt \
  --out-dir "$TRAIN_OUT"

"$MFCLEAN_PYTHON" -m torch.distributed.run \
  --standalone --nproc_per_node="$NPROC" \
  scripts/generate_plddt_rollouts.py "${COMMON[@]}" \
  --file-list data/splits/plddt_pilot_val.txt \
  --out-dir "$VAL_OUT"
