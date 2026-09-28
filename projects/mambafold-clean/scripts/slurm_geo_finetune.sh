#!/usr/bin/env bash
#SBATCH --job-name=mf-geo-ft
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --qos=long
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/slurm/mf-geo-ft-%j.out
#SBATCH --error=logs/slurm/mf-geo-ft-%j.out

# One declared geometric fine-tune from the final Run-A EMA.  The base config
# pins the architecture and data contract; every phase-specific override is
# explicit below and is persisted by train.py in the output config.json.

set -euo pipefail
ROOT="${MFCLEAN_ROOT:-${SLURM_SUBMIT_DIR:?}}"
cd "$ROOT"

CHECKPOINT="${MF_GEO_CHECKPOINT:-out/run-a-mamba3-1024-atom14/ckpt_0300000.pt}"
OUT_DIR="${MF_GEO_OUT_DIR:-out/run-a-geo-ft-v1}"

if [ ! -f "$CHECKPOINT" ]; then
  echo "missing checkpoint: $CHECKPOINT" >&2
  exit 1
fi
if [ -e "$OUT_DIR/config.json" ]; then
  echo "refusing to overwrite an existing run: $OUT_DIR" >&2
  exit 1
fi

export MF_CONFIG=configs/run_a_mamba.yaml
export MF_OUT_DIR="$OUT_DIR"

exec bash scripts/slurm_train.sh \
  --resume "$CHECKPOINT" \
  --reset_optimizer \
  --initialize_model_from_ema \
  --strict_resume \
  --expected_resume_step 300000 \
  --start_step 0 \
  --total_steps 10000 \
  --lr 1.0e-5 \
  --min_lr 1.0e-6 \
  --warmup_steps 500 \
  --lr_cooldown_steps 5000 \
  --t_schedule uniform \
  --alpha_mode ramp \
  --w_fm 1.0 \
  --w_lddt_atom 1.0 \
  --w_bond 1.0 \
  --w_angle 1.0 \
  --w_clash 1.0 \
  --clash_overlap_tolerance_A 0.4 \
  --clash_pair_chunk_size 256 \
  --ckpt_interval 1000 \
  --keep_last_checkpoints 3 \
  --wandb_name run-a-geo-ft-v1
