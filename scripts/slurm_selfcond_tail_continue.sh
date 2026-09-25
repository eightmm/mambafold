#!/usr/bin/env bash
#SBATCH --job-name=mf-sc-tail-v1
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --qos=long
#SBATCH --time=3-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/slurm/mf-sc-tail-v1-%j.out
#SBATCH --error=logs/slurm/mf-sc-tail-v1-%j.out

# Continue from the final pretraining EMA while introducing self-conditioning
# and restoring SimpleFold's less tail-suppressing timestep interpolation.
# Geometry stays off in this phase; it is reapplied only after the folding
# model has adapted to the new information path and noise distribution.

set -euo pipefail
ROOT="${MFCLEAN_ROOT:-${SLURM_SUBMIT_DIR:?}}"
cd "$ROOT"

CHECKPOINT="${MF_SC_CHECKPOINT:-out/run-a-mamba3-1024-atom14/ckpt_0300000.pt}"
OUT_DIR="${MF_SC_OUT_DIR:-out/run-a-selfcond-tail-v1}"

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
  --expected_missing_resume_keys self_cond_proj.weight \
  --expected_resume_step 300000 \
  --start_step 0 \
  --total_steps 30000 \
  --lr 1.0e-5 \
  --min_lr 1.0e-6 \
  --warmup_steps 1000 \
  --lr_cooldown_steps 15000 \
  --t_schedule logit_normal \
  --t_uniform_weight 0.02 \
  --self_conditioning \
  --self_condition_prob 0.5 \
  --alpha_mode const \
  --w_fm 1.0 \
  --w_lddt_atom 1.0 \
  --w_bond 0.0 \
  --w_angle 0.0 \
  --w_clash 0.0 \
  --ckpt_interval 1000 \
  --keep_last_checkpoints 3 \
  --keep_checkpoint_steps 10000 20000 30000 \
  --wandb_name run-a-selfcond-tail-v1
