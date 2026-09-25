#!/usr/bin/env bash
#SBATCH --job-name=mf-sc-geo-v1
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --qos=long
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/slurm/mf-sc-geo-v1-%j.out
#SBATCH --error=logs/slurm/mf-sc-geo-v1-%j.out

# Reapply the corrected geometry objective after the folding model has adapted
# to self-conditioning. Keeping self-conditioning active here matches the
# sampler path that the downstream pLDDT rollouts will observe.

set -euo pipefail
ROOT="${MFCLEAN_ROOT:-${SLURM_SUBMIT_DIR:?}}"
cd "$ROOT"

CHECKPOINT="${MF_SC_GEO_CHECKPOINT:-out/run-a-selfcond-tail-v1/ckpt_0030000.pt}"
OUT_DIR="${MF_SC_GEO_OUT_DIR:-out/run-a-selfcond-geo-v1}"

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
  --expected_resume_step 30000 \
  --start_step 0 \
  --total_steps 10000 \
  --grad_accum_steps 4 \
  --lr 5.0e-6 \
  --min_lr 5.0e-7 \
  --warmup_steps 500 \
  --lr_cooldown_steps 5000 \
  --t_schedule uniform \
  --t_uniform_weight 0.02 \
  --self_conditioning \
  --self_condition_prob 0.5 \
  --alpha_mode ramp \
  --w_fm 1.0 \
  --w_lddt_atom 1.0 \
  --w_bond 1.0 \
  --w_angle 1.0 \
  --w_clash 1.0 \
  --clash_overlap_tolerance_A 1.5 \
  --clash_margin_A 0.1 \
  --clash_huber_delta_A 0.25 \
  --clash_soft_count_tau_A 0.05 \
  --clash_pair_chunk_size 256 \
  --ckpt_interval 1000 \
  --keep_last_checkpoints 3 \
  --keep_checkpoint_steps 10000 \
  --wandb_name run-a-selfcond-geo-v1
