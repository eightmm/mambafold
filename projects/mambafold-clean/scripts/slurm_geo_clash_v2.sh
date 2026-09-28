#!/usr/bin/env bash
#SBATCH --job-name=mf-geo-clash-v2
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --qos=long
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/slurm/mf-geo-clash-v2-%j.out
#SBATCH --error=logs/slurm/mf-geo-clash-v2-%j.out

# Continue from geo-ft-v1's EMA with the corrected OpenStructure-aligned clash
# objective. This is a new run because the v1 clash definition excluded 1-3
# and 1-4 pairs, ignored unresolved canonical output atoms, and used the
# hydrogen-aware MolProbity tolerance with a heavy-atom-only model.
# Four ranks with grad_accum=4 preserve v1's effective 128 noise copies per
# optimizer step (4 ranks x 8 copies x 4) while fitting the live cluster queue.

set -euo pipefail
ROOT="${MFCLEAN_ROOT:-${SLURM_SUBMIT_DIR:?}}"
cd "$ROOT"

CHECKPOINT="${MF_GEO_CHECKPOINT:-out/run-a-geo-ft-v1/ckpt_0010000.pt}"
OUT_DIR="${MF_GEO_OUT_DIR:-out/run-a-geo-clash-v2}"

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
  --expected_resume_step 10000 \
  --start_step 0 \
  --total_steps 10000 \
  --grad_accum_steps 4 \
  --lr 5.0e-6 \
  --min_lr 5.0e-7 \
  --warmup_steps 500 \
  --lr_cooldown_steps 5000 \
  --t_schedule uniform \
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
  --wandb_name run-a-geo-clash-v2
