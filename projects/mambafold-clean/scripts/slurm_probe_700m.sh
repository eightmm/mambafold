#!/usr/bin/env bash
#SBATCH --job-name=mf-probe-700m
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --qos=veryshort
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/mf-probe-700m-%j.out
#SBATCH --error=logs/slurm/mf-probe-700m-%j.out

set -euo pipefail
ROOT="${MFCLEAN_ROOT:-${SLURM_SUBMIT_DIR:?}}"
cd "$ROOT"
set -a; source config/paths.env; set +a
mkdir -p logs/slurm data/audit/gpu_probe
export PYTHONUNBUFFERED=1

# Probe the real forward, exact all-atom lDDT backward, and AdamW step.  Batch
# here is the number of noise copies resident in one micro-step.  Testing
# 1/2/4 at the three representative length buckets determines whether the
# 700M run can retain 4 copies x 4 accumulation steps, i.e. the same 16-noise
# training contract as Run A, without risking a full eight-GPU allocation.
"$MFCLEAN_PYTHON" benchmarks/probe_train_memory.py \
  --config configs/run_a_mamba.yaml \
  --override d_res=1152,n_trunk=22,d_plm_proj=1152 \
  --crops 256,512,1024 \
  --batches 1,2,4 \
  --steps 3 \
  --loss_mode lddt_exact \
  --out data/audit/gpu_probe/train_memory_700m.json
