#!/bin/bash
# Job script — partition/gres는 submit_overfit.sh 또는 sbatch CLI로 주입
# 직접 실행 시: sbatch --partition=heavy --gres=gpu:h100:1 scripts/slurm/overfit_test.sh
#SBATCH --job-name=mambafold-overfit
#SBATCH --output=/home/jaemin/project/protein/folding/outputs/overfit/%j/slurm.out

set -e

cd /home/jaemin/project/protein/folding
mkdir -p outputs/overfit/${SLURM_JOB_ID}

VENV_PY=.venv/bin/python
module load cuda/12.8

echo "=== Environment ==="
$VENV_PY -c 'import torch; print(f"torch={torch.__version__}, cuda={torch.cuda.is_available()}")' || true

echo "=== Overfit + Viz ==="
PYTHONPATH=src PYTHONUNBUFFERED=1 $VENV_PY -u scripts/overfit.py \
    --config configs/overfit_test.yaml \
    --out_dir outputs/overfit/${SLURM_JOB_ID} \
    || true

echo "=== Done ==="
