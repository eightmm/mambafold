#!/bin/bash
#SBATCH --job-name=mambafold-overfit
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/overfit_%j.out

set -e

cd /home/jaemin/project/protein/folding
mkdir -p logs

VENV_PY=.venv/bin/python
module load cuda/12.8

echo "=== Environment ==="
$VENV_PY -c "import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}, gpu={torch.cuda.get_device_name(0)}')"

echo "=== Overfit Test (synthetic) ==="
PYTHONPATH=src $VENV_PY scripts/overfit_test.py \
    --n_steps 300 \
    --lr 3e-3

echo "=== Done ==="
