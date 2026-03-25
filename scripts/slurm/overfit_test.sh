#!/bin/bash
#SBATCH --job-name=mambafold-overfit
#SBATCH --partition=test
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=outputs/overfit/%j/slurm.out

set -e

cd /home/jaemin/project/protein/folding
mkdir -p outputs/overfit/${SLURM_JOB_ID}

VENV_PY=.venv/bin/python
module load cuda/12.8

echo "=== Environment ==="
$VENV_PY -c "import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}, gpu={torch.cuda.get_device_name(0)}')"

echo "=== Overfit + Viz ==="
PYTHONPATH=src PYTHONUNBUFFERED=1 $VENV_PY -u scripts/overfit.py \
    --config configs/overfit_test.yaml \
    --out_dir outputs/overfit/${SLURM_JOB_ID} \
    || true

echo "=== Done ==="
