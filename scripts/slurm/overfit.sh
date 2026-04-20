#!/bin/bash
# MambaFold overfit 스크립트
#
# 사용법:
#   sbatch --gres=gpu:a5000:1 scripts/slurm/overfit.sh          # test 파티션 A5000
#   sbatch --partition=heavy --gres=gpu:h100:1 scripts/slurm/overfit.sh
#   sbatch --partition=6000ada --gres=gpu:1 scripts/slurm/overfit.sh
#
#SBATCH --job-name=mf-overfit
#SBATCH --partition=test
#SBATCH --gres=gpu:a5000:1
#SBATCH --time=4:00:00
#SBATCH --output=/home/jaemin/project/protein/folding/outputs/overfit/%j/slurm.out

set -e
cd /home/jaemin/project/protein/folding
mkdir -p outputs/overfit/${SLURM_JOB_ID}

VENV_PY=.venv/bin/python
module load cuda/12.8

echo "=== Environment ==="
$VENV_PY -c "import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}, gpu={torch.cuda.get_device_name(0)}')"

echo "=== Overfit ==="
PYTHONPATH=src PYTHONUNBUFFERED=1 $VENV_PY -u scripts/overfit.py \
    --config configs/overfit.yaml \
    --out_dir outputs/overfit/${SLURM_JOB_ID} \
    "$@"

echo "=== Done ==="
