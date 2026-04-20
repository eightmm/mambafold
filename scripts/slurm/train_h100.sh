#!/bin/bash
#SBATCH --job-name=mf-train-h100
#SBATCH --partition=heavy
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=96G
#SBATCH --time=30-00:00:00
#SBATCH --output=/home/jaemin/project/protein/folding/outputs/train/%j/slurm.out

set -e
cd /home/jaemin/project/protein/folding
mkdir -p outputs/train/${SLURM_JOB_ID}

VENV_PY=.venv/bin/python
module load cuda/12.8

echo "=== Environment (single H100) ==="
$VENV_PY -c "import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}, n_gpu={torch.cuda.device_count()}')"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "=== Training (H100 80GB, batch_size=8) ==="
# H100 80GB: 6000 Ada (48GB) 대비 ~1.7x VRAM → batch 4→8로 증량
# 단일 GPU이므로 DDP/NCCL 이슈 없음
PYTHONPATH=src PYTHONUNBUFFERED=1 $VENV_PY -u scripts/train.py \
    --config configs/train_base.yaml \
    --out_dir outputs/train/${SLURM_JOB_ID} \
    --batch_size 8 \
    "$@" \
    ${RESUME:+--resume $RESUME}

echo "=== Done ==="
