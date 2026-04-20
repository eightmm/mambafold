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

echo "=== Training (single H100 80GB) ==="
# Config 선택: CONFIG 환경변수 우선, 없으면 기본 pretrain config
CONFIG="${CONFIG:-configs/train_base.yaml}"
# batch_size는 yaml에서 결정 (pretrain=8, finetune_512=4, finetune_1024=2)
echo "Config : $CONFIG"
echo "Resume : ${RESUME:-none}"

PYTHONPATH=src PYTHONUNBUFFERED=1 $VENV_PY -u scripts/train.py \
    --config "$CONFIG" \
    --out_dir outputs/train/${SLURM_JOB_ID} \
    "$@" \
    ${RESUME:+--resume $RESUME}

echo "=== Done ==="
