#!/bin/bash
#SBATCH --job-name=mf-train
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=30-00:00:00
#SBATCH --output=/home/jaemin/project/protein/folding/outputs/train/%j/slurm.out

set -e
cd /home/jaemin/project/protein/folding
mkdir -p outputs/train/${SLURM_JOB_ID}

VENV_PY=.venv/bin/python
N_GPU=${SLURM_GPUS_ON_NODE:-1}
module load cuda/12.8

echo "=== Environment (${N_GPU} GPU) ==="
$VENV_PY -c "import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}, n_gpu={torch.cuda.device_count()}')"

echo "=== Training ==="
if [ "${N_GPU}" -gt 1 ]; then
    PYTHONPATH=src PYTHONUNBUFFERED=1 .venv/bin/torchrun \
        --nproc_per_node=${N_GPU} \
        --master_port=29500 \
        scripts/train.py \
        --config configs/train_base.yaml \
        --out_dir outputs/train/${SLURM_JOB_ID} \
        ${RESUME:+--resume $RESUME}
else
    PYTHONPATH=src PYTHONUNBUFFERED=1 $VENV_PY -u scripts/train.py \
        --config configs/train_base.yaml \
        --out_dir outputs/train/${SLURM_JOB_ID} \
        ${RESUME:+--resume $RESUME}
fi

echo "=== Done ==="
