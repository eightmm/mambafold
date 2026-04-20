#!/bin/bash
#SBATCH --job-name=mf-esm
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=30-00:00:00
#SBATCH --output=/home/jaemin/project/protein/folding/outputs/esm/%j/slurm.out

set -e
cd /home/jaemin/project/protein/folding
mkdir -p outputs/esm/${SLURM_JOB_ID}

VENV_PY=.venv/bin/python
module load cuda/12.8
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=== ESM Pre-computation ==="
$VENV_PY -c "import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}')"

DATA_DIR=${DATA_DIR:-data/rcsb}
OUT_DIR=${OUT_DIR:-data/rcsb_esm}
ESM_MODEL=${ESM_MODEL:-esm3-open}

echo "data_dir : ${DATA_DIR}"
echo "out_dir  : ${OUT_DIR}"
echo "esm_model: ${ESM_MODEL}"
echo "file_list: ${FILE_LIST:-all}"

mkdir -p ${OUT_DIR}

PYTHONPATH=src PYTHONUNBUFFERED=1 $VENV_PY -u scripts/precompute_esm.py \
    --data_dir ${DATA_DIR} \
    --out_dir  ${OUT_DIR} \
    --esm_model ${ESM_MODEL} \
    --device cuda \
    --skip_existing \
    ${FILE_LIST:+--file_list $FILE_LIST} \
    "$@"

echo "=== Done ==="
