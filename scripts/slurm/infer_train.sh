#!/bin/bash
#SBATCH --job-name=mf-infer-train
#SBATCH --partition=test
#SBATCH --gres=gpu:a5000:1
#SBATCH --time=01:00:00
#SBATCH --output=outputs/infer_train/%j.out

set -e
cd /home/jaemin/project/protein/folding

CKPT="${CKPT:-outputs/train/26367/ckpt_latest.pt}"
PDB_IDS="${PDB_IDS:-2olz,1eid,7l7s,8ki5,7n4x,7uvm,5lfl,1h6b}"
OUT="${OUT:-outputs/infer_train/$(basename $(dirname $CKPT))_$(basename $CKPT .pt)}"
mkdir -p "$OUT"

echo "=== Train-set inference sanity check ==="
echo "CKPT    : $CKPT"
echo "PDB IDs : $PDB_IDS"
echo "OUT     : $OUT"
echo "Node    : $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "========================================"

PYTHONPATH=src .venv/bin/python -u scripts/infer_train.py \
    --ckpt "$CKPT" \
    --data_dir data/rcsb \
    --esm_dir data/rcsb_esm \
    --pdb_ids "$PDB_IDS" \
    --out "$OUT" \
    --n_seeds 3 \
    --n_steps 50

echo "=== Done ==="
