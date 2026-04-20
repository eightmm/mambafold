#!/bin/bash
#SBATCH --job-name=mf-infer
#SBATCH --partition=test
#SBATCH --gres=gpu:a5000:1
#SBATCH --time=01:00:00
#SBATCH --output=outputs/infer_2025/%x_%j.out

set -e
cd /home/jaemin/project/protein/folding

CKPT="outputs/train/26367/ckpt_latest.pt"
PDB_ID="${1:-10AF}"
SEQ="${2:-MAHHHHHHMSRPHVFFDITIGGSNAGRIVMELFADIVPKTAENFRCLCTGERGMGRSGKKLHYKGSKFHRVIPNFMLQGGDFTRGNGTGGESIYGEKFPDENFQEKHTGPGVLSMANAGPNTNGSQFFICTAKTEWLDGKHVVFGRVVEGMNVVKAVESKGSQSGRTSADIVIADCGQL}"
OUT="outputs/infer_2025/${PDB_ID,,}"

mkdir -p "$OUT"

echo "=== MambaFold 2025 Sequence Inference ==="
echo "Checkpoint : $CKPT"
echo "PDB ID     : $PDB_ID"
echo "Seq length : ${#SEQ}"
echo "Output     : $OUT"
echo "Node       : $(hostname)"
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "========================================="

CUDA_LAUNCH_BLOCKING=1 PYTHONPATH=src .venv/bin/python scripts/infer_seq.py \
    --ckpt "$CKPT" \
    --seq "$SEQ" \
    --pdb_id "$PDB_ID" \
    --out "$OUT" \
    --use_ema \
    --n_steps 50 \
    --n_seeds 3

echo "=== Done ==="
