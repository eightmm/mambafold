#!/bin/bash
#SBATCH --job-name=mambafold-train
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=30-00:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -e

cd /home/jaemin/project/protein/folding
mkdir -p logs checkpoints

VENV_PY=.venv/bin/python
module load cuda/12.8

echo "=== Environment ==="
$VENV_PY -c "import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}, gpu={torch.cuda.get_device_name(0)}')"

# --- Configure these ---
DATA_DIR=${DATA_DIR:-"/home/jaemin/data/afdb_pt"}
RUN_NAME=${RUN_NAME:-"mambafold-run-${SLURM_JOB_ID}"}
CKPT_DIR=${CKPT_DIR:-"checkpoints/${SLURM_JOB_ID}"}
RESUME=${RESUME:-""}

# --- Model size (small by default; scale up for production) ---
D_ATOM=${D_ATOM:-128}
D_RES=${D_RES:-256}
N_ATOM_ENC=${N_ATOM_ENC:-2}
N_TRUNK=${N_TRUNK:-8}
N_ATOM_DEC=${N_ATOM_DEC:-2}
ATOM_D_STATE=${ATOM_D_STATE:-16}
ATOM_MIMO_RANK=${ATOM_MIMO_RANK:-2}
ATOM_HEADDIM=${ATOM_HEADDIM:-32}
D_STATE=${D_STATE:-32}
MIMO_RANK=${MIMO_RANK:-4}
HEADDIM=${HEADDIM:-32}

echo "=== Training: ${RUN_NAME} ==="
echo "DATA_DIR=${DATA_DIR}"
echo "CKPT_DIR=${CKPT_DIR}"

RESUME_ARG=""
if [ -n "$RESUME" ] && [ -f "$RESUME" ]; then
    RESUME_ARG="--resume ${RESUME}"
    echo "Resuming from: ${RESUME}"
fi

PYTHONPATH=src $VENV_PY -m mambafold.train.trainer \
    --data_dir      "${DATA_DIR}" \
    --ckpt_dir      "${CKPT_DIR}" \
    --wandb_project "mambafold" \
    --run_name      "${RUN_NAME}" \
    --d_atom        ${D_ATOM} \
    --d_res         ${D_RES} \
    --n_atom_enc    ${N_ATOM_ENC} \
    --n_trunk       ${N_TRUNK} \
    --n_atom_dec    ${N_ATOM_DEC} \
    --atom_d_state  ${ATOM_D_STATE} \
    --atom_mimo_rank ${ATOM_MIMO_RANK} \
    --atom_headdim  ${ATOM_HEADDIM} \
    --d_state       ${D_STATE} \
    --mimo_rank     ${MIMO_RANK} \
    --headdim       ${HEADDIM} \
    --batch_size    4 \
    --copies_per_protein 2 \
    --max_length    256 \
    --num_workers   4 \
    --lr            1e-4 \
    --weight_decay  1e-2 \
    --total_steps   500000 \
    --warmup_steps  2000 \
    --finetune_start 250000 \
    --grad_clip     1.0 \
    --ema_decay     0.999 \
    --log_interval  50 \
    --eval_interval 500 \
    --ckpt_interval 2000 \
    ${RESUME_ARG}

echo "=== Done ==="
