#!/usr/bin/env bash
#SBATCH --job-name=mambafold-esm
#SBATCH --partition=6000ada
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

: "${DATA_DIR:?Set DATA_DIR to the Boltz-style NPZ directory}"
: "${OUT_DIR:?Set OUT_DIR to the ESM output directory}"

GPUS="${GPUS:-0,1,2,3}"
mkdir -p outputs/logs "$OUT_DIR" .cache/tmp

export DATA_DIR OUT_DIR GPUS
export TMPDIR="${TMPDIR:-$SLURM_SUBMIT_DIR/.cache/tmp}"
export PYTHONPATH="${PYTHONPATH:-src}"
export TOKENIZERS_PARALLELISM=false

run_data_dir="$DATA_DIR"
if [[ "${STAGE_LOCAL:-0}" == "1" ]]; then
    local_root="${SLURM_TMPDIR:-/tmp/${USER}/mambafold-${SLURM_JOB_ID}}"
    local_data_dir="$local_root/npz"
    mkdir -p "$local_data_dir"
    echo "Staging $DATA_DIR to $local_data_dir"
    cp -RL "$DATA_DIR"/. "$local_data_dir"/
    run_data_dir="$local_data_dir"
fi

DATA_DIR="$run_data_dir" bash scripts/precompute_afdb_swissprot_esm_4gpu.sh
