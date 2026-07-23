#!/usr/bin/env bash
#SBATCH --job-name=mambafold-esmc6b
#SBATCH --partition=6000ada
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=160G
#SBATCH --time=3-00:00:00
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

: "${DATA_DIR:?Set DATA_DIR to a Boltz-style NPZ directory}"
: "${OUT_DIR:?Set OUT_DIR to a new ESMC-6B embedding directory}"

revision="45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a"
hf_cache="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
export ESMC_6B_MODEL_DIR="${ESMC_6B_MODEL_DIR:-$hf_cache/models--biohub--ESMC-6B/snapshots/$revision}"
test -s "$ESMC_6B_MODEL_DIR/model.safetensors.index.json"

export ESM_MODEL="esmc-6b"
export GPUS="${GPUS:-0,1,2,3}"
export SHARDS="${SHARDS:-4}"
export MAX_LENGTH="${MAX_LENGTH:-1024}"
export SHARD_FILES=0
export CACHE_LAYOUT=sequence
export FAIL_ON_ERROR=1
export LOG_DIR="${LOG_DIR:-outputs/esmc6b_$(basename "$OUT_DIR")}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

bash scripts/precompute_afdb_swissprot_esm_4gpu.sh
