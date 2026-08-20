#!/usr/bin/env bash
#SBATCH --job-name=mf-casp14-esmc6b
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/esmc_casp14_70/%x-%j.out
#SBATCH --error=outputs/esmc_casp14_70/%x-%j.err

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python_bin="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
data_dir="${DATA_DIR:-data/casp_official/npz_70}"
file_list="${FILE_LIST:-data/casp_official/casp14_70_npz_files.txt}"
out_dir="${OUT_DIR:-data/casp_official/esmc6b_70}"
revision="45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a"
hf_cache="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"

mkdir -p "$out_dir" outputs/esmc_casp14_70 .cache/tmp
test -x "$python_bin"
test -s "$file_list"
test "$(wc -l < "$file_list")" -eq 70
export ESMC_6B_MODEL_DIR="${ESMC_6B_MODEL_DIR:-$hf_cache/models--biohub--ESMC-6B/snapshots/$revision}"
test -s "$ESMC_6B_MODEL_DIR/model.safetensors.index.json"

export TMPDIR="${TMPDIR:-$ROOT/.cache/tmp}"
export PYTHONPATH="${PYTHONPATH:-$ROOT/src}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

"$python_bin" -u scripts/precompute_esm.py \
    --data_dir "$data_dir" \
    --file_list "$file_list" \
    --out_dir "$out_dir" \
    --esm_model esmc-6b \
    --device cuda \
    --dtype float16 \
    --max_length 1024 \
    --cache_layout sequence \
    --fail_on_error

test "$(find "$out_dir/by_sequence" -type f -name '*.npy' | wc -l)" -eq 70
