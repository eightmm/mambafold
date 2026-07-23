#!/usr/bin/env bash
#SBATCH --job-name=mf-casp14-70-esm
#SBATCH --partition=6000ada
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/esm_casp14_70/%x-%j.out
#SBATCH --error=outputs/esm_casp14_70/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

python_bin="/home/jaemin/miniforge3/envs/folding-gpu2/bin/python"
data_dir="data/casp_official/npz_70"
file_list="data/casp_official/casp14_70_npz_files.txt"
out_dir="data/casp_official/esm_70"

mkdir -p "$out_dir" outputs/esm_casp14_70 .cache/tmp
test -x "$python_bin"
test -s "$file_list"
test "$(wc -l < "$file_list")" -eq 70

export TMPDIR="$SLURM_SUBMIT_DIR/.cache/tmp"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

"$python_bin" -u scripts/precompute_esm.py \
    --data_dir "$data_dir" \
    --file_list "$file_list" \
    --out_dir "$out_dir" \
    --esm_model esm3-open \
    --device cuda \
    --dtype float16 \
    --max_length 1024

test "$(find "$out_dir" -maxdepth 1 -type f -name '*_ch0.npy' | wc -l)" -eq 70
