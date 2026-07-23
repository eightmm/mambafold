#!/usr/bin/env bash
#SBATCH --job-name=mf-esmc6b-smoke
#SBATCH --partition=6000ada
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
out="outputs/smoke/esmc6b_${SLURM_JOB_ID}"
mkdir -p "$out"

uv run --no-sync python -u scripts/precompute_esm.py \
    --data_dir data/casp_official/npz_70 \
    --file_list data/casp_official/casp14_70_npz_files.txt \
    --out_dir "$out" \
    --esm_model esmc-6b \
    --device cuda \
    --dtype float16 \
    --max_length 1024 \
    --limit 1 \
    --fail_on_error

uv run --no-sync python - "$out" <<'PY'
import sys
from pathlib import Path

import numpy as np

files = sorted(Path(sys.argv[1]).glob("*_ch0.npy"))
if len(files) != 1:
    raise SystemExit(f"Expected one smoke embedding, found {len(files)}")
array = np.load(files[0], mmap_mode="r")
if array.ndim != 2 or array.shape[1] != 2560 or array.dtype != np.float16:
    raise SystemExit(f"Unexpected smoke embedding: shape={array.shape} dtype={array.dtype}")
print(f"smoke_embedding={files[0]} shape={array.shape} dtype={array.dtype}")
PY
