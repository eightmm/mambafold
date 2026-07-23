#!/usr/bin/env bash
#SBATCH --job-name=mambafold-eval-env
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

conda_bin="/home/jaemin/miniforge3/bin/conda"
source_env="/home/jaemin/miniforge3/envs/folding-gpu2"
target_env="$SLURM_SUBMIT_DIR/tools/mambafold-eval-env"
mamba_source="$SLURM_SUBMIT_DIR/tools/mamba-src"
mamba_commit="316ed6036538405f767782132f76caf342256d33"

mkdir -p outputs/logs tools
test -x "$conda_bin"
test -x "$source_env/bin/python"
export CUDA_HOME="/usr/local/cuda"
export PATH="$CUDA_HOME/bin:$PATH"
test -x "$CUDA_HOME/bin/nvcc"

if [[ ! -x "$target_env/bin/python" ]]; then
    "$conda_bin" create --yes --prefix "$target_env" --clone "$source_env"
fi

if ! "$target_env/bin/python" -c \
    'import importlib.metadata as m, importlib.util; raise SystemExit(0 if m.version("mamba-ssm") == "2.3.1" and importlib.util.find_spec("selective_scan_cuda") else 1)'; then
    "$target_env/bin/python" -m pip install \
        --upgrade \
        --no-build-isolation \
        "mamba-ssm @ git+https://github.com/state-spaces/mamba.git@$mamba_commit"
fi

"$target_env/bin/python" -m pip install \
    --force-reinstall \
    --no-deps \
    "transformers==4.48.1" \
    "tokenizers==0.21.4"

if [[ ! -d "$mamba_source/.git" ]]; then
    git clone --filter=blob:none https://github.com/state-spaces/mamba.git "$mamba_source"
fi
git -C "$mamba_source" fetch --quiet origin "$mamba_commit"
git -C "$mamba_source" checkout --quiet --detach "$mamba_commit"
test "$(git -C "$mamba_source" rev-parse HEAD)" = "$mamba_commit"

PYTHONPATH="$mamba_source${PYTHONPATH:+:$PYTHONPATH}" "$target_env/bin/python" - <<'PY'
import torch
from mamba_ssm.modules.mamba3 import Mamba3

assert torch.__version__.startswith("2.10.0")
assert Mamba3.__module__ == "mamba_ssm.modules.mamba3"
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print("mamba3_import_ok=true")
PY
