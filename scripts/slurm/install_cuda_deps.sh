#!/bin/bash
#SBATCH --job-name=install-mamba
#SBATCH --partition=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/install_cuda_%j.out

set -e

cd /home/jaemin/project/protein/folding
VENV_PIP=.venv/bin/pip
VENV_PY=.venv/bin/python

module load cuda/12.8
export CUDA_HOME=/appl/cuda/12.8

echo "=== Environment ==="
nvcc --version
nvidia-smi | head -5

echo "=== PyTorch CUDA check ==="
$VENV_PY -c "import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}, gpu={torch.cuda.get_device_name(0)}')"

echo "=== Installing causal-conv1d (pip, no build isolation) ==="
$VENV_PIP install --no-build-isolation "causal-conv1d>=1.4" 2>&1

echo "=== Installing mamba-ssm (pip, no build isolation) ==="
$VENV_PIP install --no-build-isolation "mamba-ssm>=2.0" 2>&1

echo "=== Verification ==="
$VENV_PY -c "
import torch
print(f'torch={torch.__version__}, cuda={torch.version.cuda}')
import causal_conv1d
print('causal_conv1d OK')
import mamba_ssm
print(f'mamba_ssm={mamba_ssm.__version__}')
from mamba_ssm import Mamba2
m = Mamba2(d_model=256, d_state=64).cuda()
x = torch.randn(1, 32, 256).cuda()
y = m(x)
print(f'Mamba2 forward OK: {x.shape} -> {y.shape}')
"

echo "=== All CUDA deps installed successfully ==="
