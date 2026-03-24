#!/bin/bash
#SBATCH --job-name=install_mamba_ssm
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/install_mamba_ssm_%j.out

cd /home/jaemin/project/protein/folding
VENV_PY=.venv/bin/python
VENV_PIP=.venv/bin/pip
module load cuda/12.8

echo "CUDA: $(nvcc --version | grep release)"
echo "PyTorch: $($VENV_PY -c 'import torch; print(torch.__version__, torch.version.cuda)')"

$VENV_PIP install git+https://github.com/state-spaces/mamba --no-build-isolation

echo "Verifying install..."
PYTHONPATH=/home/jaemin/project/protein/folding/src $VENV_PY -c "
from mamba_ssm.modules.mamba3 import Mamba3
from mambafold.model.ssm.mamba3 import Mamba3Layer, HAS_MAMBA_SSM
import torch

print('HAS_MAMBA_SSM:', HAS_MAMBA_SSM)
assert HAS_MAMBA_SSM, 'mamba-ssm not detected after install!'

# SISO
layer = Mamba3Layer(d_model=256, d_state=64, mimo_rank=1, headdim=64).cuda()
x = torch.randn(2, 32, 256, device='cuda')
y = layer(x)
print('SISO:', y.shape)

# MIMO R=4
layer_mimo = Mamba3Layer(d_model=256, d_state=64, mimo_rank=4, headdim=64).cuda()
y2 = layer_mimo(x)
print('MIMO R=4:', y2.shape)
print('Official kernel OK.')
"
