#!/bin/bash
#SBATCH --job-name=verify-mambafold
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/verify_model_%j.out

set -e

cd /home/jaemin/project/protein/folding
VENV_PY=.venv/bin/python
module load cuda/12.8

echo "=== Environment ==="
$VENV_PY -c "import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}, gpu={torch.cuda.get_device_name(0)}')"

echo "=== Imports ==="
PYTHONPATH=src $VENV_PY -c "
from mambafold.model.ssm.mamba3 import Mamba3Layer
from mambafold.model.ssm.bimamba3 import Mamba3Block, BiMamba3Block, MambaStack
from mambafold.model.blocks import RMSNorm, SwiGLU, GammaEmbedder
from mambafold.model.mambafold import MambaFoldEqM
from mambafold.losses.eqm import eqm_loss, eqm_reconstruction_scale
from mambafold.losses.lddt import soft_lddt_ca_loss
print('All imports OK')
"

echo "=== EqM time-unconditional: model does not see gamma ==="
PYTHONPATH=src $VENV_PY -c "
import torch
from mambafold.model.blocks import RMSNorm, SwiGLU
# GammaEmbedder should NOT exist
try:
    from mambafold.model.blocks import GammaEmbedder
    raise AssertionError('GammaEmbedder still exists — EqM violation!')
except ImportError:
    print('GammaEmbedder correctly absent (EqM time-unconditional) OK')
"

echo "=== BiMamba3Block with bfloat16 ==="
PYTHONPATH=src $VENV_PY -c "
import torch
from mambafold.model.ssm.bimamba3 import BiMamba3Block

x = torch.randn(2, 64, 256, device='cuda', dtype=torch.bfloat16)
mask = torch.ones(2, 64, device='cuda', dtype=torch.bool)
mask[0, 50:] = False

block = BiMamba3Block(d_model=256, d_state=64, mimo_rank=4, headdim=64).cuda().to(torch.bfloat16)
y = block(x, mask)
assert y.dtype == torch.bfloat16, f'dtype mismatch: {y.dtype}'
assert y[0, 50:].abs().max() == 0, 'padding not zeroed!'
print(f'BiMamba3Block bfloat16: {x.shape} -> {y.shape}, padding OK')
"

echo "=== MambaFoldEqM with gamma conditioning ==="
PYTHONPATH=src $VENV_PY -c "
import torch
from mambafold.data.types import ProteinBatch
from mambafold.model.mambafold import MambaFoldEqM

B, L, A = 2, 16, 15
batch = ProteinBatch(
    res_type=torch.zeros(B, L, dtype=torch.long, device='cuda'),
    atom_type=torch.zeros(B, L, A, dtype=torch.long, device='cuda'),
    res_mask=torch.ones(B, L, dtype=torch.bool, device='cuda'),
    atom_mask=torch.ones(B, L, A, dtype=torch.bool, device='cuda'),
    valid_mask=torch.ones(B, L, A, dtype=torch.bool, device='cuda'),
    ca_mask=torch.ones(B, L, dtype=torch.bool, device='cuda'),
    x_clean=torch.randn(B, L, A, 3, device='cuda'),
    x_gamma=torch.randn(B, L, A, 3, device='cuda'),
    eps=torch.randn(B, L, A, 3, device='cuda'),
    gamma=torch.rand(B, 1, 1, 1, device='cuda'),
    esm=torch.randn(B, L, 32, device='cuda'),
)

model = MambaFoldEqM(
    d_atom=64, d_res=128, d_plm=32,
    n_atom_enc=1, n_trunk=2, n_atom_dec=1,
    use_plm=True,
    atom_d_state=16, atom_mimo_rank=4, atom_headdim=32,
    d_state=32, mimo_rank=4, headdim=32,
).cuda()

n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f'MambaFoldEqM: {n_params:.2f}M parameters')

out = model(batch)
assert out.shape == (B, L, A, 3), f'wrong shape: {out.shape}'
print(f'Forward pass: {out.shape} OK')

# EqM: model output should be same regardless of gamma (time-unconditional)
import dataclasses
batch2 = dataclasses.replace(batch, gamma=torch.zeros(B, 1, 1, 1, device='cuda'))
out2 = model(batch2)
diff = (out - out2).abs().max().item()
assert diff == 0.0, f'Model depends on gamma — EqM violation! max diff={diff}'
print(f'EqM time-unconditional verified: output invariant to gamma OK')
"

echo "=== EqM loss ==="
PYTHONPATH=src $VENV_PY -c "
import torch
from mambafold.losses.eqm import eqm_loss, eqm_reconstruction_scale

B, L, A = 2, 8, 15
pred = torch.randn(B, L, A, 3, device='cuda')
x_clean = torch.randn(B, L, A, 3, device='cuda')
eps = torch.randn(B, L, A, 3, device='cuda')
gamma = torch.rand(B, 1, 1, 1, device='cuda')
mask = torch.ones(B, L, A, dtype=torch.bool, device='cuda')

loss = eqm_loss(pred, x_clean, eps, gamma, mask)
print(f'EqM loss: {loss.item():.4f} OK')

scale = eqm_reconstruction_scale(gamma)
x_hat = pred - scale * pred
print(f'Reconstruction scale shape: {scale.shape} OK')
"

echo "=== All checks passed ==="
