# MambaFold — Project Overview

## Purpose

Implement protein structure prediction combining **Equilibrium Matching (EqM)** and **Mamba-3 SSM** for time-unconditional, all-atom generation.

## Quick Links

- **Training**: `scripts/train.py` (DDP multi-GPU)
- **Validation**: `scripts/overfit.py` (small dataset)
- **Inference**: `scripts/export_inference.py`
- **Config**: `configs/train_base.yaml`, `configs/overfit_base.yaml`

## Architecture Summary

**EqM (time-unconditional)**: Model sees only noisy coords `x_γ`, learns to predict equilibrium gradient `f(x_γ) ≈ (ε - x_clean)·c(γ)`

**Pipeline**:
```
Coordinates [B, L, 14, 3]
  ↓ AtomFeatureEmbedder + SequenceFourierEmbedder
  ↓ MambaStack (Atom Encoder, per-residue)
  ↓ group_atoms_to_residues [B, L, d_atom]
  ↓ [concat ESM if use_plm] → trunk_proj
  ↓ MambaStack (Residue Trunk, BiMamba-3, heavy)
  ↓ ResidueToAtomBroadcast [B, L, 14, d_atom]
  ↓ MambaStack (Atom Decoder, per-residue)
  ↓ GradientHead
Gradient [B, L, 14, 3]
```

**Key components**:
- **AtomFeatureEmbedder** (`model/embeddings.py`) — Fourier PE + token embeddings
- **MambaStack** (`model/ssm/bimamba3.py`) — BiMamba-3 blocks with gated fusion
- **Atom Encoder/Decoder** (`model/atom_encoder.py`, `model/atom_decoder.py`) — Local attention (per-residue)
- **Grouping** (`model/grouping.py`) — Pool atoms→residues, broadcast back

## Training

**Command**:
```bash
# Single GPU
PYTHONPATH=src python -u scripts/train.py --config configs/train_base.yaml

# Multi-GPU DDP via torchrun
PYTHONPATH=src torchrun --nproc_per_node=4 scripts/train.py --config configs/train_base.yaml

# Via SLURM (auto GPU detection)
sbatch scripts/slurm/train_6000ada.sh
```

**Key files**:
- `train/trainer.py` — Model builder, cosine warmup LR, checkpoint I/O
- `train/distributed.py` — DDP setup, all-reduce, GPU monitor
- `train/engine.py` — train_step, eval_step
- `train/ema.py` — EMA (decay=0.999)

**Settings**:
- Optimizer: AdamW, lr=1e-4, warmup_steps=2000
- Precision: bfloat16 AMP (model forward), fp32 loss
- Gradient clip: 1.0
- EMA decay: 0.999

## Data

- **Source**: AFDB .pt files (`afdb_data/train/`)
- **Format**: All-atom coords [L, 14, 3] per residue
- **Processing**: Center + scale by 10.0Å
- **EqM corruption**: `x_γ = γ·x_clean + (1-γ)·ε` with γ ~ U(0,1)

Key files:
- `data/dataset.py` — Load AFDB .pt
- `data/constants.py` — Atom slots, AA vocab
- `data/transforms.py` — EqM corruption, augmentation
- `data/collate.py` — Batching, bucketing sampler

## Losses

**EqM Loss** (`losses/eqm.py`):
```
L_EqM = ||f(x_γ) - (ε - x_clean)·c(γ)||²
c(γ) = 4·c_trunc(γ, a=0.8)  [truncated at γ=0.8]
```

**CA-LDDT Loss** (`losses/lddt.py`):
```
x̂ = x_γ - scale(γ)·f(x_γ)  [1-step reconstruction]
L_LDDT = soft_lddt_ca(x̂, x_clean)
```

**Combined**: `loss = L_EqM + α·L_LDDT`

## Sampling / Inference

**Samplers** (`sampling/sampler.py`):

1. **EqMNAGSampler** — Nesterov Accelerated Gradient
   - Descends learned gradient field
   - Adaptive stopping on gradient norm
   - Parameters: `eta=0.1, mu=0.3, g_min=5e-3, max_steps=128`

2. **EqMEulerSampler** — Euler integration
   - Interpolates from γ=0→1
   - Used in overfit validation

## Implementation Status (2026-03-28)

| Component | Status |
|-----------|--------|
| Data pipeline | ✅ Complete |
| Model architecture | ✅ Complete |
| EqM & CA-LDDT losses | ✅ Complete |
| DDP training + EMA | ✅ Complete |
| Cosine warmup LR scheduler | ✅ Complete |
| NAG + Euler samplers | ✅ Complete |
| Overfit validation | ✅ Passing |
| YAML configs | ✅ Complete |
| W&B logging | ✅ Complete |
| Full training pipeline | ✅ Complete |
| ESM PLM integration | ✅ Complete |
| Export inference | ✅ Complete |

## Key Papers

- **EqM**: Wang & Du (2025) — `docs/papers/equilibrium_matching.pdf`
- **Mamba-3**: arXiv:2603.15569 — `docs/papers/mamba3.pdf`
- **SimpleFold**: Architecture reference — `docs/papers/simplefold.pdf`

## Module Navigation

- `src/mambafold/data/` — Data loading, preprocessing
- `src/mambafold/model/` — Model architecture
- `src/mambafold/losses/` — Training objectives
- `src/mambafold/sampling/` — Inference samplers
- `src/mambafold/train/` — Training loop, DDP, checkpoints
- `src/mambafold/utils/` — Geometry, metrics
