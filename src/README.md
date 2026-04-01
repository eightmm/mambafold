# src/mambafold - Source Code

## Structure

```
src/mambafold/
├── __init__.py
├── data/                    # Data pipeline
│   ├── __init__.py
│   ├── constants.py         # AA vocab (21), atom slots (15), COORD_SCALE=10.0
│   ├── types.py             # ProteinExample, ProteinBatch dataclass
│   ├── dataset.py           # AFDBDataset — load .pt, canonical atom slots
│   ├── transforms.py        # CenterScale, SO3Aug, EqMCorrupt
│   ├── collate.py           # ProteinCollator, LengthBucketSampler
│   ├── loader.py            # inf_loader (infinite dataloader)
│   └── esm.py              # EvolutionaryScalePLM, ESM embedding
├── model/                   # Model architecture
│   ├── __init__.py
│   ├── embeddings.py        # AtomFeatureEmbedder, SequenceFourierEmbedder
│   ├── ssm/                 # Mamba-3 SSM modules
│   │   ├── __init__.py
│   │   ├── mamba3.py        # Mamba3Layer wrapper
│   │   └── bimamba3.py      # BiMamba3Block, MambaStack
│   ├── blocks.py            # RMSNorm, SwiGLU
│   ├── atom_encoder.py      # AtomEncoder (local attention per-residue)
│   ├── grouping.py          # group_atoms_to_residues, ResidueToAtomBroadcast
│   ├── residue_trunk.py     # ResidueTrunk (BiMamba-3 stack)
│   └── mambafold.py         # MambaFoldEqM (end-to-end)
├── losses/                  # Loss functions
│   ├── __init__.py
│   ├── eqm.py              # EqM loss, truncated_c, eqm_reconstruction_scale
│   └── lddt.py             # soft_lddt_ca_loss (differentiable CA-LDDT)
├── sampling/                # Sampling / inference
│   ├── __init__.py
│   ├── sampler.py           # EqMNAGSampler, EqMEulerSampler
│   └── stop_criterion.py    # (optional) stopping criteria
├── train/                   # Training infrastructure
│   ├── __init__.py
│   ├── distributed.py       # setup_dist, all_reduce_mean, GPUMonitor
│   ├── trainer.py           # build_model, cosine_warmup_lr, save/load_checkpoint
│   ├── engine.py            # train_step, eval_step
│   └── ema.py              # EMA (Exponential Moving Average)
└── utils/                   # Utilities
    ├── __init__.py
    ├── geometry.py          # SO3 rotation, centroid, pairwise distances
    └── metrics.py           # ca_lddt, rmsd (optional)
```

## Implementation Status

| Module | Status | Notes |
|--------|--------|-------|
| **data/constants.py** | ✅ Complete | 21 AAs, atom14 slots, MAX_ATOMS_PER_RES=14 |
| **data/types.py** | ✅ Complete | ProteinExample, ProteinBatch dataclasses |
| **data/dataset.py** | ✅ Complete | AFDB dataset loading, canonical atom slots |
| **data/transforms.py** | ✅ Complete | Center-scale, SO3 augmentation, EqM corruption |
| **data/collate.py** | ✅ Complete | ProteinCollator, LengthBucketSampler |
| **data/loader.py** | ✅ Complete | inf_loader for infinite batching |
| **data/esm.py** | ✅ Complete | ESM3/ESMc PLM integration |
| **model/embeddings.py** | ✅ Complete | Fourier PE, atom/residue embedders |
| **model/mamba3.py** | ✅ Complete | Mamba3Layer wrapper |
| **model/bimamba3.py** | ✅ Complete | BiMamba3Block, MambaStack |
| **model/blocks.py** | ✅ Complete | RMSNorm, SwiGLU |
| **model/atom_encoder.py** | ✅ Complete | Local attention (per-residue) |
| **model/grouping.py** | ✅ Complete | Pool/broadcast atoms ↔ residues |
| **model/residue_trunk.py** | ✅ Complete | BiMamba-3 residue trunk |
| **model/mambafold.py** | ✅ Complete | MambaFoldEqM end-to-end |
| **losses/eqm.py** | ✅ Complete | EqM loss, truncated c(γ), reconstruction scale |
| **losses/lddt.py** | ✅ Complete | Differentiable CA-LDDT |
| **sampling/sampler.py** | ✅ Complete | NAG sampler, Euler sampler |
| **train/distributed.py** | ✅ Complete | DDP setup, all-reduce, GPU monitor |
| **train/trainer.py** | ✅ Complete | Model builder, LR scheduler, checkpoint I/O |
| **train/engine.py** | ✅ Complete | train_step, eval_step |
| **train/ema.py** | ✅ Complete | EMA with shadow parameters |
| **utils/geometry.py** | ✅ Complete | SO3, centroid, pairwise distances |

## Data Schema

### ProteinExample (single protein)
- `res_type: [L]` — amino acid type (0-20)
- `atom_type: [L, 14]` — atom slot index per residue
- `coords: [L, 14, 3]` — 3D coordinates (Å)
- `atom_mask: [L, 14]` — valid atom indicator
- `observed_mask: [L, 14]` — experimentally observed

### ProteinBatch (batched)
- `x_clean: [B, L, 14, 3]` — normalized clean coordinates
- `x_gamma: [B, L, 14, 3]` — EqM corrupted coordinates
- `eps: [B, L, 14, 3]` — Gaussian noise
- `gamma: [B, 1, 1, 1]` — noise level (interpolation factor)
- `esm: [B, L, d_esm]` — ESM2 embeddings (optional)
- `res_mask: [B, L]` — residue validity
- `atom_mask: [B, L, 14]` — atom validity
