# src/mambafold — Package Root

## Modules

```
mambafold/
├── data/          Data pipeline (loading, types, constants, transforms)
├── model/         Model architecture
│   └── ssm/       Mamba-3 layers (official kernel wrapper + BiMamba)
├── losses/        EqM loss, CA-LDDT auxiliary loss
├── sampling/      Samplers (NAG, Euler)
├── train/         Training loop, DDP, checkpoints, EMA
└── utils/         Geometry utilities
```

## Key Entry Points

| Task | Module |
|------|--------|
| Load AFDB data | `data/dataset.py::AFDBDataset` |
| Build model | `train/trainer.py::build_model` |
| Forward pass | `model/mambafold.py::MambaFoldEqM.forward` |
| Training step | `train/engine.py::train_step` |
| Evaluation | `train/engine.py::eval_step` |
| EqM loss | `losses/eqm.py::eqm_loss` |
| Sample structures | `sampling/sampler.py::EqMNAGSampler` |
| DDP setup | `train/distributed.py::setup_dist` |
| Checkpoints | `train/trainer.py::{save,load}_checkpoint` |

## Installation

```bash
# Via PYTHONPATH
PYTHONPATH=src python scripts/train.py ...

# Or editable install
pip install -e .
```

## GPU Requirements

Mamba-3 uses official Triton kernels → **CUDA GPU required**.
CPU-only nodes can import but cannot run forward pass.
