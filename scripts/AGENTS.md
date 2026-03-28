# scripts/ — Training and Inference Scripts

## `train.py`
**Full DDP training script** (single/multi-GPU).

```bash
# Single GPU
PYTHONPATH=src python -u scripts/train.py --config configs/train_base.yaml

# Multi-GPU (torchrun)
PYTHONPATH=src torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/train_base.yaml

# Resume
PYTHONPATH=src torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/train_base.yaml \
    --resume outputs/train/run1/ckpt_latest.pt
```

Outputs: Checkpoints, W&B logging, config.json

## `overfit.py`
**Quick validation** on ~16 examples. Train → Eval → Visualize in one script.

```bash
PYTHONPATH=src python -u scripts/overfit.py \
    --config configs/overfit_base.yaml \
    --out_dir outputs/overfit/test1
```

Outputs:
- `checkpoint.pt` — model + ema + optimizer
- `metrics.json` — per-gamma LDDT/RMSD
- `viz.png` — loss curve, LDDT, RMSD plots
- `inference.npz` — full inference results for notebooks

## `export_inference.py`
Re-export inference.npz from trained checkpoint.

```bash
PYTHONPATH=src python scripts/export_inference.py \
    --ckpt outputs/train/run1/ckpt_latest.pt \
    --data_dir afdb_data/train \
    --out outputs/train/run1/inference.npz
```

## `viz_sampler.py`
Visualize sampler trajectories (NAG, Euler).

## SLURM Scripts

See `slurm/AGENTS.md` for submission scripts.
