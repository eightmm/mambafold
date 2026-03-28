# scripts/ - Training and Evaluation Scripts

## Scripts

| Script | Purpose |
|--------|---------|
| `train.py` | Full EqM training (single/multi-GPU DDP) |
| `overfit.py` | Quick validation on small dataset |
| `export_inference.py` | Export checkpoint to inference format |
| `viz_sampler.py` | Visualize sampler trajectories |

## Training

### Single GPU
```bash
PYTHONPATH=src python -u scripts/train.py --config configs/train_base.yaml
```

### Multi-GPU (DDP via torchrun)
```bash
PYTHONPATH=src torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/train_base.yaml
```

### Resume from checkpoint
```bash
PYTHONPATH=src torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/train_base.yaml \
    --resume outputs/train/run1/ckpt_latest.pt
```

## Overfit Validation

Test model on ~16 examples to verify training loop works:

```bash
PYTHONPATH=src python -u scripts/overfit.py \
    --config configs/overfit_base.yaml \
    --data_dir afdb_data/train \
    --out_dir outputs/overfit/test1
```

Output: `config.json`, `checkpoint.pt`, `metrics.json`, `viz.png`

## SLURM Submission

### Training on 6000ada partition
```bash
sbatch scripts/slurm/train_6000ada.sh
```

Auto-detects GPU count from `SLURM_GPUS_ON_NODE` and runs either single-GPU or multi-GPU DDP.

### Test partition (quick debugging)
```bash
sbatch scripts/slurm/overfit_test.sh
```

## Configuration

All training parameters are controlled via YAML config files in `configs/`. See `configs/README.md` for details.

Key config files:
- `train_base.yaml` - Main training config
- `overfit_base.yaml` - Overfit validation config
- `overfit_test.yaml`, `overfit_plm.yaml` - Alternative overfit configs
