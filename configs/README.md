# configs/ - Configuration Files

All training is controlled via YAML configuration files. Parameters in config files can be overridden via command-line arguments.

## Config Files

| File | Purpose |
|------|---------|
| `train_base.yaml` | Main training config (full-scale) |
| `overfit_base.yaml` | Overfit validation (small dataset) |
| `overfit_test.yaml` | Quick overfit (minimal config) |
| `overfit_6000ada.yaml` | Overfit on 6000ada partition |
| `overfit_plm.yaml` | Overfit with ESM PLM enabled |

## Key Parameters

### Model
- `d_atom` - Atom feature dimension (256)
- `d_res` - Residue feature dimension (256)
- `d_plm` - PLM embedding dimension (1024)
- `n_atom_enc` - Atom encoder layers (2-4)
- `n_trunk` - Residue trunk layers (6-24)
- `n_atom_dec` - Atom decoder layers (2-4)
- `use_plm` - Enable ESM PLM conditioning (true/false)
- `d_state`, `mimo_rank`, `headdim` - SSM parameters

### Training
- `total_steps` - Total training steps
- `lr` - Learning rate (1e-4)
- `warmup_steps` - Linear warmup steps (2000)
- `batch_size` - Per-GPU batch size
- `ckpt_interval` - Checkpoint save interval (5000 steps)
- `eval_interval` - Evaluation interval (0 = off)
- `gamma_schedule` - "logit_normal" or "uniform"

### Data
- `max_length` - Max protein length (256 for training)
- `data_dir` - Path to AFDB data
- `num_workers` - DataLoader workers

## Usage

```bash
# Use config file
PYTHONPATH=src python scripts/train.py --config configs/train_base.yaml

# Override parameters
PYTHONPATH=src python scripts/train.py \
    --config configs/train_base.yaml \
    --lr 5e-5 \
    --batch_size 4 \
    --d_atom 128
```
