# configs/ — Configuration Files

YAML keys map 1:1 to script argument names. CLI flags override YAML values.

## Files

### `train_base.yaml`
Full-scale training config for multi-GPU DDP.
- `total_steps: 200000`
- `lr: 1e-4`, `warmup_steps: 2000`
- `batch_size: 8` (per-GPU)
- `max_length: 256` (pretrain)

### `overfit_base.yaml`
Template for quick validation (small dataset, ~16 proteins).

### `overfit_test.yaml`
Test partition config.
- `n_steps: 5000`
- `lr: 1e-3`
- `d_atom: 256, d_res: 256, n_trunk: 6`

### `overfit_6000ada.yaml`
Larger overfit on 6000ada.
- `d_atom: 384, d_res: 384, n_trunk: 8`
- `n_steps: 10000`

### `overfit_plm.yaml`
Overfit with ESM PLM enabled.
- `use_plm: true`

## Key Parameters

| Setting | Meaning |
|---------|---------|
| `data_dir` | AFDB data path |
| `max_length` | Max protein length (256 train, 512 finetune) |
| `batch_size` | Per-GPU batch size |
| `total_steps` | Training steps |
| `lr` | Learning rate |
| `warmup_steps` | Linear warmup to lr |
| `d_atom` | Atom token dimension |
| `d_res` | Residue token dimension |
| `n_atom_enc`, `n_trunk`, `n_atom_dec` | Layer counts |
| `use_plm` | Enable ESM PLM |
| `gamma_schedule` | "logit_normal" or "uniform" |

## Usage

```bash
# Use config
python scripts/train.py --config configs/train_base.yaml

# Override single param
python scripts/train.py --config configs/train_base.yaml --lr 5e-5

# Disable W&B
python scripts/train.py --config configs/overfit_test.yaml --no_wandb
```
