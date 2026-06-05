# configs/ — Training Configurations

Active configs: `stage1.yaml`, `stage1_ct.yaml`, `stage2.yaml`, `joint.yaml`.

YAML keys map 1:1 to `scripts/train.py` arguments. CLI flags override YAML.

## stage configs

Current production training setup:

| Setting | Value |
|---|---|
| Objective | flow matching |
| Data | `data/rcsb`, `data/splits/train.txt`, `data/splits/val.txt` |
| Length | `max_length: 1024` |
| Batch | `batch_size: 8`, `grad_accum_steps: 3` |
| PLM | `use_plm: true`, `esm_dir: data/rcsb_esm`, `d_plm: 1536` |
| Model | `d_atom=384`, `d_res=1024`, `n_trunk=16` |
| Aux losses | bond, clash, distogram, pLDDT |

## Usage

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train.sh --lr 5e-5
CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh --total_steps 100 --no_wandb
```
