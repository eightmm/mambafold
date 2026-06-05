# scripts/ — Training and Data Scripts

## `train.sh`

Preferred launcher. It defaults to `configs/stage1.yaml`, sets W&B/NCCL environment variables, and invokes `torchrun` with one process per visible GPU.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train.sh
```

Resume:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
RESUME=outputs/train/<run>/ckpt_latest.pt \
bash scripts/train.sh
```

## `train.py`

Underlying DDP entrypoint:

```bash
PYTHONPATH=src uv run torchrun --nproc_per_node=4 scripts/train.py \
  --config configs/stage1.yaml
```

Outputs checkpoints, W&B logs, and `config.json`.

## Data prep

```text
download_rcsb_cif.sh -> batch_convert_cif.py -> build_metadata.py
                                      |
                                      v
extract_deposit_dates.py -> make_val_split.py -> train.txt / val.txt / val_casp.txt
```

## ESM precompute

Use `precompute_esm.py` or the 8-GPU wrapper `precompute_esm_8gpu.sh` to populate `data/rcsb_esm`.
