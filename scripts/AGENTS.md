# scripts/

`train.sh` defaults to `configs/direct_allatom_360m.yaml`, sets W&B/NCCL
environment variables, and invokes `torchrun` with one process per visible GPU.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train.sh
PYTHONPATH=src uv run torchrun --nproc_per_node=4 scripts/train.py \
  --config configs/direct_allatom_360m.yaml
```
