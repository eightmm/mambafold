# Training

Active config:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 CONFIG=configs/direct_allatom_360m.yaml bash scripts/train.sh
```

Default scale:

- L=1024
- 4 GPUs
- per-GPU batch 4
- grad accumulation 12
- effective batch 192
- ~389.6M actual parameters
- logit-normal FM time sampling
- sampled all-atom LDDT/clash auxiliaries

Validation and checkpoints are controlled by `eval_interval` and
`ckpt_interval` in `configs/direct_allatom_360m.yaml`.

Resume:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
CONFIG=configs/direct_allatom_360m.yaml \
RESUME=outputs/train/<run>/ckpt_latest.pt \
bash scripts/train.sh
```
