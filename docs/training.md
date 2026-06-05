# Training

Active training is single-chain, PLM-conditioned, coarse-to-fine flow matching.

## Phases

| Phase | Config | Purpose |
|---|---|---|
| 1a | `configs/stage1.yaml` | C-alpha scaffold pretraining, L=512 |
| 1b | `configs/stage1_ct.yaml` | Stage 1 long-context continuation, L=1024 |
| 2 | `configs/stage2.yaml` | all-atom refiner with frozen Stage 1 |
| 3 | `configs/joint.yaml` | optional joint finetune |

## Losses

Stage 1:

```text
L = L_fm_ca + alpha(t) * L_lddt_ca + w_bond * L_ca_ca_bond + w_distogram * L_distogram
```

Stage 2:

```text
L = L_fm_non_ca + alpha(t) * L_lddt_full + w_bond * L_bond + w_clash * L_clash + w_ca_anchor * L_ca_anchor
```

Stage 2 initializes its CA slot from Stage 1, but may refine CA locally. `w_ca_anchor` limits drift.

## Launch

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 CONFIG=configs/stage1.yaml bash scripts/train.sh
```

Stage 2:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
CONFIG=configs/stage2.yaml \
STAGE1_CKPT=outputs/train/<stage1_run>/ckpt_latest.pt \
bash scripts/train.sh
```

## Checkpoints

`src/mambafold/train/trainer.py::save_checkpoint` stores `model`, `ema`, optimizer, scheduler, config args, W&B run id, and updates `ckpt_latest.pt`.
