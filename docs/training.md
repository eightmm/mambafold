# Training

The sole active training configuration is ESMC-6B conditioned and starts from
scratch with respect to the frozen ESM3 archive:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
CONFIG=configs/direct_allatom_puremamba_attn6_geo_adaln_sf360_esmc6b_gpu8.yaml \
bash scripts/train.sh
```

## Mainline contract

- maximum length 1,024;
- eight RTX 6000 Ada ranks;
- batch 9 per rank, seven-step gradient accumulation, effective batch 504;
- 18-block, width-1,024 Bi-Mamba trunk with attention every six blocks;
- no explicit pair stack;
- pinned 2,560-dimensional ESMC-6B residue embeddings;
- official Boltz-style RCSB monomers plus AFDB SwissProt structures;
- logit-normal flow-time sampling and sampled all-atom lDDT/clash losses.

The from-scratch run reached step 170,000. Its EMA is the provisional preview
under [`projects/esmc6b`](../projects/esmc6b/README.md). A separate 50,000-step
geometry fine-tune initialized from that EMA remains in progress; it is not
represented by the preview artifact and no final model selection is claimed.

Resume only a checkpoint created by the same ESMC-6B configuration:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
CONFIG=configs/direct_allatom_puremamba_attn6_geo_adaln_sf360_esmc6b_gpu8.yaml \
RESUME=outputs/train/<esmc6b-run>/ckpt_latest.pt \
bash scripts/train.sh
```

Never resume an ESM3 checkpoint: its 1,536-dimensional conditioner is
checkpoint-incompatible with ESMC-6B.
