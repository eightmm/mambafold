# MambaFold ESMC-6B track

This is the sole active MambaFold research and release track. It uses the pinned
sequence-only `biohub/ESMC-6B` revision
`45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a` and 2,560-dimensional residue
embeddings. The different embedding width makes ESMC-6B checkpoints
incompatible with ESM3 checkpoints.

## Current contract

| Field | Value |
| --- | --- |
| Configuration | `configs/direct_allatom_puremamba_attn6_geo_adaln_sf360_esmc6b_gpu8.yaml` |
| PLM | sequence-only ESMC-6B, pinned revision above |
| Training data | official Boltz-style processed RCSB plus AFDB SwissProt, with single-chain extraction |
| Cache identity | SHA-256 of canonical amino-acid sequence; repeated sequences share one embedding cache entry |
| Model parameters | 404.86M |
| Hardware | 8 x RTX 6000 Ada, 48 GB each |
| Maximum length | 1,024 residues |
| GPU-safe state size | `d_state=64` (the `d_state=128` MIMO backward kernel exceeds Ada shared-memory capacity) |
| Effective batch | 504 (9 proteins/GPU × 8 GPUs × 7 accumulation steps) |
| Training mixture | 835,570 indexed RCSB monomer examples + 268,816 AFDB SwissProt examples |

## Status and reporting rule

The from-scratch mainline reached its planned step 170,000 on 2026-08-18. Its
EMA state is distributed as a verified, deliberately provisional preview in
[`../../projects/esmc6b`](../../projects/esmc6b/README.md). A separate
50,000-step geometry fine-tuning experiment initialized from that EMA remains
in progress, so step 170,000 is not a final model selection.

The step-170,000 EMA was evaluated on the fixed 70-target CASP14 whole-chain
set with seed 0, 500-step SDE sampling, and OpenStructure 2.9.1. That set
excludes T1044 because it exceeds the 1,024-residue model limit.

| Checkpoint | GDT-TS ↑ | GDT-HA ↑ | TM-score ↑ | all-atom lDDT ↑ | backbone lDDT ↑ | RMSD (Å) ↓ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50,000 | 0.596 | 0.458 | 0.687 | 0.577 | 0.719 | 8.178 |
| 88,500 | 0.629 | 0.487 | 0.718 | 0.612 | 0.738 | 7.275 |
| **170,000 preview** | **0.682** | **0.544** | **0.761** | **0.646** | **0.769** | **6.146** |

All values are 70-target means from `ost compare-structures` with
`--fault-tolerant --min-pep-length 4 --lddt --bb-lddt --rigid-scores
--tm-score`. The machine-readable interim record is
[`../results/esmc6b_casp14_interim.json`](../results/esmc6b_casp14_interim.json).

These remain interim research results, not a frozen final model. CASP14 also
contains six exact matches to the coordinate-training corpus
(`T1029`, `T1030`, `T1034`, `T1065s2`, `T1082`, and `T1092`). The
step-170,000 inference-only EMA is distributed as a preview, while its source
training state and the ongoing geometry fine-tune are not. ESMC-6B pretraining
postdates CASP14; accordingly, CASP14 is retrospective engineering evidence
and must not be described as a temporally clean blind test.

New model comparisons use only ESMC-6B. The frozen ESM3 directory is retained
for historical reproduction and is not an active baseline.

## Reproduction entrypoints

```bash
MAMBA_SKIP_CUDA_BUILD=TRUE uv sync --extra dev
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  CONFIG=configs/direct_allatom_puremamba_attn6_geo_adaln_sf360_esmc6b_gpu8.yaml \
  bash scripts/train.sh
```

On a scheduler, preserve the same eight-rank launch and let the data loader cap
workers from the CPU allocation actually supplied by the cluster. The active
production configuration is
`configs/direct_allatom_puremamba_attn6_geo_adaln_sf360_esmc6b_gpu8.yaml`.
