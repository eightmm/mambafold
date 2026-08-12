# MambaFold ESMC-6B track

This is the active successor to the ESM3 legacy baseline.  It uses the pinned
sequence-only `biohub/ESMC-6B` revision
`45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a` and 2,560-dimensional residue
embeddings.  The different embedding width makes ESMC-6B checkpoints
incompatible with ESM3 checkpoints.

## Current contract

| Field | Value |
| --- | --- |
| Configuration | `configs/direct_allatom_puremamba_attn6_geo_adaln_sf360_esmc6b.yaml` |
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

This track is actively training from scratch. At the 2026-08-12 snapshot, the
latest retained checkpoint is step 111,500 of 170,000. Training has not
finished, and this checkpoint has not yet received the full evaluation below.

The newest complete result is the step-88,500 EMA checkpoint evaluated on all
70 fixed CASP14 whole-chain targets with seed 0, 500-step SDE sampling, and
OpenStructure 2.9.1. T1044 is excluded because it exceeds the 1,024-residue
model limit.

| Checkpoint | GDT-TS ↑ | GDT-HA ↑ | TM-score ↑ | all-atom lDDT ↑ | backbone lDDT ↑ | RMSD (Å) ↓ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50,000 | 0.596 | 0.458 | 0.687 | 0.577 | 0.719 | 8.178 |
| **88,500** | **0.629** | **0.487** | **0.718** | **0.612** | **0.738** | **7.275** |

All values are 70-target means from `ost compare-structures` with
`--fault-tolerant --min-pep-length 4 --lddt --bb-lddt --rigid-scores
--tm-score`. The machine-readable interim record is
[`../results/esmc6b_casp14_interim.json`](../results/esmc6b_casp14_interim.json).

These are interim research results, not a frozen model release. The checkpoint
is not distributed, step 111,500 has not been rescored, and ESMC-6B pretraining
postdates CASP14. Accordingly, CASP14 is used here for retrospective
architecture comparison and must not be described as a temporally clean blind
test.

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
