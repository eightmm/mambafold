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
| Hardware target | 4 x RTX 6000 Ada, 48 GB each |
| Maximum length | 1,024 residues |
| GPU-safe state size | `d_state=64` (the `d_state=128` MIMO backward kernel exceeds Ada shared-memory capacity) |

## Status and reporting rule

The track is in data-loader/preflight validation and has **no completed
training checkpoint or CASP14 result**.  Do not compare it with ESM3 or report
its performance until an EMA checkpoint, frozen target list, sampler settings,
and OpenStructure evaluation artifact have all been recorded.

## Reproduction entrypoints

```bash
MAMBA_SKIP_CUDA_BUILD=TRUE uv sync --extra dev
sbatch scripts/slurm_preflight_esmc6b_ada.sh
# After the real-data and 4-rank model preflights pass:
sbatch scripts/slurm_train_esmc6b_ada.sh
```

The Slurm scripts intentionally do not request a fixed CPU count.  The data
loader caps workers based on the CPU allocation actually supplied by the
cluster.
