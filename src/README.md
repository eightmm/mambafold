# `src/mambafold`

This package implements the shared single-chain direct all-atom MambaFold
architecture. The frozen ESM3 release uses this code for inference; the active
research path uses the same architecture with ESMC-6B conditioning.

```text
mambafold/
├── data/        RCSB/AFDB datasets, PLM embedding loading, collation, transforms
├── losses/      flow-matching, lDDT, geometry, and topology auxiliaries
├── model/       Bi-Mamba blocks and atom→token→atom model
├── sampling/    direct all-atom ODE/SDE samplers
├── train/       config, DDP, engine, logging, and checkpoint loading
└── utils/       geometry helpers
```

The public ESM3 FASTA entrypoint is
[`projects/esm3/predict_fasta.py`](../projects/esm3/predict_fasta.py). Do not
resume its checkpoint with the ESMC-6B configuration: ESM3 uses 1,536-d PLM
embeddings while ESMC-6B uses 2,560-d embeddings.
