# `src/mambafold`

This package implements the active single-chain direct all-atom MambaFold
architecture conditioned on frozen, sequence-only ESMC-6B embeddings.

```text
mambafold/
├── data/        RCSB/AFDB datasets, ESMC cache loading, collation, transforms
├── losses/      flow-matching, lDDT, geometry, and topology auxiliaries
├── model/       Bi-Mamba atom encoder/decoder and pair-free residue trunk
├── sampling/    direct all-atom ODE/SDE samplers
├── train/       configuration, DDP, engine, logging, and checkpoints
└── utils/       geometry helpers
```

The public FASTA entrypoint is
[`projects/esmc6b/predict_fasta.py`](../projects/esmc6b/predict_fasta.py).
The ESM3 project is an immutable legacy archive only. ESM3 and ESMC-6B
checkpoints are incompatible because their PLM feature widths are 1,536 and
2,560, respectively.
