# `src/mambafold`

Main package for the single-chain coarse-to-fine model.

```text
mambafold/
├── data/        Dataset, collation, transforms, batch dataclasses
├── losses/      CA/all-atom FM, lDDT, geometry, distogram
├── model/       BiMamba3 blocks and `model/fold/` architecture
├── sampling/    two-stage Euler sampler
├── train/       config, DDP, engine, logging, checkpoints
└── utils/       geometry helpers
```

Active model entry point: `mambafold.model.fold`.
