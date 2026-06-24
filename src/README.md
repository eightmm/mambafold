# `src/mambafold`

Main package for the single-chain direct all-atom model.

```text
mambafold/
├── data/        dataset, collation, transforms, batch dataclasses
├── losses/      FM, LDDT, geometry, topology auxiliaries
├── model/       BiMamba blocks and direct all-atom architecture
├── sampling/    direct all-atom Euler sampler
├── train/       config, DDP, engine, logging, checkpoints
└── utils/       geometry helpers
```
