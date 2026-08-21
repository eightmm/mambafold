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

Which parts this project actually trains is set by
[`docs/data_contract.md`](../docs/data_contract.md), not by what exists here.
Two differences matter when reading the code:

- The trunk has no attention and there is no pair stack, and neither is
  reachable by config. `bimamba3.py`'s hybrid attention layer and
  `model/fold/pair_blocks.py` / `multiplicative_update.py` were deleted rather
  than defaulted off: a disabled O(L^2) triangle path still has to be carried
  and sized, and a trunk that *could* be given attention cannot support the
  claim that a pure SSM trunk suffices. An all-attention control arm is a
  separate trunk, written for the purpose.
- The objective is flow matching plus all-atom lDDT. `losses/geometry.py` and
  `losses/stereochemistry.py` are present but carry zero weight in the baseline
  run; a geometry fine-tune is a separate run, reported separately.

`pairfree_aux_heads` is the one O(L^2) tensor left, and it is an output head:
it builds an outer sum from the residue latent on the way out and feeds nothing
back into the trunk. It is off in the baseline run.
