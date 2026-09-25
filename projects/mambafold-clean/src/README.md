# `src/mambafold`

This package implements the active single-chain direct all-atom MambaFold
architecture conditioned on frozen, sequence-only ESMC-6B embeddings.

```text
mambafold/
├── data/        RCSB/AFDB datasets, ESMC cache loading, collation, transforms
├── losses/      exact all-atom soft lDDT
├── model/       Bi-Mamba atom encoder/decoder and pair-free residue trunk
├── sampling/    one batch-first all-atom ODE/SDE sampler
├── train/       configuration, DDP, engine, logging, and checkpoints
└── utils/       geometry helpers
```

The active training contract is set by
[`docs/data_contract.md`](../docs/data_contract.md). Two differences matter
when reading the code:

- The trunk has no attention and there is no pair stack, and neither is
  reachable by config. `bimamba3.py`'s hybrid attention layer and
  `model/fold/pair_blocks.py` / `multiplicative_update.py` were deleted rather
  than defaulted off: a disabled O(L^2) triangle path still has to be carried
  and sized, and a trunk that *could* be given attention cannot support the
  claim that a pure SSM trunk suffices. That reasoning stands on its own. The
  all-attention control trunk this once promised was cancelled on cost, so no
  controlled comparison against attention exists.
- The objective is aligned all-atom flow matching plus exact all-atom soft
  lDDT. Unused auxiliary heads and losses are absent. Confidence is a separate
  later phase over a frozen folding model.

The model emits only all-atom velocity and the residue trunk latent. Sampling
uses one batch-first implementation, including when batch size is one.
