# Reference overview

These notes map the active MambaFold-ESMC-6B implementation to its main
architectural references. Paper drafting itself is maintained outside this
directory.

## SimpleFold

SimpleFold provides the primary folding comparison and the high-level
atom-encoder, residue-trunk, atom-decoder decomposition. SimpleFold-360M is the
size-matched primary baseline; SimpleFold-3B is a scale reference when an
evaluation under the same target and scoring contract is available.

MambaFold retains the direct all-atom flow-matching decomposition but uses a
pair-free Bi-Mamba residue trunk with sparse self-attention.

## Mamba-3

Mamba-3 supplies the sequence-modeling blocks used bidirectionally in the atom
encoder, residue trunk, and atom decoder. The active 18-block residue trunk
inserts self-attention every six blocks for sparse global communication.

## ESMC-6B conditioning

The active conditioner is the frozen, sequence-only `biohub/ESMC-6B` revision
pinned in [`../models/esmc6b.md`](../models/esmc6b.md). Its residue embeddings
are projected into the trunk input. ESM3 belongs only to the immutable legacy
archive and is not part of the active comparison.

## Flow objective

```text
x_t = t * x_clean + (1 - t) * epsilon
target velocity = x_clean - epsilon
```

The atom decoder predicts the velocity for every valid atom slot. Geometry,
lDDT, and topology terms supplement the masked flow-matching objective.

## Reading order

1. `simplefold_summary.md` for the folding pipeline template.
2. `mamba3_summary.md` for the sequence backbone.
3. [`../architecture.md`](../architecture.md) for the active implementation.
