# Paper Overview

These notes map the active MambaFold implementation to its main references.

## SimpleFold

SimpleFold is the architectural reference: all-atom generation with an atom encoder, residue-level global trunk, atom decoder, PLM conditioning, and structure-aware auxiliary losses.

MambaFold keeps this decomposition but replaces the expensive trunk with Mamba-3 blocks.

## Mamba-3

Mamba-3 is the sequence-modeling backbone used in the residue trunk. The goal is to keep long-range residue communication while avoiding a full quadratic attention trunk.

Current implementation uses bidirectional Mamba stacks for atom encoder/decoder and residue trunk blocks.

## Active Objective

The active training objective is flow matching:

```text
x_t = t * x_clean + (1 - t) * eps
model target = x_clean - eps
```

ESM3 embeddings are concatenated into the residue trunk input. Auxiliary heads provide distogram and pLDDT supervision.

## Reading Order

1. `simplefold_summary.md` for the folding pipeline template.
2. `mamba3_summary.md` for the sequence backbone.
