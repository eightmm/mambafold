# docs/

This directory contains project notes, implementation plans, and paper references used to design MambaFold.

## Layout

```text
docs/
├── README.md
└── papers/
    ├── OVERVIEW.md
    ├── mamba3.pdf
    ├── mamba3_summary.md
    ├── simplefold.pdf
    ├── simplefold_summary.md
    ├── equilibrium_matching.pdf
    └── equilibrium_matching_summary.md
```

## Suggested Reading Order

1. **[OVERVIEW.md](papers/OVERVIEW.md)**: how the three papers map onto the project design.
2. **[simplefold_summary.md](papers/simplefold_summary.md)**: the end-to-end folding pipeline and training setup.
3. **[mamba3_summary.md](papers/mamba3_summary.md)**: the backbone SSM architecture that may replace the Transformer trunk.
4. **[equilibrium_matching_summary.md](papers/equilibrium_matching_summary.md)**: the alternative generative objective and sampling perspective.

## How Each Paper Is Used

### SimpleFold

SimpleFold is the architectural reference point. It explains how to turn sequence information and noisy coordinates into all-atom structure updates through an atom encoder, a residue-level trunk, and an atom decoder. For MambaFold, this is the base system design.

### Mamba3

Mamba3 is the main backbone hypothesis. The project uses it as the candidate replacement for the heavy residue-level Transformer blocks in a SimpleFold-style trunk. The motivation is better scaling on long chains without giving up expressive sequence modeling.

### Equilibrium Matching

Equilibrium Matching, or EQM, is the main experimental extension. It matters because it suggests a way to train a structure generator without explicit time conditioning, potentially giving more flexible iterative refinement behavior than standard flow matching.
