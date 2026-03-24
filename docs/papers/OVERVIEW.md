# Paper Overview

## Project Goal

The goal of MambaFold is to build a single-chain protein structure prediction model that combines:

- a **SimpleFold-style all-atom generative pipeline**
- a **Mamba3-based residue backbone**
- and, eventually, **Equilibrium Matching (EQM)** as an alternative to standard flow matching

The project is not a direct reproduction of any one paper. It is a synthesis: SimpleFold provides the pipeline template, Mamba3 provides the sequence-modeling backbone, and EQM provides an experimental generative objective.

## Roles of the Three Papers

| Paper | Role in MambaFold | Why it matters |
|------|--------------------|----------------|
| **SimpleFold** | End-to-end folding pipeline | Defines the architecture split, supervision signals, and all-atom prediction target |
| **Mamba3** | Backbone architecture | Replaces the Transformer-style residue trunk with a modern state space model |
| **Equilibrium Matching** | Alternative generative framework | Provides a path beyond explicit time-conditioned flow matching |

## Design Thesis

The main design thesis is that the residue trunk should be the place where we gain most from improved sequence scaling. Proteins can be long, and residue-level global reasoning is typically the most expensive part of the model. A SimpleFold-like pipeline gives a strong structural template, but its central sequence-processing block can be reconsidered.

This leads to the following plan:

1. Start from the SimpleFold decomposition of the task.
2. Replace the heavy residue trunk with Mamba3 blocks.
3. Keep flow matching as the first stable training target.
4. Introduce EQM later as an ablation or second-generation training objective.

## 1. SimpleFold: The Pipeline Blueprint

SimpleFold is the paper that most directly informs the model layout. It matters because it cleanly separates:

- **local atom-level geometric processing**
- **global residue-level contextual reasoning**
- **all-atom coordinate decoding**

That decomposition fits this project well. Protein folding is not purely sequence modeling and not purely geometry. The atom encoder handles fine local structure and noisy coordinates. The residue trunk handles long-range sequence-scale interactions. The atom decoder maps global residue context back to atom-level updates.

The main elements borrowed from SimpleFold are:

- an **atom encoder -> residue trunk -> atom decoder** pipeline
- **all-atom structure generation** rather than only backbone prediction
- **language-model conditioning** from a frozen protein LM
- **structure-aware training**, combining coordinate/velocity supervision with quality-oriented losses such as LDDT

SimpleFold is therefore the blueprint for what the folding system should do.

## 2. Mamba3: The Backbone Replacement

Mamba3 is the paper that most strongly changes the internals of the architecture. In a SimpleFold-like design, the residue trunk is where attention cost becomes significant as sequences grow. Mamba3 is appealing because it offers a modern state space formulation intended to preserve strong sequence modeling while scaling better than full attention.

For MambaFold, Mamba3 is not just a generic efficiency swap. It is the core research question:

- Can a strong SSM replace the Transformer trunk in a structure generator?
- Does linear or sub-quadratic sequence processing help on longer proteins?
- Can residue-level global reasoning remain expressive enough without standard attention blocks?

The expected benefits are:

- better scaling with protein length
- a cleaner memory profile for long chains
- a direct test of whether recent SSM advances transfer to structural biology workloads

In practice, the residue trunk is the most natural insertion point for Mamba3 because:

- residue tokens are more compact than atom tokens
- long-range context matters strongly at the residue level
- this preserves the rest of the SimpleFold pipeline with minimal disruption

## 3. Equilibrium Matching: The Experimental Objective

EQM is the most exploratory part of the project. It is not required to build the first working model, but it is the most interesting extension once the base system is stable.

Standard flow matching is attractive because it is practical and already tied to successful generative structure models. But it also requires:

- explicit time conditioning
- a chosen noising or interpolation path
- a fixed notion of how generation is parameterized over time

EQM offers a different view. Instead of learning a time-indexed velocity field, the model learns update directions linked to an equilibrium or energy landscape. Sampling can then be interpreted more like iterative descent or refinement.

Why this may matter for proteins:

- protein refinement is naturally iterative
- adaptive compute may be useful when some targets are easy and others are hard
- partially initialized structures could be refined without forcing a strict flow-time interpretation

That said, EQM is best treated as a second-stage research question. The pragmatic plan is:

1. build the SimpleFold-style system with a Mamba3 trunk
2. train it with standard flow matching
3. benchmark stability and structural quality
4. only then compare against an EQM formulation

## Architecture Sketch

```text
Input:
  - amino acid sequence
  - noisy all-atom coordinates
  - timestep t for flow matching, or no timestep for EQM

Frozen protein LM
  -> sequence embeddings

Atom encoder
  -> local atom features from noisy coordinates and residue context

Grouping / pooling
  -> residue tokens

Residue trunk
  -> Mamba3-style SSM blocks process long-range sequence context

Ungrouping / broadcast
  -> residue context returned to atoms

Atom decoder
  -> atom-wise coordinate updates or velocity predictions
```

## Training Strategy

| Component | Baseline plan | Experimental path |
|----------|----------------|-------------------|
| Generative objective | Flow matching | Equilibrium Matching |
| Backbone | Mamba3 residue trunk | Variants and ablations |
| Sequence conditioning | Frozen protein LM | Same |
| Structure target | Full-atom coordinates | Same |
| Sampling | Standard flow integration | Iterative EQM refinement |

## Practical Reading Order

- [SimpleFold summary](./simplefold_summary.md): read first for the pipeline logic.
- [Mamba3 summary](./mamba3_summary.md): read second for the backbone replacement.
- [Equilibrium Matching summary](./equilibrium_matching_summary.md): read third for the experimental extension.
