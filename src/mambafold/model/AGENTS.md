# Model Rules

Active model package: `fold/`.

Key files:

- `fold/stage1_ca.py::MambaFoldStage1` — C-alpha flow matching with pair stack.
- `fold/stage2_atom.py::MambaFoldStage2` — all-atom conditional refiner.
- `fold/two_stage.py::TwoStageMambaFold` — Stage 1/Stage 2 wrapper.
- `bimamba3.py` — shared Mamba stack implementation.
- `embeddings.py` — shared coordinate/sequence embedders.

Do not reintroduce versioned model packages. This repo tracks one active single-chain architecture.
