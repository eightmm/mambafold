# MambaFold

MambaFold is a single-chain protein structure prediction project built around a linear-complexity state space backbone. The goal is to keep the overall folding pipeline close to recent all-atom generative models while replacing the expensive Transformer trunk with Mamba-style sequence modeling.

## Core Idea

This project combines three main ingredients:

- **SimpleFold** provides the overall folding recipe: an all-atom generative pipeline with an atom encoder, a residue-level trunk, and an atom decoder.
- **Mamba3** provides the sequence backbone: a state space model intended to scale better than quadratic attention on long protein chains.
- **Equilibrium Matching (EQM)** is an experimental alternative to standard flow matching, aimed at learning an energy-shaped update field without explicit time conditioning.

In short, the working hypothesis is:

1. Keep the strong structural inductive bias of the SimpleFold-style atom-to-residue-to-atom pipeline.
2. Replace the heavy Transformer trunk with Mamba3 blocks for better length scaling.
3. Compare standard flow matching against EQM-style training and sampling once the base model is stable.

## Model Overview

The current design follows this high-level path:

1. Encode noisy all-atom coordinates with a lightweight atom encoder.
2. Pool atom features into residue tokens.
3. Run a residue trunk based on Mamba3-style SSM blocks.
4. Broadcast residue features back to atoms.
5. Decode atom-wise coordinate updates or velocity predictions.

This keeps the architecture close to modern structure generators while making the longest sequence-processing stage cheaper than full attention.

## Why These Papers Matter

### SimpleFold

SimpleFold is the architectural template for the project. Its main relevance here is not just that it predicts structures, but that it does so with a clean decomposition between local atom geometry and residue-level global reasoning. That split maps well onto protein folding, where atom identity and residue context both matter.

The project borrows the following ideas from SimpleFold:

- all-atom coordinate generation rather than a coarse backbone-only target
- a staged atom encoder / residue trunk / atom decoder pipeline
- frozen protein language model conditioning
- structure-aware training losses such as geometric regression plus LDDT-style quality signals

### Mamba3

Mamba3 is the backbone replacement. In a standard SimpleFold-like setup, the residue trunk is the most obvious place where attention cost grows with sequence length. Mamba3 is attractive because it keeps global sequence processing but avoids the full quadratic attention pattern.

For this project, Mamba3 matters in three ways:

- it offers a sequence model that should scale more gracefully to longer proteins
- it preserves ordered, causal-style sequence processing in a way that fits residue chains naturally
- it gives a concrete research direction for testing whether modern SSMs can substitute for Transformer trunks in folding models

### Equilibrium Matching (EQM)

EQM is not the first target, but it is an important extension path. Flow matching gives a practical baseline for structure generation, but it requires explicit time conditioning and a prescribed interpolation path. EQM instead learns update directions connected to an equilibrium or energy landscape view.

If EQM works well in this setting, it could be useful because:

- inference can be framed as iterative descent rather than a fixed time-discretized flow
- compute can be adjusted dynamically at sampling time
- partially noised or partially initialized structures may be easier to refine

The likely execution order is to get a stable flow-matching model first, then introduce EQM as an ablation or second training regime.

## Repository Layout

```text
folding/
├── README.md
├── pyproject.toml
├── data/                   # Protein data (overfit and RCSB symlink)
│   ├── overfit/            # Small .pt files for validation
│   └── rcsb/               # Symlink to Boltz-preprocessed RCSB structures (.npz)
├── configs/                # Configuration files
├── docs/                   # Notes, plans, and paper summaries
├── scripts/                # Training, evaluation, and SLURM helpers
├── src/mambafold/          # Main package
│   ├── data/               # Dataset, collation, transforms, types
│   ├── losses/             # Training losses
│   ├── model/              # Encoders, trunk, decoder, SSM blocks
│   ├── sampling/           # Sampling utilities
│   ├── train/              # Training engine and trainer
│   └── utils/              # Geometry and support utilities
└── tests/                  # Unit tests
```

## Dataset

The project supports two protein structure formats, auto-detected by directory content:

### AFDBDataset (Legacy)

Reads `.pt` files from the overfit directory. Each file stores residue-level metadata and atom-level geometry:

| Key | Type | Description |
|-----|------|-------------|
| `res_names` | `list[str]` | Residue names such as `MET`, `VAL`, `LEU` |
| `res_seq_nums` | `list[int]` | Residue sequence numbers |
| `res_ins_codes` | `list[str]` | Insertion codes |
| `atom_names` | `list[list[str]]` | Atom names per residue |
| `atom_nums` | `list[list[int]]` | Atom serial numbers |
| `coords` | `list[list[list[float]]]` | 3D coordinates for each atom |
| `is_observed` | `list[list[bool]]` | Observation masks |

### RCSBDataset (Main)

Reads Boltz-style `.npz` files from `data/rcsb/` (symlink to `rcsb_processed_targets/structures`). Each `.npz` contains structured arrays:

| Field | Structure | Description |
|-------|-----------|-------------|
| `residues` | Array of records | `name` (residue type), `res_type`, `atom_idx`, `atom_num`, `is_standard`, `is_present` |
| `atoms` | Array of records | `name`, `coords` (x, y, z), `is_present` |
| `chains` | Array of records | `mol_type`, `res_idx`, `res_num` |

Atoms are stored in canonical order per residue type, with atom names recovered positionally without decoding. Only protein chains (`mol_type == 0`) and standard amino acids are used.

## Setup

The project uses `uv` as the package manager:

```bash
uv sync
```

Python should be executed via `uv run python`:

```bash
uv run python scripts/train.py --config configs/train_base.yaml
```

## Training

### Single GPU

```bash
uv run python -u scripts/train.py --config configs/train_base.yaml
```

### Multi-GPU DDP (auto-detected from SLURM_GPUS_ON_NODE)

```bash
sbatch scripts/slurm/train_6000ada.sh
```

Or manually with torchrun:

```bash
uv run torchrun --nproc_per_node=4 scripts/train.py --config configs/train_base.yaml
```

### Resume training

```bash
uv run torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/train_base.yaml \
    --resume outputs/train/run1/ckpt_latest.pt
```

## Overfit Validation

Quick validation on a small dataset to verify the model works:

```bash
sbatch scripts/slurm/overfit_test.sh
```

Or directly on a GPU node:

```bash
uv run python -u scripts/overfit.py --config configs/overfit.yaml
```

### Configuration

Key model hyperparameters in `configs/overfit.yaml`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `d_atom` | 384 | Atom feature dimension |
| `d_res` | 384 | Residue feature dimension |
| `d_state` | 64 | SSM state dimension |
| `mimo_rank` | 4 | MIMO decomposition rank |
| `headdim` | 64 | Attention head dimension |
| `n_trunk` | 8 | Number of trunk layers |
| `max_length` | 256 | Maximum sequence length |

## Architecture

The model is composed of the following core modules:

- **`model/bimamba3.py`** — BiMamba3Block and MambaStack: state space model blocks for sequence modeling
- **`model/embeddings.py`** — AtomFeatureEmbedder, SequenceFourierEmbedder: feature embeddings for atoms and residues
- **`model/mambafold.py`** — MambaFoldEqM: main model integrating encoder, trunk, and decoder

## Inference / Export

Export a trained checkpoint to inference format:

```bash
uv run python scripts/export_inference.py \
    --ckpt outputs/train/run1/ckpt_latest.pt \
    --data_dir data/rcsb \
    --out outputs/train/run1/inference.npz
```

## Additional Reading

- [Project docs](/home/jaemin/project/protein/folding/docs/README.md)
- [Paper overview](/home/jaemin/project/protein/folding/docs/papers/OVERVIEW.md)
