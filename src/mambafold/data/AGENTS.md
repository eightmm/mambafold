# data/ — Data Pipeline

## Files

### `constants.py`
Vocabulary definitions for protein structures. Referenced by all modules.

```python
NUM_AA_TYPES = 21              # 20 standard AAs + UNK
MAX_ATOMS_PER_RES = 14         # atom14 slots (heavy atoms only)
CA_ATOM_ID = 1                 # Cα = slot 1 (N=0, CA=1, C=2, O=3)
COORD_SCALE = 10.0             # Scale: Å → normalized (÷10)
NUM_ATOM_TYPES                 # Unique atom names (N, CA, C, O, CB, etc.)
NUM_PAIR_TYPES                 # Unique (residue, atom) pair types
PAIR_PAD_ID                    # Padding value for empty slots
```

### `types.py`
Data containers (dataclasses).

**ProteinExample** (single protein):
- `res_type: [L]` — AA type (0-20)
- `atom_type: [L, 14]` — atom slot index
- `coords: [L, 14, 3]` — 3D coordinates (Å)
- `atom_mask: [L, 14]` — valid atoms
- `observed_mask: [L, 14]` — experimentally observed

**ProteinBatch** (batched, with EqM corruption):
- `x_clean: [B, L, 14, 3]` — normalized ground truth
- `x_gamma: [B, L, 14, 3]` — corrupted coords (model input)
- `eps: [B, L, 14, 3]` — Gaussian noise
- `gamma: [B, 1, 1, 1]` — noise level (0 to 1)
- `esm: [B, L, d_esm]` — ESM embeddings (optional)
- `res_mask: [B, L]` — residue validity
- `atom_mask: [B, L, 14]` — atom validity
- `ca_mask: [B, L]` — Cα positions

Method: `batch.with_coords(new_x)` — update x_gamma only (for sampling loops)

### `dataset.py`
Load AFDB .pt files (pre-processed ProteinExample or dict format).

```python
AFDBDataset(data_dir, max_length=128)
    - Filter to standard 20 AAs (skip non-standard)
    - Map to canonical atom14 slots
    - Skip on load failure (handled by collate)
```

### `transforms.py`
**center_and_scale(example)**:
- Remove centroid (translation equivariance)
- Divide by COORD_SCALE (10.0)

**EqM corruption** (applied in training loop):
```python
eps = torch.randn_like(x_clean)
x_gamma = gamma * x_clean + (1 - gamma) * eps
```

### `collate.py`
Batch padding and masking. `ProteinExample` list → `ProteinBatch`.

**ProteinCollator**: Handles variable-length proteins, pads to max length.

**LengthBucketSampler**: Groups proteins by length to minimize padding waste.

### `loader.py`
**inf_loader(dataset, ...)**: Infinite DataLoader wrapper for training.

### `esm.py`
**EvolutionaryScalePLM**: ESM3/ESMc protein language model integration.
- Provides `[B, L, d_esm]` embeddings
- Disabled by default (`use_plm=False`)
- Optional for full training runs

## Data Flow

```
afdb_data/train/*.pt
  ↓ AFDBDataset.__getitem__
  ↓ ProteinExample
  ↓ center_and_scale (normalize coords)
  ↓ ProteinCollator (batch, pad, mask)
  ↓ EqM corruption (gamma * x + (1-gamma) * eps)
  ↓ ProteinBatch
  ↓ model(batch)
  ↓ gradient prediction [B, L, 14, 3]
```

## Key Points

- **valid_mask** = atom_mask ∧ observed_mask (loss uses this)
- **ca_mask** = positions with valid Cα (for LDDT metric)
- **atom_mask** = False for missing atoms (e.g., GLY has no CB)
