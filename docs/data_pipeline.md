# 데이터 파이프라인

현재 기준은 `configs/stage1.yaml`의 RCSB `.npz` + per-chain ESM3 `.npy` 파이프라인이다.

## 전체 흐름

```mermaid
flowchart TB
    R[RCSB mmCIF] --> C[batch_convert_cif.py<br/>Boltz-style npz]
    C --> N[data/rcsb/*.npz]
    N --> S[make_val_split.py<br/>train.txt / val.txt]
    N --> E[precompute_esm.py<br/>data/rcsb_esm/*_ch*.npy]
    N & S & E --> D[RCSBDataset]
    D --> T[center_and_scale<br/>random_so3_augment]
    T --> P[ProteinCollator<br/>FM corrupt x_t]
    P --> B[ProteinBatch]
    B --> M[MambaFold]
```

## Stored Files

| Path | Meaning |
|---|---|
| `data/rcsb/**/*.npz` | Boltz-style structure records |
| `data/rcsb_esm/*_ch*.npy` | per-original-chain ESM3 embeddings, shape `[L_chain, 1536]` |
| `data/splits/train.txt` | training file list |
| `data/splits/val.txt` | validation file list |
| `data/splits/val_casp.txt` | CASP-style holdout list |

## RCSBDataset

`RCSBDataset` loads all protein chains in an entry, keeps standard amino acids, and flattens chains into one residue axis while preserving:

| Field | Meaning |
|---|---|
| `chain_id` | 0-based chain index in the loaded example |
| `entity_id` | shared id for chains with identical sequences |
| `sym_id` | copy number within an entity |
| `res_seq_nums` | residue index within the original chain |
| `is_nterm` / `is_cterm` | original-chain termini |

If `len(entries) > max_length`, a random contiguous crop over the flattened chain sequence is used. Current `max_length` is `1024`.

## ESM3 Loading

For each residue, the dataset looks up:

```text
data/rcsb_esm/<npz_stem>_ch<origin_chain>.npy
```

Rows are reassembled according to the crop and returned as `ProteinExample.esm`.
The loader verifies that `esm_dir` exists and contains `.npy` files. If
`use_plm=true`, missing batch ESM is a hard model error, so run
`scripts/precompute_esm.py` until it reports `0 files to write` before training.
Very long chains are limited to the residue range covered by their precomputed
ESM rows, so random training crops do not silently fall into an unconditioned
tail region.

## Runtime Transforms

### `center_and_scale(example)`

Centers on the system-level valid-atom centroid and divides by `COORD_SCALE=10.0`.

### `random_so3_augment(example)`

Applies one SO(3) rotation to the whole system, preserving inter-chain geometry.

### `flow_corrupt(coords, atom_mask, schedule="uniform")`

```text
x_t = t * x_clean + (1 - t) * eps
t ~ Uniform(0, 1)
eps is centered over valid atoms
```

The current run uses `t_schedule: uniform` and `copies_per_protein: 1`.

## ProteinBatch Fields

| Field | Shape | Description |
|---|---|---|
| `res_type` | `B, L` | amino-acid index |
| `res_seq_nums` | `B, L` | chain-local residue index |
| `atom_type` | `B, L, A` | atom name id |
| `pair_type` | `B, L, A` | residue-atom pair id |
| `res_mask` | `B, L` | valid residue mask |
| `atom_mask` | `B, L, A` | canonical atom exists |
| `valid_mask` | `B, L, A` | canonical and observed atom, loss only |
| `ca_mask` | `B, L` | observed C-alpha mask |
| `chain_id`, `entity_id`, `sym_id` | `B, L` | chain-aware features |
| `is_nterm`, `is_cterm` | `B, L` | original-chain terminus flags |
| `x_clean` | `B, L, A, 3` | normalized ground-truth coords |
| `x_t` | `B, L, A, 3` | corrupted coords |
| `eps` | `B, L, A, 3` | sampled noise |
| `t` | `B, 1, 1, 1` | interpolation time |
| `esm` | `B, L, 1536` or `None` | ESM3 conditioning |

## ESM Precompute

```bash
PYTHONPATH=src uv run python scripts/precompute_esm.py \
  --data_dir data/rcsb \
  --out_dir data/rcsb_esm
```

The script scans `.npz` files, deduplicates identical sequences, runs ESM once per unique sequence, and writes per-chain `.npy` outputs.
