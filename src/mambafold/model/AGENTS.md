# model/ — Model Architecture

## Files

### `mambafold.py`
**MambaFoldEqM** — end-to-end forward pass.

```
x_gamma [B, L, 14, 3]
  ↓ AtomFeatureEmbedder          [B, L, 14, d_atom]
  ↓ MambaStack (atom encoder)    per-residue
  ↓ group_atoms_to_residues      [B, L, d_atom] masked avg pool
  ↓ [concat ESM if use_plm]      [B, L, d_res]
  ↓ MambaStack (residue trunk)   BiMamba-3, heavy
  ↓ ResidueToAtomBroadcast       [B, L, 14, d_atom] + skip
  ↓ MambaStack (atom decoder)    per-residue
  ↓ GradientHead                 [B, L, 14, 3]
```

**Key principle**: Model is time-unconditional (does not receive γ as input). γ is in the batch but unused in forward.

**Parameters**:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| d_atom | 256 | Atom token dimension |
| d_res | 256 | Residue token dimension |
| d_plm | 1024 | PLM embedding dimension |
| d_state | 32 | Mamba SSM state size |
| mimo_rank | 2 | MIMO rank |
| headdim | 64 | Mamba head dimension |
| n_atom_enc | 2 | Atom encoder layers |
| n_trunk | 6 | Residue trunk layers |
| n_atom_dec | 2 | Atom decoder layers |

### `embeddings.py`
**SequenceFourierEmbedder**: Positional encoding for residue positions (if d_res_pos > 0).

**AtomFeatureEmbedder**: Combines four embeddings:
1. Residue type embedding (21 AAs) → broadcast to atoms
2. Atom type embedding (N, CA, C, O, CB, ...) → per-atom
3. Pair embedding (residue, atom) type → per-atom
4. Coordinate Fourier embedding → per-atom

Output: Linear projection → `[B, L, 14, d_atom]`

### `blocks.py`
**RMSNorm**: Lightweight normalization (scale only, no bias).

**SwiGLU**: FFN variant: `w2(silu(w1(x)) * w3(x))`, more efficient than standard FFN.

### `grouping.py`
**group_atoms_to_residues**: Masked average pool atoms → residues.
```
[B, L, 14, D] → [B, L, D]
```

**ResidueToAtomBroadcast**: Project residue features to atoms + broadcast + skip connection.
```
[B, L, d_res] → [B, L, 14, d_atom]
```

### `atom_encoder.py`
**AtomEncoder**: Lightweight attention over atoms (per-residue).
- Input: `[B*L, 14, d_atom]`
- Output: `[B*L, 14, d_atom]`
- Uses MambaStack with bidirectional processing

### `atom_decoder.py`
**AtomDecoder**: Decoder with gradient head.
- Input: `[B*L, 14, d_atom]`
- MambaStack (BiMamba-3)
- **GradientHead**: LayerNorm → Linear → GELU → Linear(d_atom//2, 3)
- Output: `[B, L, 14, 3]` (gradient field)

### `residue_trunk.py`
**ResidueTrunk**: Heavy BiMamba-3 stack for long-range residue interactions.
- Input: `[B, L, d_res]`
- N BiMamba-3 blocks with SwiGLU
- Output: `[B, L, d_res]`

## ssm/ Submodule

See `ssm/AGENTS.md` for Mamba-3 and BiMamba details.
