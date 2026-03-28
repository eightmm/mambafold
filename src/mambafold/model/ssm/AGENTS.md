# model/ssm/ — Mamba-3 SSM Layers

## `mamba3.py`
**Mamba3Layer** — Wrapper around official `mamba_ssm.modules.mamba3.Mamba3`.

Key features:
- Mask support for padded sequences
- Auto-padding to chunk_size multiples
- MIMO (Multiple Input Multiple Output) mode

```python
Mamba3Layer(d_model, d_state, expand, headdim, mimo_rank)
    forward(x [B,L,D], mask [B,L]) → [B,L,D]
```

**MIMO Parameters**:
- `mimo_rank=1` → SISO (chunk_size=64)
- `mimo_rank=2` → MIMO (chunk_size=16)
- `mimo_rank=4` → MIMO (chunk_size=8)

Triton kernels require `seq_len % chunk_size == 0`, auto-padded internally.

## `bimamba3.py`
**BiMamba3Block** — Bidirectional fusion with gated merging.

```
h = RMSNorm(x)
y_f = Mamba3(forward direction)      [B, L, D]
y_b = flip → Mamba3(backward) → flip [B, L, D]

gate = sigmoid(Linear(concat([y_f, y_b])))  [B, L, D]
fused = gate * y_f + (1 - gate) * y_b      [soft gated fusion]

x = x + Linear(fused)                      [residual]
x = x + SwiGLU(RMSNorm(x))                [FFN residual]
```

**Why gated fusion**: Proteins have no N→C directionality. Gate learns which direction is more informative per position.

**_flip_by_mask**: Reverse valid positions only, keep padding at end.

**MambaStack** — Reusable N-layer BiMamba stack.
Used by: atom encoder, residue trunk, atom decoder.
