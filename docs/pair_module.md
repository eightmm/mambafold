# Pair Module — Linear Triangular Attention + Multiplicative Update

Companion to `docs/architecture.md`. Defines the pair-side computation.

References:
- SeedFold (arXiv:2512.24354v1) §3 — Linear Triangular Attention with ReLU
  feature maps, Additive/Gated variants. **Replaces softmax triangular
  attention with ϕ=ReLU; O(nd²) vs O(n²d).**
- AF2 — multiplicative triangular update (outgoing + incoming). Kept as-is.

## 1. Pair Block Composition

One `PairBlock` = three residual sub-modules.

```python
class PairBlock(nn.Module):
    def __init__(self, d_pair=192, n_heads=4):
        super().__init__()
        # Multiplicative update — same as AF2, replaced if memory tight
        self.norm_mu_out = nn.LayerNorm(d_pair)
        self.mult_update_out = TriangleMultiplicativeUpdate(d_pair, mode="outgoing")
        self.norm_mu_in  = nn.LayerNorm(d_pair)
        self.mult_update_in  = TriangleMultiplicativeUpdate(d_pair, mode="incoming")

        # Linear triangular attention (SeedFold) — replaces classical triangle attn
        self.norm_tri_start = nn.LayerNorm(d_pair)
        self.linear_tri_start = LinearTriangleAttention(
            d_pair=d_pair, n_heads=n_heads, axis="start", variant="gated",
        )
        self.norm_tri_end = nn.LayerNorm(d_pair)
        self.linear_tri_end = LinearTriangleAttention(
            d_pair=d_pair, n_heads=n_heads, axis="end", variant="gated",
        )

        # Transition FFN
        self.norm_trans = nn.LayerNorm(d_pair)
        self.transition = PairTransition(d_pair, hidden_mult=2)

    def forward(self, pair, mask):
        pair = pair + self.mult_update_out(self.norm_mu_out(pair), mask)
        pair = pair + self.mult_update_in(self.norm_mu_in(pair),  mask)
        pair = pair + self.linear_tri_start(self.norm_tri_start(pair), mask)
        pair = pair + self.linear_tri_end(self.norm_tri_end(pair),   mask)
        pair = pair + self.transition(self.norm_trans(pair))
        return pair
```

Total per block: ~6.7M params at `d_pair=192`. Stack of 6 → ~40M.

## 2. Triangle Multiplicative Update (AF2)

Standard AF2 multiplicative update — *unchanged*. Two modes (outgoing, incoming):

### Outgoing update for pair Zᵢⱼ

```text
a_ik = sigmoid(linear_a_g) ⊙ linear_a(Zᵢₖ)
b_jk = sigmoid(linear_b_g) ⊙ linear_b(Zⱼₖ)
g_ij = sigmoid(linear_g(Zᵢⱼ))
o_ij = LayerNorm(Σ_k a_ik ⊙ b_jk)
Zᵢⱼ += g_ij ⊙ linear_z(o_ij)
```

Compute: O(L³·c) where c = hidden width per head. For L=1024, c=128:
~128 GFLOPs/sample. Memory: intermediate [B,L,L,c] = ~1 GB/sample bf16.

### Incoming update

Same form but Z indices flipped — uses Zₖᵢ, Zₖⱼ instead of Zᵢₖ, Zⱼₖ.

### Implementation note
Use `einsum("bikc,bjkc->bijc", a, b)` for the core contraction. Apply mask
before contraction to zero out padding positions. Grad checkpoint these two
modules if memory tight (they hold the largest activations in the stack).

## 3. Linear Triangular Attention (SeedFold)

### 3.1 Why "Linear"
Classical triangle attention computes `softmax(QᵢKᵢᵀ + B) · Vᵢ` over every
triangle vertex, requiring full L×L attention matrix → O(L²·d) memory and
compute per axis.

SeedFold replaces softmax with a ReLU feature map ϕ, enabling associative
reordering:
```
softmax(QKᵀ)V  →  ϕ(Q) (ϕ(K)ᵀ V)   [right-product first]
```
After reordering, the inner product `ϕ(K)ᵀ V` is [d, d] not [L, L]. Total
complexity: **O(L·d²)** vs classical **O(L²·d)**. For L=1024, d_head=64:
~16 GFLOPs vs ~16 TFLOPs — *thousand-fold* compute saving plus matching
memory saving.

### 3.2 Gated Variant (default)

For pair representation Z ∈ ℝ^{B,L,L,D}, we treat each row (start-axis) and
column (end-axis) as the attention sequence.

```text
# Start-axis attention: for each "anchor" i, attend over k along axis=2 (sequence)
# Inputs:  Z ∈ [B, L_anchor, L_seq, D]
# Project to query, key, value, gate, bias:
Q = Linear(D, n_heads·d_head)(Z)                     # [B, L_anc, L_seq, h·d]
K = Linear(D, n_heads·d_head)(Z)                     # same
V = Linear(D, n_heads·d_head)(Z)                     # same
B_bias = Linear(D, n_heads)(Z_template).unsqueeze(-1) # bias per head [B, L_anc, L_seq, h, 1]

Q, K, V = relu(Q), relu(K), Z_value                  # ϕ = ReLU on Q, K
B_bias = sigmoid(B_bias)                             # ψ = sigmoid on B

# Reshape to [B, L_anc, L_seq, h, d_head]
# Aggregate along L_seq (right-product trick):
KV = einsum("bnshd, bnshk -> bnhdk", K, V)           # [B, L_anc, h, d_head, d_head]
# Each anchor i gets: Q_i · KV_i
out = einsum("bnshd, bnhdk -> bnshk", Q, KV)         # [B, L_anc, L_seq, h, d_head]
# Apply gating + concatenate heads
out = (out * B_bias).reshape(B, L_anc, L_seq, n_heads * d_head)
out = Linear(n_heads · d_head, D)(out)
# Post-processing: gated FFN (SeedFold eq. 15)
out = Linear(D, D)(sigmoid(Linear(D, D)(Z)) * LayerNorm(out))
return out
```

### 3.3 Start-axis vs End-axis

- **Start-axis** (`axis="start"`): for each pair (i, k), attend over j. Query Zᵢⱼ
  attends with keys Zᵢₖ. Captures "given anchor i, how does j relate to other k?"
- **End-axis** (`axis="end"`): swap roles. Captures "given anchor j, how does
  i relate to other k?"

Both axes used per block (AF2 convention).

### 3.4 Masking

Pair mask: `mask_ij = res_mask_i AND res_mask_j`. Padding rows/cols zeroed in
input projections so their contribution to `ϕ(K)ᵀ V` is zero. Output masked
again post-attention.

### 3.5 Memory & Compute

For L=1024, d_pair=192, n_heads=4, d_head=48:
- Q, K, V tensors: [B, L, L, 192] bf16 = 6 GB/sample (B=4 → 24 GB)
- KV intermediate: [B, L, 4, 48, 48] bf16 = 37 MB/sample (B=4 → 150 MB)
- vs classical: [B, L, L, L, 4] bf16 = 6 TB/sample (impossible)
- → Linear variant is ~1000-100,000× memory cheaper, depending on what's kept.

**Activation checkpointing**: keep block input only, recompute Q/K/V on
backward. Saves 18 GB/sample at peak. Default ON.

### 3.6 Implementation notes
- Use `torch.einsum` for clarity. Optimise with `opt_einsum` if needed.
- Add NaN guard: ReLU on Q, K can produce all-zero attention weights for
  some queries; add small ε to the denominator if normalising (Gated variant
  does not normalise, so safe).
- Triton kernel for Gated variant is an *optimisation*, not required for
  baseline correctness.

## 4. Pair Transition (FFN)

Same as AF2:

```python
class PairTransition(nn.Module):
    def __init__(self, d_pair, hidden_mult=2):
        self.lin1 = nn.Linear(d_pair, hidden_mult * d_pair)
        self.lin2 = nn.Linear(hidden_mult * d_pair, d_pair)
    def forward(self, pair):
        return self.lin2(F.relu(self.lin1(pair)))
```

`hidden_mult=2` keeps params modest at d_pair=192 (~150K per block).

## 5. Pair → Single Update

After the pair stack, reduce pair to per-residue bias:

```python
def pair_to_single(pair, res_mask):
    """
    pair: [B, L, L, d_pair]
    res_mask: [B, L]
    returns: [B, L, d_res]
    """
    mask_j = res_mask.unsqueeze(1).unsqueeze(-1).to(pair.dtype)  # [B, 1, L, 1]
    denom = mask_j.sum(dim=2).clamp(min=1)
    pair_row = (pair * mask_j).sum(dim=2) / denom                # [B, L, d_pair]
    return Linear(d_pair, d_res)(pair_row)                       # [B, L, d_res]
```

Operates on the direct all-atom model's pair tensor.

## 6. Distogram Aux Head

For aux supervision (recommended — gives pair tensor a direct learning
signal, important early in training):

```python
class DistogramHead(nn.Module):
    def __init__(self, d_pair, n_bins=64):
        self.proj = nn.Sequential(
            nn.LayerNorm(d_pair),
            nn.Linear(d_pair, n_bins),
        )
    def forward(self, pair):
        # Symmetrise pair to ensure d(i,j) == d(j,i)
        pair_sym = (pair + pair.transpose(1, 2)) / 2
        return self.proj(pair_sym)                                # [B, L, L, n_bins]
```

Distogram targets binned in [2 Å, 22 Å], 64 bins. Cross-entropy loss.

## 7. Hyperparameter Defaults

| Param | Value | Notes |
|---|---|---|
| `d_pair` | **192** | revisit after I1 memory profile |
| `n_pair_blocks` | **6** | can grow to 8 if memory permits |
| `n_heads` (LinearTriAttn) | **4** | `d_head = d_pair / n_heads / 4 = 12`? Actually use 48 — set head dim explicit. |
| `d_head` (LinearTriAttn) | **48** | per-head, gives 4 × 48 = 192 |
| `hidden_mult` (Transition) | **2** | |
| LinearTri variant | **Gated** | Additive fallback if Triton trouble |
| Distogram bins | **64** | 2–22 Å range |
| Mult update grad ckpt | **ON** | required for L=1024 |

## 8. Param Count (verified target ~40M for pair stack)

Per PairBlock at d_pair=192, n_heads=4, d_head=48:

| Sub-module | Params | Note |
|---|---|---|
| 2 × MultUpdate (out + in) | ~1.5M each = 3M | Linear in/out + gating + ouput |
| 2 × LinearTriAttn (start + end) | ~1.2M each = 2.4M | Q/K/V/bias/gating projections |
| Transition FFN | ~150K | |
| LayerNorms (×5) | ~5K | |
| **Per block** | **~5.6M** | |
| **× 6 blocks** | **~34M** | |
| Pair init (single + ESM + relpos) | ~1M | |
| Pair → single | ~0.2M | |
| Distogram head | ~12K | |
| **Pair-side total** | **~35M** | |

Slight under target (~40M) — capacity to grow `d_pair` to 224 if I1 profile
allows.

## 9. Open Implementation Questions

Answered as of 2026-05-17:

| Question | Default | Reason |
|---|---|---|
| Variant: Additive vs Gated | **Gated** | matches SeedFold main config, stronger expressivity |
| Mask before or after Q/K/V? | **Before** | zero padding rows before projection — keeps ReLU(0)=0 propagation clean |
| `n_heads`: more, smaller-d? | **4 heads × 48** | minimal head count keeps KV intermediate small |
| Bias term B in attention | **Yes** | per-head Linear(d_pair → 1), sigmoid'd — gating |
| Triton kernel | **No (PyTorch first)** | correctness baseline; optimise later |
| Activation checkpoint | **Multiplicative update only** | dominant memory, attention is already light |
