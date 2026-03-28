# losses/ — Loss Functions

## `eqm.py`

### `truncated_c(gamma, a=0.8, lam=4.0)`
EqM gradient magnitude function.

```
c(γ) = lam · c_trunc(γ, a)
     = 4         if γ ≤ 0.8    (constant in noise region)
     = 4·(1-γ)/0.2 if γ > 0.8  (→ 0 as γ→1, near data manifold)
```

**Theory**: c(1) = 0 ensures data is at energy minimum (EqM invariant).

### `eqm_reconstruction_scale(gamma, a=0.8, lam=4.0)`
Computes 1-step structure reconstruction scale.

```
x̂ = x_γ - scale · f(x_γ)

Derivation:
  f(x_γ) ≈ (ε - x_clean) · c(γ)
  x_γ = γ·x_clean + (1-γ)·ε
  scale = (1-γ)/c(γ)
```

Numerically stable via `torch.where` to handle γ→1 limit.

### `eqm_loss(pred, x_clean, eps, gamma, valid_mask, a=0.8, lam=4.0)`
Main EqM training loss.

```
L_EqM = ||f(x_γ) - (ε - x_clean)·c(γ)||²
```

Applied only to valid atoms (observed + in-mask).

## `lddt.py`

### `soft_lddt_ca_loss(pred, true, ca_mask, cutoff=1.5)`
Auxiliary CA-LDDT loss for structure quality supervision.

Compares Cα distances between predicted and ground truth structures:
```
dist_pred = pairwise_dist(pred_ca)  [B, L, L]
dist_true = pairwise_dist(true_ca)
lddt = mean_over_4_thresholds(|dist_pred - dist_true| < threshold)
loss = 1 - lddt
```

Thresholds: 0.5, 1.0, 2.0, 4.0 (in normalized coords, ×10 → Å).

## Loss Combination

| Loss | Role | Weight |
|------|------|--------|
| L_EqM | Learn gradient field (theory-grounded) | 1.0 (always) |
| L_LDDT | Structure quality supervision | 1.0 (pretrain) or 1+8·ReLU(γ-0.5) (finetune) |

Finetune α ramping: higher γ → higher LDDT weight → quality focus near data.
