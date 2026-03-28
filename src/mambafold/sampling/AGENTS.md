# sampling/ — Samplers

## `sampler.py`

### `EqMNAGSampler`
Nesterov Accelerated Gradient sampler for structure generation from trained model.
Based on paper Algorithm 2 (Eq.9, NAG Form 1).

```python
sampler = EqMNAGSampler(model, eta=0.1, mu=0.3, g_min=5e-3,
                        max_steps=128, max_disp=0.5)
coords, n_steps = sampler.sample(batch)  # [B, L, 14, 3] in Å
```

**Algorithm**:
```
x_0 ~ N(0, I)  [normalized, translation-centered]
for k = 0..max_steps:
    look = x if k==0 else x + mu*(x - x_prev)  [Nesterov lookahead]
    grad = model(batch.with_coords(look))       [f(look)]

    grad_rms = sqrt(||grad||² / n_atoms / 3)
    if grad_rms < g_min: break                  [adaptive stopping]

    step = clamp(eta * grad, -max_disp, max_disp)  [per-atom norm clamp]
    x ← x - step
    x ← remove_translation(x)                   [SE(3) invariance]
return x * COORD_SCALE  [back to Ångströms]
```

**Parameters**:

| Param | Default | Meaning |
|-------|---------|---------|
| eta | 0.1 | Step size |
| mu | 0.3 | NAG momentum |
| g_min | 5e-3 | Stopping gradient RMS threshold |
| max_steps | 128 | Max iterations |
| max_disp | 0.5 | Per-step norm clamp (normalized coords) |

### `EqMEulerSampler`
Euler integration sampler (used in overfit validation).

γ-adaptive step: descends from γ=0 (pure noise) to γ=1 (clean data).
Used for quick validation and visualization.

**Key difference from NAG**:
- Gamma-scheduled step size (adaptive per noise level)
- Fixed number of steps (N=50)
- Faster for validation, slower for quality
