# losses/ — 손실 함수

## `eqm.py`

### `truncated_c(gamma, a=0.8, lam=4.0)`
EqM gradient magnitude 함수 c(γ) 계산.

```
c(γ) = lam · c_trunc(γ, a)
     = 4           if γ ≤ 0.8    (gradient 크기 일정, noise 영역)
     = 4·(1-γ)/0.2  if γ > 0.8  (→ 0 as γ→1, data manifold 근처)
```

**c(1) = 0 조건**: real data에서 gradient가 0이 되어 data가 energy local minima가 됨
(EqM Statement 1, 2의 이론적 기반)

### `eqm_reconstruction_scale(gamma, a=0.8, lam=4.0)`
1-step 구조 복원에 쓰이는 scale 계산.

```
x̂ = x_γ - scale · f(x_γ)

유도:
  f(x_γ) ≈ (ε - x_clean) · c(γ)
  x_γ = γ·x_clean + (1-γ)·ε
  → x_clean = x_γ - (1-γ)/c(γ) · f(x_γ)

  scale = (1-γ)/c(γ):
    γ ≤ 0.8: (1-γ)/4
    γ > 0.8: 0.2/4 = 0.05  (상수! γ→1에서도 안정적)
```

γ→1 시 (1-γ)/c(γ)가 0/0 형태가 될 수 있으나, c_trunc에서는 분모도 (1-γ)에 비례하므로
극한값 = (1-a)/lam = 0.05로 안정적. `torch.where`로 수치 안정성 보장.

### `eqm_loss(pred, x_clean, eps, gamma, valid_mask, a=0.8, lam=4.0)`
EqM 훈련 손실.

```
L_EqM = ||f(x_γ) - (ε - x_clean)·c(γ)||²  (valid atom에 대한 masked mean)
```

- `valid_mask` = atom_mask & observed_mask (관측 안 된 원자 제외)
- 3D 벡터이므로 `.sum(dim=-1)` 후 mask 적용

## `lddt.py` — `soft_lddt_ca_loss`

CA-LDDT 보조 손실. 1-step 복원 x̂에서 Cα 간 거리 행렬 비교.

```python
# pred_ca = x_hat[:, :, CA_ID, :]  [B, L, 3]
# true_ca = x_clean[:, :, CA_ID, :] [B, L, 3]

dist_pred = pairwise_dist(pred_ca)   # [B, L, L]
dist_true = pairwise_dist(true_ca)
dist_err  = |dist_pred - dist_true|

# cutoff=1.5 (정규화 단위, 실제 15Å)
# 4개 threshold에서 fraction 평균
lddt = mean(dist_err < {0.5, 1.0, 2.0, 4.0} × 1/coord_scale)
loss = 1 - lddt
```

**cutoff=1.5**: `COORD_SCALE=10`으로 정규화된 좌표 기준. 실제 15Å 이내 Cα 쌍에만 적용.

## 두 손실의 역할

| 손실 | 역할 | α 가중치 |
|------|------|---------|
| `L_EqM` | gradient field 학습, 이론적 기반 | 1.0 (항상) |
| `L_LDDT` | 구조 품질 직접 감독, 빠른 수렴 | pretrain: 1.0 / finetune: 1+8·ReLU(γ-0.5) |

finetune에서 α 증가: γ가 높을수록 (data에 가까울수록) LDDT 가중치 크게 → 고품질 영역 집중.
