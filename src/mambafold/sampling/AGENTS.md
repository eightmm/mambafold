# sampling/ — EqM 샘플러

## `sampler.py` — EqMNAGSampler

완전 학습된 모델에서 단백질 구조를 생성하는 NAG gradient descent 샘플러.
논문 Algorithm 2 (Eq.9, NAG Form 1) 구현.

```python
EqMNAGSampler(model, eta=0.1, mu=0.3, g_min=5e-3, max_steps=128, max_disp=0.5)
    .sample(batch) → (coords [B,L,A,3] in Å, n_steps)
```

### 알고리즘

```
x_0 ~ N(0, I)  (pure noise, translation removed)
x_prev = x_0

for k in range(max_steps):
    look = x if k==0 else x + mu*(x - x_prev)   # Nesterov lookahead
    grad = model(batch.with_coords(look))         # f(lookahead)

    grad_rms = sqrt(||grad||² / n_valid_atoms / 3)
    if grad_rms < g_min: break                   # adaptive stopping

    step = clamp(eta * grad, -max_disp, max_disp)
    x_next = x - step                             # NAG Form 1: step from x_k
    x_next = remove_translation(x_next)           # SE(3) invariance 유지

return x * COORD_SCALE  # → Angstrom
```

### 파라미터

| 파라미터 | 기본값 | 의미 |
|---------|-------|------|
| eta | 0.1 | step size (논문: 0.0017 for 250 steps) |
| mu | 0.3 | NAG momentum (논문 최적값) |
| g_min | 5e-3 | adaptive stopping threshold (gradient RMS) |
| max_steps | 128 | 최대 반복 수 |
| max_disp | 0.5 | 정규화 좌표 기준 step 크기 clamp (=5Å) |

### overfit.py의 sampler와 차이

`overfit.py`의 `sample_eqm_euler` / `sample_eqm_nag`는 **검증/시각화용**:
- γ를 0→0.99로 스윕하며 `eqm_reconstruction_scale(γ)`로 step size 계산
- γ-adaptive step size → 완벽 모델에서 linear interpolation 경로 보장
- N=50 steps로 충분

`EqMNAGSampler`는 **프로덕션용**:
- 고정 η, adaptive compute (gradient norm 기준 조기 종료)
- batch 단위 처리 가능 (B>1)
- `batch.with_coords()` 로 x_gamma만 교체

### 주의사항

- `remove_translation`: 매 step 후 centroid 제거. EqM gradient가 translation-equivariant이므로
  누적 translation drift 방지.
- `max_disp` clamp: 큰 gradient에서 발산 방지 (특히 초기 noise 근처)
- μ=0.3 vs μ=0.35: 논문 Table 2는 μ=0.35. 두 값 모두 실용적 범위. 단백질 folding에서
  최적값은 추가 실험 필요.
