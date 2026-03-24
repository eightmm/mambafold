# Equilibrium Matching: Generative Modeling with Implicit Energy-Based Models

- **저자**: Runqian Wang, Yilun Du
- **소속**: MIT, Harvard University
- **발표**: arXiv:2510.02300 (2025년 10월, v3: 2025년 10월)
- **프로젝트**: https://raywang4.github.io/equilibrium_matching/

---

## 1. 핵심 요약

Equilibrium Matching (EqM)은 **time conditioning을 제거**하고, implicit energy landscape의 equilibrium gradient를 학습하는 generative framework이다.

### Flow Matching vs Equilibrium Matching

| 측면 | Flow Matching | Equilibrium Matching |
|------|---------------|---------------------|
| 학습 target | velocity x - ε | gradient (ε - x)·c(γ) |
| Time conditioning | **필수** (model sees t) | **없음** (time-invariant) |
| 학습하는 것 | t마다 다른 non-equilibrium dynamics | 단일 equilibrium gradient field |
| Sampling | ODE integration (고정 trajectory) | **Gradient descent** (유연) |
| Step size | 고정 (η = 1/N) | **자유** (다양한 η 가능) |
| 제약 조건 | ∫c(γ)dγ = 1 (valid path) | **c(1) = 0** (data에서 gradient 0) |
| Ground truth에서 | t=1에서만 수렴 | **항상 local minimum** |

## 2. Training

### Corruption Scheme
```
x_γ = γ·x + (1-γ)·ε

where:
  x = real sample
  ε ~ N(0, I)
  γ ~ U(0, 1) — interpolation factor (implicit, model sees it NOT)
```

### Training Objective
```
L_EqM = ||f(x_γ) - (ε - x)·c(γ)||²
```

- f는 x_γ만 입력으로 받음 (γ를 모름!)
- target gradient 방향: ε - x (noise → data, FM의 반대)
- c(γ): gradient magnitude function

### 핵심 제약: c(1) = 0
- γ=1일 때 x_γ = x (clean data) → target gradient = 0
- 이것이 energy landscape에서 **real data가 local minima**가 되는 조건

### Gradient Magnitude Function c(γ) 선택지

**1. Linear Decay**
```
c(γ) = 1 - γ
```

**2. Truncated Decay (최적)**
```
c(γ) = 1           if γ ≤ a
      = (1-γ)/(1-a) if γ > a

Best: a = 0.8
```

**3. Piecewise**
```
c(γ) = b - (b-1)/a · γ    if γ ≤ a
      = (1-γ)/(1-a)        if γ > a
```

**4. Gradient Multiplier λ**
- 위 함수에 전체 스케일링: c(γ) = λ · c_base(γ)
- **최적: λ = 4** (truncated decay, a=0.8)

### 최종 권장 설정
```
c(γ) = 4 · c_trunc(γ, a=0.8)
```

## 3. Theoretical Guarantees

### Statement 1: Data는 Zero Gradient
Perfect training 시, ground truth sample x^(i)에서:
```
||f(x^(i))||₂ ≈ 0
```
→ Real data에서 gradient가 0 (local minima)

### Statement 2: Local Minima는 Data
f(x̂) = 0인 모든 점 x̂에 대해:
```
P(x̂ ∈ X) ≈ 1
```
→ 모든 local minima는 training data에 속함

### Statement 3: Convergence Rate
L-smooth, bounded-below energy E에서 gradient descent:
```
min_{0≤k<K} ||f(x_k)||² ≤ 2(E(x₀) - E_inf) / (η·K)
```
→ O(1/K) convergence (smooth convex optimization의 표준 rate)

## 4. Sampling

### Gradient Descent (GD)
```
x_{k+1} = x_k - η · f(x_k)
```
단순 gradient descent. η는 자유롭게 조절 가능.

### Nesterov Accelerated Gradient (NAG-GD) — 권장
```
x_{k+1} = x_k - η · f(x_k + μ·(x_k - x_{k-1}))
```
- μ: look-ahead factor (최적: μ ≈ 0.3~0.35)
- 적은 step에서 GD 대비 큰 품질 향상

### ODE와의 관계
Forward Euler (step size h = 1/N) = gradient descent (η = 1/N)
→ ODE integration은 GD의 special case

### Adaptive Compute (핵심 장점)
```
while ||f(x_k)||₂ > g_min:
    x_{k+1} = x_k - η · f(x_k + μ·(x_k - x_{k-1}))
```
- Sample마다 다른 step 수 할당
- 쉬운 sample → 적은 step, 어려운 sample → 많은 step
- **~40% compute 절감** (동등 품질)

## 5. 구현 (매우 간단)

### Training Pseudocode
```python
def training_loss(f, x, c):
    eps = randn_like(x)
    gamma = rand()                    # implicit, model doesn't see this
    xg = (1 - gamma) * eps + gamma * x
    target = (eps - x) * c(gamma)
    loss = (f(xg) - target) ** 2
    return loss
```

### Sampling Pseudocode
```python
def generate(f, st, eta, mu, g_min):
    x = st                            # start from noise
    x_last = st
    grad = f(st)
    while norm(grad) > g_min:         # adaptive stopping
        x_last = x
        x = x - eta * grad            # gradient step
        grad = f(x + mu * (x - x_last))  # NAG lookahead
    return x
```

### FM → EqM 전환
기존 flow matching codebase에서:
1. Model에서 **timestep conditioning 제거** (t=0 고정)
2. Target을 `x - ε`에서 `(ε - x) · c(γ)`로 변경
3. Sampling을 ODE integration에서 gradient descent로 변경

## 6. Explicit Energy Learning (EqM-E)

Implicit gradient f 대신 explicit energy g를 학습하는 variant:

```
L_EqM-E = ||∇g(x_γ) - (ε - x)·c(γ)||²
```

### 두 가지 구성 방법

**Dot Product** (권장):
```
g(x) = x · f(x)
∇g(x) = f(x) + x^T · ∇f(x)
```

**Squared L2 Norm**:
```
g(x) = -½||f(x)||₂²
∇g(x) = -f(x) · ∇f(x)
```

→ 실험 결과: explicit energy는 implicit보다 성능 저하. **Implicit (기본) 권장.**

## 7. 성능

### ImageNet 256×256 (class-conditional)
| Model | Method | FID ↓ |
|-------|--------|-------|
| StyleGAN-XL | GAN | 2.30 |
| VDM++ | Diffusion | 2.12 |
| DiT-XL/2 | Diffusion | 2.27 |
| SiT-XL/2 | Flow Matching | 2.06 |
| **EqM-XL/2** | **EqM** | **1.90** |

### Scalability
- Model size, patch size, training epochs 모든 축에서 FM 대비 우수
- 동일 architecture (SiT backbone) — 차이는 오직 training/sampling 방식

### Unique Properties
1. **Partially noised input 처리 가능**: FM은 pure noise에서만 시작해야 함
2. **OOD detection**: energy 값으로 in/out-of-distribution 판별
3. **Composition**: 여러 conditional model의 gradient 합산으로 compositional generation

## 8. 우리 프로젝트에의 적용

### FM 대비 EqM의 잠재적 장점 (Protein Folding)

1. **Adaptive compute per protein**
   - 짧은/단순한 단백질 → 적은 sampling step
   - 긴/복잡한 단백질 → 많은 step
   - GPU 자원 효율화

2. **Timestep conditioning 제거**
   - 모델 구조 단순화 (adaptive layers의 t 의존성 불필요)
   - Mamba-3 block에서 conditioning 제거 → 더 깨끗한 설계

3. **유연한 step size**
   - 추론 시 step size 조절로 속도-품질 trade-off

4. **Partially refined structure**
   - 초기 coarse 예측에서 시작하여 refinement 가능
   - Iterative refinement pipeline과 자연스럽게 호환

### 구현 시 변경점 (SimpleFold FM → EqM)

```
SimpleFold (FM):
  - Model: v_θ(x_t, s, t) — timestep t 입력
  - Target: x - ε
  - Sampling: ODE integration t=0→1

MambaFold (EqM):
  - Model: f_θ(x_γ, s) — timestep 없음
  - Target: (ε - x) · c(γ)  with c(γ) = 4·c_trunc(γ, 0.8)
  - Sampling: NAG-GD with adaptive stopping
```

### LDDT Loss 호환성
```
기존: x̂(x_t) = x_t + (1-t) · v_θ(x_t, s, t)

EqM: gradient f는 noise→data 방향이므로:
     x̂(x_γ) = x_γ - (1/λ) · f_θ(x_γ, s)  (적절한 scaling 필요)
```

→ LDDT loss를 EqM에 맞게 조정 필요 (one-step estimate 방식 변경)

### 권장 실험 순서
1. **먼저 FM 기반으로 구현** (SimpleFold 방식, 검증된 방법)
2. **FM 동작 확인 후 EqM로 전환** 실험
   - Timestep conditioning 제거
   - Target/sampling 변경
   - 성능 비교
3. **Adaptive compute** 실험으로 inference 효율 확인

### 주의사항
- EqM은 ImageNet에서만 검증됨 → protein folding에서의 효과는 미검증
- LDDT loss와의 호환성 검증 필요
- c(γ) hyperparameter 탐색 필요 (image ≠ protein structure)
