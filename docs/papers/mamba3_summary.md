# Mamba-3: Improved Sequence Modeling using State Space Principles

- **저자**: Aakash Lahoti, Kevin Y. Li, Berlin Chen, Caitlin Wang, Aviv Bick, J. Zico Kolter, Tri Dao, Albert Gu
- **소속**: Carnegie Mellon University, Princeton University, Together AI, Cartesia AI
- **발표**: ICLR 2026 (arXiv:2603.15569, 2026년 3월)
- **코드**: https://github.com/state-spaces/mamba

---

## 1. 핵심 요약

Mamba-3는 sub-quadratic sequence model의 **quality, capability, efficiency** 세 축을 동시에 개선한 SSM이다.
세 가지 핵심 혁신:

1. **Exponential-Trapezoidal Discretization** — 더 정밀한 recurrence (2차 정확도)
2. **Complex-valued SSM** — state tracking 능력 회복 (data-dependent RoPE)
3. **MIMO (Multi-Input Multi-Output)** — inference 시 hardware 효율 극대화

## 2. 배경: SSM Recurrence

### Continuous-time SSM
```
ḣ(t) = A(t)h(t) + B(t)x(t)
y(t) = C(t)^T h(t)
```

### Mamba-2 (Exponential-Euler) Discretization
```
h_t = α_t · h_{t-1} + γ_t · B_t · x_t
y_t = C_t^T · h_t

where:
  α_t = exp(Δ_t · A_t)    ← state decay (0~1)
  γ_t = Δ_t                ← step size
```

- A_t는 scalar × identity (A_t = a_t · I)로 단순화 → matmul 가능
- Δ_t, B_t, C_t는 data-dependent (input에서 projection)
- State Space Duality (SSD): recurrence ↔ parallel matrix form 변환 가능

### SSD Parallel Form
```
Y = (L ⊙ C·B^T) · X

where L ∈ R^{T×T} is structured mask (decay 기반)
```

## 3. 혁신 1: Exponential-Trapezoidal Discretization

### 문제
Mamba-1/2의 Euler discretization은 1차 근사 (error O(Δ²))

### 해결: Generalized Trapezoidal Rule
```
h_t = α_t · h_{t-1} + β_t · B_{t-1} · x_{t-1} + γ_t · B_t · x_t

where:
  α_t = exp(Δ_t · A_t)
  β_t = (1 - λ_t) · Δ_t · exp(Δ_t · A_t)
  γ_t = λ_t · Δ_t
  λ_t ∈ [0, 1] — data-dependent scalar
```

### 핵심 포인트
- **3-term recurrence**: 이전 시점(t-1)과 현재 시점(t) 모두 사용
- **2차 정확도**: error O(Δ³), Euler의 O(Δ²) 대비 개선
- **Implicit convolution**: state-input `B_t·x_t`에 width-2 data-dependent convolution 적용
- λ_t = 1이면 Mamba-2 (Euler), λ_t = 0.5이면 classical trapezoid
- **Short causal convolution 제거 가능**: bias + trapezoidal이 대체

### SSD 확장
Mamba-3의 structured mask는 **decay matrix × 2-band convolutional matrix**의 곱으로 분해됨:
```
L = [1-semiseparable decay matrix] · [2-band matrix with β, γ weights]
```

## 4. 혁신 2: Complex-Valued SSM

### 문제
Real scalar diagonal A_t로는 **rotation dynamics** 표현 불가 → parity 등 state tracking 실패

### 해결: Complex SSM
```
ḣ(t) = Diag(A(t) + iθ(t)) · h(t) + (B(t) + iB̂(t)) · x(t)
y(t) = Re((C(t) + iĈ(t))^T · h(t))

where h(t) ∈ C^{N/2}, θ(t) ∈ R^{N/2}
```

### Real SSM 등가 변환 (Proposition 2)
Complex SSM (state dim N/2) = Real SSM (state dim N) with block-diagonal rotation:
```
h_t = exp(Δ_t·A_t) · R_t · h_{t-1} + Δ_t · B_t · x_t

R_t = BlockDiag({R(Δ_t·θ_t[i])})  ← 2×2 rotation matrices
R(θ) = [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]
```

### Data-Dependent RoPE 등가 (Proposition 3)
```
h_t = exp(Δ_t·A_t) · h_{t-1} + Δ_t · (∏_{i=0}^{t} R_i)^T · B_t · x_t
y_t = ((∏_{i=0}^{t} R_i)^T · C_t)^T · h_t
```

- 일반 RoPE: 고정 주파수 θ[i] = 10000^{-2i/N}
- Mamba-3 RoPE: **data-dependent** 회전각 Δ_t·θ_t
- B, C (= K, Q in attention analogy)에 rotary embedding 적용
- "RoPE trick"으로 효율적 계산 (Su et al. 2023)

### 결과
- Parity: **100%** (Mamba-2: 0.9%)
- Modular Arithmetic: **98.5%** (Mamba-2: 47.8%)

## 5. 혁신 3: MIMO (Multi-Input Multi-Output)

### 문제
SISO SSM decoding의 arithmetic intensity ≈ 2.5 ops/byte (H100 matmul: 295 ops/byte) → GPU 유휴 상태

### 해결: SISO → MIMO (rank R)
```
SISO: B_t ∈ R^N, x_t ∈ R^P        → outer product B_t · x_t^T
MIMO: B_t ∈ R^{N×R}, x_t ∈ R^{P×R} → matrix multiply B_t · x_t^T
```

| 측면 | SISO | MIMO (R=4) |
|------|------|------------|
| State h_t | R^{N×P} | R^{N×P} (동일!) |
| FLOPs | 5NP | ~4NPR |
| Arithmetic intensity | Θ(1) ≈ 2.5 | Θ(R) ≈ 10 |
| Decode latency | baseline | ~동일 (matmul로 overlap) |

### 학습 효율
- Naive: R² × SISO cost
- Chunked algorithm: **R × SISO cost** (chunk size = C_SISO / R)
- 실제 Triton kernel: R=4에서 2× slowdown만 발생

### Parameter 효율
- B, C: shared across heads → 자연스럽게 R 확장 (D·N → D·N·R, 미미한 증가)
- x, y, gate z: per-head → data-independent scaling vector (D·P + P·R, 곱셈 → 덧셈)
- MLP width 약간 감소로 총 param 동일하게 맞춤

## 6. Mamba-3 Block 아키텍처

### Mamba-2 대비 변경점

| 컴포넌트 | Mamba-2 | Mamba-3 |
|----------|---------|---------|
| Discretization | Exponential-Euler | **Exponential-Trapezoidal** |
| State transition | Real scalar diagonal | **Complex scalar diagonal** |
| State update | Outer product (SISO) | **Matrix multiply (MIMO, optional)** |
| Short convolution | 필수 | **제거** (bias + trapezoidal로 대체) |
| BC Normalization | 없음 | **RMSNorm after B, C projection** |
| B, C Biases | 없음 | **Head-specific, channel-wise bias** |
| Post-gate RMSNorm | 있음 (stability) | **제거** (BCNorm이 안정화) |
| Positional encoding | 없음/고정 | **Data-dependent RoPE** |

### 전체 구조
```
Input x
  │
  ├── Linear projection → [x_proj, B_proj, C_proj, Δ_proj, A_proj, θ_proj]
  │
  ├── BCNorm(B_proj) + bias → B
  ├── BCNorm(C_proj) + bias → C
  ├── Data-dependent RoPE on B, C using θ
  │
  ├── Exponential-Trapezoidal SSM:
  │     h_t = α_t·h_{t-1} + β_t·B_{t-1}·x_{t-1} + γ_t·B_t·x_t
  │     y_t = C_t^T · h_t
  │
  ├── [Optional MIMO projection]
  │
  └── Gate × y → output

Overall: Llama-style, alternating [Mamba-3 block, SwiGLU block] with pre-norm
```

## 7. 성능 결과 (1.5B scale, 100B tokens)

| Model | Val PPL ↓ | Avg Downstream ↑ |
|-------|-----------|-------------------|
| Transformer | 10.51 | 55.4 |
| Mamba-2 | 10.47 | 55.7 |
| GDN | 10.45 | 55.8 |
| **Mamba-3 SISO** | **10.35** | **56.4** |
| **Mamba-3 MIMO (R=4)** | **10.24** | **57.6** |

- Mamba-3 state size 64 ≈ Mamba-2 state size 128 (동일 perplexity, 절반 latency)

## 8. 우리 프로젝트에의 적용

### 왜 Mamba-3인가
- 단백질 서열: 수백~수천 residue → O(L²) attention 비효율
- Mamba-3: O(L) compute, O(1) memory → 긴 서열에 유리
- Complex SSM으로 rotational dynamics (단백질 구조의 회전 대칭) 자연스럽게 표현
- Data-dependent RoPE가 SimpleFold의 axial RoPE 역할 대체 가능

### 구현 시 핵심
1. **Mamba-3 block = Exponential-Trapezoidal SSM + data-dependent RoPE + BCNorm + bias**
2. **Short convolution 불필요** (bias + trapezoidal이 대체)
3. **MIMO (R=4) 권장** — 동일 latency에서 성능 향상
4. **SwiGLU block과 interleave** (Llama-style)
5. **Adaptive layer conditioning** 추가 필요 (timestep t or noise level)
6. `mamba-ssm>=2.0` 패키지에 Triton kernel 제공

### 주의사항
- Mamba-3는 causal (autoregressive) 설계 → protein folding은 bidirectional이 유리할 수 있음
  - 방안 1: Forward + Backward Mamba-3 결합 (BiMamba)
  - 방안 2: Chunk 단위 bidirectional processing
- State size N 선택: N=64이 Mamba-2 N=128과 동급 → 절반 latency
