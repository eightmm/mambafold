# Paper Overview - MambaFold 프로젝트 참고 논문

## 프로젝트 목표
**Linear-complexity 아키텍처(Mamba3 SSM)를 backbone으로 사용하여, flow matching / equilibrium matching 기반의 single-chain protein structure prediction 모델을 구축한다.**

## 논문 3편의 역할

| 논문 | 역할 | 핵심 기여 |
|------|------|-----------|
| **Mamba-3** | Backbone 아키텍처 | Linear-time SSM, O(L) complexity로 긴 단백질 서열 처리 |
| **SimpleFold** | Folding 파이프라인 | Flow matching 기반 protein folding의 전체 구조 (데이터→학습→추론) |
| **Equilibrium Matching** | 생성 프레임워크 대안 | Flow matching 대비 time-unconditional, energy landscape 기반 접근 |

## 핵심 설계 아이디어

### 1. Backbone: Mamba-3 SSM (Transformer 대체)
- SimpleFold는 Transformer를 사용하지만, 우리는 **Mamba-3 SSM block**으로 대체
- 이유: 단백질 서열 길이 수백~수천 residue → quadratic attention 비효율적
- Mamba-3의 핵심: exponential-trapezoidal discretization + complex SSM (data-dependent RoPE) + MIMO

### 2. Folding Pipeline: SimpleFold 방식 차용
- **Flow matching**: noise → all-atom 3D 구조 생성
- **Atom Encoder → Residue Trunk → Atom Decoder** 구조
- ESM2 PLM으로 sequence conditioning
- Loss: flow matching loss + LDDT structural loss

### 3. 생성 프레임워크: Flow Matching vs Equilibrium Matching
- 기본: SimpleFold식 **flow matching** (검증된 방법)
- 대안: **Equilibrium Matching** - time conditioning 제거, energy landscape에서 gradient descent로 sampling
- EqM 장점: adaptive compute, 유연한 step size, partially noised input 처리 가능

## 모델 아키텍처 청사진

```
Input: amino acid sequence s ∈ R^{Nr}
       noisy coords x_t ∈ R^{Na × 3}
       timestep t (flow matching) 또는 없음 (EqM)

┌─────────────────────────────────────────┐
│  Frozen ESM2 PLM → sequence embedding e │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  Atom Encoder (lightweight Mamba blocks) │
│  - Fourier positional embedding of x_t   │
│  - Local attention/SSM within residue    │
│  → atom tokens a ∈ R^{Na × d_a}         │
└──────────────┬──────────────────────────┘
               │ Grouping (avg pool per residue)
┌──────────────▼──────────────────────────┐
│  Residue Trunk (heavy Mamba-3 blocks)    │
│  - Concat with ESM2 embedding            │
│  - Mamba-3 SSM layers (NOT transformer)  │
│  - Adaptive layers conditioned on t      │
│  → residue tokens r ∈ R^{Nr × d_r}      │
└──────────────┬──────────────────────────┘
               │ Ungrouping (broadcast to atoms)
┌──────────────▼──────────────────────────┐
│  Atom Decoder (lightweight Mamba blocks) │
│  - Skip connection from atom encoder     │
│  - → predicted velocity v̂_t             │
└─────────────────────────────────────────┘

Output: velocity field v̂_t ∈ R^{Na × 3}
```

## 학습 전략 요약

| 항목 | SimpleFold 방식 | 우리 프로젝트 |
|------|-----------------|---------------|
| 생성 프레임워크 | Flow matching | Flow matching (기본) / EqM (실험) |
| Backbone | Transformer + adaptive layers | Mamba-3 SSM + adaptive layers |
| Sequence conditioning | ESM2-3B (frozen) | ESM2 (frozen) |
| 좌표 표현 | Full-atom (all heavy atoms) | Full-atom |
| Positional embedding | 4D axial RoPE | Mamba-3 data-dependent RoPE 활용 |
| Loss | ℓ_FM + α(t)·ℓ_LDDT | 동일 |
| 2-stage training | Pretrain (all data) → Finetune (PDB+SwissProt) | 단계적 적용 |
| Data augmentation | SO(3) random rotation | 동일 |
| Timestep sampling | logistic-normal (t→1 oversampling) | 동일 |

## 상세 논문 정리
- [Mamba-3 상세](./mamba3_summary.md)
- [SimpleFold 상세](./simplefold_summary.md)
- [Equilibrium Matching 상세](./equilibrium_matching_summary.md)
