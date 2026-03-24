# SimpleFold: Folding Proteins is Simpler than You Think

- **저자**: Yuyang Wang, Jiarui Lu, Navdeep Jaitly, Josh Susskind, Miguel Angel Bautista
- **소속**: Apple
- **발표**: arXiv:2509.18480 (2025년 9월, v4: 2025년 12월)
- **코드**: https://github.com/apple/ml-simplefold

---

## 1. 핵심 요약

SimpleFold는 **domain-specific 모듈 없이** 범용 transformer block + flow matching으로 protein folding을 수행하는 최초의 모델이다.

**제거한 것들** (AlphaFold2 대비):
- MSA (Multiple Sequence Alignment) ✗
- Pair representation ✗
- Triangle updates ✗
- Equivariant geometric modules ✗

**대신 사용한 것들**:
- Standard transformer blocks + adaptive layers
- Flow matching generative objective
- Frozen ESM2-3B PLM for sequence conditioning
- SO(3) data augmentation (equivariance 대신)

## 2. Flow Matching for Protein Folding

### Linear Interpolant Path
```
x_t = t·x + (1-t)·ε

where:
  x ∈ R^{Na×3}  — ground truth all-atom coordinates
  ε ~ N(0, I)   — Gaussian noise
  t ∈ [0, 1]    — timestep (0=noise, 1=data)

Target velocity: v_t = x - ε
```

### 학습 목표
```
ℓ_FM = E_{x,s,ε,t} [ (1/Na) · ||v_θ(x_t, s, t) - (x - ε)||² ]
```

v_θ는 noisy 구조 x_t, sequence s, timestep t를 입력받아 velocity field를 예측한다.

### LDDT Structural Loss
Flow matching loss만으로는 local structure quality가 부족 → LDDT loss 추가:

```
ℓ_LDDT = E[ Σ_{i≠j} 1(δ_ij < C) · σ(||δ_ij - δ̂_ij^t||) / Σ 1(δ_ij < C) ]

where:
  δ_ij = ||x_i - x_j||       — ground truth pairwise distance
  δ̂_ij^t = ||x̂_i - x̂_j||   — predicted pairwise distance
  x̂(x_t) = x_t + (1-t)·v_θ(x_t, s, t)  — one-step Euler estimate
  C = distance cutoff (이웃 원자 범위)
  σ(·) = nonlinear function on distance errors
```

### 총 Loss
```
ℓ = ℓ_FM + α(t) · ℓ_LDDT

Pretrain:  α(t) = 1 (전체 flow 과정에서 균일)
Finetune:  α(t) = 1 + 8·ReLU(t - 0.5) (t→1에서 최대 5까지 증가)
```

### Timestep Resampling
```
p(t) = 0.98 · LN(0.8, 1.7) + 0.02 · U(0, 1)

LN = logistic-normal distribution
U = uniform distribution
```

- t→1 (clean data) 근처를 과다 샘플링
- 이유: 단백질은 coarse-to-fine 계층 (secondary → backbone → side chain)
- t→1에서 side chain 세부 구조 학습이 중요

## 3. 아키텍처

### 전체 구조: Atom Encoder → Residue Trunk → Atom Decoder

```
┌───────────────────────────────┐
│         Atom Encoder          │  ← lightweight (적은 block)
│  Input: x_t (noisy coords)   │
│         + atomic features     │
│  Fourier PE of coordinates    │
│  Local attention (residue별)  │
│  Output: atom tokens a        │
│          ∈ R^{Na × d_a}      │
└───────────┬───────────────────┘
            │ Grouping: avg pool per residue
            ▼
┌───────────────────────────────┐
│        Residue Trunk          │  ← heavy (대부분의 param)
│  Input: residue tokens r      │
│         + ESM2 embedding e    │
│  (concat along channel dim)   │
│  Full self-attention          │
│  Adaptive layers (timestep t) │
│  Output: updated r            │
│          ∈ R^{Nr × d_r}      │
└───────────┬───────────────────┘
            │ Ungrouping: broadcast to atoms
            ▼                    + skip from encoder
┌───────────────────────────────┐
│        Atom Decoder           │  ← lightweight
│  Local attention (residue별)  │
│  Output: velocity v̂_t        │
│          ∈ R^{Na × 3}        │
└───────────────────────────────┘
```

### 공통 Building Block (모든 모듈 공유)
```
Standard Transformer Block + Adaptive Layers:

Input → AdaLN(Scale, Shift from t) → MHA → Scale → Residual
      → AdaLN(Scale, Shift from t) → SwiGLU → Scale → Residual

Features:
- QK-normalization (학습 안정성)
- SwiGLU (standard FFN 대체)
- Adaptive layers: timestep t로부터 scale/shift 파라미터 생성
```

### Grouping / Ungrouping
- **Grouping**: 같은 residue에 속한 atom token들을 average pooling → residue token
- **Ungrouping**: residue token을 해당 AA type의 atom 수만큼 broadcast
  - Atom encoder의 output을 skip connection으로 더함 (atom 구분용)

### Positional Embedding
- **Residue Trunk**: 1D RoPE — residue index n에 대해 `e^{iθn}`
- **Atom Encoder/Decoder**: 4D axial RoPE
  - 3D: reference conformer의 3D 좌표 (rule-based cheminformatic prediction)
  - 1D: 해당 atom이 속한 residue index
  - 각 축이 hidden dim의 1/4씩 담당

### Sequence Conditioning
- **ESM2-3B** (frozen) → per-residue embedding e ∈ R^{Nr × d_e}
- Residue trunk 입력 시 residue token과 channel-wise concatenation
- Text-to-image 모델에서 CLIP embedding 역할과 동일

## 4. Sampling (Inference)

### Stochastic SDE Sampling
```
dx_t = v_θ(x_t, s, t) dt + (1/2)·w(t)·s_θ(x_t, t, c) dt + τ·√w(t) dW̄_t

where:
  s_θ(x_t, s, t) = (t·v_θ - x_t) / (1-t)  ← score function
  w(t) = 2(1-t)/(t+η)                       ← diffusion coefficient
  τ = stochasticity scale
  W̄_t = reverse-time Wiener process
```

### τ (stochasticity) 설정
| 용도 | τ | 특성 |
|------|---|------|
| 구조 예측 (folding) | 0.01 | 거의 deterministic, 최고 정확도 |
| MD ensemble 생성 | 0.6 | 중간 다양성 |
| Multi-state conformation | 0.8 | 높은 다양성 |

### 초기화
- x_0 ~ N(0, I) — 순수 Gaussian noise에서 시작
- t=0 → t=1로 Euler-Maruyama integration

## 5. Confidence Module (pLDDT)

- Folding 모델 학습 완료 후 **별도 학습** (folding 파라미터 frozen)
- 4 layers transformer blocks (adaptive layer 없음)
- Input: folding 모델의 최종 residue tokens r (t=1에서)
- Output: per-residue LDDT 예측 (0~100)
- Target: 50 bins으로 discretize, cross-entropy loss
- 학습 시 SimpleFold로 on-the-fly 구조 생성 (200 steps, τ=0.3)
- Pearson correlation 0.77 with actual LDDT-Cα

## 6. Training Strategy

### 데이터
| Source | 크기 | 설명 |
|--------|------|------|
| PDB | ~160K | 실험 구조 (May 2020 cutoff) |
| AFDB SwissProt | ~270K | High-quality distilled (pLDDT>85, std<15) |
| AFESM representative | ~1.9M | Cluster 대표 (pLDDT>0.8) |
| AFESM-E (3B only) | ~8.6M | 클러스터당 최대 10개 (pLDDT>0.8) |

### 2-Stage Training

**Stage 1: Pre-training**
- Data: 전체 (~2M, 3B는 ~8.7M)
- Max sequence length: **256** residues (crop)
- α(t) = 1 (균일 LDDT weight)
- Batch: 512 (1.6B: 1024, 3B: 3072)
- Optimizer: AdamW, lr=0.0001, 5000-step linear warmup
- EMA decay: 0.999

**Stage 2: Fine-tuning**
- Data: PDB + SwissProt만 (high quality)
- Max sequence length: **512** residues
- α(t) = 1 + 8·ReLU(t - 0.5) (t→1에서 LDDT weight 증가)
- Batch per GPU 절반 (longer sequences)
- 동일 optimizer 설정

**Stage 3: pLDDT Training** (별도)
- Data: PDB + SwissProt
- Folding model frozen
- On-the-fly structure generation

### Data Augmentation
- **SO(3) random rotation**: 구조 target에 랜덤 회전 적용
  - Equivariant architecture 대신 data augmentation으로 회전 대칭 학습

### Batch 구성 (중요)
```
한 GPU에서 같은 protein을 Bc번 복사 (다른 timestep t로)
Bp개의 서로 다른 protein이 서로 다른 GPU에
Effective batch = Bc × Bp
```
→ 동일 protein의 여러 timestep 동시 학습 → 안정적 gradient

## 7. Model Configurations

| Model | Params | Trunk Depth | Trunk Width | Atom Enc/Dec Depth |
|-------|--------|-------------|-------------|-------------------|
| SimpleFold-100M | 100M | shallow | narrow | minimal |
| SimpleFold-360M | 360M | ↓ | ↓ | ↓ |
| SimpleFold-700M | 700M | ↓ | ↓ | ↓ |
| SimpleFold-1.1B | 1.1B | ↓ | ↓ | ↓ |
| SimpleFold-1.6B | 1.6B | ↓ | ↓ | ↓ |
| SimpleFold-3B | 3B | deepest | widest | largest |

- Scaling: depth + width of all modules 함께 증가
- 100M이 ESMFold 성능의 ~90% 달성 → 효율적

## 8. 성능 (핵심 수치)

### CAMEO22 (SimpleFold-3B)
| Metric | SimpleFold-3B | ESMFold | AlphaFold2 |
|--------|---------------|---------|------------|
| TM-score | 0.837/0.916 | 0.853/0.933 | 0.863/0.942 |
| GDT-TS | 0.802/0.867 | 0.826/0.875 | 0.844/0.903 |

### CASP14 (SimpleFold-3B)
| Metric | SimpleFold-3B | ESMFold | AlphaFold2 |
|--------|---------------|---------|------------|
| TM-score | **0.720/0.792** | 0.701/0.792 | 0.845/0.907 |
| GDT-TS | **0.639/0.703** | 0.622/0.711 | 0.783/0.855 |

- CASP14에서 ESMFold를 **능가** (TM-score, GDT-TS)
- AlphaFold2 대비는 아직 격차 (MSA + regression objective의 이점)
- CASP14→CAMEO22 성능 하락이 다른 모델 대비 적음 (robustness)

## 9. 우리 프로젝트에의 적용

### 직접 차용할 것
1. **전체 파이프라인**: flow matching + LDDT loss
2. **Atom Encoder → Residue Trunk → Atom Decoder** 구조
3. **ESM2 frozen PLM** conditioning
4. **2-stage training**: pretrain (all data, crop 256) → finetune (high quality, 512)
5. **Timestep resampling**: logistic-normal, t→1 oversampling
6. **SO(3) data augmentation**
7. **Batch 구성**: 같은 protein 여러 timestep

### 변경할 것 (Mamba-3 적용)
1. **Residue Trunk**: Transformer → **Mamba-3 SSM blocks**
   - Self-attention → SSM recurrence (linear complexity)
   - Adaptive layers는 유지 (timestep conditioning)
2. **Atom Encoder/Decoder**: Transformer → **lightweight Mamba blocks**
   - Local attention → local SSM window
3. **Positional Embedding**:
   - SimpleFold의 1D RoPE → Mamba-3의 **data-dependent RoPE** 활용
   - 4D axial RoPE → 적절한 변환 필요 (SSM context에서)

### 데이터 매핑 (afdb_data → SimpleFold format)
우리 `.pt` 파일의 key와 SimpleFold input의 대응:

| `.pt` key | SimpleFold 용도 |
|-----------|----------------|
| `res_names` | AA sequence s (3-letter → 1-letter 변환) |
| `coords` | Ground truth x (all-atom 3D coords) |
| `atom_names` | Atom type features, grouping/ungrouping 기준 |
| `res_seq_nums` | Residue indexing (RoPE용) |
| `is_observed` | Missing atom masking |

### 고려사항
- SimpleFold는 **full-atom** (모든 heavy atom) 생성 → 우리 데이터와 일치
- Sequence length 제한: pretrain 256, finetune 512 → crop 전략 필요
- ESM2 embedding 계산: offline pre-computation 권장 (inference 비용 절감)
