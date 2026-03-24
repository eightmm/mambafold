# MambaFold Implementation Plan

> CCG Synthesis: Claude (orchestrator) + Codex (architecture) + Gemini (design review)
> 작성일: 2026-03-22

---

## 0. 설계 결정 요약 (Conflicts Resolved)

| 설계 항목 | Codex 의견 | Gemini 의견 | **최종 결정** | 이유 |
|-----------|-----------|-------------|--------------|------|
| Bidirectional | BiMamba (gated fusion) | BiMamba + sparse attention hybrid | **BiMamba (gated fusion)** | v1에서 단순하게, hybrid는 Phase 2 실험 |
| c(γ) 함수 | 4·c_trunc(γ,0.8) (EqM 논문 그대로) | c(γ)=1-γ (linear decay, LDDT 단순화) | **4·c_trunc(γ,0.8)** (기본) + **1-γ ablation** | 논문 검증된 설정 우선, linear decay는 비교 실험 |
| Atom Encoder/Decoder | BiMamba per-residue | Lightweight attention (≤14 atoms) | **Lightweight attention** | atom 수 적음, O(14²) trivial, 순서 없는 atom에 SSM 부적합 |
| Adaptive layers (EqM) | mode="none" (제거) | Noise estimator (input norm 기반) | **mode="none" (v1)** → noise estimator (v2) | EqM 원논문이 time-unconditional, 먼저 순수 EqM 검증 |
| 구현 순서 | EqM-first | FM baseline → EqM 전환 | **EqM-first** | FM은 fallback, 목표가 EqM이므로 직행 |
| LDDT for EqM | x̂ = x_γ - scale·f(x_γ), analytical scale | x̂ = x_γ - f(x_γ) (linear decay 시) | **Analytical scale** (c_trunc 호환) | c_trunc에서도 안정적으로 작동 |
| Positional encoding | Fourier + Mamba-3 RoPE | Fourier 3D + Mamba-3 RoPE 1D | **합의**: Fourier(3D coords) + Mamba-3 data-dependent RoPE(1D sequence) |

---

## 1. 디렉토리 구조 및 모듈 분해

```
src/mambafold/
├── __init__.py
├── data/
│   ├── __init__.py
│   ├── constants.py         # AA vocab, atom slot table, MAX_ATOMS=15
│   ├── types.py             # ProteinExample, ProteinBatch dataclass
│   ├── dataset.py           # AFDBDataset (.pt 로딩)
│   ├── transforms.py        # Canonicalize, CenterScale, RandomCrop, SO3Aug, EqMCorrupt
│   ├── collate.py           # ProteinCollator, LengthBucketSampler
│   └── esm.py              # FrozenESMEncoder, ESMCacheReader
├── model/
│   ├── __init__.py
│   ├── embeddings.py        # CoordinateFourierEmbed, AtomFeatureEmbed, ResidueFeatureEmbed
│   ├── ssm/
│   │   ├── __init__.py
│   │   ├── mamba3.py        # Mamba3Layer (official kernel wrapper)
│   │   └── bimamba3.py      # BiMamba3Block (forward+reverse+gate fusion)
│   ├── blocks.py            # RMSNorm, SwiGLU, ConditioningAdapter
│   ├── atom_encoder.py      # AtomEncoder (lightweight local attention)
│   ├── grouping.py          # group_atoms_to_residues, broadcast_residues_to_atoms
│   ├── residue_trunk.py     # ResidueTrunk (heavy BiMamba-3 stack)
│   ├── atom_decoder.py      # AtomDecoder (local attention) + GradientHead
│   └── mambafold.py         # MambaFoldEqM (end-to-end)
├── losses/
│   ├── __init__.py
│   ├── eqm.py              # truncated_c, eqm_loss, reconstruction_scale
│   └── lddt.py             # soft_lddt_ca_loss (differentiable)
├── sampling/
│   ├── __init__.py
│   └── sampler.py           # EqMNAGSampler (adaptive stopping)
├── train/
│   ├── __init__.py
│   ├── engine.py            # train_step, eval_step
│   └── ema.py              # EMA weights
└── utils/
    ├── __init__.py
    ├── geometry.py          # rotation, centroid, pairwise distances
    └── metrics.py           # ca_lddt, all_atom_rmsd
```

```
scripts/
├── precompute_esm.py        # Offline ESM2 embedding 캐시
├── train_eqm.py             # 학습 entrypoint
├── sample_structure.py      # 추론
└── slurm/
    ├── precompute_esm.sh
    ├── train.sh
    └── run_tests.sh

configs/
├── base.yaml                # 기본 하이퍼파라미터
├── debug.yaml               # 디버그용 (작은 모델, 적은 데이터)
└── full.yaml                # 전체 학습용

tests/
├── test_forward_shapes.py   # 모델 forward shape 검증
├── test_eqm.py              # EqM loss/target 수학 검증
├── test_sampler.py          # NAG sampler 수렴 검증
└── test_data.py             # Dataset/collate 검증
```

---

## 2. 구현 순서 (Phase별)

### Phase 0: 데이터 파이프라인 (의존성 없음)
1. `data/constants.py` — AA↔ID, atom slot table (atom14 + OXT = 15 slots)
2. `data/types.py` — ProteinExample, ProteinBatch dataclass
3. `data/dataset.py` — AFDBDataset: .pt 로딩, standard AA 필터, canonical atom slot 매핑
4. `utils/geometry.py` — random_rotation, centroid, pairwise_distances
5. `data/transforms.py` — CanonicalizeHeavyAtoms, CenterAndScale, RandomCrop, RandomSO3, EqMCorrupt
6. `data/collate.py` — padding, masking, bucket sampler
7. `tests/test_data.py` — 검증
8. `scripts/precompute_esm.py` + `data/esm.py` — ESM2 offline 캐시

### Phase 1: 모델 코어 (data 완료 후)
9. `model/ssm/mamba3.py` — Mamba3Layer wrapper (mamba-ssm 패키지)
10. `model/ssm/bimamba3.py` — BiMamba3Block (forward+reverse+gate)
11. `model/blocks.py` — RMSNorm, SwiGLU, ConditioningAdapter
12. `model/embeddings.py` — Fourier PE, atom/residue feature embed
13. `model/atom_encoder.py` — lightweight local attention (per-residue)
14. `model/grouping.py` — masked avg pool / broadcast
15. `model/residue_trunk.py` — BiMamba-3 stack + SwiGLU interleave
16. `model/atom_decoder.py` — local attention + gradient head
17. `model/mambafold.py` — end-to-end MambaFoldEqM
18. `tests/test_forward_shapes.py` — shape 검증

### Phase 2: Loss & Sampling
19. `losses/eqm.py` — EqM loss, c(γ) 함수, reconstruction scale
20. `losses/lddt.py` — differentiable CA-LDDT
21. `sampling/sampler.py` — EqMNAGSampler
22. `tests/test_eqm.py`, `tests/test_sampler.py`

### Phase 3: Training Pipeline
23. `train/ema.py` — EMA
24. `train/engine.py` — train_step, eval_step
25. `scripts/train_eqm.py` — main entrypoint
26. `configs/base.yaml`, `debug.yaml`
27. `scripts/slurm/train.sh`

### Phase 4: 추론 & 평가
28. `scripts/sample_structure.py` — NAG sampling → PDB 출력
29. `utils/metrics.py` — TM-score, GDT-TS, RMSD (외부 tool 호출)

---

## 3. 핵심 아키텍처 상세

### 3.1 Tensor Shapes

**Base config**: d_atom=256, d_res=768, ssm_state=64, mimo_rank=4, n_atom_enc=4, n_trunk=24, n_atom_dec=4, A=15 (max atoms/residue)

| Stage | Shape | 설명 |
|-------|-------|------|
| Raw batch | `res_type: [B, L]`, `coords: [B, L, A, 3]`, `atom_mask: [B, L, A]` | L=residues, A=15 |
| EqM corruption | `x_γ: [B, L, A, 3]`, `ε: [B, L, A, 3]`, `γ: [B, 1, 1, 1]` | |
| ESM embedding | `esm: [B, L, d_esm]` | d_esm=1280 (ESM2-650M) |
| Fourier PE | `coord_feat: [B, L, A, d_fourier]` | d_fourier=128 |
| Atom token | `[B, L, A, d_atom]` → reshape `[B*L, A, d_atom]` | per-residue 처리 |
| Atom encoder out | `[B, L, A, d_atom]` | local attention |
| Grouping | masked mean over A → `[B, L, d_atom]` | |
| Trunk input | `cat([res_tok, esm], -1)` → proj → `[B, L, d_res]` | |
| BiMamba-3 trunk | `[B, L, d_res]` (24 layers) | bidirectional |
| Ungrouping | broadcast → `[B, L, A, d_atom]` + skip connection | |
| Atom decoder | `[B*L, A, d_atom]` → local attention | |
| Gradient head | → `[B, L, A, 3]` | EqM gradient output |

### 3.2 BiMamba3Block (Residue Trunk의 핵심)

```python
class BiMamba3Block(nn.Module):
    """Bidirectional Mamba-3 with gated fusion."""
    def __init__(self, d_model, ssm_state=64, mimo_rank=4):
        self.mamba_f = Mamba3Layer(d_model, ssm_state, mimo_rank)  # forward
        self.mamba_b = Mamba3Layer(d_model, ssm_state, mimo_rank)  # backward
        self.gate_proj = nn.Linear(2 * d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = RMSNorm(d_model)
        self.swiglu = SwiGLU(d_model)

    def forward(self, x, mask):
        # x: [B, L, D], mask: [B, L]
        y_f = self.mamba_f(x, mask)
        y_b = flip_masked(self.mamba_b(flip_masked(x, mask), flip_masked(mask)), mask)
        gate = torch.sigmoid(self.gate_proj(torch.cat([y_f, y_b], dim=-1)))
        y = gate * y_f + (1 - gate) * y_b
        x = x + self.out_proj(y)              # residual
        x = x + self.swiglu(self.norm(x))     # FFN residual
        return x
```

### 3.3 Atom Encoder (Lightweight Local Attention)

```python
class AtomEncoder(nn.Module):
    """Per-residue local attention over atoms (≤15 atoms)."""
    def __init__(self, d_atom, n_layers=4, n_heads=4):
        self.layers = nn.ModuleList([
            LocalAttnBlock(d_atom, n_heads) for _ in range(n_layers)
        ])

    def forward(self, atom_tok, atom_mask):
        # atom_tok: [B*L, A, d_atom], atom_mask: [B*L, A]
        for layer in self.layers:
            atom_tok = layer(atom_tok, atom_mask)
        return atom_tok
```

### 3.4 MambaFoldEqM (End-to-End)

```python
class MambaFoldEqM(nn.Module):
    def forward(self, batch):
        B, L, A = batch.res_type.shape[0], batch.res_type.shape[1], MAX_ATOMS

        # 1. Atom embedding (Fourier PE + atom features)
        atom0 = self.atom_embed(batch.res_type, batch.atom_type,
                                batch.x_gamma, batch.atom_mask)     # [B, L, A, d_atom]

        # 2. Atom encoder (per-residue local attention)
        atom = self.atom_encoder(
            atom0.reshape(B*L, A, -1), batch.atom_mask.reshape(B*L, A)
        ).reshape(B, L, A, -1)

        # 3. Grouping: atoms → residues
        res0 = group_atoms_to_residues(atom, batch.atom_mask)       # [B, L, d_atom]

        # 4. Residue trunk (heavy BiMamba-3)
        trunk_in = self.trunk_proj(torch.cat([res0, batch.esm], dim=-1))
        res = self.residue_trunk(trunk_in, batch.res_mask)          # [B, L, d_res]

        # 5. Ungrouping: residues → atoms (+ skip)
        dec_in = atom + self.res_to_atom(res, batch.atom_mask)

        # 6. Atom decoder (per-residue local attention)
        dec = self.atom_decoder(
            dec_in.reshape(B*L, A, -1), batch.atom_mask.reshape(B*L, A)
        ).reshape(B, L, A, -1)

        # 7. Gradient head → [B, L, A, 3]
        grad = self.grad_head(dec) * batch.atom_mask.unsqueeze(-1)
        return grad
```

---

## 4. EqM Loss & LDDT Adaptation

### 4.1 EqM Loss

```python
def truncated_c(gamma, a=0.8, lam=4.0):
    """c(γ) = λ * c_trunc(γ, a)"""
    c = torch.where(gamma <= a, torch.ones_like(gamma),
                    (1 - gamma) / (1 - a))
    return lam * c

def eqm_loss(pred, x_clean, eps, gamma, valid_mask):
    """L_EqM = ||f(x_γ) - (ε - x) * c(γ)||²"""
    c = truncated_c(gamma)                              # [B, 1, 1, 1]
    target = (eps - x_clean) * c
    diff = (pred - target) ** 2                         # [B, L, A, 3]
    diff = diff * valid_mask.unsqueeze(-1)
    return diff.sum() / valid_mask.sum().clamp(min=1) / 3
```

### 4.2 LDDT를 위한 Structure Reconstruction

```python
def eqm_reconstruction_scale(gamma, a=0.8, lam=4.0):
    """x̂ = x_γ - scale * f(x_γ) 에서 scale 계산.

    f(x_γ) ≈ (ε - x) * c(γ)
    x_γ = γx + (1-γ)ε
    x̂ = x_γ - ((1-γ)/c(γ)) * f(x_γ)

    c_trunc에서 stable하게:
      γ ≤ a: scale = (1-γ) / (lam * 1) = (1-γ)/lam
      γ > a: scale = (1-γ) / (lam * (1-γ)/(1-a)) = (1-a)/lam
    """
    scale = torch.where(
        gamma <= a,
        (1 - gamma) / lam,
        torch.tensor((1 - a) / lam, device=gamma.device)
    )
    return scale  # [B, 1, 1, 1]
```

### 4.3 Combined Training Step

```python
def train_step(model, batch, optimizer, alpha_mode="const"):
    pred = model(batch)                                  # [B, L, A, 3]

    # EqM main loss
    loss_eqm = eqm_loss(pred, batch.x_clean, batch.eps,
                         batch.gamma, batch.valid_mask)

    # Structure reconstruction for LDDT
    scale = eqm_reconstruction_scale(batch.gamma)
    x_hat = batch.x_gamma - scale * pred

    # CA-LDDT loss (C-alpha only, cheaper)
    loss_lddt = soft_lddt_ca_loss(x_hat, batch.x_clean, batch.ca_mask)

    # Combined
    if alpha_mode == "const":
        alpha = 1.0
    else:  # "ramp" (finetune)
        alpha = 1.0 + 8.0 * F.relu(batch.gamma - 0.5)
        alpha = alpha.mean()

    loss = loss_eqm + alpha * loss_lddt

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {"loss": loss.item(), "eqm": loss_eqm.item(), "lddt": loss_lddt.item()}
```

---

## 5. NAG Sampler

```python
class EqMNAGSampler:
    def __init__(self, model, eta=0.1, mu=0.3, g_min=5e-3,
                 max_steps=128, max_disp=0.5, coord_scale=10.0):
        self.model = model
        self.eta = eta          # step size
        self.mu = mu            # NAG momentum
        self.g_min = g_min      # stopping threshold
        self.max_steps = max_steps
        self.max_disp = max_disp
        self.coord_scale = coord_scale

    @torch.no_grad()
    def sample(self, seq_batch):
        """seq_batch: sequence info (res_type, atom_type, atom_mask, esm, res_mask)"""
        shape = seq_batch.atom_mask.shape + (3,)  # [B, L, A, 3]
        x = torch.randn(shape, device=seq_batch.device)
        x = remove_translation(x, seq_batch.atom_mask)
        x_prev = x.clone()

        for k in range(self.max_steps):
            # NAG lookahead
            look = x if k == 0 else x + self.mu * (x - x_prev)
            grad = self.model(seq_batch.with_coords(look))
            grad = grad * seq_batch.atom_mask.unsqueeze(-1)

            # Adaptive stopping
            grad_rms = (grad ** 2).sum() / seq_batch.atom_mask.sum().clamp(1) / 3
            if grad_rms.sqrt() < self.g_min:
                break

            # Clamped gradient step
            step = (self.eta * grad).clamp(-self.max_disp, self.max_disp)
            x_next = remove_translation(x - step, seq_batch.atom_mask)
            x_prev, x = x, x_next

        return x * self.coord_scale  # back to Angstrom
```

---

## 6. Training Pipeline

### 6.1 데이터 전처리 순서
1. `precompute_esm.py` 실행 → ESM2 embedding 캐시 (seq_hash → .pt)
2. Dataset 로딩: raw .pt → standard AA 필터 → canonical atom slot (A=15) → mask 생성
3. 좌표 정규화: centroid 제거, `/coord_scale(10.0)` 나누기
4. Random crop: pretrain L≤256, finetune L≤512
5. SO(3) augmentation: random rotation on clean coords
6. EqM corruption: ε ~ N(0,I), γ ~ U(0,1), x_γ = γx + (1-γ)ε

### 6.2 학습 설정

| 항목 | Pretrain | Finetune |
|------|----------|----------|
| Data | 전체 train set | High-quality subset (pLDDT>85) |
| Max length | 256 residues | 512 residues |
| Batch | valid_atoms ≤ 32K per step | 절반 |
| Copies/protein | 2~4 (다른 γ) | 동일 |
| α(γ) | 1.0 (const) | 1 + 8·ReLU(γ-0.5) |
| Optimizer | AdamW, lr=1e-4, β=(0.9,0.95), wd=0.05 | 동일 |
| Warmup | 5K steps linear | 없음 |
| Schedule | Cosine decay | 동일 |
| Grad clip | 1.0 | 1.0 |
| Precision | bf16 | bf16 |
| EMA | 0.999 | 0.999 |

### 6.3 Validation Metrics
- EqM MSE (training loss)
- CA-lDDT on one-step reconstruction (x̂ quality)
- Sampled CA-lDDT after NAG (full inference quality)
- Gradient RMS at convergence (sampling quality indicator)

### 6.4 Inference 설정
- η=0.1, μ=0.3, g_min=5e-3, max_steps=128
- Per-step displacement clamp: 0.5 (normalized units)
- 결과를 coord_scale(10.0) 곱하여 Å 단위로 복원

---

## 7. Model Configurations

### Debug (빠른 검증용)
```yaml
model:
  d_atom: 128
  d_res: 256
  ssm_state: 32
  mimo_rank: 2
  n_atom_enc: 2
  n_trunk: 4
  n_atom_dec: 2
  n_heads_atom: 4

data:
  max_length: 64
  coord_scale: 10.0
  max_atoms_per_res: 15

training:
  batch_atoms: 4096
  lr: 1e-4
  epochs: 10
```

### Base (메인 학습)
```yaml
model:
  d_atom: 256
  d_res: 768
  ssm_state: 64
  mimo_rank: 4
  n_atom_enc: 4
  n_trunk: 24
  n_atom_dec: 4
  n_heads_atom: 8
  d_esm: 1280  # ESM2-650M

data:
  max_length: 256  # pretrain
  coord_scale: 10.0
  max_atoms_per_res: 15

training:
  batch_atoms: 32768
  lr: 1e-4
  warmup: 5000
  ema: 0.999
  grad_clip: 1.0
  copies_per_protein: 4
```

---

## 8. Risk Areas & Mitigations

| Risk | 심각도 | Mitigation |
|------|--------|------------|
| mamba-ssm 패키지 호환성 | High | Mamba3Layer wrapper로 격리, fallback to Mamba-2 |
| Causal bias (unidirectional) | High | BiMamba gated fusion 필수 (day 1) |
| EqM coordinate scale 불안정 | High | center+scale(10Å), step clamp, γ∈(1e-4, 1-1e-4) |
| γ→1 근처 division instability | High | analytical scale 사용, raw (1-γ)/c(γ) 금지 |
| Full-atom memory | Medium | A=15 고정, CA-only LDDT, atom-count batching |
| ESM bottleneck | Medium | Offline precompute |
| Chirality/bond length 위반 | Medium | Phase 2에서 geometric penalty loss 추가 |
| SE(3) invariance | Medium | SO(3) data augmentation + translation removal |
| 비표준 residue | Low | v1에서 skip or UNK 매핑, constants.py에 격리 |

---

## 9. Ablation 실험 계획 (Phase 2 이후)

| 실험 | 목적 |
|------|------|
| c(γ)=1-γ vs 4·c_trunc(0.8) | linear decay의 LDDT 단순화 효과 |
| BiMamba vs BiMamba + sparse attention (1:4) | long-range contact 정확도 |
| MIMO R=1 vs R=4 | 추론 효율 vs 품질 |
| ConditioningAdapter: none vs noise_estimator | EqM에서 implicit conditioning 효과 |
| FM baseline vs EqM | 생성 프레임워크 비교 |
| Adaptive compute ON/OFF | 추론 효율 (step 절감량) |

---

## 10. 마일스톤

| 마일스톤 | 완료 기준 | 예상 의존성 |
|----------|----------|------------|
| M0: Data pipeline | .pt → batch tensor, ESM cache 완성, test 통과 | 없음 |
| M1: Model forward | random input → gradient output, shape 검증 통과 | M0 |
| M2: Loss & training | loss 수렴 확인 (debug config), overfitting 1 sample | M1 |
| M3: Full training | 256-crop pretrain 완료, CA-lDDT > 0.3 | M2 |
| M4: Sampling | NAG sampler로 구조 생성, 시각적 확인 | M3 |
| M5: Evaluation | CAMEO22/CASP14 벤치마크 | M4 |
| M6: Ablations | 위 실험 테이블 완료 | M5 |
