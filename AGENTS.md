# MambaFold — Project Overview

## 목적 및 동기

단백질 구조 예측 모델을 **Equilibrium Matching (EqM)** 프레임워크와 **Mamba-3 SSM**을 결합해 구현한다.

### 왜 EqM인가?
기존 Diffusion/Flow Matching은:
- 모델이 timestep t를 입력으로 받아야 함 (time-conditional)
- 고정된 ODE 궤적으로만 샘플링 가능
- 각 noise level마다 별도 dynamics를 학습

EqM은:
- **time-unconditional**: 모델이 γ를 보지 않음. 단일 equilibrium gradient field 학습
- **gradient descent로 샘플링**: step size, optimizer, adaptive compute 자유롭게 선택
- **data가 energy local minima**: 완벽 학습 시 real sample에서 f(x)=0 (이론 보장)
- ImageNet에서 FID 1.90 달성 (SiT-XL/2 FM의 2.06 대비 우수)

### 왜 Mamba-3인가?
- 단백질 시퀀스는 길이가 수백~수천 residue → Transformer O(L²) 비용 부담
- Mamba-3: O(L) 선형 시간, MIMO(Multiple Input Multiple Output) 확장으로 표현력 향상
- Bidirectional: 단백질은 N→C 방향성 없음 → forward + backward SSM gated fusion
- 공식 구현(mamba-ssm) Triton/TileLang 커널 사용 → GPU에서만 실행 가능

---

## 아키텍처 요약

```
입력: x_γ = γ·x_clean + (1-γ)·ε  (noisy coordinates)
출력: f(x_γ) ≈ (ε - x_clean)·c(γ)  (EqM gradient)

Forward pass:
  x_γ [B,L,A,3]
    → AtomFeatureEmbedder            [B, L, A, d_atom]
         (res_type + atom_type + pair_type embedding + CoordinateFourierEmbed)
    → MambaStack (Atom Encoder)      [B*L, A, d_atom]  (per-residue)
    → group_atoms_to_residues        [B, L, d_atom]    (masked avg pool)
    → (PLM concat if use_plm)
    → trunk_proj                     [B, L, d_res]
    → MambaStack (Residue Trunk)     [B, L, d_res]     (BiMamba-3, heavy)
    → ResidueToAtomBroadcast         [B, L, A, d_atom] (broadcast + skip)
    → MambaStack (Atom Decoder)      [B*L, A, d_atom]  (per-residue)
    → GradHead (LayerNorm→Linear→GELU→Linear→3)
    → f(x_γ) [B, L, A, 3]

MambaStack = N × BiMamba3Block
BiMamba3Block = pre-norm + Mamba3(forward) + Mamba3(backward) + gated fusion + SwiGLU FFN
```

### 텐서 형태 (overfit config: d_atom=256, d_res=256, n_trunk=6)
| 단계 | Shape |
|------|-------|
| 입력 coords | `[B, L, A, 3]` A=15 (atom slots) |
| Atom tokens | `[B, L, A, 256]` |
| Residue tokens | `[B, L, 256]` |
| Trunk out | `[B, L, 256]` |
| Gradient out | `[B, L, A, 3]` |

---

## EqM 훈련

```
x_γ = γ·x_clean + (1-γ)·ε,  γ ~ U(0,1),  ε ~ N(0,I)
L_EqM = ||f(x_γ) - (ε - x_clean)·c(γ)||²

c(γ) = 4 · c_trunc(γ, a=0.8):
    = 4        if γ ≤ 0.8
    = 4·(1-γ)/0.2  if γ > 0.8  (→ 0 as γ→1)

보조 손실: CA-LDDT (구조 품질 직접 감독)
  x̂ = x_γ - scale(γ) · f(x_γ)   (1-step 복원)
  loss = L_EqM + α · L_LDDT
```

---

## 샘플링 (추론)

```
NAG-GD (논문 Eq.9, Form 1):
  y_k      = x_k + μ·(x_k - x_{k-1})         # Nesterov lookahead
  x_{k+1}  = x_k - η·f(y_k)                   # step from x_k

  μ=0.35 (논문 최적), adaptive stopping: ||f(x_k)||_rms < g_min

overfit.py 내 Euler sampler (γ-adaptive step):
  velocity = (x_hat - x) / (1-γ) = -f(x)/c(γ)
  x_{k+1}  = x_k + dγ·velocity = x_k - (dγ/c(γ))·f(x_k)
  → 완벽 모델에서 linear interpolation 경로 추종, N=50 steps로 충분
```

---

## 데이터

- **소스**: AlphaFold DB (AFDB) .pt 파일, `afdb_data/train/`
- **표준 20종 아미노산** + UNK (비표준 → skip or UNK)
- **Atom layout**: atom14 (14 heavy atoms) + OXT terminal oxygen = **15 slots (A=15)**
- **좌표 정규화**: centroid 제거 + `/10.0` (COORD_SCALE=10Å)
- **EqM corruption**: 훈련 시 γ를 50개 균일 격자에서 순환 샘플링

---

## 현재 구현 상태 (2026-03-25)

| 컴포넌트 | 상태 |
|---------|------|
| Data pipeline | ✅ 완성 (dataset, types, constants, transforms, collate) |
| Model forward | ✅ 완성 (atom embed → trunk → decoder → grad head) |
| EqM loss | ✅ 완성 (truncated_c, eqm_loss, reconstruction_scale) |
| CA-LDDT aux loss | ✅ 완성 |
| Train step (engine) | ✅ 완성 (bf16, AMP, grad clip) |
| EMA | ✅ 완성 |
| Overfit 검증 | ✅ **PASS** (job 22627: loss drop 94%, LDDT=0.996 at γ=0.99) |
| YAML config | ✅ 완성 (configs/overfit_*.yaml) |
| W&B logging | ✅ 완성 (train loss, eval metrics, sampler comparison) |
| EqM Euler sampler | ✅ (γ-adaptive step) |
| EqM NAG sampler | ✅ (Form 1, μ=0.35) |
| 3D visualization | ✅ notebooks/viz3d.ipynb |
| Full training pipeline | 🔲 미구현 (scripts/train_eqm.py 필요) |
| ESM PLM integration | 🔲 코드 있음, overfit에서 use_plm=False |
| CAMEO/CASP 평가 | 🔲 미구현 |

---

## 실행 방법

```bash
# GPU 노드에서 overfit 테스트
sbatch scripts/slurm/overfit_test.sh

# 직접 실행 (GPU 노드에서만)
PYTHONPATH=src .venv/bin/python scripts/overfit.py \
    --config configs/overfit_test.yaml \
    --out_dir outputs/overfit/test1

# W&B 없이 실행
PYTHONPATH=src .venv/bin/python scripts/overfit.py \
    --config configs/overfit_test.yaml \
    --no_wandb

# 기존 체크포인트에서 inference.npz 재생성
PYTHONPATH=src .venv/bin/python scripts/export_inference.py \
    --ckpt outputs/overfit/22627/checkpoint.pt \
    --data_dir afdb_data/train \
    --out outputs/overfit/22627/inference.npz
```

---

## 클러스터 환경

- **Master node**: 147.46.139.205 (GPU 없음, SLURM 스케줄러만)
- **test partition**: 2080Ti×4, A5000×4 (2시간 제한)
- **6000ada partition**: RTX 6000 Ada×8, 2노드 (30일 제한) ← 메인 학습용
- **heavy partition**: H100×1, Pro 6000×3 (30일 제한)
- **Python env**: `.venv` (uv, pyproject.toml 기준)
- **Mamba-3 커널**: CUDA 필수 (master 노드에서 import 불가)

---

## 핵심 논문

- **EqM**: arXiv:2510.02300 (Wang & Du, 2025) → `docs/papers/equilibrium_matching.pdf`
- **Mamba-3**: arXiv:2603.15569 → `docs/papers/mamba3.pdf`
- **SimpleFold**: `docs/papers/simplefold.pdf` (아키텍처 참고)
