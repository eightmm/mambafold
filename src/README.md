# src/ - 소스 코드

## 구조

```
src/mambafold/
├── __init__.py
├── data/                    # 데이터 파이프라인 (구현 완료)
│   ├── __init__.py
│   ├── constants.py         # AA vocab(21), atom slot table(15), coord scale
│   ├── types.py             # ProteinExample, ProteinBatch dataclass
│   ├── dataset.py           # AFDBDataset — .pt 로딩, canonical atom slot 매핑
│   ├── transforms.py        # CenterScale, SO3Aug, EqMCorrupt
│   ├── collate.py           # ProteinCollator, LengthBucketSampler
│   └── esm.py              # (TODO) ESM2 embedding 캐시/로딩
├── model/                   # 모델 (구현 예정)
│   ├── __init__.py
│   ├── embeddings.py        # (TODO) Fourier PE, atom/residue features
│   ├── ssm/                 # (TODO) Mamba-3 wrapper
│   │   ├── mamba3.py        # Mamba3Layer wrapper
│   │   └── bimamba3.py      # BiMamba3Block (forward+reverse+gate)
│   ├── blocks.py            # (TODO) RMSNorm, SwiGLU, ConditioningAdapter
│   ├── atom_encoder.py      # (TODO) local attention per-residue
│   ├── grouping.py          # (TODO) atoms↔residues pooling/broadcast
│   ├── residue_trunk.py     # (TODO) heavy BiMamba-3 stack
│   ├── atom_decoder.py      # (TODO) local attention + gradient head
│   └── mambafold.py         # (TODO) MambaFoldEqM end-to-end
├── losses/                  # (TODO) Loss 함수
│   ├── eqm.py              # EqM loss, c(γ), reconstruction scale
│   └── lddt.py             # differentiable CA-LDDT
├── sampling/                # (TODO) 추론
│   └── sampler.py           # EqMNAGSampler
├── train/                   # (TODO) 학습 엔진
│   ├── engine.py            # train_step, eval_step
│   └── ema.py              # EMA weights
└── utils/                   # 유틸리티 (일부 구현 완료)
    ├── __init__.py
    ├── geometry.py          # SO3 rotation, centroid, pairwise distances (구현 완료)
    └── metrics.py           # (TODO) ca_lddt, rmsd
```

## 구현 상태

| 모듈 | 상태 | 검증 |
|------|------|------|
| data/constants.py | 완료 | AA 21종, atom 37종, TRP 14 atoms 확인 |
| data/types.py | 완료 | ProteinExample/Batch dataclass |
| data/dataset.py | 완료 | AFDB test 20,571개 로딩 성공 |
| data/transforms.py | 완료 | SO3 det=1.0, EqM corrupt 정상 |
| data/collate.py | 완료 | Batch(B=4, L=128, A=15) 정상 |
| utils/geometry.py | 완료 | SO3, centroid, pairwise dist |
| model/* | 미구현 | Phase 1 |
| losses/* | 미구현 | Phase 2 |

## 데이터 스키마

### ProteinExample (단일 단백질)
- `res_type: [L]` — AA type ID (0~20)
- `atom_type: [L, 15]` — atom slot ID
- `coords: [L, 15, 3]` — 3D coordinates (Å)
- `atom_mask: [L, 15]` — valid atom slots
- `observed_mask: [L, 15]` — experimentally observed

### ProteinBatch (배치)
- `x_clean: [B, L, 15, 3]` — normalized clean coords
- `x_gamma: [B, L, 15, 3]` — EqM corrupted coords
- `eps: [B, L, 15, 3]` — noise
- `gamma: [B, 1, 1, 1]` — interpolation factor
- `esm: [B, L, d_esm]` — ESM2 embeddings (optional)
