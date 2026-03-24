# docs/ - 문서 및 참고 자료

## 구조

```
docs/
├── README.md
└── papers/
    ├── OVERVIEW.md                      # 전체 논문 관계 및 프로젝트 설계 청사진
    ├── mamba3.pdf                       # Mamba-3 원본 PDF
    ├── mamba3_summary.md                # Mamba-3 상세 정리
    ├── simplefold.pdf                   # SimpleFold 원본 PDF
    ├── simplefold_summary.md            # SimpleFold 상세 정리
    ├── equilibrium_matching.pdf         # Equilibrium Matching 원본 PDF
    └── equilibrium_matching_summary.md  # Equilibrium Matching 상세 정리
```

## 읽는 순서

1. **[OVERVIEW.md](papers/OVERVIEW.md)** — 프로젝트 목표, 논문 3편의 역할, 아키텍처 청사진
2. **[simplefold_summary.md](papers/simplefold_summary.md)** — folding 파이프라인 전체 구조 (우리 프로젝트의 뼈대)
3. **[mamba3_summary.md](papers/mamba3_summary.md)** — backbone SSM 아키텍처 (Transformer 대체)
4. **[equilibrium_matching_summary.md](papers/equilibrium_matching_summary.md)** — 생성 프레임워크 대안 (FM → EqM 실험용)

## 참고 논문 요약

### Mamba-3 (ICLR 2026)
- **역할**: Backbone 아키텍처 (Transformer 대체)
- Sub-quadratic SSM: O(L) compute, O(1) memory
- 3대 혁신: exponential-trapezoidal discretization, complex SSM (data-dependent RoPE), MIMO
- 1.5B scale에서 Transformer/Mamba-2/GDN 대비 최고 성능

### SimpleFold (Apple, 2025)
- **역할**: Protein folding 파이프라인 전체 설계
- Flow matching으로 noise → all-atom 3D 구조 생성
- Atom Encoder → Residue Trunk → Atom Decoder 구조
- ESM2 PLM conditioning, LDDT loss, 2-stage training
- Domain-specific 모듈 없이 ESMFold 수준 달성 (CASP14에서 능가)

### Equilibrium Matching (MIT/Harvard, 2025)
- **역할**: Flow matching 대안 생성 프레임워크
- Time conditioning 제거, energy landscape의 equilibrium gradient 학습
- Gradient descent로 sampling (adaptive compute, 유연한 step size)
- ImageNet FID 1.90 (FM 대비 우수)
