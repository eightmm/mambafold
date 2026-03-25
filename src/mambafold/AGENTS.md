# src/mambafold — 패키지 루트

## 모듈 구조

```
mambafold/
├── data/       데이터 파이프라인 (로딩, 타입, 상수, 전처리)
├── model/      모델 아키텍처 (embedding, SSM, encoder, decoder)
│   └── ssm/    Mamba-3 레이어 (공식 커널 wrapper + BiMamba)
├── losses/     EqM loss, CA-LDDT auxiliary loss
├── sampling/   EqMNAGSampler (추론용 gradient descent 샘플러)
├── train/      train_step, eval_step, EMA
└── utils/      geometry (translation removal, pairwise dist)
```

## 의존 관계

```
scripts/ → train/, losses/, model/, data/, sampling/
model/   → data/constants, data/types
losses/  → (독립)
sampling/→ model/, data/types, utils/
train/   → model/, losses/, data/types
```

## 주요 진입점

| 목적 | 경로 |
|------|------|
| 모델 생성 | `model/mambafold.py::MambaFoldEqM` |
| 훈련 1 step | `train/engine.py::train_step` |
| 평가 1 step | `train/engine.py::eval_step` |
| EqM loss | `losses/eqm.py::eqm_loss` |
| NAG 샘플링 | `sampling/sampler.py::EqMNAGSampler` |
| 데이터 | `data/dataset.py::AFDBDataset` |

## 설치 / PYTHONPATH

```bash
PYTHONPATH=src python scripts/overfit.py ...
# 또는 editable install (pyproject.toml)
pip install -e .
```

## GPU 요구사항

Mamba-3 공식 Triton 커널 사용 → **CUDA GPU 필수**.
Master node (CPU only)에서는 import 가능하지만 forward pass 불가.
