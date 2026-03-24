# MambaFold - Mamba 기반 Single-Chain Protein Structure Prediction

Mamba3 (State Space Model) 아키텍처를 활용한 단일 체인 단백질 3D 구조 예측 모델.

## 프로젝트 구조

```
folding/
├── README.md
├── pyproject.toml          # uv 기반 프로젝트 설정
├── afdb_data/              # AFDB 데이터 (symlink, 읽기 전용)
│   ├── train/              # 901,497개 .pt 파일
│   ├── val/                # 44,175개 .pt 파일
│   ├── test/               # 20,571개 .pt 파일
│   └── errors/             # 에러 로그
├── src/mambafold/          # 메인 소스 코드
│   ├── model/              # Mamba backbone + structure module
│   ├── data/               # Dataset, DataLoader, feature engineering
│   └── utils/              # geometry, metrics, training utils
├── configs/                # YAML 설정 파일
├── scripts/                # 학습/평가 스크립트
│   └── slurm/              # SLURM 제출 스크립트
├── tests/                  # 테스트 코드
└── docs/                   # 문서 및 참고 논문
    └── papers/             # Mamba3, SimpleFold PDF
```

## 데이터

AFDB(AlphaFold DB)에서 추출한 단일 체인 단백질 구조 데이터.
각 `.pt` 파일은 다음 정보를 포함:

| Key | Type | 설명 |
|-----|------|------|
| `res_names` | list[str] | 잔기 이름 (e.g., MET, VAL, LEU) |
| `res_seq_nums` | list[int] | 잔기 번호 |
| `res_ins_codes` | list[str] | Insertion code |
| `atom_names` | list[list[str]] | 잔기별 원자 이름 (N, CA, C, O, CB, ...) |
| `atom_nums` | list[list[int]] | 원자 번호 |
| `coords` | list[list[list[float]]] | 원자별 3D 좌표 (x, y, z) |
| `is_observed` | list[list[bool]] | 관측 여부 |

## 환경 설정

```bash
uv sync
```

## 학습

```bash
sbatch scripts/slurm/train.sh
```
