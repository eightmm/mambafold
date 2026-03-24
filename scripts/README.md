# scripts/ - 실행 스크립트

## 구조

```
scripts/
├── README.md
├── slurm/          # SLURM 클러스터 제출 스크립트
│   └── (sbatch 스크립트)
├── train.py        # 학습 엔트리포인트
├── evaluate.py     # 평가/추론 스크립트
└── preprocess.py   # 데이터 전처리 (필요시)
```

## SLURM 사용법
- `6000ada` 파티션: RTX 6000 Ada x8 (메인 학습용)
- `heavy` 파티션: H100 (대형 실험)
- `test` 파티션: 빠른 디버깅 (2시간 제한)

```bash
sbatch scripts/slurm/train.sh
```
