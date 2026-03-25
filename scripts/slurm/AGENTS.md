# scripts/slurm/ — SLURM 배치 스크립트

## `overfit_test.sh`

test 파티션 (A5000 GPU) 에서 overfit 검증 실행.

```bash
sbatch scripts/slurm/overfit_test.sh
```

**리소스**: `test` 파티션, A5000 × 1, CPU 4, 메모리 16G, 시간 1:30:00

**실행 내용**:
1. CUDA 12.8 모듈 로드
2. `.venv/bin/python` 으로 환경 확인
3. `overfit.py --config configs/overfit_test.yaml --out_dir outputs/overfit/${SLURM_JOB_ID}`

**출력**: `outputs/overfit/<job_id>/slurm.out`

---

## 파티션별 스크립트 계획

| 스크립트 | 파티션 | 용도 |
|---------|--------|------|
| `overfit_test.sh` | test (A5000) | ✅ 구현됨. overfit 검증 |
| `overfit_6000ada.sh` | 6000ada | 🔲 미구현. 큰 모델 overfit |
| `train.sh` | 6000ada | 🔲 미구현. 전체 학습 |
| `run_tests.sh` | test | 🔲 미구현. 단위 테스트 |

---

## 공통 패턴

```bash
#!/bin/bash
#SBATCH --partition=<파티션>
#SBATCH --gres=gpu:<gpu_type>:<count>
#SBATCH --output=outputs/<subdir>/%j/slurm.out

cd /home/jaemin/project/protein/folding
VENV_PY=.venv/bin/python
module load cuda/12.8
PYTHONPATH=src $VENV_PY scripts/<script>.py --config configs/<config>.yaml ...
```

---

## 클러스터 파티션 요약

| 파티션 | GPU | 시간 제한 | 용도 |
|--------|-----|---------|------|
| test | 2080Ti×4, A5000×4 | 2시간 | 빠른 검증 |
| 6000ada | RTX 6000 Ada×8 (2노드) | 30일 | 메인 학습 |
| heavy | H100×1, Pro6000×3 | 30일 | 대형 모델 |
| cpu_only | - | 3일 | 전처리 |

**주의**: master node에서 GPU 연산 불가. 모든 forward pass는 SLURM으로 제출.
