# tests/ - 테스트 코드

## 주의사항
- **GPU가 필요한 테스트는 반드시 SLURM 노드에서 실행**
- Master 노드에는 CUDA가 없으므로 직접 실행 불가

## 테스트 실행

```bash
# SLURM 노드에서 실행
srun --partition=test --gres=gpu:1 --time=00:30:00 python -m pytest tests/

# 또는 sbatch로 제출
sbatch scripts/slurm/run_tests.sh
```

## 테스트 구성
- `test_data.py`: Dataset 로딩, feature 전처리 검증
- `test_model.py`: 모델 forward pass, shape 검증
- `test_utils.py`: geometry, metrics 유틸 검증
