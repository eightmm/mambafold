# configs/ — 실험 설정 파일

YAML 키는 `scripts/overfit.py` argparse dest 이름과 1:1 매핑.
CLI 플래그(`--lr 1e-4`)가 YAML 값을 override.

## 파일 목록

### `overfit_base.yaml` — 기본 설정 (템플릿)
모든 파라미터의 기본값. 다른 config 파일의 베이스.

### `overfit_test.yaml` — test 파티션용
`sbatch scripts/slurm/overfit_test.sh`에서 사용.
- A5000 GPU, 1:30h 제한
- 5000 steps, lr=1e-3, d=256 모델
- wandb_tags: [overfit, test-partition, a5000]

### `overfit_6000ada.yaml` — RTX 6000 Ada용
더 큰 모델 (d=384, n_trunk=8) + 더 많은 steps (10000).
- wandb_tags: [overfit, 6000ada, large-model]

## 파라미터 설명

```yaml
# 데이터
data_dir: afdb_data/train       # .pt 파일 디렉토리

# 출력
out_dir: null                   # null → outputs/overfit/{SLURM_JOB_ID|timestamp}

# 훈련
n_steps: 5000                   # 총 훈련 step 수 (50 gammas × n_cycles)
lr: 1.0e-3                      # AdamW learning rate
grad_clip: 1.0                  # gradient clipping

# 모델 크기
d_atom: 256                     # atom token 차원
d_res: 256                      # residue token 차원
d_state: 32                     # Mamba SSM state 크기
mimo_rank: 2                    # MIMO rank (chunk_size = 32//rank = 16)
headdim: 64                     # Mamba head dimension
n_atom_enc: 2                   # atom encoder BiMamba 레이어 수
n_trunk: 6                      # residue trunk BiMamba 레이어 수
n_atom_dec: 2                   # atom decoder BiMamba 레이어 수

# W&B
wandb_project: mambafold
wandb_name: null                # null → job_id 또는 timestamp 자동 사용
wandb_tags: []                  # 실험 태그
wandb_offline: false            # true → 오프라인 모드 (.wandb 파일)
no_wandb: false                 # true → W&B 완전 비활성화
```

## 사용법

```bash
# YAML만 사용
python overfit.py --config configs/overfit_test.yaml

# YAML + CLI override (lr만 변경)
python overfit.py --config configs/overfit_test.yaml --lr 3e-4

# W&B 없이
python overfit.py --config configs/overfit_test.yaml --no_wandb
```

## 앞으로 추가될 config (계획)

| 파일 | 용도 |
|------|------|
| `train_base.yaml` | 전체 데이터셋 학습 (전체 학습 설정) |
| `train_finetune.yaml` | finetune (alpha_mode=ramp, max_length=512) |
| `debug.yaml` | 빠른 디버그 (d=64, n_trunk=2, n_steps=100) |
