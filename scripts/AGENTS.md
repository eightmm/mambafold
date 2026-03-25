# scripts/ — 실행 스크립트

## `overfit.py` — 메인 overfit 검증 스크립트

1개 단백질에 대해 **훈련 → 평가 → EqM 샘플링 → 시각화**를 한 번에 수행.
모델이 데이터에 overfit 할 수 있는지 검증 (PASS 기준: loss drop ≥ 50%).

```bash
PYTHONPATH=src python scripts/overfit.py \
    --config configs/overfit_test.yaml \
    --out_dir outputs/overfit/<job_id>
```

**출력 파일** (`--out_dir` 아래):
| 파일 | 내용 |
|------|------|
| `config.json` | 실험 설정 전체 |
| `checkpoint.pt` | model + ema + optimizer state |
| `metrics.json` | per-gamma LDDT/RMSD + Euler/NAG sampler RMSD |
| `viz.png` | 6패널 시각화 (loss curve, loss by gamma, LDDT, RMSD, Cα overlay) |
| `inference.npz` | 전체 inference 결과 (노트북 로드용) |

**inference.npz 내용**:
- `x_clean_ca`, `x_noisy_ca`, `x_hat_ca` [G, S, L, 3] — per-gamma 1-step 결과
- `euler_traj_ca`, `euler_final_ca`, `euler_rmsd` — Euler ODE 샘플러 결과
- `nag_traj_ca`, `nag_final_ca`, `nag_rmsd` — NAG 샘플러 결과

**W&B 로깅**:
- `train/loss`: 매 step
- `eval/rmsd_vs_gamma`, `eval/lddt_vs_gamma`: line chart
- `eval/euler_rmsd_mean`, `eval/nag_rmsd_mean`: 샘플러 비교
- `eval/viz`: viz.png 이미지

**YAML + CLI 우선순위**: `--config`로 기본값 로드, CLI 플래그로 override.
`--no_wandb`: W&B 비활성화.

---

## `export_inference.py` — GPU inference → .npz 저장

기존 체크포인트에서 inference.npz를 재생성. 노트북 시각화를 위해 사용.

```bash
PYTHONPATH=src python scripts/export_inference.py \
    --ckpt outputs/overfit/22627/checkpoint.pt \
    --data_dir afdb_data/train \
    --out outputs/overfit/22627/inference.npz \
    --n_sample_steps 50 --n_sample_seeds 3
```

overfit.py와 동일한 sampler 구현 포함 (eqm_euler, eqm_nag).
**GPU 노드에서만 실행 가능** (Mamba-3 커널).

---

## `slurm/` — SLURM 배치 스크립트

→ `slurm/AGENTS.md` 참조

---

## 미구현 (계획)

| 스크립트 | 용도 |
|---------|------|
| `train_eqm.py` | 전체 데이터셋 훈련 (multi-GPU, LengthBucketSampler) |
| `precompute_esm.py` | ESM 임베딩 오프라인 캐시 생성 |
| `sample_structure.py` | 체크포인트 + 시퀀스 → PDB 출력 |
