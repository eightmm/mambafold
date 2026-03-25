# notebooks/ — 시각화 노트북

**중요**: 노트북은 CPU(master node)에서 실행. 모델 추론 없이 `.npz` 파일만 로드.
Mamba-3 커널은 CUDA 필수이므로 노트북에서 모델 forward 금지.

## `viz3d.ipynb` — 3D 구조 시각화

`outputs/overfit/<job_id>/inference.npz`를 로드해 다음을 시각화:

### 셀 구성

| 셀 | 내용 |
|----|------|
| 1 | `%matplotlib widget` — 3D 인터랙티브 활성화 |
| 2 | imports (numpy, matplotlib) |
| 3~4 | Config: `JOB_ID`, npz 경로 설정 |
| 5~6 | 데이터 로드: `x_clean_ca`, `x_noisy_ca`, `x_hat_ca`, euler/nag 결과 |
| 7~8 | **1-step 3D 인터랙티브**: 특정 γ에서 GT vs Noisy vs Reconstructed |
| 9~10 | **EqM Euler 궤적**: noise→structure Blues colormap |
| 11~12 | **EqM NAG 궤적**: noise→structure Purples colormap |
| 13~14 | **3-way 비교**: 1-step vs Euler vs NAG (side-by-side 3D), `viz3d_compare.png` 저장 |
| 15~16 | **RMSD/LDDT 곡선**: γ별 성능 + Euler/NAG 수평선 |
| 17~18 | **Euler/NAG step 슬라이더**: ipywidgets로 궤적 재생 |
| 19~20 | **γ 슬라이더**: 1-step 결과 γ별 탐색 |

### 실행 전 필요 조건

1. GPU 노드에서 `overfit.py` 실행 완료 → `inference.npz` 생성
2. `JOB_ID` 변수를 실제 job ID로 변경
3. Jupyter 커널: **"MambaFold (.venv)"** 선택
   ```bash
   # 커널 등록 (최초 1회)
   .venv/bin/python -m ipykernel install --user --name mambafold --display-name "MambaFold (.venv)"
   ```
4. `%matplotlib widget` 사용 → `ipympl` 필요 (pyproject.toml에 포함됨)

### inference.npz 구조

```
x_clean_ca    [G, S, L, 3]   γ별 ground truth Cα (Å)
x_noisy_ca    [G, S, L, 3]   γ별 noisy input Cα (Å)
x_hat_ca      [G, S, L, 3]   γ별 1-step reconstruction Cα (Å)
ca_mask       [L]             Cα 유효 마스크
gammas        [G]             γ grid (50개)
rmsds         [G, S]          per-gamma per-seed RMSD (Å)
lddts         [G, S]          per-gamma per-seed hard LDDT
euler_traj_ca [DS, T, L, 3]  Euler 궤적 (DS=seeds, T=steps)
euler_final_ca [DS, L, 3]    Euler 최종 구조
euler_gammas  [T+1]           Euler γ schedule
euler_rmsd    [DS]            Euler 최종 RMSD (Å)
nag_traj_ca   [DS, T, L, 3]  NAG 궤적
nag_final_ca  [DS, L, 3]     NAG 최종 구조
nag_gammas    [T+1]           NAG γ schedule
nag_rmsd      [DS]            NAG 최종 RMSD (Å)
```
