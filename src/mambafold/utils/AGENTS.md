# utils/ — 유틸리티

## `geometry.py`

### `remove_translation(coords, atom_mask)`
좌표에서 centroid 제거. SE(3) invariance 유지를 위해 샘플링 루프 매 step 적용.

```python
# coords: [B, L, A, 3], atom_mask: [B, L, A]
centroid = (coords * mask).sum(dim=(1,2)) / mask.sum(dim=(1,2)).clamp(1)
return coords - centroid.unsqueeze(1).unsqueeze(1)
```

샘플링 시 gradient step이 미세한 translation drift를 누적할 수 있음 →
`remove_translation`으로 매 step 보정.

### 기타 함수 (geometry.py)
- pairwise distances 계산 (LDDT 계산에 활용)
- 필요 시 random SO(3) rotation (데이터 augmentation용, 현재 overfit에서 미사용)
