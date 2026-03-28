# utils/ — Utilities

## `geometry.py`

**remove_translation(coords, atom_mask)** — Remove centroid for SE(3) invariance.

```python
centroid = (coords * mask).sum(dim=(1,2)) / mask.sum(dim=(1,2))
return coords - centroid
```

Applied every step during sampling to prevent translation drift.

**pairwise_distances(coords, valid_mask)** — Compute all-to-all distances for LDDT.

**random_rotation(coords, atom_mask)** — Random SO(3) rotation for data augmentation.

## `metrics.py`
CA-LDDT, TM-score, GDT-TS (optional external tools).
