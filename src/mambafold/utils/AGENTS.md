# utils/ — Utilities

## `geometry.py`

- `random_rotation_matrix` and `apply_rotation` support coordinate augmentation.
- `masked_centroid` is shared by augmentation and per-example sampler centering.
- `weighted_rigid_align` aligns clean coordinates to detached predictions for
  the flow-matching target.
