# Inference

Inference uses one direct all-atom Euler trajectory.

```bash
PYTHONPATH=src uv run python benchmarks/run_inference.py \
  --ckpt outputs/train/<run>/ckpt_latest.pt \
  --ids benchmarks/sets/t1_quick.txt \
  --out benchmarks/results/<run>_t1 \
  --esm_dir data/rcsb_esm \
  --n_steps 50
```

Outputs:

- `<target>_pred.pdb`
- `<target>_pred_seed<i>.pdb`
- `<target>_gt.pdb`

Predicted per-residue confidence is written as per-atom B-factors in
`*_pred*.pdb`. Ground-truth PDBs only emit observed atoms, so all-atom scoring
does not compare against missing side-chain placeholders.

Score with SimpleFold-style metrics:

```bash
tools/scoring_venv/bin/python benchmarks/score_simplefold_metrics.py \
  --in_dir benchmarks/results/<run>_t1 \
  --out benchmarks/results/<run>_t1/scores.json
```
