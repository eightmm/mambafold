# tests/ - Unit Tests

## Important
GPU tests must run on SLURM compute nodes (master node has no CUDA).

## Running Tests

```bash
# On GPU node via srun
srun --partition=test --gres=gpu:1 --time=00:30:00 python -m pytest tests/

# Or submit via sbatch
sbatch scripts/slurm/run_tests.sh
```

## Test Files
- `test_data.py` — Dataset loading, data transforms
- `test_model.py` — Model forward pass, shape validation
- `test_forward_shapes.py` — End-to-end shape pipeline
- `test_eqm.py` — EqM loss math verification
- `test_sampler.py` — NAG/Euler sampler validation
- `test_utils.py` — Geometry utilities, metrics
