#!/usr/bin/env bash
#SBATCH --job-name=mf-esmc6b-data
#SBATCH --partition=cpu_only
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --time=3-00:00:00
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export PYTHONPATH="${PYTHONPATH:-src}"

uv run --no-sync python -u - <<'PY'
from mambafold.data.loader import build_dataloaders
from mambafold.train.config import parse_args
from mambafold.train.distributed import resolve_dataloader_workers

config = "configs/direct_allatom_puremamba_attn6_geo_adaln_sf360_esmc6b_gpu8.yaml"
args, _ = parse_args(["--config", config])

# The dataset constructors build/reuse the content-validated chain indexes.
# Fit index workers to the CPUs granted by Slurm's default allocation. The
# script intentionally does not request a CPU count.
requested_index_workers = args.length_cache_workers
effective, available, source = resolve_dataloader_workers(
    requested_index_workers, world_size=1
)
args.length_cache_workers = max(1, effective)
print(
    f"index_workers requested={requested_index_workers} "
    f"effective={args.length_cache_workers} available_cpus={available} source={source}",
    flush=True,
)

# Keep the one-batch preflight single-process.
args.num_workers = 0
loader, _sampler, _val_loader, dataset = build_dataloaders(args, is_dist=False)
batch = next(iter(loader))
if batch is None or batch.esm is None:
    raise RuntimeError("ESMC preflight produced an empty batch")
if batch.res_type.shape[0] != args.batch_size:
    raise RuntimeError(
        f"ESMC preflight dropped samples: batch={batch.res_type.shape[0]} "
        f"expected={args.batch_size}"
    )
if batch.esm.ndim != 3 or batch.esm.shape[-1] != 2560:
    raise RuntimeError(f"Unexpected ESMC batch shape: {tuple(batch.esm.shape)}")
print(dataset.summary(), flush=True)
print(
    f"preflight_ok batch={tuple(batch.res_type.shape)} "
    f"esm={tuple(batch.esm.shape)} dtype={batch.esm.dtype}",
    flush=True,
)
PY
