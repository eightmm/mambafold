# scripts/slurm/ — SLURM Batch Scripts

## `train_h100.sh` ⭐ main
Single H100 training (no DDP, no NCCL hangs). Takes `CONFIG` and `RESUME`
env vars; extra args are forwarded to `scripts/train.py`.

```bash
# Stage A pretrain (fresh)
sbatch scripts/slurm/train_h100.sh
# defaults to CONFIG=configs/pretrain_256.yaml

# Stage A resume
RESUME=outputs/train/<run>/ckpt_latest.pt sbatch scripts/slurm/train_h100.sh

# Stage B finetune_512 (stage transition)
CONFIG=configs/finetune_512.yaml \
  RESUME=outputs/train/<pretrain_run>/ckpt_latest.pt \
  sbatch scripts/slurm/train_h100.sh --reset_optimizer

# Stage C finetune_1024
CONFIG=configs/finetune_1024.yaml \
  RESUME=outputs/train/<512_run>/ckpt_latest.pt \
  sbatch scripts/slurm/train_h100.sh --reset_optimizer
```

## `train.sh` — legacy multi-GPU
4× RTX 6000 Ada via torchrun. Kept for reference; NCCL hangs observed,
prefer `train_h100.sh` unless you really need multi-GPU throughput.

## `overfit.sh`
Single-GPU overfit sanity on a small dataset (~16 proteins).

```bash
sbatch scripts/slurm/overfit.sh   # uses configs/overfit.yaml
```

## `infer_train.sh`
Train-set inference: 8 proteins × 3 seeds × (Euler + NAG), computes
Cα/all-atom RMSD + per-atom lDDT, saves Kabsch-aligned all-atom PDBs
with B-factor = pLDDT.

```bash
# default (uses latest 450k checkpoint from 26367)
sbatch scripts/slurm/infer_train.sh

# explicit
CKPT=outputs/train/<run>/ckpt_latest.pt \
  OUT=outputs/infer_train/<tag> \
  PDB_IDS="1eid,7uvm,5lfl,8ki5" \
  sbatch scripts/slurm/infer_train.sh
```

## `infer_2025.sh`
Novel-sequence inference on A5000 (test partition). Downloads the
mmCIF from RCSB when the PDB ID exists.

```bash
sbatch scripts/slurm/infer_2025.sh <PDB_ID> <SEQUENCE>
# e.g. sbatch scripts/slurm/infer_2025.sh 10AF "MAHHHH..."
```

## `precompute_esm.sh`
Bulk ESM3-open embedding cache (`data/rcsb_esm/`). 2-phase with dedup.

## Partitions

| Partition | GPUs | Time | Use |
|-----------|------|------|-----|
| `test` | 2080Ti×4, A5000×4 | 2h | Inference, quick validation |
| `6000ada` | RTX 6000 Ada×8 (2 nodes) | 30d | Legacy multi-GPU training |
| `heavy` | H100×1, RTX 6000 Pro×3 | 30d | Main training (H100) |

**Rules**:
- GPU code must run on compute nodes — master node has no CUDA.
- Avoid `--cpus-per-task` unless explicitly needed.
- Don't specify `#SBATCH --ntasks=` — SLURM schedules the default.
