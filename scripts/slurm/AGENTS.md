# scripts/slurm/ — SLURM Batch Scripts

## `overfit_test.sh`
Quick overfit validation on test partition.

```bash
sbatch scripts/slurm/overfit_test.sh
```

Resources: A5000 GPU, 4 CPUs, 16GB RAM, 1:30h time limit.

## `overfit_6000ada.sh`
Overfit on 6000ada partition (larger model).

```bash
sbatch scripts/slurm/overfit_6000ada.sh
```

## `train_6000ada.sh`
Full training on 6000ada partition.

Auto-detects GPU count from `SLURM_GPUS_ON_NODE`:
- Single GPU → single-process training
- Multi-GPU → DDP via `torchrun --nproc_per_node=$N_GPU`

```bash
sbatch scripts/slurm/train_6000ada.sh

# Resume
RESUME=outputs/train/run1/ckpt_latest.pt sbatch scripts/slurm/train_6000ada.sh
```

## `overfit_plm.yaml`
Overfit with ESM PLM enabled.

## Partitions

| Partition | GPUs | Time | Use |
|-----------|------|------|-----|
| test | 2080Ti×4, A5000×4 | 2h | Quick validation |
| 6000ada | RTX 6000 Ada×8 (2 nodes) | 30d | Main training |
| heavy | H100×1, 6000 Pro×3 | 30d | Large models |

**Note**: GPU code must run on compute nodes (master node has no CUDA).
