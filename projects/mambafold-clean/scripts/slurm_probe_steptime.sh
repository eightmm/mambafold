#!/usr/bin/env bash
#SBATCH --job-name=mf-steptime
#SBATCH --partition=test
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/mf-steptime-%j.out
#SBATCH --error=logs/slurm/mf-steptime-%j.out
#
# Where does a training step actually go? Two measurements of the same work
# disagree by 3.7x: probe_train_memory's fit (194 ms + 135.9 ms/1000 residues)
# predicts 298 ms for a 768-residue step, while probe_overfit measured 1091 ms.
# Collation is ruled out at 2.1 ms and the lDDT chunking is ruled out because
# the memory probe does more of it. This splits the step into collate,
# host-to-device, forward+metric syncs, backward and optimizer.
#
# Single GPU, so gpu2's broken NCCL is irrelevant here.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
set -a; source config/paths.env; set +a
mkdir -p logs/slurm data/audit/gpu_probe
export PYTHONUNBUFFERED=1

echo "node=$(hostname) job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$MFCLEAN_PYTHON" benchmarks/probe_overfit.py \
  --config configs/run_a_mamba.yaml \
  --data_dir data/rcsb_train \
  --file_list data/audit/gpu_probe_filelist.txt \
  --esm_dir data/rcsb_esmc6b \
  --n_proteins 4 --max_length 256 --copies 4 --steps 120 --profile \
  --out data/audit/gpu_probe/steptime_a5000.json
echo "  rc=$?"
