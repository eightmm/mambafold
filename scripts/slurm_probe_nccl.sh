#!/usr/bin/env bash
#SBATCH --job-name=mf-nccl
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/slurm/mf-nccl-%j.out
#SBATCH --error=logs/slurm/mf-nccl-%j.out
#
# Is NCCL usable on this node at all? Partition and --gres come from the
# submit line so the same script covers every card type:
#
#   sbatch -p test    --gres=gpu:a5000:2 scripts/slurm_probe_nccl.sh
#   sbatch -p 6000ada --gres=gpu:2       scripts/slurm_probe_nccl.sh

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
set -a; source config/paths.env; set +a
mkdir -p logs/slurm
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1

echo "node=$(hostname) job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
"$MFCLEAN_PYTHON" -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda)"

run() {  # run <label> <env assignments...>
  local label="$1"; shift
  echo; echo "--- $label ---"
  env "$@" timeout 300 "$MFCLEAN_PYTHON" -m torch.distributed.run \
    --standalone --nproc_per_node=2 --tee 3 benchmarks/probe_nccl.py
  echo "  rc=$?"
}

run "nccl + device_id (what train.py does now)"      DIST_BACKEND=nccl PROBE_DEVICE_ID=1
run "nccl, no device_id (what it did before)"        DIST_BACKEND=nccl PROBE_DEVICE_ID=0
run "nccl + device_id, P2P and SHM disabled"         DIST_BACKEND=nccl PROBE_DEVICE_ID=1 NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1
run "gloo (no NCCL at all)"                          DIST_BACKEND=gloo
echo; echo "=== done ==="
