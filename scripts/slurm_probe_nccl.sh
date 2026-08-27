#!/usr/bin/env bash
#SBATCH --job-name=mf-nccl
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
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

echo "node=$(hostname) job=${SLURM_JOB_ID:-none} gres=${SLURM_GPUS_ON_NODE:-?}"
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
"$MFCLEAN_PYTHON" -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda)"

# The interconnect matters here: RTX 6000 Ada has no NVLink, so every rank pair
# talks over PCIe or through host memory, and which of those NCCL picks is the
# thing under test.
echo; echo "=== nvidia-smi topo -m ==="
nvidia-smi topo -m 2>&1 || echo "  (topo unavailable)"
echo; echo "=== /dev/shm ==="
df -h /dev/shm 2>&1 | tail -2

# Rank counts to sweep. Two ranks pass on gpu3 and eight hang on the first
# barrier there, so the count is a variable, not a constant: where it starts
# hanging separates a transport that never works from a ring that cannot be
# built across all eight cards. Override with PROBE_NPROCS="2 4 8".
NPROCS="${PROBE_NPROCS:-2 4 8}"

run() {  # run <nproc> <label> <env assignments...>
  local np="$1"; local label="$2"; shift 2
  echo; echo "--- [${np} ranks] $label ---"
  env "$@" timeout "${PROBE_TIMEOUT:-300}" "$MFCLEAN_PYTHON" -m torch.distributed.run \
    --standalone --nproc_per_node="$np" --tee 3 benchmarks/probe_nccl.py 2>&1 \
    | grep -vE "^\\[default[0-9]+\\]:.*NCCL INFO (Channel|Connected|comm |ENV/Plugin)"
  echo "  rc=${PIPESTATUS[0]}"
}

for NP in $NPROCS; do
  if [ "${SLURM_GPUS_ON_NODE:-0}" -lt "$NP" ] 2>/dev/null; then
    echo; echo "--- [${NP} ranks] SKIPPED: only ${SLURM_GPUS_ON_NODE:-?} GPUs allocated ---"
    continue
  fi
  run "$NP" "nccl, default transports"        DIST_BACKEND=nccl PROBE_DEVICE_ID=1
  run "$NP" "nccl, P2P disabled"              DIST_BACKEND=nccl PROBE_DEVICE_ID=1 NCCL_P2P_DISABLE=1
  run "$NP" "nccl, SHM disabled"              DIST_BACKEND=nccl PROBE_DEVICE_ID=1 NCCL_SHM_DISABLE=1
  run "$NP" "nccl, P2P and SHM disabled"      DIST_BACKEND=nccl PROBE_DEVICE_ID=1 NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1
  run "$NP" "nccl, socket over loopback"      DIST_BACKEND=nccl PROBE_DEVICE_ID=1 NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_SOCKET_IFNAME=lo
  run "$NP" "gloo (no NCCL at all)"           DIST_BACKEND=gloo
done
echo; echo "=== done ==="
