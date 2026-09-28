#!/usr/bin/env bash
#SBATCH --job-name=mf-smoke-ddp
#SBATCH --partition=test
#SBATCH --gres=gpu:a5000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/mf-smoke-ddp-%j.out
#SBATCH --error=logs/slurm/mf-smoke-ddp-%j.out
#
# Two-rank DDP smoke. The point is the failure modes that only exist above one
# rank and that do not raise:
#
#   * the per-interval packed metric all-reduce. Every rank packs a sorted key
#     set into one tensor; if the sets ever diverge the collective HANGS rather
#     than erroring, and the first place anyone would meet that is an eight-rank
#     production run.
#   * distributed_max_int -> pad_to_length, which must put every rank on one
#     TileLang shape. A rank that compiles a shape alone strands the others
#     inside a collective until the job times out.
#   * no_sync() on the intermediate accumulation micro-step.
#   * prewarm, which compiles every binned shape before step 1.
#   * checkpoint save, then resume, then the step counter continuing.
#
# A hang is a real outcome here, so the run is wrapped in a timeout: it fails
# loudly instead of holding the allocation for the full wall clock.

set -uo pipefail
ROOT="${MFCLEAN_ROOT:-${SLURM_SUBMIT_DIR:?}}"
cd "$ROOT"
set -a; source config/paths.env; set +a
mkdir -p logs/slurm
export PYTHONUNBUFFERED=1
# A SIGSEGV kills the interpreter before it can raise, so the elastic launcher
# reports only "exitcode -11" with no Python frame. faulthandler installs a
# signal handler that prints the C-level and Python stacks from inside the
# crashing process, which is the only way to see where.
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
export NCCL_DEBUG=WARN

# Backend is overridable because it has to be. `benchmarks/probe_nccl.py` — which
# imports nothing from this project — segfaults on gpu2 for every NCCL variant
# tried (with device_id, without, and with P2P and SHM disabled) while gloo
# passes init, barrier and a value-checked all_reduce. NCCL is broken on that
# node, not here. This smoke exists to exercise *our* distributed code paths, so
# it runs on whatever backend the node can actually carry; the production
# partition gets its own NCCL verdict from the probe.
export DIST_BACKEND="${DIST_BACKEND:-gloo}"
echo "DIST_BACKEND=$DIST_BACKEND"

CONFIG=configs/smoke_ddp.yaml
OUT=out/smoke-ddp-${SLURM_JOB_ID:-local}
rc=0

echo "node=$(hostname) job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo
echo "=== 0. one rank under torchrun — is this DDP-specific at all? ==="
timeout 900 "$MFCLEAN_PYTHON" -m torch.distributed.run \
  --standalone --nproc_per_node=1 --tee 3 \
  scripts/train.py --config "$CONFIG" --out_dir "${OUT}-1rank" --no_wandb --total_steps 6
z=$?
if [ $z -eq 124 ]; then echo "0: TIMED OUT"; rc=1
elif [ $z -ne 0 ]; then echo "0: FAILED rc=$z"; rc=1
else echo "0: ok"; fi

echo
echo "=== A. 2-rank DDP, steps 1..30, checkpoint at 10/20/30 ==="
timeout 3000 "$MFCLEAN_PYTHON" -m torch.distributed.run \
  --standalone --nproc_per_node=2 --tee 3 \
  scripts/train.py --config "$CONFIG" --out_dir "$OUT" --no_wandb
a=$?
if [ $a -eq 124 ]; then echo "A: TIMED OUT — probable collective hang"; rc=1
elif [ $a -ne 0 ]; then echo "A: FAILED rc=$a"; rc=1
else echo "A: ok"; fi

echo
echo "=== checkpoints written ==="
ls -la "$OUT" 2>/dev/null | tail -6

CKPT=$(ls -1t "$OUT"/*.pt 2>/dev/null | head -1)
if [ -z "${CKPT:-}" ]; then
  echo "B: SKIPPED — no checkpoint to resume from"; rc=1
else
  echo
  echo "=== B. resume from $CKPT, run to 40 steps ==="
  timeout 1800 "$MFCLEAN_PYTHON" -m torch.distributed.run \
    --standalone --nproc_per_node=2 --tee 3 \
    scripts/train.py --config "$CONFIG" --out_dir "$OUT" --no_wandb \
    --resume "$CKPT" --total_steps 40
  b=$?
  if [ $b -eq 124 ]; then echo "B: TIMED OUT — probable collective hang"; rc=1
  elif [ $b -ne 0 ]; then echo "B: FAILED rc=$b"; rc=1
  else echo "B: ok"; fi
fi

echo
echo "=== summary ==="
echo "  0 single rank             : $([ ${z:-1} -eq 0 ] && echo ok || echo FAILED)"
echo "  A 2-rank DDP + checkpoint : $([ ${a:-1} -eq 0 ] && echo ok || echo FAILED)"
echo "  B resume + continue       : $([ "${b:-1}" -eq 0 ] && echo ok || echo FAILED)"
exit $rc
