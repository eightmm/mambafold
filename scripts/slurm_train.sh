#!/usr/bin/env bash
#SBATCH --job-name=mf-train
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --qos=verylong
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/slurm/mf-train-%j.out
#SBATCH --error=logs/slurm/mf-train-%j.out
#
# Run A. One node, eight RTX 6000 Ada, DDP via torchrun.
#
#   sbatch scripts/slurm_train.sh
#   sbatch scripts/slurm_train.sh --resume out/run-a/latest.pt
#
# --- why these numbers -------------------------------------------------------
#
# The 6000ada nodes here report CPUTot=64 and RealMemory=250000, so both numbers
# below are bounded by the node and not chosen freely.
#
# cpus-per-task 48. The loader runs `num_workers: 8` PER RANK, so eight ranks
# spawn 64 worker processes plus eight mains — 1.33x oversubscribed on 48 cores.
# That is deliberate and it is affordable: one example costs 105.7 ms on average
# (p90 238, p99 336, max 462, measured over 1,016 real chains) while a step
# consumes one example per micro-step, which leaves roughly 26x more supply than
# demand. Even after the oversubscription the margin is an order of magnitude.
# Asking for all 64 would instead queue behind any other job on the node.
# `num_workers` stays at 8 rather than being tuned, because the first run should
# read `data_wait` out of the log before anyone tunes against a guess.
#
# mem 200G of the node's 250G. Eight ranks x eight workers each hold a decoded
# record plus its ESMC-6B rows; the embeddings are 2560-wide and a 1024-residue
# chain is ~10 MB before the collator copies it.
#
# gres gpu:8 and no --nodes: memory says one node is enough. The measured fit is
# 4.89 GiB + 2.538 MiB/residue (job 57848), and `copies 8 x 1024` = 8,192
# residues projects to 28.2 GiB per rank once the EMA copy and DDP's gradient
# buckets are added — 59% of a 47.4 GiB card.
#
# qos verylong / time 7d. The 6000ada partition advertises MaxTime=30 days but
# the QOS caps it, and a request above the cap does not fail — it sits PENDING
# with reason QOSMaxWallDurationPerJobLimit forever. The ceilings available to
# this account are veryshort 4h, short 12h, normal 1d, long 3d, verylong 7d,
# and verylong allows two concurrent jobs, which is enough to chain a run.
#
# 300k steps should fit: extrapolating the A5000 measurement (422 ms at 768
# residues with the real loss) to a typical 2,560-residue micro-step and two
# micro-steps per optimiser step puts an Ada step near 850 ms, so about three
# days. That is an estimate on a card this has never run on, so read the first
# logged step time and re-plan if it disagrees. If a run needs more than seven
# days, chain it:
#
#   sbatch scripts/slurm_train.sh --resume out/<run>/ckpt_latest.pt
#
# signal B:USR1@300 gives the trainer five minutes to checkpoint before the wall
# clock kills it; scripts/train.py writes .requeue_requested when it catches it.

set -euo pipefail
ROOT="${MFCLEAN_ROOT:-${SLURM_SUBMIT_DIR:?}}"
cd "$ROOT"
set -a; source config/paths.env; set +a
mkdir -p logs/slurm out

CONFIG="${MF_CONFIG:-configs/run_a_mamba.yaml}"
OUT_DIR="${MF_OUT_DIR:-out/run-a-mamba3-1024-atom14}"
NPROC="${SLURM_GPUS_ON_NODE:-8}"

echo "node=$(hostname) job=${SLURM_JOB_ID:-none} gpus=$NPROC cpus=${SLURM_CPUS_PER_TASK:-?}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# The chain index must already exist or rank 0 probes ~500k chains at startup and
# every restart repeats it. Refuse rather than silently pay it: the cache key is
# derived from the corpus and the admission filters, so a miss here means the
# config changed since the index was built and the right fix is to rebuild it,
# not to absorb an hour per attempt.
"$MFCLEAN_PYTHON" - "$CONFIG" <<'PYCHECK'
import sys, yaml
from pathlib import Path
sys.path.insert(0, "src")
from mambafold.data.dataset import RCSBDataset
from mambafold.data.length_cache import _DEFAULT_CACHE_DIR, _cache_path

cfg = yaml.safe_load(open(sys.argv[1]))
if not cfg.get("extract_monomer_chains"):
    sys.exit(0)
missing = []
for src in cfg["train_sources"]:
    ds = RCSBDataset(
        data_dir=src["data_dir"], max_length=cfg["max_length"],
        file_list=src.get("file_list"), esm_dir=src.get("esm_dir"),
        single_chain_only=cfg.get("single_chain_only", False),
        extract_monomer_chains=False,
        dedup_homomer_chains=cfg.get("dedup_homomer_chains", False),
        min_obs_ratio=cfg.get("min_obs_ratio", 0.0),
    )
    base = _cache_path(ds, Path(_DEFAULT_CACHE_DIR))
    path = base.with_name("chainidx_" + base.name)
    print(f"  chain index {src['name']:8} {'ok' if path.exists() else 'MISSING'}  {path.name}")
    if not path.exists():
        missing.append(src["name"])
if missing:
    sys.exit(
        f"chain index missing for {missing}; run "
        f"`sbatch scripts/slurm_prebuild_loader_caches.sh {sys.argv[1]}` first"
    )
PYCHECK

# Preflight: is NCCL usable on the node we landed on? It is not everywhere —
# on gpu2 every NCCL variant segfaults (with device_id, without, and with P2P
# and SHM disabled) while gloo passes, established with a probe that imports
# nothing from this project. The danger is not a crash, which is loud and
# immediate; it is falling back to gloo unnoticed, because gloo all-reduces
# 370M parameters over CPU and would stretch this run by days with nothing in
# the log to say why. So the probe decides, and gloo needs an explicit opt-in.
if [ "${DIST_BACKEND:-nccl}" = "nccl" ] && [ "$NPROC" -gt 1 ]; then
  echo
  echo "=== preflight: NCCL on $(hostname) ==="
  if timeout 300 "$MFCLEAN_PYTHON" -m torch.distributed.run \
       --standalone --nproc_per_node=2 benchmarks/probe_nccl.py >/dev/null 2>&1; then
    echo "  NCCL ok"
  elif [ "${MF_ALLOW_GLOO:-0}" = "1" ]; then
    echo "  NCCL FAILED — falling back to gloo because MF_ALLOW_GLOO=1."
    echo "  Expect gradient all-reduce over CPU on 370M parameters: much slower."
    export DIST_BACKEND=gloo
  else
    echo "  NCCL FAILED on this node. Refusing to start."
    echo "  Reproduce:  sbatch -p \$PARTITION --gres=gpu:2 scripts/slurm_probe_nccl.sh"
    echo "  To run anyway on gloo, and accept a much slower run:"
    echo "      MF_ALLOW_GLOO=1 sbatch scripts/slurm_train.sh"
    exit 1
  fi
fi

exec "$MFCLEAN_PYTHON" -m torch.distributed.run \
  --standalone --nproc_per_node="$NPROC" \
  scripts/train.py --config "$CONFIG" --out_dir "$OUT_DIR" "$@"
