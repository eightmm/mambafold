#!/usr/bin/env bash
#SBATCH --job-name=mf-rollout-smoke
#SBATCH --partition=test
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/mf-rollout-smoke-%j.out
#SBATCH --error=logs/slurm/mf-rollout-smoke-%j.out
#
# Exercise scripts/rollout.py end to end before it matters.
#
# The rollout is the only path from a checkpoint to a number, and with the
# transformer control arm cancelled it is the only evidence this project
# produces — so the worst time to run it for the first time is after a 300k-step
# run has finished. It has never executed on a GPU: the CASP embeddings it needs
# only just landed (113/113), and it has no checkpoint of its own.
#
# The weights here come from the two-rank DDP smoke, so the structures are
# meaningless. What is being tested is the plumbing: checkpoint load with the
# "module." prefix stripped, EMA selection, sequence-only example construction,
# length-grouped batching, the sampler, PDB output, and the reference-free
# geometry report. A geometry number from a 40-step model is expected to be bad;
# a crash or an empty report is the failure this is looking for.
#
# Single GPU, so gpu2's broken NCCL does not apply.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
set -a; source config/paths.env; set +a
mkdir -p logs/slurm
export PYTHONUNBUFFERED=1

# Defaults reproduce the original plumbing smoke. Override to point the same
# comparison at a real checkpoint — the ODE/SDE split is a property of the
# sampler, so the script that exercises both is the one to reuse.
CKPT="${MF_CKPT:-out/smoke-ddp-57859/ckpt_latest.pt}"
RCONFIG="${MF_CONFIG:-configs/smoke_ddp.yaml}"
FASTA="${MF_FASTA:-benchmarks/external_testsets/casp16_single_chain_21.fasta}"
STEPS="${MF_STEPS:-20}"
LIMIT="${MF_LIMIT:-6}"
OUT="${MF_OUT:-out/rollout-smoke-${SLURM_JOB_ID:-local}}"
rc=0

echo "node=$(hostname) job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -e "$CKPT" ] || { echo "no smoke checkpoint at $CKPT"; exit 1; }

for method in ode sde; do
  echo; echo "=== $method, $LIMIT targets from $(basename "$FASTA"), $STEPS steps ==="
  "$MFCLEAN_PYTHON" scripts/rollout.py \
    --config "$RCONFIG" \
    --checkpoint "$CKPT" \
    --fasta "$FASTA" \
    --esm_dir data/casp_esmc6b \
    --out_dir "${OUT}-${method}" \
    --method "$method" --n_steps "$STEPS" --limit "$LIMIT" --max_batch_residues 2048
  r=$?
  echo "  rc=$r"; [ $r -ne 0 ] && rc=1
done

echo; echo "=== written structures ==="
for method in ode sde; do
  n=$(ls -1 "${OUT}-${method}/structures"/*.pdb 2>/dev/null | wc -l)
  echo "  $method: $n pdb"
  f=$(ls -1 "${OUT}-${method}/structures"/*.pdb 2>/dev/null | head -1)
  [ -n "$f" ] && { echo "    $(basename "$f"): $(grep -c '^ATOM' "$f") ATOM records"; head -2 "$f" | sed 's/^/    /'; }
done
exit $rc
