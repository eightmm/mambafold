#!/bin/bash
#SBATCH --job-name=mf-gpu-probe
#SBATCH --partition=test
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/mf-gpu-probe-%j.out
#SBATCH --error=logs/mf-gpu-probe-%j.out
#
# First GPU work this repository has ever run. Three stages, cheapest first, so
# a failure in an early one is not paid for by a long later one:
#
#   1. the CUDA-gated correctness tests, which have never executed. They check
#      that a padded batch matches each sequence run alone. Every masking defect
#      in this model has that signature and none of them raise — they produce a
#      plausible loss curve and stall for no reason.
#   2. memory and step time on the model the config actually builds. Every
#      figure currently in the config predates A=14, mimo_rank 4, the 25 AdaLN
#      modules, the two cross mixers and the per-slot decoder context.
#   3. an overfit probe on real chains: can the architecture memorise four
#      structures. Not a quality measurement.
#
# The A5000 is 24 GB against the 6000 Ada's 47 GB, so absolute VRAM does not
# transfer. The per-residue slope and the step time do, and both are what the
# stale fit needs replaced. Shared memory is 100 KB on sm_86 and the config's
# d_state 64 / mimo_rank 4 needs 76,112 B, so the kernels are the same ones.

# Deliberately not `set -e`. Each stage runs independently and records its exit
# code; the job fails at the end if any of them did. A `set -e` here cost a
# whole GPU allocation once, when a broken test fixture in stage 1 killed the
# script before the memory sweep and the overfit probe ever started.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
declare -A STATUS
run_stage() {  # run_stage <name> <command...>
  local name="$1"; shift
  echo; echo "=== $name ==="
  if "$@"; then STATUS[$name]=ok; else STATUS[$name]="FAILED(rc=$?)"; fi
}
set -a; source config/paths.env; set +a
mkdir -p logs data/audit/gpu_probe

PY="${MFCLEAN_PYTHON:?}"
# The overfit stage reads a 200-entry slice of the admitted set. Without it the
# dataset builds a chain index over all 158,210 records, which is about an hour
# and a half and has nothing to do with what this probe measures.
FILE_LIST=data/audit/gpu_probe_filelist.txt

echo "=== node $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader

run_stage "1. CUDA-gated correctness tests" \
  "$PY" -m pytest -q tests/test_padding_equivalence.py -rs

run_stage "2. memory + step time, model built from configs/run_a_mamba.yaml" \
  "$PY" benchmarks/probe_train_memory.py \
  --config configs/run_a_mamba.yaml \
  --crops 256,512,1024 --batches 1,2,4,8 --steps 4 \
  --loss_mode lddt_exact \
  --out data/audit/gpu_probe/train_memory_a5000.json

run_stage "3. overfit probe on real chains" \
  "$PY" benchmarks/probe_overfit.py \
  --config configs/run_a_mamba.yaml \
  --data_dir data/rcsb_train \
  --file_list "$FILE_LIST" \
  --esm_dir data/rcsb_esmc6b \
  --n_proteins 4 --max_length 256 --copies 4 --steps 400 \
  --out data/audit/gpu_probe/overfit_a5000.json

echo
echo "=== summary ==="
rc=0
for name in "${!STATUS[@]}"; do
  printf '  %-52s %s\n' "$name" "${STATUS[$name]}"
  [[ ${STATUS[$name]} == ok ]] || rc=1
done
exit "$rc"
