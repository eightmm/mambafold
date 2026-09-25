#!/usr/bin/env bash
#SBATCH --job-name=mf-plddt-probe
#SBATCH --partition=test
#SBATCH --gres=gpu:a5000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --qos=veryshort
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/mf-plddt-probe-%j.out
#SBATCH --error=logs/slurm/mf-plddt-probe-%j.out

set -euo pipefail
ROOT="${MFCLEAN_ROOT:-${SLURM_SUBMIT_DIR:?}}"
cd "$ROOT"
set -a; source config/paths.env; set +a

CHECKPOINT="${MF_PLDDT_PROBE_CHECKPOINT:-out/run-a-selfcond-geo-v1/ckpt_0010000.pt}"
EXPECTED_STEP="${MF_PLDDT_PROBE_EXPECTED_STEP:-10000}"
CONFIG="${MF_PLDDT_CONFIG:-configs/plddt_selfcond_sde500.yaml}"
FILE_LIST="${MF_PLDDT_PROBE_FILE_LIST:-data/splits/plddt_pilot_train.txt}"
TARGET_MODE="${MF_PLDDT_TARGET_MODE:-file}"
END_INDEX="${MF_PLDDT_PROBE_END_INDEX:-1}"
N_STEPS="${MF_PLDDT_PROBE_N_STEPS:-500}"
N_SEEDS="${MF_PLDDT_PROBE_N_SEEDS:-2}"
SEED_BATCH_SIZE="${MF_PLDDT_PROBE_SEED_BATCH_SIZE:-2}"
TMPROOT="$(mktemp -d /tmp/mf-plddt-probe.XXXXXX)"
trap 'rm -rf "$TMPROOT"' EXIT

case "$TARGET_MODE" in
  file) TARGET_FLAG="--file-list" ;;
  chain) TARGET_FLAG="--chain-list" ;;
  *) echo "MF_PLDDT_TARGET_MODE must be file or chain, found: $TARGET_MODE" >&2; exit 2 ;;
esac

"$MFCLEAN_PYTHON" scripts/generate_plddt_rollouts.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --expected-checkpoint-step "$EXPECTED_STEP" \
  "$TARGET_FLAG" "$FILE_LIST" \
  --out-dir "$TMPROOT/rollouts" \
  --start-index 0 --end-index "$END_INDEX" \
  --n-steps "$N_STEPS" --n-seeds "$N_SEEDS" --seed-batch-size "$SEED_BATCH_SIZE"

PYTHONPATH=src "$MFCLEAN_PYTHON" - "$TMPROOT/rollouts" <<'PY'
import sys
from pathlib import Path

from mambafold.confidence.rollout import PLDDTRolloutDataset

root = Path(sys.argv[1])
dataset = PLDDTRolloutDataset([root / "manifest-r00000-of-00001.json"])
protein_lddt = []
residue_lddt = []
median_steps = []
normal_fractions = []
for record in dataset:
    ca_step = (record["pred_ca_A"][1:] - record["pred_ca_A"][:-1]).norm(dim=-1)
    median_steps.append(float(ca_step.median()))
    normal_fractions.append(float(((ca_step - 3.8).abs() < 0.5).float().mean()))
    valid_target = record["plddt_target"][record["target_mask"]]
    protein_lddt.append(float(valid_target.mean() * 100.0))
    residue_lddt.extend((valid_target * 100.0).tolist())

import numpy as np

protein = np.asarray(protein_lddt)
residue = np.asarray(residue_lddt)
percentiles = [0, 10, 25, 50, 75, 90, 100]
print(
    f"probe records={len(dataset)} median_ca_step_A="
    f"{np.median(median_steps):.4f} min_normal_ca_fraction={min(normal_fractions):.4f}"
)
print(
    "protein_lddt_percentiles="
    + repr(dict(zip(percentiles, np.percentile(protein, percentiles).round(3).tolist())))
)
print(
    "protein_lddt_bins="
    + repr(
        {
            "<50": int((protein < 50).sum()),
            "50-70": int(((protein >= 50) & (protein < 70)).sum()),
            "70-90": int(((protein >= 70) & (protein < 90)).sum()),
            ">=90": int((protein >= 90).sum()),
        }
    )
)
print(
    "residue_lddt_percentiles="
    + repr(dict(zip(percentiles, np.percentile(residue, percentiles).round(3).tolist())))
)
bad_geometry = [
    index
    for index, (median, normal) in enumerate(zip(median_steps, normal_fractions))
    if not 3.0 <= median <= 4.6 or normal < 0.8
]
if bad_geometry:
    raise RuntimeError(
        "rollout geometry gate failed for record indices "
        f"{bad_geometry[:16]} ({len(bad_geometry)}/{len(dataset)})"
    )
PY
