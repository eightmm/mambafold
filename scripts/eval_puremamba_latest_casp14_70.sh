#!/usr/bin/env bash
#SBATCH --job-name=mf-casp14-70
#SBATCH --partition=6000ada
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/eval/logs/%x-%j.out
#SBATCH --error=outputs/eval/logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

python_bin="tools/mambafold-eval-env/bin/python"
scoring_python="tools/scoring_venv/bin/python"
checkpoint="outputs/train/direct_puremamba_attn6_geo_adaln_sf360_mixed_v1/ckpt_0120000.pt"
ids="data/casp_official/casp14_70_whole_ids_exact.txt"
data_dir="data/casp_official/npz_70"
esm_dir="data/casp_official/esm_70"
out_dir="outputs/eval/casp14_70_puremamba_attn6_geo_adaln_sf360_mixed_v1_ckpt120000_sde500_20260714_job${SLURM_JOB_ID}"

mkdir -p "$out_dir" outputs/eval/logs .cache/tmp
test -x "$python_bin"
test -x "$scoring_python"
test -s "$checkpoint"
test -s "$ids"
test "$(wc -l < "$ids")" -eq 70
test "$(find "$esm_dir" -maxdepth 1 -type f -name '*_ch0.npy' | wc -l)" -eq 70

export TMPDIR="$SLURM_SUBMIT_DIR/.cache/tmp"
export PYTHONPATH="$SLURM_SUBMIT_DIR/tools/mamba-src:$SLURM_SUBMIT_DIR/src"
export PATH="$SLURM_SUBMIT_DIR/tools/mambafold-eval-env/bin:/usr/local/cuda/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

command -v ptxas
ptxas --version

echo "checkpoint=$checkpoint"
echo "ids=$ids"
echo "out_dir=$out_dir"
echo "contract=SimpleFold CASP14 70 whole targets; T1044 excluded; EMA; SDE500; seed0"

"$python_bin" benchmarks/run_inference.py \
    --ckpt "$checkpoint" \
    --ids "$ids" \
    --out "$out_dir" \
    --data_dir "$data_dir" \
    --esm_dir "$esm_dir" \
    --max_length 1024 \
    --n_steps 500 \
    --sampler sde \
    --sde_tau 0.01 \
    --sde_eps 0.01 \
    --sde_w_cutoff 0.99 \
    --sde_log_timesteps \
    --n_seeds 1 \
    --seed_offset 0

"$python_bin" - "$out_dir/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
manifest = json.loads(manifest_path.read_text())
expected = 70
actual = manifest.get("n_predicted")
if actual != expected:
    raise SystemExit(f"Expected {expected} predictions, got {actual}")
print(f"[verify] predictions={actual}/{expected}", flush=True)
PY

"$scoring_python" benchmarks/score_simplefold_metrics.py \
    --in_dir "$out_dir" \
    --out "$out_dir/scores.json"

"$python_bin" benchmarks/score_local_geometry.py \
    --in_dir "$out_dir" \
    --out "$out_dir/local_geometry.json"

"$python_bin" - "$out_dir" "$checkpoint" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
checkpoint = Path(sys.argv[2])
scores = json.loads((out / "scores.json").read_text())
geometry = json.loads((out / "local_geometry.json").read_text())
source_tar = Path("data/casp_official/raw/casp14.targ.whole.4invitees.tgz")

digest = hashlib.sha256()
with source_tar.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)

summary = {
    "benchmark": "SimpleFold CASP14 70 whole targets",
    "target_count_requested": 70,
    "target_count_scored": scores["n"],
    "excluded_target": "T1044 (>1000 residues)",
    "checkpoint": str(checkpoint),
    "checkpoint_step": 120000,
    "use_ema": True,
    "sampler": "sde",
    "n_steps": 500,
    "seed": 0,
    "max_length": 1024,
    "source_tar_sha256": digest.hexdigest(),
    "tm_score": scores["tm_score"],
    "gdt_ts": scores["gdt_ts"],
    "lddt": scores["lddt"],
    "lddt_ca": scores["lddt_ca"],
    "rmsd": scores["rmsd"],
    "aa_rmsd": scores["aa_rmsd"],
    "pred_bond_mae_A_mean": geometry["pred_bond_mae_A_mean"],
    "pred_clashes_per_1k_atoms_mean": geometry["pred_clashes_per_1k_atoms_mean"],
}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2), flush=True)
PY
