#!/usr/bin/env bash
set -euo pipefail

cd /NHNHOME/WORKSPACE/0526040024_A/jaemin/mambafold

CKPT="outputs/train/direct_pair_global20k_v1/ckpt_0010000.pt"
OUT="outputs/eval/casp14_70_direct_pair_global20k_ckpt10000_sde500_20260630_now"
mkdir -p "$OUT" .cache/tmp

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export TMPDIR="$PWD/.cache/tmp"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=src

echo "[start] $(date -Is)"
echo "[infer] ckpt=${CKPT}"
echo "[infer] out=${OUT}"
echo "[infer] gpu=${CUDA_VISIBLE_DEVICES}"

uv run python benchmarks/run_inference.py \
  --ckpt "$CKPT" \
  --ids data/casp_official/casp14_70_whole_ids_exact.txt \
  --out "$OUT" \
  --data_dir data/casp_official/npz \
  --esm_dir data/casp_official/esm \
  --max_length 2048 \
  --n_steps 500 \
  --sampler sde \
  --sde_tau 0.01 \
  --sde_eps 0.01 \
  --sde_w_cutoff 0.99 \
  --sde_log_timesteps \
  --n_seeds 1

echo "[score] SimpleFold-style"
tools/scoring_venv/bin/python benchmarks/score_simplefold_metrics.py \
  --in_dir "$OUT" \
  --out "$OUT/scores.json"

echo "[score] local geometry"
uv run python benchmarks/score_local_geometry.py \
  --in_dir "$OUT" \
  --out "$OUT/local_geometry.json"

python - <<'PY'
import json
from pathlib import Path

out = Path("outputs/eval/casp14_70_direct_pair_global20k_ckpt10000_sde500_20260630_now")
scores = json.loads((out / "scores.json").read_text())
geom = json.loads((out / "local_geometry.json").read_text())

def pair(metric):
    v = scores[metric]
    return f"{v['mean']:.3f}/{v['median']:.3f}"

summary = {
    "out_dir": str(out),
    "ckpt": "ckpt_0010000.pt",
    "n": scores["n"],
    "tm_score": pair("tm_score"),
    "gdt_ts": pair("gdt_ts"),
    "lddt": pair("lddt"),
    "lddt_ca": pair("lddt_ca"),
    "rmsd": pair("rmsd"),
    "aa_rmsd": pair("aa_rmsd"),
    "pred_bond_mae_A_mean": round(geom["pred_bond_mae_A_mean"], 4),
    "pred_clashes_per_1k_atoms_mean": round(geom["pred_clashes_per_1k_atoms_mean"], 3),
}
(out / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY

echo "[done] $(date -Is)"
