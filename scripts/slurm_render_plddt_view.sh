#!/usr/bin/env bash
#SBATCH --job-name=mf-plddt-view-html
#SBATCH --partition=cpu_only
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --qos=veryshort
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm/mf-plddt-view-html-%j.out
#SBATCH --error=logs/slurm/mf-plddt-view-html-%j.out

set -euo pipefail
ROOT="${MFCLEAN_ROOT:-${SLURM_SUBMIT_DIR:?}}"
cd "$ROOT"
set -a; source config/paths.env; set +a

PDB="${MF_PLDDT_VIEW_PDB:?set MF_PLDDT_VIEW_PDB}"
OUT="${MF_PLDDT_VIEW_HTML:?set MF_PLDDT_VIEW_HTML}"
test -s "$PDB"
exec "$MFCLEAN_PYTHON" scripts/render_plddt_html.py --pdb "$PDB" --out "$OUT"
