#!/usr/bin/env bash
# Ordered driver for the mambafold-clean data pipeline.
#
# Every stage is idempotent and resumable, so this can be rerun after an
# interruption without losing work. Stages may also be run individually; the
# numbering is the dependency order, not a suggestion.
#
#   ./pipeline/run_all.sh              run every stage
#   ./pipeline/run_all.sh 04 05        run only these stages
#
# Requires nothing but config/paths.env and network access for stage 00.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=../config/paths.env
source config/paths.env

PY="${MFCLEAN_PYTHON:?set MFCLEAN_PYTHON in config/paths.env}"
WORKERS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

REQUESTED=("$@")

# No arguments means every stage. Deciding that here, rather than inside a
# helper that only sees its own argument count, keeps the bare `run_all.sh`
# invocation from silently doing nothing.
wanted() {
  local stage="$1"
  [ "${#REQUESTED[@]}" -eq 0 ] && return 0
  local requested
  for requested in "${REQUESTED[@]}"; do
    [ "$requested" = "$stage" ] && return 0
  done
  return 1
}

run() {
  local stage="$1"; shift
  wanted "$stage" || return 0
  echo
  echo "############ stage $stage  [$(date -u +%H:%M:%S)]"
  "$@"
}

run 00 bash pipeline/00_fetch_external.sh

run 01 "$PY" pipeline/01_fetch_rcsb_metadata.py \
  --npz_dir "$BOLTZ_STRUCTURES" \
  --out_tsv data/splits/rcsb_entry_metadata.tsv

run 02 "$PY" pipeline/02_build_admission_splits.py \
  --npz_dir "$BOLTZ_STRUCTURES" \
  --boltz_manifest "$BOLTZ_MANIFEST" \
  --entry_meta data/splits/rcsb_entry_metadata.tsv \
  --release_cutoff "$RELEASE_CUTOFF" \
  --max_resolution "$MAX_RESOLUTION" \
  --resolution_source "$RESOLUTION_SOURCE" \
  --out_dir data/splits

run 03 "$PY" pipeline/03_link_structures.py \
  --boltz_structures "$BOLTZ_STRUCTURES"

run 04 "$PY" pipeline/04_build_fasta.py --workers "$WORKERS"

run 05 "$PY" pipeline/05_link_rcsb_embeddings.py \
  --legacy_esmc_rcsb "$LEGACY_ESMC_RCSB"

run 06 "$PY" pipeline/06_convert_afdb_v4.py \
  --tar data/external/afdb_v4/swissprot_cif_v4.tar \
  --id_list data/external/simplefold_swissprot_list.csv \
  --out_dir data/afdb_swissprot_v4/npz \
  --manifest data/afdb_swissprot_v4/manifest.tsv \
  --sequences data/afdb_swissprot_v4/sequences.tsv \
  --workers "$WORKERS"

run 07 "$PY" pipeline/07_link_afdb_embeddings.py \
  --v4_sequences data/afdb_swissprot_v4/sequences.tsv \
  --legacy_esm_dir "$LEGACY_ESMC_AFDB" \
  --dest_esm_dir data/afdb_esmc6b \
  --v6_npz_dir "$LEGACY_AFDB_V6_NPZ" \
  --verify_samples 3000 \
  --out_dir data/afdb_swissprot_v4/sequence_audit

# Stage 07's guards are cheap and blind in one direction: a same-length sequence
# change passes them. 07b settles it by comparing every accession, and removes
# the links that turn out to pair a record with another protein's embedding.
run 07b "$PY" pipeline/07b_verify_afdb_sequences.py \
  --v4_sequences data/afdb_swissprot_v4/sequences.tsv \
  --v6_npz_dir "$LEGACY_AFDB_V6_NPZ" \
  --dest_esm_dir data/afdb_esmc6b \
  --legacy_esm_dir "$LEGACY_ESMC_AFDB" \
  --out_dir data/afdb_swissprot_v4/sequence_audit \
  --workers "$WORKERS"

run 08 "$PY" pipeline/08_leakage_gates.py

run 09 "$PY" pipeline/09_freeze_manifests.py

# The exact gate is necessary and not sufficient: a training sequence one
# substitution from a target is not an exact match and is still leakage.
run 10 "$PY" pipeline/10_homology_gate.py \
  --mmseqs "$MMSEQS_BIN" --threads "$WORKERS"

echo
echo "############ pipeline complete  [$(date -u +%H:%M:%S)]"
