#!/bin/bash
# Parallel RCSB mmCIF fetcher. Resumable (skips existing), retries 3×.
# Usage:  bash scripts/download_rcsb_cif.sh <id_list.txt> <out_dir> [parallel=32]

set -u
ID_LIST="${1:?need id list (one pdb_id per line, .npz optional)}"
OUT_DIR="${2:?need out dir}"
PAR="${3:-32}"
FAIL_LOG="${OUT_DIR}/_failed.txt"

mkdir -p "$OUT_DIR"
: > "$FAIL_LOG"

fetch_one() {
    local raw="$1"
    local id="${raw%.npz}"          # strip optional .npz suffix
    id="${id,,}"                    # lowercase
    local shard="${id:1:2}"
    local dest_dir="${OUT_DIR}/${shard}"
    local dest="${dest_dir}/${id}.cif.gz"
    [[ -s "$dest" ]] && return 0    # already have non-empty file
    mkdir -p "$dest_dir"
    curl -sSLf --max-time 30 --retry 3 --retry-delay 2 \
        "https://files.rcsb.org/download/${id^^}.cif.gz" -o "$dest" \
        || { echo "$id" >> "$FAIL_LOG"; rm -f "$dest"; return 1; }
}
export -f fetch_one
export OUT_DIR FAIL_LOG

total=$(wc -l < "$ID_LIST")
echo "Downloading $total ids into $OUT_DIR (parallel=$PAR)"
start=$SECONDS
cat "$ID_LIST" | xargs -P "$PAR" -I{} bash -c 'fetch_one "$@"' _ {}
elapsed=$((SECONDS - start))

got=$(find "$OUT_DIR" -name '*.cif.gz' 2>/dev/null | wc -l)
failed=$(wc -l < "$FAIL_LOG")
echo "Done in ${elapsed}s. got=$got failed=$failed"
