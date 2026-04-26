#!/bin/bash
# Download CASP15 multimer benchmark sources.
#
# Outputs:
#   benchmarks/casp15/raw/casp15.seq.txt                 # all CASP15 sequences (single file)
#   benchmarks/casp15/raw/casp15.targets.oligo.tar.gz    # multimer reference structures (tarball)
#   benchmarks/casp15/raw/oligo/                         # extracted oligomer GT structures
#
# Run-once. Re-running is a no-op for files that already exist.
#
# Usage:
#   bash benchmarks/casp15/download.sh

set -euo pipefail
cd "$(dirname "$0")"

RAW=raw
mkdir -p "$RAW/oligo"

SEQ_URL="https://predictioncenter.org/download_area/CASP15/sequences/casp15.seq.txt"
OLIGO_URL="https://predictioncenter.org/download_area/CASP15/targets/casp15.targets.oligo.ALL_09.13.2025.tar.gz"

fetch() {
    local url="$1" out="$2"
    if [[ -s "$out" ]]; then
        echo "[skip] $out (already present)"
    else
        echo "[fetch] $url → $out"
        curl -fsSL --retry 3 -o "$out" "$url"
    fi
}

fetch "$SEQ_URL"   "$RAW/casp15.seq.txt"
fetch "$OLIGO_URL" "$RAW/casp15.targets.oligo.tar.gz"

# Extract oligomer reference structures into raw/oligo/ (idempotent — only extract once)
if [[ -z "$(ls -A "$RAW/oligo" 2>/dev/null)" ]]; then
    echo "[extract] $RAW/casp15.targets.oligo.tar.gz → $RAW/oligo/"
    tar -xzf "$RAW/casp15.targets.oligo.tar.gz" -C "$RAW/oligo/" --strip-components=0
fi

echo
echo "[done] CASP15 sources at benchmarks/casp15/$RAW/"
echo "        - sequences : $RAW/casp15.seq.txt  ($(wc -l < $RAW/casp15.seq.txt) lines)"
echo "        - oligomers : $RAW/oligo/  ($(find $RAW/oligo -type f | wc -l) files)"
echo
echo "next: .venv/bin/python benchmarks/casp15/parse_targets.py"
