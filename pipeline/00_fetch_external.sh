#!/usr/bin/env bash
# Stage 00 — fetch the two external inputs the corpus is built from.
#
#   data/external/simplefold_swissprot_list.csv  the published AFDB subset
#   data/external/afdb_v4/swissprot_cif_v4.tar   the frozen AlphaFold DB v4 archive
#
# Both are idempotent; the ~39 GB archive resumes, so an interrupted run costs
# only the time already spent.
#
# Two hazards this stage is written around.
#
# 1. aria2c preallocates the output and fills it out of order, so a partial
#    download can already have the final apparent size while most of it is
#    holes. Completeness is therefore decided by the absence of the .aria2
#    control file, never by size alone.
# 2. Because of (1), no other tool may resume an aria2c-owned file. wget -c and
#    curl -C - both resume from the apparent size and would treat a hole-filled
#    file as finished. If a control file is present and aria2c is unavailable
#    — as on the compute nodes, which have curl and wget but not aria2c — this
#    stage waits or fails rather than corrupting the archive.
#
# The practical consequence: run the archive download where aria2c exists (the
# login node, ideally under tmux so it survives a disconnect), and let the
# batch job wait on it.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../config/paths.env
source "$ROOT/config/paths.env"

LIST="$ROOT/data/external/simplefold_swissprot_list.csv"
TAR_DIR="$ROOT/data/external/afdb_v4"
TAR="$TAR_DIR/swissprot_cif_v4.tar"
CONTROL="${TAR}.aria2"
WAIT_MINUTES="${AFDB_WAIT_MINUTES:-240}"
mkdir -p "$(dirname "$LIST")" "$TAR_DIR"

echo "== SimpleFold AFDB id list"
if [ -s "$LIST" ]; then
  echo "   present: $(grep -c '^AF-' "$LIST") ids"
else
  curl -fsSL --retry 5 --retry-delay 5 -o "$LIST" "$SIMPLEFOLD_LIST_URL"
  echo "   fetched: $(grep -c '^AF-' "$LIST") ids"
fi

complete() {
  [ -f "$TAR" ] && [ ! -f "$CONTROL" ] && [ "$(stat -c %s "$TAR")" = "$AFDB_V4_TAR_BYTES" ]
}

echo "== AlphaFold DB v4 SwissProt archive"
if complete; then
  echo "   already complete"
else
  have_aria2c=$(command -v aria2c || true)
  downloading=$(pgrep -f "aria2c.*swissprot_cif_v4" >/dev/null 2>&1 && echo yes || echo no)

  if [ "$downloading" = no ] && [ -n "$have_aria2c" ]; then
    echo "   downloading with aria2c"
    aria2c -c -x 16 -s 16 -k 50M --file-allocation=none \
      --console-log-level=warn --summary-interval=120 \
      -d "$TAR_DIR" -o "$(basename "$TAR")" "$AFDB_V4_TAR_URL"
  elif [ "$downloading" = no ] && [ ! -f "$CONTROL" ] && [ ! -f "$TAR" ]; then
    # Nothing started yet and no aria2c: a plain sequential fetch is safe
    # because no control file exists and no other writer owns the file.
    echo "   no aria2c; downloading sequentially with curl (slower)"
    curl -fL --retry 10 --retry-delay 10 --retry-all-errors -C - \
      -o "$TAR" "$AFDB_V4_TAR_URL"
  else
    if [ -z "$have_aria2c" ] && [ -f "$CONTROL" ]; then
      echo "   an aria2c-owned partial archive is present but aria2c is not"
      echo "   installed here; waiting rather than resuming it with another tool."
    else
      echo "   another downloader owns the archive; waiting"
    fi
    deadline=$(( $(date +%s) + WAIT_MINUTES * 60 ))
    until complete; do
      if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "FATAL: archive still incomplete after ${WAIT_MINUTES} minutes."
        echo "       Run this stage where aria2c exists, for example:"
        echo "         tmux new -d -s afdbv4 'bash pipeline/00_fetch_external.sh'"
        exit 1
      fi
      sleep 120
      echo "   waiting: used=$(du -s --block-size=1M "$TAR" 2>/dev/null | cut -f1)MiB control=$([ -f "$CONTROL" ] && echo present || echo gone)"
    done
    echo "   completed by the other downloader"
  fi
fi

actual=$(stat -c %s "$TAR")
echo "   bytes=$actual expected=$AFDB_V4_TAR_BYTES"
[ "$actual" = "$AFDB_V4_TAR_BYTES" ] || { echo "FATAL: archive size mismatch"; exit 1; }
[ -f "$CONTROL" ] && { echo "FATAL: control file still present; archive is incomplete"; exit 1; }
echo "   members=$(tar tf "$TAR" | wc -l)"
