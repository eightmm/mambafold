#!/usr/bin/env python
"""Stage 05 — give the admitted RCSB corpus its sequence-addressed ESMC-6B cache.

The embeddings were computed once by the previous track and are addressed by the
SHA-256 of the canonical sequence, so every admitted sequence that also occurred
there is reused as a hard link and nothing is recomputed.

Sequences with no cached embedding are written out rather than skipped quietly.
They are exactly the records a later ESMC-6B pass has to cover before they can
be trained on, and most of them are the entries the previous track never held.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline" / "lib"))

from corpus_links import read_fasta, sync_embeddings  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--legacy_esmc_rcsb", required=True)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    fasta = root / "data" / "audit" / "rcsb-admitted.fasta"
    if not fasta.exists():
        raise FileNotFoundError(f"{fasta} missing; run stage 04 first")

    records = read_fasta(fasta)
    stats, missing = sync_embeddings(
        records, Path(args.legacy_esmc_rcsb), root / "data" / "rcsb_esmc6b"
    )
    print(json.dumps(stats, indent=2))

    audit = root / "data" / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    (audit / "rcsb_missing_embeddings.txt").write_text(
        "\n".join(missing) + ("\n" if missing else "")
    )
    (audit / "rcsb_embeddings.json").write_text(json.dumps(stats, indent=2) + "\n")
    if missing:
        target = audit / 'rcsb_missing_embeddings.txt'
        print(f"{len(missing)} chains need a fresh ESMC-6B pass -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
