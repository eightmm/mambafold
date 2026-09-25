#!/usr/bin/env python
"""Stage 04 — enumerate the protein chains of the admitted and training corpora.

The FASTA is not decoration. It is the key for the sequence-addressed embedding
cache in stage 05 and the query set for the leakage gate in stage 08, so it has
to be rebuilt whenever the admission rule changes.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD_METADATA = ROOT / "pipeline" / "lib" / "build_metadata.py"

# One target. There is no train/val split, so a separate "training" FASTA would
# be a byte-identical copy of the admitted one — and the leakage gates screening
# the smaller of the two was how 5,744 admitted entries went unscreened.
TARGETS = [
    ("admitted_all", "rcsb-admitted"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    farm = root / "data" / "rcsb_train"
    audit = root / "data" / "audit"
    audit.mkdir(parents=True, exist_ok=True)

    for list_name, out_name in TARGETS:
        fasta = audit / f"{out_name}.fasta"
        tsv = audit / f"{out_name}.tsv"
        # A stale FASTA is worse than none: stage 05 would key the cache off a
        # corpus that no longer exists.
        fasta.unlink(missing_ok=True)
        tsv.unlink(missing_ok=True)
        cmd = [
            sys.executable,
            str(BUILD_METADATA),
            "--npz_dir", str(farm),
            "--file_list", str(root / "data" / "splits" / f"{list_name}.txt"),
            "--out_tsv", str(tsv),
            "--out_fasta", str(fasta),
            "--workers", str(args.workers),
        ]
        print(f"== {list_name} -> {fasta.name}", flush=True)
        subprocess.run(cmd, check=True, env={**__import__("os").environ,
                                             "PYTHONPATH": str(root / "src")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
