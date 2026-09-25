#!/usr/bin/env python
"""Stage 03 — materialise the structure farms as symlinks into the Boltz snapshot.

Admitted records and post-cutoff records land in separate directories on
purpose: a training run pointed at ``data/rcsb_train`` cannot reach a
post-cutoff structure through its data directory, whatever a config says. The
post-cutoff pool is the confirmatory evaluation set, not training data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline" / "lib"))

from corpus_links import read_ids, sync_farm  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--boltz_structures", required=True)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    splits = root / "data" / "splits"
    src = Path(args.boltz_structures)

    admitted = read_ids(splits / "admitted_all.txt")
    heldout = read_ids(splits / "heldout_post_cutoff.txt")
    print(f"admitted={len(admitted)} heldout={len(heldout)}", flush=True)

    report = {
        "boltz_structures": str(src),
        "train_farm": sync_farm(admitted, src, root / "data" / "rcsb_train"),
        "heldout_farm": sync_farm(heldout, src, root / "data" / "rcsb_heldout_post_cutoff"),
    }
    print(json.dumps(report, indent=2))

    out = root / "data" / "audit" / "structure_farms.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
