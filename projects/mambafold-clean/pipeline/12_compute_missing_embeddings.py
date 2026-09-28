#!/usr/bin/env python
"""Stage 12 — compute the ESMC-6B embeddings the corpus is missing.

The link stages reuse embeddings the previous track already computed. What they
cannot supply, this fills: sequences that track never held, and AFDB records
whose UniProt sequence changed between AlphaFold DB v4 and v6 so the cached
embedding belongs to a different protein.

The work list comes from the cache, not from whichever stage flagged a record.
That distinction is not academic — stage 07 and stage 07b each reject on their
own grounds and neither list is a superset of the other, so a flag-driven
version missed a record that stage 07 rejected and stage 07b cleared. Asking
the cache which sequences are absent cannot miss that case.

Needs a GPU and the pinned ESMC-6B revision, which is why it sits outside the
CPU pipeline and `run_all.sh` never calls it. `precompute_esm` skips outputs
that already exist, so a rerun costs only the scan.

Usage:
    python pipeline/12_compute_missing_embeddings.py --source rcsb --device cuda
    python pipeline/12_compute_missing_embeddings.py --source afdb --device cuda
    python pipeline/12_compute_missing_embeddings.py --source all  --dry_run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRECOMPUTE = ROOT / "pipeline" / "lib" / "precompute_esm.py"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pipeline" / "lib"))

from precompute_esm import get_protein_chains  # noqa: E402

from mambafold.data.sequence_cache import sequence_embedding_path  # noqa: E402

# Each source names where its records live, where its embeddings go, and how to
# enumerate the sequences that must be covered.
SOURCES = {
    "rcsb": {
        "records": "data/rcsb_train",
        "cache": "data/rcsb_esmc6b",
        "shard": False,
    },
    "afdb": {
        "records": "data/afdb_swissprot_v4/npz",
        "cache": "data/afdb_esmc6b",
        "shard": True,
    },
}


def gap_for(root: Path, source: str, min_length: int) -> list[Path]:
    """Records holding at least one sequence with no embedding in the cache."""
    spec = SOURCES[source]
    records = root / spec["records"]
    cache = root / spec["cache"]
    paths = sorted(records.rglob("*.npz") if spec["shard"] else records.glob("*.npz"))

    missing: list[Path] = []
    for path in paths:
        for sequence in get_protein_chains(path, strict=False):
            # Chains the loader will never turn into an example need no
            # embedding, and demanding one reports a healthy corpus as broken.
            if len(sequence) < min_length:
                continue
            if not sequence_embedding_path(cache, sequence).exists():
                missing.append(path.relative_to(records))
                break
    return missing


def run(root: Path, source: str, args) -> int:
    spec = SOURCES[source]
    records = root / spec["records"]
    cache = root / spec["cache"]

    print(f"== {source}: scanning {records}", flush=True)
    missing = gap_for(root, source, args.min_length)
    print(f"   {len(missing)} records need an embedding", flush=True)
    if not missing:
        print("   nothing to compute", flush=True)
        return 0

    audit = root / "data" / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    file_list = audit / f"{source}_embedding_gap.txt"
    file_list.write_text("\n".join(str(p) for p in missing) + "\n")
    print(f"   file list -> {file_list}", flush=True)
    if args.dry_run:
        return 0

    cmd = [
        sys.executable, str(PRECOMPUTE),
        "--data_dir", str(records),
        "--file_list", str(file_list),
        "--out_dir", str(cache),
        "--device", args.device,
        "--max_length", str(args.max_length),
        "--shard_idx", str(args.shard_idx),
        "--shard_count", str(args.shard_count),
    ]
    print("   " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env={**os.environ, "PYTHONPATH": str(root / "src")})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--source", default="all", choices=[*SOURCES, "all"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--min_length", type=int, default=20,
                    help="matches RCSBDataset; shorter chains never become examples")
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--shard_count", type=int, default=1)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    names = list(SOURCES) if args.source == "all" else [args.source]
    for name in names:
        run(root, name, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
