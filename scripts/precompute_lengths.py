#!/usr/bin/env python
"""Pre-build the per-file length cache used by length-bucketed batching.

Probes every training file once (validity + true example length) and writes a
cache keyed by the length/validity filters. The training loader auto-builds this
on the first bucketing run; running it ahead of time avoids the startup scan.

Usage:
    PYTHONPATH=src uv run python scripts/precompute_lengths.py \
        --config configs/direct_allatom_360m.yaml [--length_cache_workers 16]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mambafold.data.dataset import RCSBDataset  # noqa: E402
from mambafold.data.length_cache import build_length_cache  # noqa: E402
from mambafold.train.config import parse_args  # noqa: E402


def main():
    args, _ = parse_args()
    esm_dir = getattr(args, "esm_dir", None)
    workers = getattr(args, "length_cache_workers", 8)
    extract = bool(getattr(args, "extract_monomer_chains", False))
    # Constructing with extract_monomer_chains builds (and caches) the chain index.
    ds = RCSBDataset(
        data_dir=args.data_dir,
        max_length=args.max_length,
        file_list=getattr(args, "file_list", None),
        esm_dir=esm_dir,
        single_chain_only=bool(getattr(args, "single_chain_only", False)),
        extract_monomer_chains=extract,
        chain_index_workers=workers,
    )
    if extract:
        lens = sorted(t[2] for t in ds.chain_index)
        n = len(lens)
        print(f"[precompute] extract_monomer_chains: {n} monomer chains from "
              f"{len(ds.files)} files; len min={lens[0]} median={lens[n // 2]} max={lens[-1]}")
    else:
        cache = build_length_cache(ds, num_workers=workers)
        lens = sorted(cache.values())
        n = len(lens)
        print(f"[precompute] single-chain entries: valid={n}/{len(ds.files)}; "
              f"len min={lens[0]} median={lens[n // 2]} max={lens[-1]}")


if __name__ == "__main__":
    main()
