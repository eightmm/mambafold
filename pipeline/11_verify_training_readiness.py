#!/usr/bin/env python
"""Stage 11 — check both training sources are actually loadable and embedded.

Earlier stages prove the corpus is *admitted* and *linked*. Neither shows that a
training step can consume it. A record can pass admission and still be useless:
its structure may not parse, its chains may canonicalise to nothing, or its
embedding may exist at the right path with the wrong number of rows.

The last case is the dangerous one. The loader resolves embeddings by sequence
hash and silently drops an example when the lookup fails, so a systematic
mismatch shows up as a quietly smaller corpus rather than an error. This stage
reads every record and checks the embedding against the sequence it must
correspond to.

Checked per record, for both RCSB and AFDB:

* the ``.npz`` opens and carries the Boltz fields the loader reads
* at least one protein chain canonicalises to a non-empty sequence
* every such sequence resolves to an embedding
* that embedding has ``min(len(sequence), max_length)`` rows and the PLM width
  the model config declares

Parallel, because reading ~430k structure records serially takes hours.

Usage:
    python pipeline/11_verify_training_readiness.py --workers 16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pipeline" / "lib"))

from precompute_esm import get_protein_chains  # noqa: E402

from mambafold.data.sequence_cache import sequence_embedding_path  # noqa: E402

_CFG: dict = {}


def _init(cache_dir: str, embed_dim: int, max_length: int, min_length: int) -> None:
    _CFG.update(cache=cache_dir, embed_dim=embed_dim,
                max_length=max_length, min_length=min_length)


def _check(path_str: str) -> tuple[str, str, int, int]:
    """Return (verdict, entry_id, n_chains, n_residues_total)."""
    path = Path(path_str)
    try:
        chains = get_protein_chains(path, strict=True)
    except Exception:  # noqa: BLE001
        return "unreadable_record", path.stem, 0, 0
    if not chains:
        return "no_protein_chain", path.stem, 0, 0

    # Apply the loader's own length gate before demanding an embedding. A
    # chain shorter than `min_length` is never turned into an example, so a
    # missing embedding for it is correct, not a gap — several RCSB entries
    # carry single-residue protein chains that would otherwise be reported as
    # broken corpus.
    chains = [s for s in chains if len(s) >= _CFG["min_length"]]
    if not chains:
        return "below_min_length", path.stem, 0, 0

    total = 0
    for sequence in chains:
        total += len(sequence)
        embedding = sequence_embedding_path(_CFG["cache"], sequence)
        if not embedding.exists():
            return "missing_embedding", path.stem, len(chains), total
        try:
            array = np.load(embedding, mmap_mode="r")
        except Exception:  # noqa: BLE001
            return "unreadable_embedding", path.stem, len(chains), total
        expected = min(len(sequence), _CFG["max_length"])
        if array.ndim != 2 or array.shape[0] != expected:
            return "embedding_row_mismatch", path.stem, len(chains), total
        if array.shape[1] != _CFG["embed_dim"]:
            return "embedding_width_mismatch", path.stem, len(chains), total
    return "ok", path.stem, len(chains), total


def verify(name: str, files: list[Path], cache: str, embed_dim: int,
           max_length: int, min_length: int, workers: int) -> dict:
    print(f"== {name}: {len(files)} records, {workers} workers", flush=True)
    verdicts: Counter[str] = Counter()
    offenders: dict[str, list[str]] = {}
    chains = residues = 0
    done = 0
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init,
        initargs=(cache, embed_dim, max_length, min_length)
    ) as pool:
        for verdict, entry_id, n_chains, n_res in pool.map(
            _check, [str(p) for p in files], chunksize=256
        ):
            verdicts[verdict] += 1
            if verdict == "ok":
                chains += n_chains
                residues += n_res
            elif len(offenders.setdefault(verdict, [])) < 20:
                offenders[verdict].append(entry_id)
            done += 1
            if done % 50000 == 0:
                print(f"   {done}/{len(files)}  {dict(verdicts)}", flush=True)

    report = {
        "records": len(files),
        "verdicts": dict(verdicts),
        "trainable_records": verdicts["ok"],
        "protein_chains": chains,
        "total_residues": residues,
        "examples": offenders,
    }
    print(f"   {json.dumps({k: v for k, v in report.items() if k != 'examples'})}", flush=True)
    for verdict, ids in offenders.items():
        print(f"   {verdict}: {ids[:5]}", flush=True)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--embed_dim", type=int, default=2560, help="ESMC-6B width")
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--min_length", type=int, default=20,
                    help="matches RCSBDataset; shorter chains never become examples")
    ap.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK") or max(1, (os.cpu_count() or 2) - 1)),
    )
    ap.add_argument("--limit", type=int, default=0, help="check only the first N of each source")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    sources = [
        ("rcsb", sorted((root / "data" / "rcsb_train").glob("*.npz")),
         str(root / "data" / "rcsb_esmc6b")),
        ("afdb_v4", sorted((root / "data" / "afdb_swissprot_v4" / "npz").rglob("*.npz")),
         str(root / "data" / "afdb_esmc6b")),
    ]

    summary = {
        "embed_dim": args.embed_dim,
        "max_length": args.max_length,
        "min_length": args.min_length,
        "sources": {},
    }
    for name, files, cache in sources:
        if args.limit:
            files = files[: args.limit]
        summary["sources"][name] = verify(
            name, files, cache, args.embed_dim, args.max_length,
            args.min_length, args.workers
        )

    totals = summary["sources"]
    summary["combined"] = {
        "records": sum(s["records"] for s in totals.values()),
        "trainable_records": sum(s["trainable_records"] for s in totals.values()),
        "protein_chains": sum(s["protein_chains"] for s in totals.values()),
        "total_residues": sum(s["total_residues"] for s in totals.values()),
    }

    out = root / "data" / "audit" / "training_readiness.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["combined"], indent=2))
    print(f"report -> {out}")

    blocking = sum(
        count
        for source in totals.values()
        for verdict, count in source["verdicts"].items()
        if verdict not in ("ok", "no_protein_chain", "below_min_length")
    )
    if blocking:
        print(f"WARNING: {blocking} records cannot be trained on as they stand", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
