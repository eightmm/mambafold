#!/usr/bin/env python
"""Stage 13 — build the loader's chain index before training, not during it.

With `extract_monomer_chains`, the training unit is a chain, and the dataset
has to know which chains exist and how long each one is before it can bucket
them. `RCSBDataset.__init__` works that out by probing every record: it walks
the residues, resolves each chain's ESM embedding, clamps the length to the
embedding's row count, and checks a valid observed-atom crop exists. Over this
corpus that is ~430k records and ~765k chains, which is over an hour of CPU.

Left to the training job it is paid at startup, by every DDP rank at once, and
again after every restart. So it is built here instead, once, on a compute
node — and the training job finds a cache and starts in seconds.

The cache key is a hash of the filters that change the answer (max_length,
min_length, min_obs_ratio, esm_dir, single_chain_only) plus the file set, so
this stage must be given the *same* values the training config uses. Pass the
training config and they are read from it.

The stage also reports the chain-length distribution of what actually survives
those filters. That is a stricter population than the FASTA-level distribution
in `data/audit/chain_length_distribution.json` — a chain with no embedding or
no valid crop is admitted corpus but not a training example — and it is the
population the crop and batch size have to be chosen against.

Usage:
    python pipeline/13_prebuild_loader_caches.py --config configs/run_a_mamba.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mambafold.data.dataset import RCSBDataset  # noqa: E402
from mambafold.data.length_cache import _cache_path, build_chain_index  # noqa: E402

BUCKETS = ((0, 128), (129, 256), (257, 384), (385, 512), (513, 768), (769, 1024))


def summarize(lengths: list[int]) -> dict:
    """Chain-length distribution plus what a fixed crop would actually see."""
    if not lengths:
        return {"chains": 0}
    ordered = sorted(lengths)
    n = len(ordered)
    total = sum(ordered)

    def quantile(p: float) -> int:
        return ordered[min(n - 1, int(round(p * (n - 1))))]

    counts: Counter[str] = Counter()
    for length in ordered:
        for low, high in BUCKETS:
            if low <= length <= high:
                counts[f"{low}-{high}"] += 1
                break
        else:
            counts[f">{BUCKETS[-1][1]}"] += 1
    return {
        "chains": n,
        "residues": total,
        "mean": round(total / n, 1),
        "median": quantile(0.5),
        "p90": quantile(0.9),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
        "max": ordered[-1],
        "buckets": {
            key: {"chains": counts[key], "pct": round(100 * counts[key] / n, 2)}
            for key in sorted(counts, key=lambda k: int(k.strip(">").split("-")[0]))
        },
        # A batch pads to its longest member, so the padding waste of a bucketed
        # batch is bounded by how tightly the bucket holds. This is the ceiling:
        # what a single un-bucketed batch padded to the global max would cost.
        "padding_waste_at_fixed_crop": round(1 - total / (n * ordered[-1]), 4),
    }


def build_source(name: str, source: dict, args, workers: int) -> dict:
    data_dir = ROOT / source["data_dir"]
    esm_dir = source.get("esm_dir")
    file_list = source.get("file_list")
    print(f"== {name}: {data_dir}", flush=True)

    started = time.time()
    # Constructed exactly the way `loader._build_rcsb_dataset` constructs it.
    # `min_length` and `min_obs_ratio` are deliberately not passed: the loader
    # does not pass them either, and both feed the cache key — setting them
    # here would produce a cache training cannot find, which is the one failure
    # this stage must not have. It would not error, it would silently rebuild.
    dataset = RCSBDataset(
        data_dir=str(data_dir),
        max_length=args.max_length,
        file_list=str(ROOT / file_list) if file_list else None,
        esm_dir=str(ROOT / esm_dir) if esm_dir else None,
        single_chain_only=args.single_chain_only,
        dedup_homomer_chains=args.dedup_homomer_chains,
        extract_monomer_chains=False,  # index built explicitly below
        chain_index_workers=workers,
    )
    index = build_chain_index(dataset, num_workers=workers, force=args.force)
    elapsed = round(time.time() - started, 1)

    base = _cache_path(dataset, Path(".cache/length_cache"))
    cache_file = base.with_name("chainidx_" + base.name)
    lengths = [entry[2] for entry in index]
    entries_with_chains = len({entry[0] for entry in index})

    report = {
        "data_dir": str(data_dir.relative_to(ROOT)),
        "esm_dir": esm_dir,
        "file_list": file_list,
        # Read back off the dataset rather than restated, so the report cannot
        # claim filters the cache was not actually built under.
        "min_length": dataset.min_length,
        "min_obs_ratio": dataset.min_obs_ratio,
        "records": len(dataset.files),
        "records_contributing_a_chain": entries_with_chains,
        "records_yielding_nothing": len(dataset.files) - entries_with_chains,
        "cache_file": str(cache_file),
        "seconds": elapsed,
        "distribution": summarize(lengths),
    }
    print(json.dumps({k: v for k, v in report.items() if k != "distribution"}, indent=2),
          flush=True)
    print(json.dumps(report["distribution"], indent=2), flush=True)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="the training config this cache must match")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="rebuild even if the cache exists")
    ap.add_argument("--out", default="data/audit/loader_caches.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    sources = cfg.get("train_sources")
    if not sources:
        print(f"{args.config} declares no train_sources", file=sys.stderr)
        return 2

    import os

    workers = args.workers or int(
        os.environ.get("SLURM_CPUS_PER_TASK") or max(1, (os.cpu_count() or 2) - 1)
    )

    # Everything the cache key depends on comes from the training config, so a
    # cache built here is the cache training finds. A mismatch here would not
    # error — it would silently rebuild at startup, which is the cost this
    # stage exists to avoid.
    class Filters:
        max_length = int(cfg.get("max_length", 1024))
        single_chain_only = bool(cfg.get("single_chain_only", False))
        dedup_homomer_chains = bool(cfg.get("dedup_homomer_chains", False))
        force = args.force

    if not cfg.get("extract_monomer_chains", False):
        print(
            "warning: the config does not set extract_monomer_chains; the chain "
            "index built here is then unused at training time",
            file=sys.stderr,
            flush=True,
        )

    summary = {
        "config": args.config,
        "filters": {
            "max_length": Filters.max_length,
            "single_chain_only": Filters.single_chain_only,
            "dedup_homomer_chains": Filters.dedup_homomer_chains,
            "min_length": "RCSBDataset default (see per-source report)",
            "min_obs_ratio": "RCSBDataset default (see per-source report)",
        },
        "workers": workers,
        "sources": {},
    }
    for source in sources:
        name = source.get("name") or Path(source["data_dir"]).name
        summary["sources"][name] = build_source(name, source, Filters, workers)

    every_length = []
    for report in summary["sources"].values():
        distribution = report["distribution"]
        every_length.append((distribution.get("chains", 0), distribution.get("residues", 0)))
    summary["combined"] = {
        "chains": sum(c for c, _ in every_length),
        "residues": sum(r for _, r in every_length),
    }

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["combined"], indent=2))
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
