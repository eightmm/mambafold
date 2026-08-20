"""Per-file length cache for length-bucketed batching.

Bucketing needs each file's *actual* example length (what the collator pads to)
— not the metadata sum-of-chains length, which is wrong for single-chain
training. We get the exact value by running `RCSBDataset.probe_length`
(= `_canonicalize(...).seq_len`) over every file once and caching the result.

The cache is keyed by a hash of the filters that affect length/validity
(max_length, min_length, min_obs_ratio, single_chain_only, esm presence, and the
file set), so a config change rebuilds it. Files that fail the filters
(multi-chain, missing ESM, low observation) are omitted — the bucket sampler
then only emits valid indices, which `__getitem__` returns without its
skip-to-next-valid fallback, keeping the bucketed length aligned with the
batch's real content.
"""

from __future__ import annotations

import hashlib
import json
import os
from multiprocessing import Pool
from pathlib import Path

from mambafold.data.dataset import RCSBDataset

_DEFAULT_CACHE_DIR = Path(".cache/length_cache")
_ESM_CACHE_SCHEMA = 2  # sequence-addressed cache with occurrence fallback
_CHAIN_VALIDITY_SCHEMA = 3  # observed-atom-valid crop is required

# ── Worker (one probe dataset per process; built without scanning files) ─────
_PROBE: RCSBDataset | None = None


def _make_probe(cfg: dict) -> RCSBDataset:
    ds = RCSBDataset.__new__(RCSBDataset)  # bypass __init__ file scan
    ds.data_dir = Path(cfg["data_dir"])
    ds.max_length = cfg["max_length"]
    ds.min_length = cfg["min_length"]
    ds.min_obs_ratio = cfg["min_obs_ratio"]
    ds.esm_dir = Path(cfg["esm_dir"]) if cfg["esm_dir"] else None
    ds.single_chain_only = cfg["single_chain_only"]
    return ds


def _init_worker(cfg: dict) -> None:
    global _PROBE
    _PROBE = _make_probe(cfg)


def _probe_one(path: str):
    return path, _PROBE.probe_length(path)


def _probe_chains_one(path: str):
    return path, _PROBE.probe_chains(path)


def _dataset_cfg(dataset: RCSBDataset) -> dict:
    return {
        "data_dir": str(dataset.data_dir),
        "max_length": dataset.max_length,
        "min_length": dataset.min_length,
        "min_obs_ratio": dataset.min_obs_ratio,
        "esm_dir": str(dataset.esm_dir) if dataset.esm_dir else None,
        "esm_cache_schema": _ESM_CACHE_SCHEMA,
        "chain_validity_schema": _CHAIN_VALIDITY_SCHEMA,
        "single_chain_only": dataset.single_chain_only,
    }


def _cache_path(dataset: RCSBDataset, cache_dir: Path) -> Path:
    cfg = _dataset_cfg(dataset)
    # Hash the filters + the file set so any change rebuilds the cache.
    h = hashlib.sha1()
    h.update(json.dumps(cfg, sort_keys=True).encode())
    h.update(str(len(dataset.files)).encode())
    h.update("\n".join(str(f) for f in dataset.files).encode())
    name = Path(cfg["data_dir"]).name
    cache_name = (
        f"len_{name}_L{cfg['max_length']}_sc{int(cfg['single_chain_only'])}_"
        f"{h.hexdigest()[:12]}.json"
    )
    return cache_dir / cache_name


def build_length_cache(
    dataset: RCSBDataset,
    num_workers: int = 8,
    cache_dir: Path | str = _DEFAULT_CACHE_DIR,
    force: bool = False,
) -> dict[str, int]:
    """Return {file_path_str: length} for valid files, building/caching as needed."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(dataset, cache_dir)

    if path.exists() and not force:
        with open(path) as f:
            return json.load(f)

    files = [str(f) for f in dataset.files]
    cfg = _dataset_cfg(dataset)
    print(f"[length_cache] building over {len(files)} files ({num_workers} workers) → {path}")
    result: dict[str, int] = {}
    with Pool(num_workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        for i, (p, L) in enumerate(pool.imap_unordered(_probe_one, files, chunksize=64)):
            if L is not None:
                result[p] = int(L)
            if (i + 1) % 20000 == 0:
                print(f"[length_cache]   {i + 1}/{len(files)} probed, {len(result)} valid")
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        json.dump(result, f)
    tmp.replace(path)
    print(f"[length_cache] done: {len(result)}/{len(files)} valid → {path}")
    return result


def index_lengths_for_dataset(
    dataset: RCSBDataset,
    num_workers: int = 8,
    cache_dir: Path | str = _DEFAULT_CACHE_DIR,
) -> dict[int, int]:
    """Map dataset index → true length for valid files (the bucket sampler input)."""
    by_path = build_length_cache(dataset, num_workers=num_workers, cache_dir=cache_dir)
    out: dict[int, int] = {}
    for i, f in enumerate(dataset.files):
        L = by_path.get(str(f))
        if L is not None:
            out[i] = L
    return out


def build_chain_index(
    dataset: RCSBDataset,
    num_workers: int = 8,
    cache_dir: Path | str = _DEFAULT_CACHE_DIR,
    force: bool = False,
) -> list[tuple[int, int, int]]:
    """Monomer-extraction index: [(file_idx, chain_origin, length), ...] over every
    valid protein chain of every file. Cached (keyed by the same filters)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    base = _cache_path(dataset, cache_dir)
    path = base.with_name("chainidx_" + base.name)

    if path.exists() and not force:
        with open(path) as f:
            return [tuple(t) for t in json.load(f)]

    files = [str(f) for f in dataset.files]
    path_to_idx = {str(f): i for i, f in enumerate(dataset.files)}
    cfg = _dataset_cfg(dataset)
    print(f"[chain_index] building over {len(files)} files ({num_workers} workers) → {path}")
    index: list[tuple[int, int, int]] = []
    with Pool(num_workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        for i, (p, chains) in enumerate(
            pool.imap_unordered(_probe_chains_one, files, chunksize=32)
        ):
            fi = path_to_idx[p]
            for origin, L in chains:
                index.append((fi, int(origin), int(L)))
            if (i + 1) % 20000 == 0:
                print(f"[chain_index]   {i + 1}/{len(files)} files, {len(index)} chains")
    index.sort()  # determinism (by file_idx, origin)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        json.dump(index, f)
    tmp.replace(path)
    print(f"[chain_index] done: {len(index)} monomer chains from {len(files)} files → {path}")
    return index
