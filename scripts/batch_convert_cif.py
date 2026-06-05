"""Batch convert sharded mmCIF (.cif.gz) → MambaFold .npz.

Reuses pdb_to_npz.convert() but reads from a local gzip instead of
hitting the RCSB network. Skips files whose .npz already exists.

Usage:
    PYTHONPATH=src python scripts/batch_convert_cif.py \\
        --cif_dir data/rcsb_cif --out_dir data/rcsb \\
        --workers 32

The input layout is `<cif_dir>/<shard>/<pdb_id>.cif.gz` (the pattern written
by scripts/download_rcsb_cif.sh). Output mirrors the same sharding.
"""

import argparse
import gzip
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pdb_to_npz import convert, parse_cif  # noqa: E402


def _convert_one(args):
    cif_path, out_path = args
    try:
        if out_path.exists() and out_path.stat().st_size > 0:
            return ("skip", str(cif_path), None)
        with gzip.open(cif_path, "rt") as fh:
            cif_text = fh.read()
        structure = parse_cif(cif_text)
        arrays = convert(structure, cif_path.stem)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **arrays)
        return ("ok", str(cif_path), None)
    except Exception as e:  # noqa: BLE001
        return ("fail", str(cif_path), f"{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2))
    ap.add_argument("--limit", type=int, default=0, help="debug: cap files processed")
    ap.add_argument("--fail_log", default=None)
    args = ap.parse_args()

    cif_dir = Path(args.cif_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fail_log = Path(args.fail_log or out_dir / "_convert_failed.tsv")

    cif_paths = sorted(cif_dir.rglob("*.cif.gz"))
    if args.limit:
        cif_paths = cif_paths[: args.limit]
    print(f"Found {len(cif_paths)} cif files")

    jobs = []
    for p in cif_paths:
        pdb_id = p.stem.replace(".cif", "")   # "101m.cif.gz" → stem=".cif" issue? stem="101m.cif"
        pdb_id = p.name.removesuffix(".cif.gz")
        out_path = out_dir / p.parent.name / f"{pdb_id}.npz"
        jobs.append((p, out_path))

    t0 = time.time()
    ok = skip = fail = 0
    fail_lines = []
    with mp.Pool(args.workers) as pool, \
         open(fail_log, "w") as flog:
        flog.write("cif_path\terror\n")
        for i, (status, path, err) in enumerate(
                pool.imap_unordered(_convert_one, jobs, chunksize=16), 1):
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                flog.write(f"{path}\t{err}\n")
                if fail <= 5:
                    fail_lines.append(f"  {path}: {err}")
            if i % 5000 == 0 or i == len(jobs):
                dt = time.time() - t0
                rate = i / max(dt, 1e-6)
                print(f"[{i}/{len(jobs)}] ok={ok} skip={skip} fail={fail} "
                      f"rate={rate:.0f}/s eta={(len(jobs)-i)/max(rate,1e-6):.0f}s",
                      flush=True)
    dt = time.time() - t0
    print(f"\nDone in {dt:.0f}s. ok={ok} skip={skip} fail={fail}")
    print(f"Fail log: {fail_log}")
    if fail_lines:
        print("First failures:")
        print("\n".join(fail_lines))


if __name__ == "__main__":
    main()
