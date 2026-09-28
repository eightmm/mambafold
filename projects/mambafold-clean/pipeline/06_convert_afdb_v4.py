#!/usr/bin/env python
"""Stream ``swissprot_cif_v4.tar`` and convert the SimpleFold subset to npz.

SimpleFold trains its distillation stream on AFDB SwissProt **v4**
(``configs/data/pdb_sp.yaml`` -> ``data/swissprot_pdb_v4_boltz``) restricted to
its published ``swissprot_list.csv``. Our earlier corpus resolved the same
accessions through the AlphaFold API and therefore received **v6** coordinates
(``data/afdb_swissprot/manifest.tsv`` records ``model_created 2025-08-01``),
which both breaks recipe parity and leaves the prediction model's training
cutoff unknown. This script rebuilds the v4 corpus from the frozen EBI archive.

The tar holds every SwissProt entry as ``<id>.cif.gz`` at the archive root, so
one sequential pass extracts only the listed subset. Existing npz outputs are
skipped, which makes the run resumable.

The ESMC-6B cache is addressed by the SHA-256 of the canonical sequence, so a
v4 record reuses the cached embedding whenever its sequence matches the v6
record for the same accession. ``sequences.tsv`` records the per-entry sequence
so ``compare_afdb_v4_v6_sequences.py`` can verify that reuse.

Usage:
    uv run --no-sync python scripts/convert_afdb_v4_tar.py \
      --tar data/external/afdb_v4/swissprot_cif_v4.tar \
      --id_list data/external/simplefold/swissprot_list.csv \
      --out_dir data/afdb_swissprot_v4/npz \
      --manifest data/afdb_swissprot_v4/manifest.tsv \
      --sequences data/afdb_swissprot_v4/sequences.tsv
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pipeline" / "lib"))

from pdb_to_npz import convert, parse_cif  # noqa: E402

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def read_id_list(path: Path) -> set[str]:
    ids: set[str] = set()
    for raw in path.read_text().splitlines():
        token = raw.strip().split(",")[0]
        if not token or token.lower().startswith("af-") is False:
            continue
        ids.add(token)
    return ids


def shard_dir(out_dir: Path, entry_id: str) -> Path:
    # Mirror the existing v6 layout: two-character shard from the accession.
    accession = entry_id.split("-")[1] if "-" in entry_id else entry_id
    return out_dir / accession[:2].lower()


def sequence_from_arrays(arrays: dict) -> str:
    residues = arrays["residues"]
    names = residues["name"]
    letters = []
    for name in names:
        key = name.decode() if isinstance(name, bytes) else str(name)
        letters.append(THREE_TO_ONE.get(key.upper(), "X"))
    return "".join(letters)


def build_member_index(
    tar_path: Path, wanted: set[str], index_path: Path
) -> list[tuple[str, int, int]]:
    """One sequential pass recording where each wanted member's data begins.

    The archive is complete and therefore seekable, so a single scan buys random
    access for every worker afterwards. Doing it per worker instead would mean
    re-reading 39 GB per process.
    """
    if index_path.exists():
        rows = [tuple(r) for r in json.loads(index_path.read_text())]
        print(f"member index: reusing {len(rows)} entries from {index_path}", flush=True)
        return rows

    rows: list[tuple[str, int, int]] = []
    started = time.time()
    with tarfile.open(tar_path, mode="r|") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = Path(member.name).name
            if not name.endswith(".cif.gz"):
                continue
            entry_id = name[: -len(".cif.gz")]
            if entry_id in wanted:
                rows.append((entry_id, member.offset_data, member.size))
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(rows))
    print(
        f"member index: {len(rows)} entries in {time.time() - started:.0f}s -> {index_path}",
        flush=True,
    )
    return rows


_WORKER: dict = {}


def _worker_init(tar_path: str, out_dir: str) -> None:
    _WORKER["handle"] = open(tar_path, "rb")  # noqa: SIM115 - lives for the process
    _WORKER["out_dir"] = Path(out_dir)


def _worker_convert(row: tuple[str, int, int]) -> tuple[str, str, str]:
    """Return (entry_id, sequence, error). Exactly one of sequence/error is set."""
    entry_id, offset, size = row
    out_dir = _WORKER["out_dir"]
    npz_path = shard_dir(out_dir, entry_id) / f"{entry_id}.npz"
    if npz_path.exists():
        try:
            with np.load(npz_path, allow_pickle=True) as existing:
                return entry_id, sequence_from_arrays(existing), ""
        except Exception as exc:  # noqa: BLE001
            return entry_id, "", f"pre-existing unreadable: {type(exc).__name__}: {exc}"
    try:
        handle = _WORKER["handle"]
        handle.seek(offset)
        raw = handle.read(size)
        cif_text = gzip.decompress(raw).decode("utf-8", errors="ignore")
        arrays = convert(parse_cif(cif_text), entry_id, verbose=False)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        # numpy appends ".npz" to any path that does not already end in it, so
        # the temp name must keep that suffix last.
        tmp_path = npz_path.with_suffix(".tmp.npz")
        np.savez(tmp_path, **arrays)
        tmp_path.replace(npz_path)
        return entry_id, sequence_from_arrays(arrays), ""
    except Exception as exc:  # noqa: BLE001
        return entry_id, "", f"{type(exc).__name__}: {exc}"


def convert_parallel(args, wanted: set[str], done: set[str], manifest, sequences) -> dict:
    out_dir = Path(args.out_dir)
    index = build_member_index(
        Path(args.tar), wanted, out_dir.parent / "member_index.json"
    )
    todo = [row for row in index if row[0] not in done]
    print(
        f"parallel convert: {len(todo)} of {len(index)} members, {args.workers} workers",
        flush=True,
    )

    converted = failed = 0
    started = time.time()
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_init,
        initargs=(str(args.tar), str(out_dir)),
    ) as pool:
        for entry_id, sequence, error in pool.map(_worker_convert, todo, chunksize=64):
            if error:
                failed += 1
                manifest.write(f"{entry_id}\t\tfail\t\t{error}\n")
                continue
            npz_path = shard_dir(out_dir, entry_id) / f"{entry_id}.npz"
            sequences.write(f"{entry_id}\t{sequence}\n")
            manifest.write(f"{entry_id}\t{npz_path}\tok\t{len(sequence)}\t\n")
            converted += 1
            if converted % args.report_every == 0:
                manifest.flush()
                sequences.flush()
                elapsed = time.time() - started
                rate = converted / max(elapsed, 1e-9) * 60
                print(
                    f"[{elapsed:.0f}s] converted={converted} failed={failed} "
                    f"rate={rate:.0f}/min",
                    flush=True,
                )
    return {"converted": converted, "failed": failed, "seen": len(index)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", required=True)
    ap.add_argument("--id_list", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--sequences", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--report_every", type=int, default=5000)
    ap.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK") or max(1, (os.cpu_count() or 2) - 1)),
        help="parallel converters; 1 falls back to the sequential stream. "
             "Defaults to the Slurm allocation, not the node's core count, so a "
             "batch job does not oversubscribe its cgroup.",
    )
    args = ap.parse_args()

    wanted = read_id_list(Path(args.id_list))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    sequences_path = Path(args.sequences)
    sequences_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if manifest_path.exists():
        for line_number, line in enumerate(manifest_path.read_text().splitlines()):
            if line_number == 0:
                continue
            parts = line.split("\t")
            if len(parts) >= 3 and parts[2] == "ok":
                done.add(parts[0])
    print(f"wanted={len(wanted)} already_converted={len(done)}", flush=True)

    manifest_new = not manifest_path.exists()
    manifest = manifest_path.open("a")
    if manifest_new:
        manifest.write("entry_id\tnpz_path\tstatus\tn_residues\terror\n")
    sequences = sequences_path.open("a")
    if not sequences_path.stat().st_size:
        sequences.write("entry_id\tsequence\n")

    if args.workers > 1:
        stats = convert_parallel(args, wanted, done, manifest, sequences)
        manifest.close()
        sequences.close()
        print(
            f"Done. seen={stats['seen']} converted={stats['converted']} "
            f"skipped={len(done)} failed={stats['failed']} "
            f"missing_from_tar={len(wanted) - stats['seen']}",
            flush=True,
        )
        return 0

    seen = converted = skipped = failed = 0
    started = time.time()
    with tarfile.open(args.tar, mode="r|") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = Path(member.name).name
            if not name.endswith(".cif.gz"):
                continue
            entry_id = name[: -len(".cif.gz")]
            if entry_id not in wanted:
                continue
            seen += 1
            if entry_id in done:
                skipped += 1
                continue

            target_dir = shard_dir(out_dir, entry_id)
            npz_path = target_dir / f"{entry_id}.npz"
            if npz_path.exists():
                # The record survived an interrupted run. Recover its sequence
                # from the npz rather than marking it ok with no sequences.tsv
                # row: `done` is rebuilt from ok rows, so the entry would never
                # be revisited and its sequence would be permanently lost.
                skipped += 1
                try:
                    with np.load(npz_path, allow_pickle=True) as existing:
                        seq = sequence_from_arrays(existing)
                    sequences.write(f"{entry_id}\t{seq}\n")
                    manifest.write(f"{entry_id}\t{npz_path}\tok\t{len(seq)}\tpre-existing\n")
                except Exception as exc:  # noqa: BLE001
                    manifest.write(
                        f"{entry_id}\t{npz_path}\tfail\t\tpre-existing unreadable: "
                        f"{type(exc).__name__}: {exc}\n"
                    )
                continue

            handle = tar.extractfile(member)
            if handle is None:
                failed += 1
                manifest.write(f"{entry_id}\t\tfail\t\tno stream\n")
                continue
            try:
                cif_text = gzip.decompress(handle.read()).decode("utf-8", errors="ignore")
                structure = parse_cif(cif_text)
                arrays = convert(structure, entry_id, verbose=False)
                target_dir.mkdir(parents=True, exist_ok=True)
                # numpy appends ".npz" to any path that does not already end
                # in it, so the temp name must keep that suffix last.
                tmp_path = npz_path.with_suffix(".tmp.npz")
                np.savez(tmp_path, **arrays)
                tmp_path.replace(npz_path)
                seq = sequence_from_arrays(arrays)
                sequences.write(f"{entry_id}\t{seq}\n")
                manifest.write(f"{entry_id}\t{npz_path}\tok\t{len(seq)}\t\n")
                converted += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                manifest.write(f"{entry_id}\t\tfail\t\t{type(exc).__name__}: {exc}\n")

            if converted and converted % args.report_every == 0:
                manifest.flush()
                sequences.flush()
                elapsed = time.time() - started
                print(
                    f"[{elapsed:.0f}s] seen={seen} converted={converted} "
                    f"skipped={skipped} failed={failed}",
                    flush=True,
                )
            if args.limit and converted >= args.limit:
                break

    manifest.close()
    sequences.close()
    print(
        f"Done. seen={seen} converted={converted} skipped={skipped} failed={failed} "
        f"missing_from_tar={len(wanted) - seen}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
