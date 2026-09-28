#!/usr/bin/env python
"""Stage 07b — verify every reused AFDB embedding against the v6 record.

Stage 07 reuses the previously computed ESMC-6B embeddings by joining on the
UniProt accession, on the premise that an AlphaFold DB version change alters the
prediction and not the sequence. That premise is mostly true and measurably not
always true: stage 07's random sample of 3,000 accessions found 7 whose v4 and
v6 sequences differ at the *same* length, which the row-count guard cannot see.
A same-length difference means the cached embedding encodes a different protein,
so the link is silently wrong.

At roughly 0.2% of 269,003 records that is several hundred wrong embeddings —
few enough to be invisible in aggregate metrics, many enough to be a defect.
This stage removes the guesswork: it compares every accession, drops the links
that do not match, and writes the list that has to be recomputed.

Reading 269k structure records is I/O-bound, so the comparison is parallel. The
v6 records are only read, never modified.

Usage:
    python pipeline/07b_verify_afdb_sequences.py \
      --v4_sequences data/afdb_swissprot_v4/sequences.tsv \
      --v6_npz_dir "$LEGACY_AFDB_V6_NPZ" \
      --dest_esm_dir data/afdb_esmc6b \
      --out_dir data/afdb_swissprot_v4/sequence_audit \
      --workers 16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pipeline" / "lib"))

from precompute_esm import get_protein_chains  # noqa: E402

from mambafold.data.sequence_cache import sequence_embedding_path  # noqa: E402


def accession(entry_id: str) -> str:
    parts = entry_id.split("-")
    return parts[1] if len(parts) > 1 else entry_id


def read_v4_sequences(path: Path) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    for index, line in enumerate(path.read_text().splitlines()):
        if index == 0 and line.startswith("entry_id\t"):
            continue
        entry_id, _, sequence = line.partition("\t")
        if entry_id and sequence:
            out[accession(entry_id)] = (entry_id, sequence)
    return out


_V4: dict[str, tuple[str, str]] = {}


def _init(v4_path: str) -> None:
    _V4.update(read_v4_sequences(Path(v4_path)))


def _compare(npz_path_str: str) -> tuple[str, str] | None:
    """Return (entry_id, verdict) for accessions present in the v4 corpus."""
    path = Path(npz_path_str)
    acc = accession(path.stem)
    record = _V4.get(acc)
    if record is None:
        return None
    entry_id, v4_sequence = record
    chains = list(get_protein_chains(path, strict=False))
    if not chains:
        return entry_id, "v6_unreadable"
    return entry_id, ("match" if chains[0] == v4_sequence else "mismatch")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4_sequences", required=True)
    ap.add_argument("--v6_npz_dir", required=True)
    ap.add_argument("--dest_esm_dir", required=True)
    ap.add_argument(
        "--legacy_esm_dir",
        default=None,
        help="the cache stage 07 borrowed from; needed to tell a borrowed "
             "embedding from one stage 07c computed at the same path",
    )
    ap.add_argument("--legacy_prefix", default="AF-")
    ap.add_argument("--legacy_suffix", default="-F1-model_v6_ch0.npy")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK") or max(1, (os.cpu_count() or 2) - 1)),
    )
    ap.add_argument(
        "--keep_links",
        action="store_true",
        help="report only; do not remove the links that failed verification",
    )
    args = ap.parse_args()

    v4 = read_v4_sequences(Path(args.v4_sequences))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"v4 records: {len(v4)}", flush=True)

    paths = [str(p) for p in Path(args.v6_npz_dir).rglob("*.npz")]
    print(f"v6 records to read: {len(paths)} with {args.workers} workers", flush=True)

    matched: list[str] = []
    mismatched: list[str] = []
    unreadable: list[str] = []
    started = time.time()
    done = 0
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init, initargs=(args.v4_sequences,)
    ) as pool:
        for result in pool.map(_compare, paths, chunksize=256):
            if result is None:
                continue
            entry_id, verdict = result
            if verdict == "match":
                matched.append(entry_id)
            elif verdict == "mismatch":
                mismatched.append(entry_id)
            else:
                unreadable.append(entry_id)
            done += 1
            if done % 50000 == 0:
                elapsed = time.time() - started
                print(
                    f"  {done} compared  {elapsed:.0f}s  "
                    f"mismatch={len(mismatched)} unreadable={len(unreadable)}",
                    flush=True,
                )

    compared = set(matched) | set(mismatched) | set(unreadable)
    absent = [eid for _acc, (eid, _seq) in v4.items() if eid not in compared]

    # A borrowed embedding that failed verification encodes a different protein,
    # which is worse than no embedding at all: the loader would silently train
    # on it. Remove it and let the recompute list carry the entry.
    #
    # Only *borrowed* ones. Once stage 07c computes an embedding from the v4
    # sequence it lands at this same path, and it is correct precisely because
    # the sequences disagree. A borrowed file is a hard link and still shares an
    # inode with its source in the legacy cache; a computed one does not. That
    # distinction is what keeps this stage idempotent — without it, a rerun
    # deletes the very embeddings the previous run computed.
    removed = kept_recomputed = 0
    dest_root = Path(args.dest_esm_dir)
    legacy_dir = Path(args.legacy_esm_dir) if args.legacy_esm_dir else None
    if not args.keep_links:
        by_entry = {eid: (acc, seq) for acc, (eid, seq) in v4.items()}
        for entry_id in mismatched + unreadable:
            record = by_entry.get(entry_id)
            if record is None:
                continue
            acc, sequence = record
            link = sequence_embedding_path(dest_root, sequence)
            if not link.exists():
                continue
            borrowed = False
            if legacy_dir is not None:
                legacy = legacy_dir / f"{args.legacy_prefix}{acc}{args.legacy_suffix}"
                if legacy.exists() and legacy.stat().st_ino == link.stat().st_ino:
                    borrowed = True
            if borrowed:
                link.unlink()
                removed += 1
            else:
                kept_recomputed += 1

    needs = sorted(set(mismatched) | set(unreadable) | set(absent))
    (out_dir / "needs_new_embedding_full.txt").write_text(
        "\n".join(needs) + ("\n" if needs else "")
    )
    (out_dir / "sequence_mismatch_ids.txt").write_text(
        "\n".join(sorted(mismatched)) + ("\n" if mismatched else "")
    )

    report = {
        "scope": "every v4 accession compared against its v6 record",
        "v4_records": len(v4),
        "compared": len(compared),
        "sequence_match": len(matched),
        "sequence_mismatch": len(mismatched),
        "v6_unreadable": len(unreadable),
        "absent_from_v6": len(absent),
        "links_removed": removed,
        "recomputed_embeddings_kept": kept_recomputed,
        "needs_new_embedding": len(needs),
        "mismatch_rate": round(len(mismatched) / max(len(compared), 1), 6),
    }
    (out_dir / "full_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if mismatched:
        print(
            f"{len(mismatched)} accessions changed sequence between AFDB v4 and v6; "
            "their embeddings must be recomputed from the v4 sequences",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
