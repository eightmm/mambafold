#!/usr/bin/env python
"""Create a sequence-addressed hard-link view of a legacy occurrence cache.

The operation is non-destructive and consumes no additional data blocks when
the source and destination are on the same filesystem.  After validation, the
legacy ``<stem>_ch<index>.npy`` names can be removed separately to reclaim the
duplicate directory entries and blocks owned only by duplicate occurrences.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from precompute_esm import get_protein_chains, read_fasta

from mambafold.data.sequence_cache import sequence_embedding_path


def iter_occurrences(args, files: list[Path]):
    if args.single_chain_fasta:
        stems = {path.stem for path in files}
        for header, sequence in read_fasta(Path(args.single_chain_fasta)):
            stem, separator, _chain_name = header.rpartition("_")
            if separator and stem in stems:
                yield sequence, Path(args.esm_dir) / f"{stem}_ch0.npy"
        return

    for path in files:
        for origin, sequence in enumerate(get_protein_chains(path, strict=True)):
            yield sequence, Path(args.esm_dir) / f"{path.stem}_ch{origin}.npy"


def validate_embedding(path: Path, sequence: str, args) -> None:
    array = np.load(path, mmap_mode="r")
    expected_rows = min(len(sequence), args.max_length)
    if array.ndim != 2 or array.shape[0] != expected_rows:
        raise ValueError(f"shape={array.shape}, expected rows={expected_rows}")
    if args.embedding_dim and array.shape[1] != args.embedding_dim:
        raise ValueError(
            f"embedding dim={array.shape[1]}, expected={args.embedding_dim}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--esm_dir", required=True)
    parser.add_argument("--file_list", default=None)
    parser.add_argument("--single_chain_fasta", default=None)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--embedding_dim", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--fail_on_error", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    esm_dir = Path(args.esm_dir)
    if args.file_list:
        files = [
            data_dir / line.strip()
            for line in Path(args.file_list).read_text().splitlines()
            if line.strip()
        ]
    else:
        files = sorted(data_dir.rglob("*.npz"))
    if args.limit:
        files = files[:args.limit]

    associations = unique = reused = missing = invalid = 0
    validated: set[Path] = set()
    for sequence, occurrence_path in iter_occurrences(args, files):
        associations += 1
        sequence_path = sequence_embedding_path(esm_dir, sequence)
        if sequence_path.exists():
            try:
                if sequence_path not in validated:
                    validate_embedding(sequence_path, sequence, args)
                    validated.add(sequence_path)
                reused += 1
            except Exception as exc:
                invalid += 1
                print(f"invalid {sequence_path}: {exc}", flush=True)
            continue
        if not occurrence_path.exists():
            missing += 1
            continue
        try:
            validate_embedding(occurrence_path, sequence, args)
            sequence_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(occurrence_path, sequence_path)
                validated.add(sequence_path)
                unique += 1
            except FileExistsError:
                reused += 1
        except Exception as exc:
            invalid += 1
            print(f"invalid {occurrence_path}: {exc}", flush=True)

        if associations % 10000 == 0:
            print(
                f"[{associations}] unique={unique} reused={reused} "
                f"missing={missing} invalid={invalid}",
                flush=True,
            )

    print(
        f"Done. associations={associations} unique={unique} reused={reused} "
        f"missing={missing} invalid={invalid}",
        flush=True,
    )
    if args.fail_on_error and (missing or invalid):
        raise SystemExit(f"migration incomplete: missing={missing} invalid={invalid}")


if __name__ == "__main__":
    main()
