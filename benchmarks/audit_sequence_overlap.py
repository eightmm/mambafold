#!/usr/bin/env python
"""Audit exact sequence overlap between benchmark and training FASTA files.

This is the first of two leakage gates. Passing it does not establish that a
target is homology-clean; run the MMseqs2 screen described in
``benchmarks/BENCHMARK_POLICY.md`` before making a generalization claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FastaRecord:
    identifier: str
    sequence: str


def read_fasta(path: Path) -> list[FastaRecord]:
    records: list[FastaRecord] = []
    identifier: str | None = None
    chunks: list[str] = []

    def append_record() -> None:
        if identifier is None:
            return
        sequence = "".join(chunks).upper()
        if not sequence:
            raise ValueError(f"empty sequence for {identifier!r} in {path}")
        invalid = sorted(set(sequence) - set("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        if invalid:
            raise ValueError(f"invalid sequence characters for {identifier!r} in {path}: {invalid}")
        records.append(FastaRecord(identifier=identifier, sequence=sequence))

    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            append_record()
            identifier = line[1:].split(maxsplit=1)[0]
            if not identifier:
                raise ValueError(f"empty FASTA identifier at {path}:{line_number}")
            chunks = []
        elif identifier is None:
            raise ValueError(f"sequence before FASTA header at {path}:{line_number}")
        else:
            chunks.append("".join(line.split()))
    append_record()

    identifiers = [record.identifier for record in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"duplicate FASTA identifiers in {path}")
    return records


def sequence_digest(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_exact_overlap(
    targets: list[FastaRecord],
    training_sets: list[tuple[str, list[FastaRecord]]],
) -> dict[str, Any]:
    training_by_sequence: dict[str, list[dict[str, str]]] = {}
    training_records = 0
    for source, records in training_sets:
        training_records += len(records)
        for record in records:
            training_by_sequence.setdefault(record.sequence, []).append(
                {"source": source, "identifier": record.identifier}
            )

    matches = []
    for target in targets:
        training_matches = training_by_sequence.get(target.sequence)
        if training_matches:
            matches.append(
                {
                    "target_id": target.identifier,
                    "sequence_sha256": sequence_digest(target.sequence),
                    "training_matches": training_matches,
                }
            )

    return {
        "target_records": len(targets),
        "training_records": training_records,
        "exact_overlap_targets": len(matches),
        "exact_clean_targets": len(targets) - len(matches),
        "matches": matches,
    }


def write_fasta(path: Path, records: list[FastaRecord]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        for record in records:
            handle.write(f">{record.identifier}\n")
            for start in range(0, len(record.sequence), 80):
                handle.write(record.sequence[start : start + 80] + "\n")


def write_ids(path: Path, records: list[FastaRecord]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        for record in records:
            handle.write(record.identifier + "\n")


def build_report(target_path: Path, training_paths: list[Path]) -> dict[str, Any]:
    targets = read_fasta(target_path)
    training_sets = [(path.name, read_fasta(path)) for path in training_paths]
    return {
        "schema_version": 1,
        "scope": "exact_sequence_only_not_homology_clean",
        "target_fasta": {
            "filename": target_path.name,
            "sha256": file_digest(target_path),
        },
        "training_fastas": [
            {"filename": path.name, "sha256": file_digest(path)} for path in training_paths
        ],
        "result": audit_exact_overlap(targets, training_sets),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument(
        "--training",
        type=Path,
        action="append",
        required=True,
        help="Training-source FASTA; repeat for RCSB, AFDB, and other sources.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--write-exact-clean-fasta",
        type=Path,
        help=(
            "Optionally write targets with exact matches removed. "
            "Still requires homology screening."
        ),
    )
    parser.add_argument(
        "--write-exact-clean-ids",
        type=Path,
        help=(
            "Optionally write exact-clean target IDs for the scorer's --target-ids. "
            "Still requires homology screening."
        ),
    )
    args = parser.parse_args()

    outputs = [args.out]
    if args.write_exact_clean_fasta:
        outputs.append(args.write_exact_clean_fasta)
    if args.write_exact_clean_ids:
        outputs.append(args.write_exact_clean_ids)
    for output in outputs:
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")

    report = build_report(args.targets, args.training)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        handle.write(json.dumps(report, indent=2) + "\n")

    if args.write_exact_clean_fasta or args.write_exact_clean_ids:
        matched_ids = {item["target_id"] for item in report["result"]["matches"]}
        clean = [
            record for record in read_fasta(args.targets) if record.identifier not in matched_ids
        ]
        if args.write_exact_clean_fasta:
            write_fasta(args.write_exact_clean_fasta, clean)
        if args.write_exact_clean_ids:
            write_ids(args.write_exact_clean_ids, clean)

    result = report["result"]
    print(
        f"exact overlaps: {result['exact_overlap_targets']}/{result['target_records']}; "
        f"report: {args.out}"
    )


if __name__ == "__main__":
    main()
