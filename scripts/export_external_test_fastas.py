#!/usr/bin/env python
"""Export the repository's fixed external benchmark inputs as public FASTA."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mambafold.data.constants import AA_3TO1  # noqa: E402

STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class FastaSet:
    name: str
    output_name: str
    source: Path
    expected_count: int
    kind: str


COPY_SETS = (
    FastaSet(
        "CASP15 strict single-chain",
        "casp15_single_chain_22.fasta",
        Path("casp_official/casp15_single_chain/sequences.fasta"),
        22,
        "folding",
    ),
    FastaSet(
        "CASP16 strict single-chain",
        "casp16_single_chain_21.fasta",
        Path("casp_official/casp16_single_chain/sequences.fasta"),
        21,
        "folding",
    ),
    FastaSet(
        "SimpleFold CAMEO22",
        "cameo22_183.fasta",
        Path("simplefold_official/testsets/cameo22/sequences.fasta"),
        183,
        "folding",
    ),
    FastaSet(
        "SimpleFold Apo",
        "apo_90.fasta",
        Path("simplefold_official/testsets/apo/sequences.fasta"),
        90,
        "two-state",
    ),
    FastaSet(
        "SimpleFold CoDNaS",
        "codnas_77.fasta",
        Path("simplefold_official/testsets/codnas/sequences.fasta"),
        77,
        "two-state",
    ),
)


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    identifier: str | None = None
    chunks: list[str] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if identifier is not None:
                records.append((identifier, "".join(chunks).upper()))
            identifier = line[1:].split()[0]
            chunks = []
        elif identifier is None:
            raise ValueError(f"sequence before FASTA header: {path}")
        else:
            chunks.append(line)
    if identifier is not None:
        records.append((identifier, "".join(chunks).upper()))
    return records


def write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    lines = []
    for identifier, sequence in records:
        lines.append(f">{identifier}\n")
        lines.extend(f"{sequence[i:i + 80]}\n" for i in range(0, len(sequence), 80))
    path.write_text("".join(lines))


def validate_records(
    records: list[tuple[str, str]], expected_count: int, name: str
) -> None:
    identifiers = [identifier for identifier, _ in records]
    if len(records) != expected_count or len(set(identifiers)) != expected_count:
        raise ValueError(
            f"{name}: expected {expected_count} unique records, found "
            f"{len(records)} records and {len(set(identifiers))} unique IDs"
        )
    for identifier, sequence in records:
        invalid = set(sequence) - STANDARD_AA
        if invalid or not 10 <= len(sequence) <= 1024:
            raise ValueError(
                f"{name}/{identifier}: length={len(sequence)} invalid={sorted(invalid)}"
            )


def casp14_records(data_root: Path) -> list[tuple[str, str]]:
    ids_path = data_root / "casp_official/casp14_70_whole_ids_exact.txt"
    npz_root = data_root / "casp_official/npz_70"
    records = []
    for identifier in ids_path.read_text().splitlines():
        target_id = identifier.strip().lower()
        if not target_id:
            continue
        path = npz_root / target_id[1:3] / f"{target_id}.npz"
        data = np.load(path)
        protein_chains = [chain for chain in data["chains"] if int(chain["mol_type"]) == 0]
        if len(protein_chains) != 1:
            raise ValueError(f"{target_id}: expected one protein chain")
        chain = protein_chains[0]
        start = int(chain["res_idx"])
        end = start + int(chain["res_num"])
        sequence = "".join(
            AA_3TO1[str(residue["name"])]
            for residue in data["residues"][start:end]
            if bool(residue["is_standard"]) and str(residue["name"]) in AA_3TO1
        )
        records.append((target_id, sequence))
    return records


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO / "data")
    parser.add_argument(
        "--out", type=Path, default=REPO / "benchmarks/external_testsets"
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = []
    records = casp14_records(args.data_root)
    validate_records(records, 70, "CASP14 whole")
    output = args.out / "casp14_70.fasta"
    write_fasta(output, records)
    manifest.append(
        {
            "name": "CASP14 whole",
            "kind": "folding",
            "records": 70,
            "fasta": output.name,
            "sha256": sha256(output),
        }
    )

    for item in COPY_SETS:
        source = args.data_root / item.source
        records = read_fasta(source)
        validate_records(records, item.expected_count, item.name)
        output = args.out / item.output_name
        write_fasta(output, records)
        manifest.append(
            {
                "name": item.name,
                "kind": item.kind,
                "records": item.expected_count,
                "fasta": output.name,
                "sha256": sha256(output),
            }
        )

    (args.out / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "sets": manifest}, indent=2) + "\n"
    )
    print(f"exported {sum(item['records'] for item in manifest)} FASTA records")


if __name__ == "__main__":
    main()
