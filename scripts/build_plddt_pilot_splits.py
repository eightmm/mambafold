#!/usr/bin/env python
"""Build deterministic exact-sequence-disjoint pLDDT pilot splits.

The pilot deliberately uses single-chain, pre-cutoff RCSB entries already in
the folding corpus. It measures whether final-sampler latents can rank the
folding model's stochastic errors; it is not a new clean folding benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    name: str | None = None
    parts: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith(">"):
            if name is not None:
                records[name] = "".join(parts)
            name = line[1:].split()[0]
            parts = []
        elif line:
            parts.append(line)
    if name is not None:
        records[name] = "".join(parts)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path, default=Path("data/audit/rcsb-admitted.fasta"))
    parser.add_argument("--tsv", type=Path, default=Path("data/audit/rcsb-admitted.tsv"))
    parser.add_argument(
        "--admitted", type=Path, default=Path("data/splits/admitted_all.txt")
    )
    parser.add_argument(
        "--train-out", type=Path, default=Path("data/splits/plddt_pilot_train.txt")
    )
    parser.add_argument(
        "--val-out", type=Path, default=Path("data/splits/plddt_pilot_val.txt")
    )
    parser.add_argument(
        "--audit-out", type=Path, default=Path("data/audit/plddt_pilot_splits.json")
    )
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-val", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=730019)
    args = parser.parse_args()

    outputs = (args.train_out, args.val_out, args.audit_out)
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing outputs: {existing}")
    if args.n_train < 1 or args.n_val < 1:
        raise ValueError("n_train and n_val must be positive")

    fasta = _read_fasta(args.fasta)
    admitted = {line.strip() for line in args.admitted.read_text().splitlines() if line.strip()}
    rows_by_entry: dict[str, list[tuple[str, int]]] = {}
    with args.tsv.open(encoding="utf-8") as handle:
        header = next(handle).rstrip("\n").split("\t")
        columns = {name: index for index, name in enumerate(header)}
        required = {"pdb_id", "chain", "seq_len"}
        if not required.issubset(columns):
            raise ValueError(f"TSV missing columns: {sorted(required - set(columns))}")
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            entry = fields[columns["pdb_id"]]
            chain = fields[columns["chain"]]
            length = int(fields[columns["seq_len"]])
            rows_by_entry.setdefault(entry, []).append((chain, length))

    # One structure per exact sequence prevents repeated hemoglobin-like entries
    # from dominating this small pilot and makes train/validation sequence-disjoint.
    by_sequence: dict[str, str] = {}
    for entry, rows in rows_by_entry.items():
        filename = f"{entry}.npz"
        if filename not in admitted or len(rows) != 1:
            continue
        chain, length = rows[0]
        if not 20 <= length <= 1024:
            continue
        sequence = fasta.get(f"{entry}_{chain}")
        if sequence is None or len(sequence) != length:
            continue
        incumbent = by_sequence.get(sequence)
        if incumbent is None or filename < incumbent:
            by_sequence[sequence] = filename

    candidates = sorted(by_sequence.values())
    random.Random(args.seed).shuffle(candidates)
    required_count = args.n_train + args.n_val
    if len(candidates) < required_count:
        raise RuntimeError(f"need {required_count} candidates, found {len(candidates)}")
    val = sorted(candidates[: args.n_val])
    train = sorted(candidates[args.n_val : required_count])

    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.train_out.write_text("\n".join(train) + "\n", encoding="utf-8")
    args.val_out.write_text("\n".join(val) + "\n", encoding="utf-8")
    audit = {
        "schema_version": 1,
        "purpose": "pLDDT pilot only; not a clean folding benchmark",
        "population": "single-chain pre-cutoff RCSB, one entry per exact sequence",
        "seed": args.seed,
        "n_candidates": len(candidates),
        "n_train": len(train),
        "n_val": len(val),
        "exact_sequence_overlap": 0,
        "inputs": {
            path.name: _sha256(path) for path in (args.fasta, args.tsv, args.admitted)
        },
        "outputs": {
            args.train_out.name: _sha256(args.train_out),
            args.val_out.name: _sha256(args.val_out),
        },
    }
    args.audit_out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
