#!/usr/bin/env python
"""Build full-RCSB, exact-sequence-disjoint chain splits for pLDDT.

Unlike the earlier pilot split, every valid protein chain from an admitted
pre-cutoff RCSB entry is eligible.  Exact duplicate sequences collapse to the
best-resolved occurrence, so homomers and repeatedly deposited proteins do not
overweight confidence training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from mambafold.data.constants import AA_3TO1, RESIDUE_ATOMS
from mambafold.data.dataset import RCSBDataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _chain_sequences_and_resolution(path: Path) -> dict[int, tuple[str, int]]:
    """Return canonical full sequence and resolved canonical atoms by origin."""
    result: dict[int, tuple[str, int]] = {}
    with np.load(path) as data:
        residues, atoms = data["residues"], data["atoms"]
        origin = -1
        for chain in data["chains"]:
            if int(chain["mol_type"]) != RCSBDataset.MOL_TYPE_PROTEIN:
                continue
            origin += 1
            start = int(chain["res_idx"])
            end = start + int(chain["res_num"])
            sequence: list[str] = []
            resolved = 0
            for residue in residues[start:end]:
                name = str(residue["name"])
                if not bool(residue["is_standard"]) or name not in AA_3TO1:
                    continue
                sequence.append(AA_3TO1[name])
                usable = min(int(residue["atom_num"]), len(RESIDUE_ATOMS.get(name, ())))
                atom_start = int(residue["atom_idx"])
                resolved += int(
                    np.count_nonzero(atoms["is_present"][atom_start : atom_start + usable])
                )
            result[origin] = ("".join(sequence), resolved)
    return result


def _write_split(path: Path, rows: list[tuple[str, int, int]]) -> None:
    path.write_text(
        "".join(f"{filename}\t{origin}\t{length}\n" for filename, origin, length in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/rcsb_train"))
    parser.add_argument("--esm-dir", type=Path, default=Path("data/rcsb_esmc6b"))
    parser.add_argument("--admitted", type=Path, default=Path("data/splits/admitted_all.txt"))
    parser.add_argument(
        "--train-out", type=Path, default=Path("data/splits/plddt_rcsb_chain_train.tsv")
    )
    parser.add_argument(
        "--val-out", type=Path, default=Path("data/splits/plddt_rcsb_chain_val.tsv")
    )
    parser.add_argument(
        "--audit-out", type=Path, default=Path("data/audit/plddt_rcsb_chain_splits.json")
    )
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--min-length", type=int, default=20)
    parser.add_argument("--min-obs-ratio", type=float, default=0.0)
    parser.add_argument("--n-val", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=730019)
    parser.add_argument("--chain-index-workers", type=int, default=8)
    args = parser.parse_args()

    outputs = (args.train_out, args.val_out, args.audit_out)
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing outputs: {existing}")
    if args.n_val < 1:
        raise ValueError("n_val must be positive")

    dataset = RCSBDataset(
        str(args.data_dir),
        max_length=args.max_length,
        min_length=args.min_length,
        min_obs_ratio=args.min_obs_ratio,
        file_list=str(args.admitted),
        esm_dir=str(args.esm_dir),
        extract_monomer_chains=True,
        dedup_homomer_chains=False,
        chain_index_workers=args.chain_index_workers,
    )
    assert dataset.chain_index is not None
    targets_by_file: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for file_index, origin, length in dataset.chain_index:
        targets_by_file[file_index].append((origin, length))

    # sequence -> (resolved canonical atoms, filename, origin, effective length)
    by_sequence: dict[str, tuple[int, str, int, int]] = {}
    missing_origins = 0
    for position, file_index in enumerate(sorted(targets_by_file), start=1):
        path = Path(dataset.files[file_index])
        chains = _chain_sequences_and_resolution(path)
        try:
            filename = path.relative_to(args.data_dir).as_posix()
        except ValueError:
            filename = path.name
        for origin, length in targets_by_file[file_index]:
            chain = chains.get(origin)
            if chain is None:
                missing_origins += 1
                continue
            sequence, resolved = chain
            if len(sequence) < args.min_length:
                missing_origins += 1
                continue
            candidate = (resolved, filename, origin, length)
            incumbent = by_sequence.get(sequence)
            if incumbent is None or resolved > incumbent[0] or (
                resolved == incumbent[0] and (filename, origin) < (incumbent[1], incumbent[2])
            ):
                by_sequence[sequence] = candidate
        if position % 20000 == 0:
            print(
                f"[split] files={position}/{len(targets_by_file)} "
                f"unique_sequences={len(by_sequence)}",
                flush=True,
            )

    candidates = [
        (filename, origin, length) for _, filename, origin, length in by_sequence.values()
    ]
    candidates.sort()
    random.Random(args.seed).shuffle(candidates)
    if len(candidates) <= args.n_val:
        raise RuntimeError(f"need more than {args.n_val} candidates, found {len(candidates)}")
    val = sorted(candidates[: args.n_val])
    train = sorted(candidates[args.n_val :])

    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    _write_split(args.train_out, train)
    _write_split(args.val_out, val)
    audit = {
        "schema_version": 1,
        "purpose": "standalone pLDDT head training; not a clean folding benchmark",
        "population": (
            "valid monomer chains extracted from pre-cutoff admitted RCSB; "
            "one best-resolved occurrence per exact full sequence"
        ),
        "seed": args.seed,
        "max_length": args.max_length,
        "min_length": args.min_length,
        "min_obs_ratio": args.min_obs_ratio,
        "n_indexed_chains": len(dataset.chain_index),
        "n_unique_sequences": len(candidates),
        "n_train": len(train),
        "n_val": len(val),
        "missing_index_origins": missing_origins,
        "exact_sequence_overlap": 0,
        "inputs": {args.admitted.name: _sha256(args.admitted)},
        "outputs": {
            args.train_out.name: _sha256(args.train_out),
            args.val_out.name: _sha256(args.val_out),
        },
    }
    args.audit_out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
