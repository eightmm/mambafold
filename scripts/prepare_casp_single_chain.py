#!/usr/bin/env python
"""Build strict single-chain CASP15/16 benchmark inputs from official files.

The inference sequence always comes from the official CASP FASTA.  Experimental
coordinates are aligned onto that sequence only to provide atom topology and a
ground-truth mask for MambaFold's existing inference loader.  Original PDB files
are retained separately for OpenStructure scoring.

CASP15 publishes TS evaluation-domain coordinates, while CASP16 also publishes
whole monomer coordinates.  Consequently, CASP15's primary references are EUs
and CASP16's primary references are whole targets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import re
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from Bio.Align import PairwiseAligner
from Bio.PDB import PDBParser

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mambafold.data.constants import (  # noqa: E402
    AA_3TO1,
    AA_TO_ID,
    BOLTZ_ATOMS_DTYPE,
    BOLTZ_CHAINS_DTYPE,
    BOLTZ_RESIDUES_DTYPE,
    RESIDUE_ATOMS,
)

AA1_TO_3 = {one: three for three, one in AA_3TO1.items()}
CANONICAL_AA = frozenset(AA1_TO_3)
DOMAIN_SUFFIX = re.compile(r"-D\d+$")


@dataclass(frozen=True)
class EditionFiles:
    target_table: str
    sequence_file: str
    whole_archive: str | None
    domain_archive: str


FILES = {
    15: EditionFiles(
        target_table="targets.csv",
        sequence_file="casp15.seq.txt",
        whole_archive=None,
        domain_archive="casp15.targets.TS-domains.public_12.20.2022.tar.gz",
    ),
    16: EditionFiles(
        target_table="targets.csv",
        sequence_file="casp16.T1.seq.txt",
        whole_archive="casp16.targets_monomer.tgz",
        domain_archive="casp16.targets_monomer_trimmed2domains.tgz",
    ),
}


@dataclass
class ParsedPDB:
    payload: bytes
    residues: list
    sequence: str
    ca_fraction: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_target_table(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="cp1252", newline="") as handle:
        rows = csv.DictReader(handle, delimiter=";")
        return {
            row["Target"].strip(): {
                key: value.strip()
                for key, value in row.items()
                if key is not None and isinstance(value, str)
            }
            for row in rows
        }


def read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    current: str | None = None
    chunks: list[str] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current is not None:
                records[current] = "".join(chunks).upper()
            current = line[1:].split()[0]
            chunks = []
        elif current is None:
            raise ValueError(f"Sequence before FASTA header in {path}")
        else:
            chunks.append(line)
    if current is not None:
        records[current] = "".join(chunks).upper()
    return records


def reference_target_id(reference_id: str) -> str:
    return DOMAIN_SUFFIX.sub("", reference_id)


def read_archive(path: Path) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    with tarfile.open(path, "r:gz") as archive:
        members = [
            member
            for member in archive.getmembers()
            if (member.isfile() or member.islnk()) and member.name.endswith(".pdb")
        ]
        for member in members:
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"Cannot read {member.name} from {path}")
            reference_id = Path(member.name).name.removesuffix(".pdb")
            payloads[reference_id] = handle.read()
    return payloads


def parse_single_chain_pdb(payload: bytes, reference_id: str) -> ParsedPDB:
    text = payload.decode("utf-8", errors="replace")
    structure = PDBParser(QUIET=True).get_structure(reference_id, io.StringIO(text))
    model = next(structure.get_models())
    protein_chains: list[list] = []
    for chain in model.get_chains():
        residues = [
            residue
            for residue in chain.get_residues()
            if not residue.get_id()[0].strip() and residue.get_resname().strip() in AA_3TO1
        ]
        if residues:
            protein_chains.append(residues)
    if len(protein_chains) != 1:
        raise ValueError(f"expected one protein chain, found {len(protein_chains)}")
    residues = protein_chains[0]
    sequence = "".join(AA_3TO1[residue.get_resname().strip()] for residue in residues)
    ca_fraction = sum(residue.has_id("CA") for residue in residues) / len(residues)
    return ParsedPDB(payload=payload, residues=residues, sequence=sequence, ca_fraction=ca_fraction)


def align_residues(official_sequence: str, parsed: ParsedPDB) -> tuple[dict[int, object], float]:
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -5.0
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(official_sequence, parsed.sequence)[0]
    mapping: dict[int, object] = {}
    matches = 0
    aligned_pairs = 0
    for official_idx, pdb_idx in alignment.indices.T:
        if official_idx < 0 or pdb_idx < 0:
            continue
        aligned_pairs += 1
        if official_sequence[official_idx] == parsed.sequence[pdb_idx]:
            mapping[int(official_idx)] = parsed.residues[int(pdb_idx)]
            matches += 1
    identity = matches / aligned_pairs if aligned_pairs else 0.0
    return mapping, identity


def build_npz(sequence: str, mappings: list[dict[int, object]]) -> dict[str, np.ndarray]:
    residue_records = []
    atom_records = []
    atom_index = 0
    for residue_index, one_letter in enumerate(sequence):
        residue_name = AA1_TO_3[one_letter]
        canonical_atoms = RESIDUE_ATOMS[residue_name]
        sources = [mapping[residue_index] for mapping in mappings if residue_index in mapping]
        atom_start = atom_index
        any_present = False
        for atom_name in canonical_atoms:
            record = np.zeros(1, dtype=BOLTZ_ATOMS_DTYPE)[0]
            for source in sources:
                if source.has_id(atom_name):
                    record["coords"] = source[atom_name].get_coord().astype(np.float32)
                    record["is_present"] = True
                    any_present = True
                    break
            atom_records.append(record)
            atom_index += 1
        residue_record = np.zeros(1, dtype=BOLTZ_RESIDUES_DTYPE)[0]
        residue_record["name"] = residue_name
        residue_record["res_type"] = AA_TO_ID[residue_name]
        residue_record["res_idx"] = residue_index
        residue_record["atom_idx"] = atom_start
        residue_record["atom_num"] = len(canonical_atoms)
        residue_record["is_standard"] = True
        residue_record["is_present"] = any_present
        residue_records.append(residue_record)

    chain = np.zeros(1, dtype=BOLTZ_CHAINS_DTYPE)
    chain[0]["name"] = "A"
    chain[0]["mol_type"] = 0
    chain[0]["res_idx"] = 0
    chain[0]["res_num"] = len(sequence)
    chain[0]["atom_idx"] = 0
    chain[0]["atom_num"] = len(atom_records)
    return {
        "residues": np.asarray(residue_records, dtype=BOLTZ_RESIDUES_DTYPE),
        "atoms": np.asarray(atom_records, dtype=BOLTZ_ATOMS_DTYPE),
        "chains": chain,
        "bonds": np.zeros(0),
        "connections": np.zeros(0),
        "interfaces": np.zeros(0),
        "mask": np.ones(len(sequence), dtype=bool),
        "coords": np.zeros((len(atom_records), 3), dtype=np.float32),
        "ensemble": np.zeros(0),
    }


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_fasta(path: Path, target_ids: list[str], sequences: dict[str, str]) -> None:
    lines: list[str] = []
    for target_id in target_ids:
        lines.extend(
            (f">{target_id.lower()} official_CASP_target={target_id}", sequences[target_id])
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def prepare(edition: int, raw_dir: Path, out_dir: Path) -> dict[str, int]:
    files = FILES[edition]
    table_path = raw_dir / files.target_table
    sequence_path = raw_dir / files.sequence_file
    domain_path = raw_dir / files.domain_archive
    required = [table_path, sequence_path, domain_path]
    whole_path = raw_dir / files.whole_archive if files.whole_archive else None
    if whole_path is not None:
        required.append(whole_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing official input files: {missing}")

    out_dir.mkdir(parents=True, exist_ok=True)
    for legacy_name in (
        "simplefold_100_750.fasta",
        "simplefold_100_750_ids.txt",
        "length_matched_100_750.fasta",
        "length_matched_100_750_ids.txt",
    ):
        (out_dir / legacy_name).unlink(missing_ok=True)

    metadata = read_target_table(table_path)
    sequences = read_fasta(sequence_path)
    domain_payloads = read_archive(domain_path)
    whole_payloads = read_archive(whole_path) if whole_path else {}
    primary_payloads = whole_payloads if edition == 16 else domain_payloads

    refs_by_target: dict[str, list[tuple[str, bytes]]] = {}
    for reference_id, payload in primary_payloads.items():
        target_id = reference_target_id(reference_id)
        if not target_id.startswith("T1"):
            continue
        refs_by_target.setdefault(target_id, []).append((reference_id, payload))

    target_rows: list[dict[str, object]] = []
    reference_rows: list[dict[str, object]] = []
    included: list[str] = []
    simplefold_protocol: list[str] = []
    primary: list[str] = []

    for target_id in sorted(set(metadata) | set(sequences)):
        row = metadata.get(target_id)
        sequence = sequences.get(target_id)
        reasons: list[str] = []
        if not target_id.startswith("T1"):
            reasons.append("not_standard_T1_target")
        if row is None:
            reasons.append("missing_target_metadata")
        else:
            if row.get("Oligo.State") != "A1":
                reasons.append("not_A1")
            if row.get("Cancellation Date") not in {"", "-"}:
                reasons.append("cancelled")
        if sequence is None:
            reasons.append("missing_official_sequence")
        elif not set(sequence) <= CANONICAL_AA:
            reasons.append("noncanonical_sequence")
        elif not 20 <= len(sequence) <= 1024:
            reasons.append("length_outside_20_1024")
        references = refs_by_target.get(target_id, [])
        if not references:
            reasons.append("missing_primary_reference")

        parsed_refs: list[tuple[str, ParsedPDB, dict[int, object], float]] = []
        if not reasons and sequence is not None:
            for reference_id, payload in sorted(references):
                try:
                    parsed = parse_single_chain_pdb(payload, reference_id)
                    mapping, identity = align_residues(sequence, parsed)
                    if parsed.ca_fraction < 0.90:
                        raise ValueError(f"CA fraction {parsed.ca_fraction:.3f} < 0.90")
                    if identity < 0.95:
                        raise ValueError(f"sequence identity {identity:.3f} < 0.95")
                    parsed_refs.append((reference_id, parsed, mapping, identity))
                except Exception as exc:  # noqa: BLE001
                    reasons.append(f"invalid_reference:{reference_id}:{type(exc).__name__}")
        is_included = not reasons and bool(parsed_refs)
        observed_ca = {
            index
            for _, _parsed, mapping, _identity in parsed_refs
            for index, residue in mapping.items()
            if residue.has_id("CA")
        }
        coordinate_coverage = len(observed_ca) / len(sequence) if sequence else 0.0
        # CASP15 structures are evaluation domains, so completeness is measured
        # inside each EU above. CASP16 whole targets can be measured directly
        # against their complete official sequence.
        is_high_coverage = is_included and (edition == 15 or coordinate_coverage >= 0.90)
        is_simplefold_protocol = (
            is_high_coverage and sequence is not None and 50 <= len(sequence) <= 1000
        )
        target_rows.append(
            {
                "edition": edition,
                "target_id": target_id,
                "length": len(sequence) if sequence else "",
                "oligo_state": row.get("Oligo.State", "") if row else "",
                "primary_reference_kind": "whole" if edition == 16 else "domain_EU",
                "primary_reference_count": len(parsed_refs),
                "coordinate_CA_coverage": (
                    f"{coordinate_coverage:.6f}" if parsed_refs else ""
                ),
                "included_20_1024": int(is_included),
                "simplefold_protocol_50_1000": int(is_simplefold_protocol),
                "high_coverage_0.90": int(is_high_coverage),
                "reason": ";".join(reasons),
            }
        )
        if not is_included or sequence is None:
            continue

        included.append(target_id)
        if is_simplefold_protocol:
            simplefold_protocol.append(target_id)
        if is_high_coverage:
            primary.append(target_id)
        npz = build_npz(sequence, [mapping for _, _, mapping, _ in parsed_refs])
        normalized_id = target_id.lower()
        npz_path = out_dir / "inference_npz" / normalized_id[1:3] / f"{normalized_id}.npz"
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, **npz)

        for reference_id, parsed, mapping, identity in parsed_refs:
            kind = "whole" if edition == 16 else "domains"
            reference_path = out_dir / "references" / kind / f"{reference_id}.pdb"
            reference_path.parent.mkdir(parents=True, exist_ok=True)
            reference_path.write_bytes(parsed.payload)
            reference_rows.append(
                {
                    "edition": edition,
                    "target_id": target_id,
                    "prediction_id": normalized_id,
                    "reference_id": reference_id,
                    "reference_kind": "whole" if edition == 16 else "domain_EU",
                    "pdb_residues": len(parsed.residues),
                    "mapped_residues": len(mapping),
                    "sequence_identity": f"{identity:.6f}",
                    "CA_fraction": f"{parsed.ca_fraction:.6f}",
                    "reference_path": str(reference_path.relative_to(out_dir)),
                }
            )

    # Retain all valid domain/EU references for included CASP16 whole targets.
    if edition == 16:
        for reference_id, payload in sorted(domain_payloads.items()):
            target_id = reference_target_id(reference_id)
            if target_id not in included or not target_id.startswith("T1"):
                continue
            try:
                parsed = parse_single_chain_pdb(payload, reference_id)
                mapping, identity = align_residues(sequences[target_id], parsed)
                if parsed.ca_fraction < 0.90 or identity < 0.95:
                    continue
            except Exception:
                continue
            reference_path = out_dir / "references" / "domains" / f"{reference_id}.pdb"
            reference_path.parent.mkdir(parents=True, exist_ok=True)
            reference_path.write_bytes(payload)
            reference_rows.append(
                {
                    "edition": edition,
                    "target_id": target_id,
                    "prediction_id": target_id.lower(),
                    "reference_id": reference_id,
                    "reference_kind": "domain_EU",
                    "pdb_residues": len(parsed.residues),
                    "mapped_residues": len(mapping),
                    "sequence_identity": f"{identity:.6f}",
                    "CA_fraction": f"{parsed.ca_fraction:.6f}",
                    "reference_path": str(reference_path.relative_to(out_dir)),
                }
            )

    normalized_eligible = [target_id.lower() for target_id in included]
    normalized_ids = [target_id.lower() for target_id in primary]
    normalized_simplefold = [target_id.lower() for target_id in simplefold_protocol]
    (out_dir / "inference_ids.txt").write_text("\n".join(normalized_ids) + "\n")
    (out_dir / "all_eligible_ids.txt").write_text("\n".join(normalized_eligible) + "\n")
    (out_dir / "simplefold_ids.txt").write_text("\n".join(normalized_simplefold) + "\n")
    (out_dir / "high_coverage_0.90_ids.txt").write_text("\n".join(normalized_ids) + "\n")
    write_fasta(out_dir / "sequences.fasta", primary, sequences)
    write_fasta(out_dir / "all_eligible.fasta", included, sequences)
    write_fasta(out_dir / "simplefold_input.fasta", simplefold_protocol, sequences)
    write_fasta(out_dir / "high_coverage_0.90.fasta", primary, sequences)
    write_tsv(
        out_dir / "manifest.tsv",
        target_rows,
        [
            "edition",
            "target_id",
            "length",
            "oligo_state",
            "primary_reference_kind",
            "primary_reference_count",
            "coordinate_CA_coverage",
            "included_20_1024",
            "simplefold_protocol_50_1000",
            "high_coverage_0.90",
            "reason",
        ],
    )
    write_tsv(
        out_dir / "reference_manifest.tsv",
        reference_rows,
        [
            "edition",
            "target_id",
            "prediction_id",
            "reference_id",
            "reference_kind",
            "pdb_residues",
            "mapped_residues",
            "sequence_identity",
            "CA_fraction",
            "reference_path",
        ],
    )
    primary_targets = set(primary)
    write_tsv(
        out_dir / "primary_reference_manifest.tsv",
        [row for row in reference_rows if row["target_id"] in primary_targets],
        [
            "edition",
            "target_id",
            "prediction_id",
            "reference_id",
            "reference_kind",
            "pdb_residues",
            "mapped_residues",
            "sequence_identity",
            "CA_fraction",
            "reference_path",
        ],
    )
    with (out_dir / "sources.sha256").open("w") as handle:
        for path in required:
            handle.write(f"{sha256_file(path)}  {path.name}\n")
    primary_kind = "whole-chain PDB" if edition == 16 else "official domain/EU PDB"
    (out_dir / "DATASET.md").write_text(
        f"# CASP{edition} strict single-chain benchmark\n\n"
        f"- Primary reference: {primary_kind}\n"
        "- Eligibility: official protein target, stoichiometry A1, not cancelled, "
        "canonical sequence, 20-1024 residues, one protein chain in the reference.\n"
        "- Primary set: eligible targets with at least 90% coordinate coverage; "
        "for CASP15 this is measured within each official evaluation domain.\n"
        "- Inference sequence: official CASP FASTA, never reconstructed from coordinates.\n"
        "- SimpleFold comparison: `simplefold_input.fasta`, restricted to the paper's "
        "CASP14 length protocol of 50-1000 residues.\n"
        "- Scoring: use original PDBs under `references/` with OpenStructure 2.9.1.\n"
        "- `all_eligible*` is the lower-coverage sensitivity slice.\n"
    )
    return {
        "included": len(included),
        "primary": len(primary),
        "simplefold_protocol": len(simplefold_protocol),
        "references": len(reference_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edition", type=int, choices=sorted(FILES), required=True)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    raw_dir = args.raw_dir or Path(f"data/casp_official/raw/casp{args.edition}")
    out_dir = args.out_dir or Path(f"data/casp_official/casp{args.edition}_single_chain")
    counts = prepare(args.edition, raw_dir, out_dir)
    print(
        f"CASP{args.edition}: eligible={counts['included']} primary={counts['primary']} "
        f"simplefold_protocol_50_1000={counts['simplefold_protocol']} "
        f"references={counts['references']} "
        f"out={out_dir}"
    )


if __name__ == "__main__":
    main()
