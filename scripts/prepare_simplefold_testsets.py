#!/usr/bin/env python
"""Prepare the exact public SimpleFold CAMEO22, Apo, and CoDNaS test sets.

The SimpleFold archives define the exact inference inputs. Reference structures
and two-state metadata come from the EigenFold repository used by SimpleFold's
evaluation implementation. CAMEO22's revised 8QCW-A target is sourced directly
from RCSB because the upstream EigenFold checkout contains the older 8AHP-A.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from Bio.PDB import PDBIO, MMCIFParser, Select

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from mambafold.data.constants import AA_3TO1  # noqa: E402
from scripts.prepare_casp_single_chain import (  # noqa: E402
    align_residues,
    build_npz,
    parse_single_chain_pdb,
    sha256_file,
    write_fasta,
    write_tsv,
)

MODELS = (
    "simplefold_100M",
    "simplefold_360M",
    "simplefold_700M",
    "simplefold_1.1B",
    "simplefold_1.6B",
    "simplefold_3B",
)
PRIMARY_REFERENCE_MIN_IDENTITY = 0.95
SAMPLE_PATTERN = re.compile(r"(?P<target>.+)_sampled_(?P<sample>\d+)\.cif$")


@dataclass(frozen=True)
class Task:
    name: str
    split_file: str
    expected_targets: int
    expected_samples: int
    pair_column: str | None


TASKS = {
    "cameo22": Task("cameo22", "cameo2022.csv", 183, 1, None),
    "apo": Task("apo", "apo.csv", 90, 5, "holo"),
    "codnas": Task("codnas", "codnas.csv", 77, 5, "other"),
}


def archive_index(zip_path: Path, task: Task) -> dict[str, dict[str, dict[int, str]]]:
    """Return model -> target -> sample -> archive member."""
    index: dict[str, dict[str, dict[int, str]]] = {model: {} for model in MODELS}
    prefix = f"{task.name}_predictions/"
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            if not member.startswith(prefix) or member.startswith("__MACOSX/"):
                continue
            parts = Path(member).parts
            if len(parts) != 3 or parts[1] not in index:
                continue
            match = SAMPLE_PATTERN.fullmatch(parts[2])
            if match is None:
                continue
            target_id = match.group("target")
            sample = int(match.group("sample"))
            index[parts[1]].setdefault(target_id, {})[sample] = member

    expected_samples = set(range(task.expected_samples))
    target_sets = []
    for model, targets in index.items():
        if len(targets) != task.expected_targets:
            raise ValueError(
                f"{task.name}/{model}: expected {task.expected_targets} targets, "
                f"found {len(targets)}"
            )
        bad = {
            target: sorted(set(samples) ^ expected_samples)
            for target, samples in targets.items()
            if set(samples) != expected_samples
        }
        if bad:
            raise ValueError(f"{task.name}/{model}: sample mismatch {bad}")
        target_sets.append(set(targets))
    if any(targets != target_sets[0] for targets in target_sets[1:]):
        raise ValueError(f"{task.name}: target IDs differ across model sizes")
    return index


def prediction_sequence(payload: bytes, target_id: str) -> str:
    structure = MMCIFParser(QUIET=True).get_structure(
        target_id, io.StringIO(payload.decode("utf-8", errors="replace"))
    )
    model = next(structure.get_models())
    chains = []
    for chain in model.get_chains():
        residues = [
            residue
            for residue in chain.get_residues()
            if not residue.get_id()[0].strip()
            and residue.get_resname().strip() in AA_3TO1
        ]
        if residues:
            chains.append(residues)
    if len(chains) != 1:
        raise ValueError(f"{target_id}: expected one prediction chain, found {len(chains)}")
    return "".join(AA_3TO1[residue.get_resname().strip()] for residue in chains[0])


def split_target_id(name: str) -> str:
    return Path(name).stem.replace(".", "_").lower()


def eigenfold_structure_path(eigenfold: Path, name: str) -> Path:
    stem = Path(name).stem
    pdb_id = stem.split(".")[0].lower()
    return eigenfold / "structures" / pdb_id[:2] / name


class _ChainSelect(Select):
    def __init__(self, chain_id: str):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return int(chain.id == self.chain_id)

    def accept_residue(self, residue):
        return int(
            not residue.get_id()[0].strip()
            and residue.get_resname().strip() in AA_3TO1
        )


def rcsb_chain_pdb(cif_path: Path, chain_id: str) -> bytes:
    structure = MMCIFParser(QUIET=True).get_structure("rcsb", cif_path)
    output = io.StringIO()
    writer = PDBIO()
    writer.set_structure(structure)
    writer.save(output, select=_ChainSelect(chain_id))
    payload = output.getvalue().encode()
    if not payload.startswith(b"ATOM"):
        raise ValueError(f"No chain {chain_id} protein atoms in {cif_path}")
    return payload


def read_split(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        return {split_target_id(row["name"]): row for row in csv.DictReader(handle)}


def extract_predictions(
    zip_path: Path,
    index: dict[str, dict[str, dict[int, str]]],
    out_dir: Path,
) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        for model, targets in index.items():
            model_dir = out_dir / "simplefold_predictions" / model
            model_dir.mkdir(parents=True, exist_ok=True)
            for target_id, samples in sorted(targets.items()):
                for sample, member in sorted(samples.items()):
                    path = model_dir / f"{target_id}_sampled_{sample}.cif"
                    path.write_bytes(archive.read(member))


def prepare_task(
    task: Task,
    raw_dir: Path,
    eigenfold: Path,
    out_dir: Path,
    extract_official_predictions: bool,
) -> dict[str, int]:
    zip_path = raw_dir / f"{task.name}_predictions.zip"
    split_path = eigenfold / "splits" / task.split_file
    if not zip_path.is_file() or not split_path.is_file():
        raise FileNotFoundError(f"Missing {zip_path} or {split_path}")

    index = archive_index(zip_path, task)
    target_ids = sorted(index["simplefold_3B"])
    split = read_split(split_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        sequences = {
            target_id.lower(): prediction_sequence(
                archive.read(index["simplefold_3B"][target_id][0]), target_id
            )
            for target_id in target_ids
        }

    manifest: list[dict[str, object]] = []
    reference_manifest: list[dict[str, object]] = []
    final_ids: list[str] = []
    fasta_sequences: dict[str, str] = {}

    for archive_target_id in target_ids:
        target_id = archive_target_id.lower()
        sequence = sequences[target_id]
        row = split.get(target_id)
        reference_payloads: list[tuple[str, str, bytes]] = []
        metadata_source = task.split_file

        if row is None and task.name == "cameo22" and target_id == "8qcw_a":
            cif_path = raw_dir / "cameo22" / "8qcw.cif"
            reference_payloads.append(("state1", "8qcw.A.pdb", rcsb_chain_pdb(cif_path, "A")))
            metadata_source = "RCSB 8QCW-A (revised SimpleFold target)"
        elif row is None:
            raise ValueError(f"{task.name}: no metadata for {target_id}")
        else:
            first_name = row["name"]
            first_path = eigenfold_structure_path(eigenfold, first_name)
            if not first_path.is_file():
                raise FileNotFoundError(first_path)
            reference_payloads.append(("state1", first_name, first_path.read_bytes()))
            if task.pair_column is not None:
                second_name = row[task.pair_column]
                second_path = eigenfold_structure_path(eigenfold, second_name)
                if not second_path.is_file():
                    raise FileNotFoundError(second_path)
                reference_payloads.append(("state2", second_name, second_path.read_bytes()))

        mappings = []
        coverages = []
        identities = []
        for state, reference_name, payload in reference_payloads:
            parsed = parse_single_chain_pdb(payload, reference_name)
            mapping, identity = align_residues(sequence, parsed)
            exact_sequence_match = identity >= PRIMARY_REFERENCE_MIN_IDENTITY
            # The primary structure defines the inference target and must match its
            # sequence. CoDNaS deliberately includes some homologous alternate-state
            # pairs, so applying this threshold to state2 would alter the benchmark.
            if state == "state1" and not exact_sequence_match:
                raise ValueError(
                    f"{task.name}/{target_id}/{reference_name}: identity={identity:.3f}"
                )
            ca_indices = {
                index for index, residue in mapping.items() if residue.has_id("CA")
            }
            coverage = len(ca_indices) / len(sequence)
            mappings.append(mapping)
            coverages.append(coverage)
            identities.append(identity)

            reference_dir = out_dir / "references" / target_id
            reference_dir.mkdir(parents=True, exist_ok=True)
            reference_path = reference_dir / f"{state}_{reference_name}"
            reference_path.write_bytes(payload)
            reference_manifest.append(
                {
                    "target_id": target_id,
                    "state": state,
                    "reference_name": reference_name,
                    "sequence_identity": f"{identity:.6f}",
                    "exact_sequence_match_0.95": int(exact_sequence_match),
                    "CA_coverage": f"{coverage:.6f}",
                    "sha256": sha256_file(reference_path),
                    "reference_path": str(reference_path.relative_to(out_dir)),
                }
            )

        arrays = build_npz(sequence, [mappings[0]])
        npz_path = out_dir / "inference_npz" / target_id[1:3] / f"{target_id}.npz"
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, **arrays)
        final_ids.append(target_id)
        fasta_sequences[target_id] = sequence
        manifest.append(
            {
                "task": task.name,
                "target_id": target_id,
                "length": len(sequence),
                "states": len(reference_payloads),
                "samples_per_model": task.expected_samples,
                "simplefold_models": len(MODELS),
                "min_reference_identity": f"{min(identities):.6f}",
                "min_CA_coverage": f"{min(coverages):.6f}",
                "metadata_source": metadata_source,
            }
        )

    (out_dir / "inference_ids.txt").write_text("\n".join(final_ids) + "\n")
    write_fasta(out_dir / "sequences.fasta", final_ids, fasta_sequences)
    write_fasta(out_dir / "simplefold_input.fasta", final_ids, fasta_sequences)
    write_tsv(
        out_dir / "manifest.tsv",
        manifest,
        [
            "task",
            "target_id",
            "length",
            "states",
            "samples_per_model",
            "simplefold_models",
            "min_reference_identity",
            "min_CA_coverage",
            "metadata_source",
        ],
    )
    write_tsv(
        out_dir / "reference_manifest.tsv",
        reference_manifest,
        [
            "target_id",
            "state",
            "reference_name",
            "sequence_identity",
            "exact_sequence_match_0.95",
            "CA_coverage",
            "sha256",
            "reference_path",
        ],
    )
    shutil.copy2(split_path, out_dir / f"eigenfold_{task.split_file}")
    if extract_official_predictions:
        extract_predictions(zip_path, index, out_dir)

    source_lines = [f"{sha256_file(zip_path)}  {zip_path.name}"]
    source_lines.append(f"{sha256_file(split_path)}  EigenFold/{task.split_file}")
    if task.name == "cameo22":
        revised = raw_dir / "cameo22" / "8qcw.cif"
        source_lines.append(f"{sha256_file(revised)}  RCSB/8qcw.cif")
    (out_dir / "sources.sha256").write_text("\n".join(source_lines) + "\n")

    contract = (
        "one structure per target; OpenStructure 2.9.1 folding metrics"
        if task.name == "cameo22"
        else "five samples per target; maximum TM-score to each state and ensemble metrics"
    )
    reference_note = (
        "the EigenFold checkout used by SimpleFold's released evaluator; "
        "8QCW-A is the revised RCSB reference"
        if task.name == "cameo22"
        else "the exact official EigenFold metadata pairs used by SimpleFold's "
        "released evaluator"
    )
    (out_dir / "DATASET.md").write_text(
        f"# SimpleFold {task.name} benchmark\n\n"
        f"- Targets: {len(final_ids)}\n"
        f"- Length: {min(map(len, fasta_sequences.values()))}-"
        f"{max(map(len, fasta_sequences.values()))}\n"
        f"- Contract: {contract}.\n"
        "- Input sequence: reconstructed from the official SimpleFold-3B prediction "
        "artifact and cross-checked against EigenFold metadata where available.\n"
        f"- References: {reference_note}.\n"
        "- Reference identity: state1 must have at least 95% sequence identity. "
        "Alternate states retain the exact official pairing; some CoDNaS state2 "
        "structures are homologous rather than sequence-identical.\n"
        "- Model outputs are evaluated separately by model size; they are never pooled.\n"
    )
    return {
        "targets": len(final_ids),
        "references": len(reference_manifest),
        "predictions": len(final_ids) * task.expected_samples * len(MODELS),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/simplefold_official/raw"))
    parser.add_argument("--eigenfold-dir", type=Path, required=True)
    parser.add_argument(
        "--out-root", type=Path, default=Path("data/simplefold_official/testsets")
    )
    parser.add_argument(
        "--task", choices=[*TASKS, "all"], default="all"
    )
    parser.add_argument(
        "--extract-official-predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    selected = TASKS.values() if args.task == "all" else (TASKS[args.task],)
    for task in selected:
        counts = prepare_task(
            task,
            args.raw_dir,
            args.eigenfold_dir,
            args.out_root / task.name,
            args.extract_official_predictions,
        )
        print(
            f"{task.name}: targets={counts['targets']} references={counts['references']} "
            f"official_predictions={counts['predictions']}"
        )


if __name__ == "__main__":
    main()
