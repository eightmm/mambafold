#!/usr/bin/env python3
"""Score one model on one frozen external dataset with OpenStructure.

CASP15 is distributed as official domain/EU references.  Those references are
scored independently and combined within each target using mapped-residue
weights.  Dataset means are then unweighted target means.  CASP16 uses the
official whole-chain references and CAMEO22 uses the frozen state-1 reference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

METRICS = ("oligo_gdtts", "oligo_gdtha", "tm_score", "lddt", "bb_lddt", "rmsd")
COMPARE_STRUCTURE_ARGS = (
    "--fault-tolerant",
    "--min-pep-length",
    "4",
    "--lddt",
    "--bb-lddt",
    "--rigid-scores",
    "--tm-score",
)
MODEL_NAMES = (
    "mambafold_esmc6b_step170000",
    "simplefold_360m",
    "esmfold_v1",
    "dplm2_bit_650m",
)
DEFAULT_CASP14_REFERENCE_SUBDIR = Path("references/casp14_full70")


@dataclass(frozen=True)
class Reference:
    target_id: str
    pair_id: str
    canonical_id: str
    path: Path
    weight: int
    reference_kind: str


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_target_filter(path: Path) -> tuple[set[str], dict[str, Any]]:
    identifiers = [
        line.split("#", 1)[0].strip().lower()
        for line in path.read_text().splitlines()
        if line.split("#", 1)[0].strip()
    ]
    if not identifiers:
        raise SystemExit(f"Target filter is empty: {path}")
    if len(identifiers) != len(set(identifiers)):
        raise SystemExit(f"Target filter contains duplicate IDs: {path}")
    return set(identifiers), {
        "filename": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "target_ids": sorted(identifiers),
    }


def apply_target_filter(
    references: list[Reference], path: Path, dataset: str
) -> tuple[list[Reference], dict[str, Any]]:
    requested_ids, metadata = read_target_filter(path)
    reference_ids = {reference.target_id.lower() for reference in references}
    unknown_ids = sorted(requested_ids - reference_ids)
    if unknown_ids:
        raise SystemExit(f"Target filter contains IDs absent from {dataset}: {unknown_ids}")
    selected = [
        reference for reference in references if reference.target_id.lower() in requested_ids
    ]
    return selected, metadata


def load_mapping(evaluation_root: Path, dataset_key: str) -> dict[str, dict[str, Any]]:
    manifest = json.loads((evaluation_root / "inputs/manifest.json").read_text())
    rows = [row for row in manifest["mapping"] if row["dataset"] == dataset_key]
    mapping = {str(row["fasta_id"]).lower(): row for row in rows}
    if len(mapping) != len(rows):
        raise SystemExit(f"Duplicate FASTA IDs in mapping for {dataset_key}")
    return mapping


def load_references(
    repo_root: Path,
    evaluation_root: Path,
    dataset: str,
    *,
    casp14_reference_dir: Path | None = None,
) -> list[Reference]:
    if dataset == "casp14":
        mapping = load_mapping(evaluation_root, "casp14_70")
        reference_root = casp14_reference_dir or evaluation_root / DEFAULT_CASP14_REFERENCE_SUBDIR
        if not reference_root.is_dir():
            raise SystemExit(
                "Missing model-independent CASP14 reference directory: "
                f"{reference_root}. Populate it with <target>_gt.pdb files or pass "
                "--casp14-reference-dir."
            )
        references = []
        for fasta_id, mapping_row in sorted(mapping.items()):
            reference_path = reference_root / f"{fasta_id}_gt.pdb"
            if not reference_path.is_file():
                raise SystemExit(f"Missing frozen CASP14 reference PDB: {reference_path}")
            references.append(
                Reference(
                    target_id=fasta_id,
                    pair_id=fasta_id,
                    canonical_id=str(mapping_row["canonical_id"]),
                    path=reference_path,
                    weight=int(mapping_row["length"]),
                    reference_kind="whole",
                )
            )
        return references
    if dataset == "casp15":
        dataset_key = "casp15_single_chain_22"
        dataset_root = repo_root / "data/casp_official/casp15_single_chain"
        rows = read_tsv(dataset_root / "primary_reference_manifest.tsv")
        rows = [row for row in rows if row["reference_kind"] == "domain_EU"]
        fasta_field = "prediction_id"
    elif dataset == "casp16":
        dataset_key = "casp16_single_chain_21"
        dataset_root = repo_root / "data/casp_official/casp16_single_chain"
        rows = read_tsv(dataset_root / "primary_reference_manifest.tsv")
        rows = [row for row in rows if row["reference_kind"] == "whole"]
        fasta_field = "prediction_id"
    elif dataset == "cameo22":
        dataset_key = "cameo22_183"
        dataset_root = repo_root / "data/simplefold_official/testsets/cameo22"
        rows = read_tsv(dataset_root / "reference_manifest.tsv")
        rows = [row for row in rows if row["state"] == "state1"]
        fasta_field = "target_id"
    else:
        raise SystemExit(f"Unsupported dataset: {dataset}")

    mapping = load_mapping(evaluation_root, dataset_key)
    references: list[Reference] = []
    for row in rows:
        fasta_id = row[fasta_field].lower()
        if fasta_id not in mapping:
            raise SystemExit(f"Reference FASTA ID missing from canonical mapping: {fasta_id}")
        target_id = row["target_id"]
        pair_id = row.get("reference_id") or target_id
        weight = int(row.get("mapped_residues") or mapping[fasta_id]["length"])
        reference_path = dataset_root / row["reference_path"]
        if not reference_path.is_file():
            raise SystemExit(f"Missing reference PDB: {reference_path}")
        references.append(
            Reference(
                target_id=target_id,
                pair_id=pair_id,
                canonical_id=str(mapping[fasta_id]["canonical_id"]),
                path=reference_path,
                weight=weight,
                reference_kind=row.get("reference_kind") or row.get("state") or "reference",
            )
        )

    expected_targets = {row["fasta_id"].lower() for row in mapping.values()}
    reference_targets = {
        next(
            fasta_id
            for fasta_id, mapping_row in mapping.items()
            if mapping_row["canonical_id"] == reference.canonical_id
        )
        for reference in references
    }
    if reference_targets != expected_targets:
        missing = sorted(expected_targets - reference_targets)
        extra = sorted(reference_targets - expected_targets)
        raise SystemExit(f"Reference target mismatch: missing={missing}, extra={extra}")
    return references


def prediction_index(evaluation_root: Path, model: str) -> dict[str, Path]:
    seed_dir = evaluation_root / "predictions_per_target" / model / "seed_0"
    if not seed_dir.is_dir():
        raise SystemExit(f"Missing seed-0 prediction directory: {seed_dir}")
    indexed: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(seed_dir.rglob("*.pdb")):
        stem = path.stem
        if model == "simplefold_360m" and stem.endswith("_sampled_0"):
            stem = stem.removesuffix("_sampled_0")
        if not stem.startswith("seq_"):
            continue
        if stem in indexed:
            duplicates[stem].extend((indexed[stem], path))
        else:
            indexed[stem] = path
    if duplicates:
        rendered = {key: sorted({str(path) for path in paths}) for key, paths in duplicates.items()}
        raise SystemExit(f"Duplicate prediction PDBs: {rendered}")
    return indexed


def safe_pair_id(pair_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", pair_id)
    if not safe or safe in {".", ".."}:
        raise SystemExit(f"Unsafe pair ID: {pair_id!r}")
    return safe


def coordinate_lines(path: Path) -> tuple[list[str], list[str]]:
    lines = path.read_text().splitlines(keepends=True)
    atom_lines = [line for line in lines if line.startswith("ATOM  ")]
    if not atom_lines:
        raise SystemExit(f"PDB contains no ATOM records: {path}")
    return lines, atom_lines


def stage_pdb(source: Path, destination: Path) -> dict[str, Any]:
    """Link valid PDBs, or copy while filling otherwise blank chain IDs."""
    lines, atom_lines = coordinate_lines(source)
    atom_chains = {line[21:22] if len(line) > 21 else " " for line in atom_lines}
    named_atom_chains = {chain for chain in atom_chains if chain.strip()}
    coordinate = [
        line
        for line in lines
        if line.startswith("ATOM  ") or line.startswith("HETATM") or line.startswith("TER   ")
    ]
    blank_count = sum(1 for line in coordinate if len(line) <= 21 or not line[21:22].strip())

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if blank_count == 0:
        destination.symlink_to(source.resolve())
        return {"normalized": False, "blank_chain_records": 0, "assigned_chain": None}

    if len(named_atom_chains) > 1:
        raise SystemExit(
            f"Cannot assign blank chain records in multi-chain PDB {source}: "
            f"named chains={sorted(named_atom_chains)}"
        )
    assigned_chain = next(iter(named_atom_chains), "A")
    normalized: list[str] = []
    for line in lines:
        is_coordinate = (
            line.startswith("ATOM  ") or line.startswith("HETATM") or line.startswith("TER   ")
        )
        if is_coordinate and (len(line) <= 21 or not line[21:22].strip()):
            had_newline = line.endswith("\n")
            body = line[:-1] if had_newline else line
            body = body.ljust(22)
            line = body[:21] + assigned_chain + body[22:] + ("\n" if had_newline else "")
        normalized.append(line)
    destination.write_text("".join(normalized))
    return {
        "normalized": True,
        "blank_chain_records": blank_count,
        "assigned_chain": assigned_chain,
    }


def valid_result(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        result = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if result.get("status") != "SUCCESS":
        return None
    if any(not isinstance(result.get(metric), (int, float)) for metric in METRICS):
        return None
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_cached_result(
    result_path: Path,
    identity_path: Path,
    expected_identity: dict[str, Any],
) -> dict[str, Any] | None:
    """Return a score only when it was produced for these exact inputs."""
    result = valid_result(result_path)
    if result is None or not identity_path.is_file():
        return None
    try:
        recorded_identity = json.loads(identity_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return result if recorded_identity == expected_identity else None


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def default_ost_path() -> Path:
    """Resolve OpenStructure from the environment or the current ``PATH``."""
    configured = os.environ.get("OPENSTRUCTURE_OST")
    if configured:
        return Path(configured).expanduser()
    discovered = shutil.which("ost")
    return Path(discovered) if discovered else Path("ost")


def score(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve()
    evaluation_root = args.evaluation_root.resolve()
    out_dir = args.out_dir.resolve()
    casp14_reference_dir = getattr(args, "casp14_reference_dir", None)
    if casp14_reference_dir is not None:
        casp14_reference_dir = casp14_reference_dir.expanduser().resolve()
    references = load_references(
        repo_root,
        evaluation_root,
        args.dataset,
        casp14_reference_dir=casp14_reference_dir,
    )
    target_filter = None
    target_ids_path = getattr(args, "target_ids", None)
    if target_ids_path is not None:
        references, target_filter = apply_target_filter(references, target_ids_path, args.dataset)
    predictions = prediction_index(evaluation_root, args.model)
    ost = args.ost.resolve()
    if not ost.is_file():
        raise SystemExit(f"OpenStructure executable not found: {ost}")
    version = subprocess.run(
        [str(ost), "--version"], check=True, capture_output=True, text=True
    ).stdout.strip()

    expected_target_ids = sorted({reference.target_id for reference in references})
    available_target_ids = sorted(
        {reference.target_id for reference in references if reference.canonical_id in predictions}
    )
    selected = [reference for reference in references if reference.canonical_id in predictions]
    if not selected:
        raise SystemExit(f"No predictions available for {args.model} on {args.dataset}")

    inputs_dir = out_dir / "inputs"
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    chain_normalization_count = 0
    for reference in selected:
        pair_id = safe_pair_id(reference.pair_id)
        prediction_input = inputs_dir / f"{pair_id}_pred.pdb"
        reference_input = inputs_dir / f"{pair_id}_gt.pdb"
        prediction_stage = stage_pdb(predictions[reference.canonical_id], prediction_input)
        reference_stage = stage_pdb(reference.path, reference_input)
        prediction_sha256 = file_sha256(prediction_input)
        reference_sha256 = file_sha256(reference_input)
        chain_normalization_count += int(prediction_stage["normalized"])
        chain_normalization_count += int(reference_stage["normalized"])
        manifest_rows.append(
            {
                "target_id": reference.target_id,
                "pair_id": pair_id,
                "canonical_id": reference.canonical_id,
                "weight": reference.weight,
                "reference_kind": reference.reference_kind,
                "prediction_source": str(predictions[reference.canonical_id].resolve()),
                "reference_source": str(reference.path.resolve()),
                "prediction_input": str(prediction_input),
                "reference_input": str(reference_input),
                "prediction_sha256": prediction_sha256,
                "reference_sha256": reference_sha256,
                "prediction_chain_stage": prediction_stage,
                "reference_chain_stage": reference_stage,
            }
        )

    input_manifest = {
        "schema_version": 2,
        "dataset": args.dataset,
        "model": args.model,
        "target_filter": target_filter,
        "expected_target_count": len(expected_target_ids),
        "available_prediction_target_count": len(available_target_ids),
        "reference_pair_count": len(manifest_rows),
        "chain_normalization_count": chain_normalization_count,
        "expected_target_ids": expected_target_ids,
        "available_target_ids": available_target_ids,
        "rows": manifest_rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "input_manifest.json").write_text(json.dumps(input_manifest, indent=2) + "\n")

    failures: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    for index, row in enumerate(manifest_rows, start=1):
        pair_id = row["pair_id"]
        output = raw_dir / f"{pair_id}.json"
        identity_path = raw_dir / f"{pair_id}.inputs.json"
        expected_identity = {
            "schema_version": 1,
            "dataset": args.dataset,
            "model": args.model,
            "pair_id": pair_id,
            "canonical_id": row["canonical_id"],
            "prediction_sha256": row["prediction_sha256"],
            "reference_sha256": row["reference_sha256"],
            "openstructure_version": version,
            "compare_structure_args": list(COMPARE_STRUCTURE_ARGS),
        }
        result = valid_cached_result(output, identity_path, expected_identity)
        if result is None:
            if output.exists():
                output.unlink()
            if identity_path.exists():
                identity_path.unlink()
            command = [
                str(ost),
                "compare-structures",
                "-m",
                row["prediction_input"],
                "-r",
                row["reference_input"],
                "-o",
                str(output),
                *COMPARE_STRUCTURE_ARGS,
            ]
            print(f"[{index}/{len(manifest_rows)}] SCORE {pair_id}", flush=True)
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            result = valid_result(output)
            if completed.returncode != 0 or result is None:
                failures.append(
                    {
                        "target_id": row["target_id"],
                        "pair_id": pair_id,
                        "returncode": completed.returncode,
                        "stdout": completed.stdout,
                        "stderr": completed.stderr,
                    }
                )
                continue
            identity_path.write_text(json.dumps(expected_identity, indent=2) + "\n")
        else:
            print(f"[{index}/{len(manifest_rows)}] RESUME {pair_id}", flush=True)

        reference_row: dict[str, Any] = {
            "target_id": row["target_id"],
            "pair_id": pair_id,
            "weight": row["weight"],
            "reference_kind": row["reference_kind"],
        }
        for metric in METRICS:
            reference_row[metric] = float(result[metric])
        reference_rows.append(reference_row)

    successful_by_pair = {row["pair_id"]: row for row in reference_rows}
    expected_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest_rows:
        expected_by_target[row["target_id"]].append(row)

    target_rows: list[dict[str, Any]] = []
    incomplete_targets: list[str] = []
    for target_id, target_manifest_rows in sorted(expected_by_target.items()):
        if any(row["pair_id"] not in successful_by_pair for row in target_manifest_rows):
            incomplete_targets.append(target_id)
            continue
        total_weight = sum(int(row["weight"]) for row in target_manifest_rows)
        target_row: dict[str, Any] = {
            "target_id": target_id,
            "reference_count": len(target_manifest_rows),
            "total_weight": total_weight,
        }
        for metric in METRICS:
            target_row[metric] = (
                sum(
                    float(successful_by_pair[row["pair_id"]][metric]) * int(row["weight"])
                    for row in target_manifest_rows
                )
                / total_weight
            )
        target_rows.append(target_row)

    evaluation_complete = not failures and len(target_rows) == len(expected_target_ids)
    summary = {
        "schema_version": 2,
        "generated_at": datetime.now().astimezone().isoformat(),
        "dataset": args.dataset,
        "model": args.model,
        "target_filter": target_filter,
        "aggregation": (
            "mapped-residue-weighted domain/EU mean within target, then unweighted target mean"
            if args.dataset == "casp15"
            else "unweighted target mean"
        ),
        "expected_target_count": len(expected_target_ids),
        "available_prediction_target_count": len(available_target_ids),
        "missing_prediction_target_ids": sorted(
            set(expected_target_ids) - set(available_target_ids)
        ),
        "reference_pair_count": len(manifest_rows),
        "successful_reference_pair_count": len(reference_rows),
        "successful_target_count": len(target_rows),
        "incomplete_target_ids": incomplete_targets,
        "evaluation_complete_for_available_predictions": (
            not failures and len(target_rows) == len(available_target_ids)
        ),
        "evaluation_complete": evaluation_complete,
        "openstructure": {
            "version": version,
            "executable": str(ost),
            "command": (
                "ost compare-structures -m MODEL_FILE -r REFERENCE_FILE -o OUTPUT_FILE "
                "--fault-tolerant --min-pep-length 4 --lddt --bb-lddt "
                "--rigid-scores --tm-score"
            ),
        },
        "chain_normalization_count": chain_normalization_count,
        "metrics": {
            metric: summarize([float(row[metric]) for row in target_rows]) for metric in METRICS
        },
        "reference_rows": reference_rows,
        "target_rows": target_rows,
        "failures": failures,
    }
    (out_dir / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "model": args.model,
                "targets": f"{len(target_rows)}/{len(available_target_ids)}",
                "failures": len(failures),
                "means": {metric: summary["metrics"][metric]["mean"] for metric in METRICS},
            },
            indent=2,
        ),
        flush=True,
    )
    return 0 if evaluation_complete else 1


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=("casp14", "casp15", "casp16", "cameo22"), required=True
    )
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=repo_root / "outputs/eval/external_compare_esmc6b",
    )
    parser.add_argument(
        "--casp14-reference-dir",
        type=Path,
        help=(
            "directory containing <target>_gt.pdb CASP14 references "
            "(default: EVALUATION_ROOT/references/casp14_full70)"
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--target-ids",
        type=Path,
        help=(
            "optional exact/homology-clean target ID file; filtered IDs become "
            "the complete expected evaluation set"
        ),
    )
    parser.add_argument(
        "--ost",
        type=Path,
        default=default_ost_path(),
        help="OpenStructure executable (default: OPENSTRUCTURE_OST or ost from PATH)",
    )
    args = parser.parse_args()
    raise SystemExit(score(args))


if __name__ == "__main__":
    main()
