#!/usr/bin/env python3
"""Measure named-atom chirality and coarse torsion states in benchmark PDBs."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

CHI1_GAMMA = {"SER": "OG", "THR": "OG1", "CYS": "SG", "VAL": "CG1", "ILE": "CG1"}
STANDARD = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}


def read_pdb(path: Path) -> list[tuple[tuple[str, int, str], str, dict[str, np.ndarray]]]:
    residues: dict[tuple[str, int, str], tuple[str, dict[str, np.ndarray]]] = {}
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM") or line[16] not in (" ", "A"):
            continue
        residue = line[17:20].strip()
        if residue not in STANDARD:
            continue
        key = (line[21], int(line[22:26]), line[26])
        if key not in residues:
            residues[key] = (residue, {})
        elif residues[key][0] != residue:
            raise ValueError(f"conflicting residue identity in {path}: {key}")
        atom = line[12:16].strip()
        coords = np.array([float(line[i : i + 8]) for i in (30, 38, 46)])
        if not np.isfinite(coords).all():
            raise ValueError(f"non-finite atom coordinates in {path}: {key}/{atom}")
        residues[key][1].setdefault(atom, coords)
    if not residues:
        raise ValueError(f"no standard protein atoms in {path}")
    return [(key, name, atoms) for key, (name, atoms) in residues.items()]


def signed_volume(atoms: dict[str, np.ndarray], names: tuple[str, str, str, str]) -> float:
    a, b, c, d = (atoms[name] for name in names)
    return float(np.dot(np.cross(a - b, c - b), d - b))


def dihedral(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    axis = c - b
    axis_length = np.linalg.norm(axis)
    if axis_length < 1e-8:
        raise ValueError("undefined dihedral with zero-length center bond")
    axis /= axis_length
    left = a - b
    right = d - c
    left -= np.dot(left, axis) * axis
    right -= np.dot(right, axis) * axis
    if np.linalg.norm(left) < 1e-8 or np.linalg.norm(right) < 1e-8:
        raise ValueError("undefined dihedral with collinear atoms")
    return math.degrees(math.atan2(np.dot(np.cross(axis, left), right), np.dot(left, right)))


def angular_distance(angle: float, center: float) -> float:
    return abs((angle - center + 180.0) % 360.0 - 180.0)


def chi1_angles(path: Path) -> dict[tuple[str, int, str], tuple[str, float]]:
    result = {}
    for key, name, atoms in read_pdb(path):
        if name in ("GLY", "ALA", "PRO"):
            continue
        gamma = CHI1_GAMMA.get(name, "CG")
        names = ("N", "CA", "CB", gamma)
        if all(atom in atoms for atom in names):
            result[key] = (name, dihedral(*(atoms[atom].copy() for atom in names)))
    return result


def diagnose(path: Path) -> dict[str, int]:
    residues = read_pdb(path)
    counts: Counter[str] = Counter()
    for _, name, atoms in residues:
        centers = []
        if name != "GLY":
            centers.append(("ca", ("N", "CA", "C", "CB")))
        if name in ("ILE", "THR"):
            centers.append(("cb", ("CA", "CB", "CG1" if name == "ILE" else "OG1", "CG2")))
        for label, names in centers:
            if not all(atom in atoms for atom in names):
                continue
            volume = signed_volume(atoms, names)
            counts[f"{label}_checked"] += 1
            counts[f"{label}_inverted"] += volume < 0.0
            counts[f"{label}_near_planar"] += abs(volume) < 0.5
        if name not in ("GLY", "ALA", "PRO"):
            gamma = CHI1_GAMMA.get(name, "CG")
            names = ("N", "CA", "CB", gamma)
            if all(atom in atoms for atom in names):
                angle = dihedral(*(atoms[atom].copy() for atom in names))
                deviation = min(angular_distance(angle, center) for center in (-60, 60, 180))
                counts["chi1_checked"] += 1
                counts["chi1_far_from_three_states"] += deviation > 40.0
    for (key_a, _, a), (key_b, _, b) in zip(residues, residues[1:]):
        if key_a[0] != key_b[0] or key_b[1] != key_a[1] + 1:
            continue
        if not all(atom in a for atom in ("CA", "C")) or not all(atom in b for atom in ("N", "CA")):
            continue
        if np.linalg.norm(a["C"] - b["N"]) > 1.8:
            continue
        angle = dihedral(a["CA"].copy(), a["C"].copy(), b["N"].copy(), b["CA"].copy())
        counts["omega_checked"] += 1
        counts["omega_cis"] += angular_distance(angle, 0) <= 30.0
        counts["omega_far_from_cis_trans"] += (
            min(angular_distance(angle, 0), angular_distance(angle, 180)) > 30.0
        )
    return dict(counts)


def summarize(rows: dict[str, dict[str, int]]) -> dict:
    labels = ("ca", "cb", "chi1", "omega")
    bad = ("inverted", "inverted", "far_from_three_states", "far_from_cis_trans")
    result = {"target_count": len(rows), "targets": rows}
    for label, problem in zip(labels, bad):
        checked = f"{label}_checked"
        flagged = f"{label}_{problem}"
        count = sum(row.get(checked, 0) for row in rows.values())
        flags = sum(row.get(flagged, 0) for row in rows.values())
        result[flagged] = {
            "count": flags,
            "checked": count,
            "fraction": flags / count if count else None,
        }
    for label in ("ca", "cb"):
        checked = sum(row.get(f"{label}_checked", 0) for row in rows.values())
        count = sum(row.get(f"{label}_near_planar", 0) for row in rows.values())
        result[f"{label}_near_planar"] = {"count": count, "checked": checked}
    return result


def reference_chi1_accuracy(baseline_root: Path, guided_root: Path, cohort: str) -> dict:
    base = baseline_root / cohort
    manifest = json.loads((base / "pairs-admitted/manifest.json").read_text())
    targets: dict[str, dict[str, float]] = {}
    cached: dict[tuple[str, str], dict[tuple[str, int, str], tuple[str, float]]] = {}
    for row in manifest["rows"]:
        target = row["target_id"].lower()
        pair = row["pair_id"]
        reference = chi1_angles(base / "pairs-admitted" / f"{pair}_gt.pdb")
        target_row = targets.setdefault(
            target, {"compared": 0, "baseline_within_30": 0, "guided_within_30": 0}
        )
        for arm, root in (("baseline", baseline_root), ("guided", guided_root)):
            key = (arm, target)
            if key not in cached:
                cached[key] = chi1_angles(root / cohort / "rollout/structures" / f"{target}.pdb")
        matched = [
            (residue_key, angle)
            for residue_key, (name, angle) in reference.items()
            if all(
                residue_key in cached[(arm, target)]
                and cached[(arm, target)][residue_key][0] == name
                for arm in ("baseline", "guided")
            )
        ]
        target_row["compared"] += len(matched)
        for arm in ("baseline", "guided"):
            prediction = cached[(arm, target)]
            target_row[f"{arm}_within_30"] += sum(
                angular_distance(prediction[residue_key][1], angle) <= 30.0
                for residue_key, angle in matched
            )
    for target, row in targets.items():
        if row["compared"] == 0:
            raise ValueError(f"no reference χ1 comparisons for {cohort}/{target}")
        for arm in ("baseline", "guided"):
            row[f"{arm}_fraction_within_30"] = row[f"{arm}_within_30"] / row["compared"]
    return {
        "target_count": len(targets),
        "reference_pair_count": len(manifest["rows"]),
        "targets": targets,
        "mean_fraction_within_30": {
            arm: sum(row[f"{arm}_fraction_within_30"] for row in targets.values()) / len(targets)
            for arm in ("baseline", "guided")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--guided-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = {
        "schema_version": 1,
        "definitions": {
            "chirality": (
                "Named-atom signed volume < 0, using N-CA-C-CB and "
                "CA-CB-(CG1 or OG1)-CG2; near-planar if |volume| < 0.5 A^3."
            ),
            "chi1": (
                "N-CA-CB-gamma dihedral > 40 degrees from each of -60, +60, "
                "and 180; excludes Pro, Gly, Ala. Coarse three-state diagnostic, "
                "not a rotamer-library outlier call."
            ),
            "omega": (
                "CA-C-N-CA dihedral > 30 degrees from both cis and trans; "
                "consecutive residues with C-N <= 1.8 A only."
            ),
        },
        "cohorts": {},
    }
    for cohort in ("casp15", "casp16"):
        arms = {}
        for arm, root in (("guided", args.guided_root), ("baseline", args.baseline_root)):
            structures = root / cohort / "rollout" / "structures"
            rows = {p.stem.lower(): diagnose(p) for p in structures.glob("*.pdb")}
            if arm == "baseline":
                selected = set(arms["guided"]["targets"])
                rows = {target: row for target, row in rows.items() if target in selected}
            arms[arm] = summarize(rows)
        if set(arms["baseline"]["targets"]) != set(arms["guided"]["targets"]):
            raise ValueError(f"unpaired target sets for {cohort}")
        expected = 19 if cohort == "casp15" else 18
        if arms["baseline"]["target_count"] != expected:
            raise ValueError(f"wrong admitted target count for {cohort}")
        reference_dir = args.baseline_root / cohort / "pairs-admitted"
        reference_rows = {
            p.name.removesuffix("_gt.pdb"): diagnose(p) for p in reference_dir.glob("*_gt.pdb")
        }
        expected_pairs = 28 if cohort == "casp15" else 18
        if len(reference_rows) != expected_pairs:
            raise ValueError(f"wrong reference pair count for {cohort}")
        arms["reference_orientation_check"] = summarize(reference_rows)
        arms["native_chi1_agreement"] = reference_chi1_accuracy(
            args.baseline_root, args.guided_root, cohort
        )
        if set(arms["native_chi1_agreement"]["targets"]) != set(arms["guided"]["targets"]):
            raise ValueError(f"χ1 reference target set differs for {cohort}")
        output["cohorts"][cohort] = arms
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
