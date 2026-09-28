#!/usr/bin/env python3
"""Compare external PDB pLDDT with residue-level hard lDDT-Cα labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mambafold.confidence.targets import hard_lddt_ca  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ca_residues(path: Path, chain_id: str) -> list[tuple[str, np.ndarray, float]]:
    model = next(PDBParser(QUIET=True).get_structure(path.stem, str(path)).get_models())
    if chain_id not in model:
        raise ValueError(f"{path}: missing chain {chain_id}")
    rows = []
    for residue in model[chain_id]:
        if residue.id[0] != " " or "CA" not in residue:
            continue
        letter = seq1(residue.resname, custom_map={"UNK": "X"})
        if len(letter) != 1:
            continue
        atom = residue["CA"]
        rows.append((letter, atom.coord.astype(np.float32), float(atom.bfactor)))
    if not rows:
        raise ValueError(f"{path}: chain {chain_id} has no Cα atoms")
    return rows


def align_ca(
    prediction: list[tuple[str, np.ndarray, float]],
    reference: list[tuple[str, np.ndarray, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    from Bio.Align import PairwiseAligner

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -0.5
    try:
        aligner.end_insertion_score = 0
        aligner.end_deletion_score = 0
    except AttributeError:
        aligner.target_end_gap_score = 0
        aligner.query_end_gap_score = 0
    alignment = aligner.align(
        "".join(row[0] for row in prediction),
        "".join(row[0] for row in reference),
    )[0]
    matched = [
        (int(i), int(j))
        for i, j in alignment.indices.T
        if i >= 0 and j >= 0 and prediction[i][0] == reference[j][0]
    ]
    if not matched:
        raise ValueError("prediction/reference have no matching Cα residues")
    pred_xyz = np.stack([prediction[i][1] for i, _ in matched])
    ref_xyz = np.stack([reference[j][1] for _, j in matched])
    scores = np.asarray([prediction[i][2] / 100.0 for i, _ in matched])
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("prediction has invalid pLDDT B-factors")
    return pred_xyz, ref_xyz, scores, len(matched)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-dir", required=True, type=Path)
    parser.add_argument("--scores-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    scores_path = args.scores_dir / "summary.json"
    scores = json.loads(scores_path.read_text())
    if not scores.get("evaluation_complete"):
        raise ValueError(f"OpenStructure evaluation is incomplete: {scores_path}")
    rows = []
    all_predictions = []
    all_targets = []
    for item in scores["rows"]:
        target = item["target_id"]
        raw = json.loads((args.scores_dir / "raw" / f"{target}.json").read_text())
        chain_map = raw["chain_mapping"]
        if len(chain_map) != 1:
            raise ValueError(f"{target}: expected one chain mapping, got {chain_map}")
        reference_chain, prediction_chain = next(iter(chain_map.items()))
        prediction = ca_residues(args.pairs_dir / f"{target}_pred.pdb", prediction_chain)
        reference = ca_residues(args.pairs_dir / f"{target}_gt.pdb", reference_chain)
        pred_xyz, ref_xyz, plddt, matched = align_ca(prediction, reference)
        pred_tensor = torch.from_numpy(pred_xyz).unsqueeze(0)
        ref_tensor = torch.from_numpy(ref_xyz).unsqueeze(0)
        labels, valid = hard_lddt_ca(
            pred_tensor, ref_tensor, torch.ones((1, matched), dtype=torch.bool)
        )
        valid_np = valid.squeeze(0).numpy()
        prediction_values = plddt[valid_np]
        target_values = labels.squeeze(0).numpy()[valid_np]
        if not len(prediction_values):
            raise ValueError(f"{target}: no residues with lDDT labels")
        all_predictions.append(prediction_values)
        all_targets.append(target_values)
        rows.append({
            "target_id": target,
            "prediction_residues": len(prediction),
            "reference_ca_residues": len(reference),
            "aligned_identical_residues": matched,
            "labeled_residues": len(prediction_values),
            "mean_plddt": float(prediction_values.mean()),
            "mean_lddt_ca": float(target_values.mean()),
            "mae": float(np.abs(prediction_values - target_values).mean()),
        })

    pred = np.concatenate(all_predictions)
    true = np.concatenate(all_targets)
    bins = np.minimum((pred * 10).astype(int), 9)
    gaps = [
        abs(float(pred[bins == index].mean() - true[bins == index].mean()))
        if (bins == index).any() else 0.0
        for index in range(10)
    ]
    counts = np.bincount(bins, minlength=10)
    summary = {
        "schema_version": 1,
        "metric": "residue hard lDDT-Ca versus PDB pLDDT B-factor / 100",
        "pairs_manifest_sha256": sha256(args.pairs_dir / "manifest.json"),
        "openstructure_summary_sha256": sha256(scores_path),
        "targets": len(rows),
        "labeled_residues": len(pred),
        "mae": float(np.abs(pred - true).mean()),
        "rmse": float(np.sqrt(np.square(pred - true).mean())),
        "pearson": float(np.corrcoef(pred, true)[0, 1]),
        "ece_10": float(np.dot(counts / len(pred), gaps)),
        "mean_plddt": float(pred.mean()),
        "mean_lddt_ca": float(true.mean()),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
