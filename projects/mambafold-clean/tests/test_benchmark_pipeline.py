from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from benchmarks.score_plddt_external import align_ca
from benchmarks.stage_rollout_references import main as stage_main
from scripts.rollout import benchmark_seeds


def test_benchmark_seed_is_not_target_or_process_hash_dependent() -> None:
    names = ["7qsw_A", "8day_A", "7qsw_A"]
    first = benchmark_seeds(42, names)
    assert first == benchmark_seeds(42, names)
    assert first[0] != first[1]
    assert first[0] == first[2]
    assert first != benchmark_seeds(43, names)


def test_plddt_alignment_skips_unobserved_reference_residue() -> None:
    point = lambda index: np.array([index, 0, 0], dtype=np.float32)
    prediction = [(aa, point(i), 75.0) for i, aa in enumerate("ACDEFG")]
    reference = [(aa, point(i), 0.0) for i, aa in enumerate("ACEFG")]

    pred_ca, ref_ca, plddt, matched = align_ca(prediction, reference)

    assert matched == 5
    assert pred_ca[:, 0].tolist() == [0, 1, 3, 4, 5]
    assert ref_ca[:, 0].tolist() == [0, 1, 2, 3, 4]
    assert np.allclose(plddt, 0.75)


def test_stage_rollout_references_builds_hashed_pairs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    rollout_dir = tmp_path / "rollout"
    structures = rollout_dir / "structures"
    structures.mkdir(parents=True)
    prediction = structures / "t1.pdb"
    prediction.write_text("ATOM\n")
    (rollout_dir / "rollout.json").write_text(
        json.dumps({"per_target": {"t1": {"length": 1}}})
    )
    references = tmp_path / "references"
    references.mkdir()
    reference = references / "t1_gt.pdb"
    reference.write_text("ATOM\n")
    out_dir = tmp_path / "pairs"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage_rollout_references.py",
            "--rollout-dir",
            str(rollout_dir),
            "--reference-dir",
            str(references),
            "--out-dir",
            str(out_dir),
        ],
    )
    stage_main()

    assert (out_dir / "t1_pred.pdb").resolve() == prediction.resolve()
    assert (out_dir / "t1_gt.pdb").resolve() == reference.resolve()
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["targets"] == 1
    assert len(manifest["rows"][0]["prediction_sha256"]) == 64
