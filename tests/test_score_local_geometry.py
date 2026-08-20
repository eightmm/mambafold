"""Focused tests for ground-truth-free structure-validity metrics."""

import numpy as np

from benchmarks.score_local_geometry import (
    _ca_chiral_volumes,
    _nonlocal_ca_metrics,
    _nonlocal_ca_segment_metrics,
    _segment_pair_distances,
)


def _pdb_with_cb(z: float):
    return {
        "residues": {
            ("A", 1, ""): {
                "CA": np.array([0.0, 0.0, 0.0], dtype=np.float32),
                "N": np.array([1.0, 0.0, 0.0], dtype=np.float32),
                "C": np.array([0.0, 1.0, 0.0], dtype=np.float32),
                "CB": np.array([0.0, 0.0, z], dtype=np.float32),
            }
        },
        "atoms": [],
    }


def test_ca_chirality_volume_changes_sign_under_reflection():
    assert _ca_chiral_volumes(_pdb_with_cb(1.0)) == [1.0]
    assert _ca_chiral_volumes(_pdb_with_cb(-1.0)) == [-1.0]


def test_ca_chirality_skips_residue_without_cb():
    pdb = _pdb_with_cb(1.0)
    del pdb["residues"][("A", 1, "")]["CB"]
    assert _ca_chiral_volumes(pdb) == []


def test_nonlocal_ca_metrics_detect_sequence_distant_self_overlap():
    pdb = {"residues": {}, "atoms": []}
    for index in range(1, 16):
        pdb["residues"][("A", index, "")] = {
            "CA": np.array([float(index * 5), 0.0, 0.0], dtype=np.float32)
        }
    pdb["residues"][("A", 15, "")]["CA"] = np.array([5.5, 0.0, 0.0], dtype=np.float32)

    metrics = _nonlocal_ca_metrics(pdb, seq_sep=12)

    assert metrics["nonlocal_ca_min_A"] == 0.5
    assert metrics["nonlocal_ca_clashes_lt_2A"] == 1
    assert metrics["nonlocal_ca_clashes_lt_3A"] == 1
    assert metrics["nonlocal_ca_clashes_lt_3p6A"] == 1
    assert np.isclose(metrics["nonlocal_ca_penetration_rms_A"], 3.1)


def test_segment_pair_distance_detects_interior_crossing():
    p0 = np.array([[-1.0, 0.0, 0.0]], dtype=np.float32)
    p1 = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    q0 = np.array([[0.0, -1.0, 0.0]], dtype=np.float32)
    q1 = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

    distance = _segment_pair_distances(p0, p1, q0, q1)

    np.testing.assert_allclose(distance, [0.0], atol=1e-7)


def test_nonlocal_segment_metrics_catch_crossing_missed_by_endpoints():
    pdb = {"residues": {}, "atoms": []}
    coordinates = {
        1: (-1.0, 0.0, 0.0),
        2: (1.0, 0.0, 0.0),
        15: (0.0, -1.0, 0.0),
        16: (0.0, 1.0, 0.0),
    }
    for residue, xyz in coordinates.items():
        pdb["residues"][("A", residue, "")] = {"CA": np.asarray(xyz, dtype=np.float32)}

    point_metrics = _nonlocal_ca_metrics(pdb, seq_sep=12)
    segment_metrics = _nonlocal_ca_segment_metrics(pdb, seq_sep=12)

    assert point_metrics["nonlocal_ca_min_A"] > 1.0
    assert segment_metrics["nonlocal_ca_segment_pairs"] == 1
    assert segment_metrics["nonlocal_ca_segment_min_A"] == 0.0
    assert segment_metrics["nonlocal_ca_segment_clashes_lt_0p5A"] == 1
    assert segment_metrics["nonlocal_ca_segment_clashes_lt_1A"] == 1
    assert segment_metrics["nonlocal_ca_segment_clashes_lt_2A"] == 1
    assert segment_metrics["nonlocal_ca_segment_clashes_lt_2p5A"] == 1
    assert segment_metrics["nonlocal_ca_segment_clashes_lt_3A"] == 1
    assert segment_metrics["nonlocal_ca_segment_penetration_rms_A"] == 2.5


def test_nonlocal_segment_metrics_reject_implausibly_long_edge():
    pdb = {"residues": {}, "atoms": []}
    for residue, xyz in {
        1: (0.0, 0.0, 0.0),
        2: (42.0, 0.0, 0.0),
        15: (20.0, -1.0, 0.0),
        16: (20.0, 1.0, 0.0),
    }.items():
        pdb["residues"][("A", residue, "")] = {"CA": np.asarray(xyz, dtype=np.float32)}

    metrics = _nonlocal_ca_segment_metrics(pdb, seq_sep=12, max_edge_A=6.0)

    assert metrics["nonlocal_ca_segment_pairs"] == 0
