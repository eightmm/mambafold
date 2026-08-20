from pathlib import Path

import numpy as np
import pytest

import mambafold.data.dataset as dataset_module
from mambafold.data.dataset import RCSBDataset, _valid_observation_crop_starts


def test_chain_index_does_not_scan_forward_after_invalid_sample(monkeypatch):
    dataset = RCSBDataset.__new__(RCSBDataset)
    dataset.extract_monomer_chains = True
    dataset.chain_index = [(0, 2, 128), (1, 0, 256)]
    dataset.files = [Path("first.npz"), Path("second.npz")]
    loaded = []

    def fake_load(path):
        loaded.append(path)
        return object()

    monkeypatch.setattr(np, "load", fake_load)
    monkeypatch.setattr(dataset, "_canonicalize", lambda *_args, **_kwargs: None)

    assert dataset[0] is None
    assert loaded == [Path("first.npz")]


def test_file_index_reports_path_without_scanning_forward(monkeypatch):
    dataset = RCSBDataset.__new__(RCSBDataset)
    dataset.extract_monomer_chains = False
    dataset.chain_index = None
    dataset.files = [Path("broken.npz"), Path("valid.npz")]
    loaded = []

    def fake_load(path):
        loaded.append(path)
        raise OSError("broken fixture")

    monkeypatch.setattr(np, "load", fake_load)

    with pytest.raises(RuntimeError, match=r"broken\.npz"):
        dataset[0]
    assert loaded == [Path("broken.npz")]


def test_observation_crop_starts_match_runtime_threshold():
    residue_dtype = np.dtype([("name", "U3"), ("atom_idx", "i4"), ("atom_num", "i4")])
    residues = np.array(
        [("ALA", 0, 2), ("ALA", 2, 2), ("ALA", 4, 2)],
        dtype=residue_dtype,
    )
    atoms = np.array(
        [(False,), (False,), (True,), (True,), (True,), (False,)],
        dtype=np.dtype([("is_present", "?")]),
    )
    entries = [(0, 0, 0), (1, 1, 1), (2, 2, 2)]

    assert _valid_observation_crop_starts(
        residues, atoms, entries, max_length=2, min_obs_ratio=0.75
    ) == [1]


def test_monomer_canonicalize_does_not_walk_unselected_chains(monkeypatch):
    residue_dtype = np.dtype(
        [
            ("name", "U3"),
            ("is_standard", "?"),
            ("atom_idx", "i4"),
            ("atom_num", "i4"),
        ]
    )
    residues = np.array(
        [("ALA", True, 0, 0)] * 100 + [("GLY", True, 0, 0)] * 5,
        dtype=residue_dtype,
    )
    chains = np.array(
        [(0, 0, 100), (0, 100, 5)],
        dtype=np.dtype([("mol_type", "i4"), ("res_idx", "i4"), ("res_num", "i4")]),
    )
    atoms = np.empty(
        0,
        dtype=np.dtype([("coords", "f4", (3,)), ("is_present", "?")]),
    )

    dataset = RCSBDataset.__new__(RCSBDataset)
    dataset.max_length = 1024
    dataset.min_length = 1
    dataset.min_obs_ratio = 0.5
    dataset.esm_dir = None
    dataset.single_chain_only = True

    original = dataset_module._residue_seq_num
    visited = []

    def track_residue(residue, fallback):
        visited.append(fallback)
        return original(residue, fallback)

    monkeypatch.setattr(dataset_module, "_residue_seq_num", track_residue)
    example = dataset._canonicalize(
        {"residues": residues, "atoms": atoms, "chains": chains},
        only_chain_origin=1,
    )

    assert example is not None
    assert example.seq_len == 5
    assert visited == list(range(5))
