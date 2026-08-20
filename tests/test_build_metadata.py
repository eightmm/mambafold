import numpy as np

from scripts.build_metadata import _scan_one, canonical_chain, resolve_paths


def test_canonical_chain_matches_training_residue_filter():
    residues = np.array(
        [
            ("ALA", True, True),
            ("MSE", False, True),
            ("UNK", True, True),
            ("GLY", True, False),
            ("NOT", True, True),
            ("VAL", True, True),
        ],
        dtype=[("name", "U3"), ("is_standard", "?"), ("is_present", "?")],
    )

    sequence, observed = canonical_chain(residues)

    assert sequence == "AGV"
    assert observed == 2


def test_scan_exports_the_loader_canonical_sequence(tmp_path):
    accepted = [("ALA", True, True)] * 4 + [("GLY", True, False)] * 4
    accepted += [("VAL", True, True)] * 4
    residues = np.array(
        accepted + [("MSE", False, True), ("UNK", True, True), ("NOT", True, True)],
        dtype=[("name", "U3"), ("is_standard", "?"), ("is_present", "?")],
    )
    chains = np.array(
        [(0, "A", 0, len(residues))],
        dtype=[("mol_type", "i4"), ("name", "U4"), ("res_idx", "i4"), ("res_num", "i4")],
    )
    path = tmp_path / "1abc.npz"
    np.savez(path, chains=chains, residues=residues)

    record = _scan_one(path)

    assert record["error"] == ""
    assert record["fasta"] == [">1abc_A\nAAAAGGGGVVVV\n"]
    assert record["rows"] == [
        {
            "pdb_id": "1abc",
            "chain": "A",
            "seq_len": 12,
            "n_standard": 12,
            "n_observed": 8,
        }
    ]


def test_resolve_paths_applies_file_list_in_declared_order(tmp_path):
    data_dir = tmp_path / "npz"
    data_dir.mkdir()
    for name in ("a.npz", "b.npz", "unused.npz"):
        (data_dir / name).touch()
    file_list = tmp_path / "train.txt"
    file_list.write_text("b.npz\n# comment\na.npz\n")

    assert resolve_paths(data_dir, file_list) == [
        data_dir / "b.npz",
        data_dir / "a.npz",
    ]


def test_resolve_paths_rejects_duplicate_and_missing_entries(tmp_path):
    data_dir = tmp_path / "npz"
    data_dir.mkdir()
    (data_dir / "a.npz").touch()
    duplicates = tmp_path / "duplicates.txt"
    duplicates.write_text("a.npz\na.npz\n")

    try:
        resolve_paths(data_dir, duplicates)
    except ValueError as exc:
        assert "duplicate entries" in str(exc)
    else:
        raise AssertionError("duplicate file-list entries should fail")

    missing = tmp_path / "missing.txt"
    missing.write_text("absent.npz\n")
    try:
        resolve_paths(data_dir, missing)
    except FileNotFoundError as exc:
        assert "1 file-list entries are missing" in str(exc)
    else:
        raise AssertionError("missing file-list entries should fail")
