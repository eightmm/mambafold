from scripts.prepare_casp_single_chain import (
    align_residues,
    build_npz,
    parse_single_chain_pdb,
    read_fasta,
    reference_target_id,
)


def _pdb(sequence_rows: str) -> bytes:
    return (sequence_rows + "END\n").encode()


def test_reference_target_id_only_removes_domain_suffix():
    assert reference_target_id("T1207-D1") == "T1207"
    assert reference_target_id("T1228v1-D4") == "T1228v1"
    assert reference_target_id("T1207") == "T1207"


def test_read_fasta_uses_first_header_token(tmp_path):
    path = tmp_path / "targets.fasta"
    path.write_text(">T1 description\nACD\nEF\n\n>T2\nGG\n")
    assert read_fasta(path) == {"T1": "ACDEF", "T2": "GG"}


def test_official_sequence_topology_keeps_missing_residue():
    payload = _pdb(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 10.00           N  \n"
        "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 10.00           C  \n"
        "ATOM      3  N   ASP A   3       2.000   0.000   0.000  1.00 10.00           N  \n"
        "ATOM      4  CA  ASP A   3       3.000   0.000   0.000  1.00 10.00           C  \n"
    )
    parsed = parse_single_chain_pdb(payload, "T1")
    mapping, identity = align_residues("ACD", parsed)
    arrays = build_npz("ACD", [mapping])

    assert identity == 1.0
    assert len(arrays["residues"]) == 3
    middle = arrays["residues"][1]
    atom_slice = arrays["atoms"][
        int(middle["atom_idx"]) : int(middle["atom_idx"] + middle["atom_num"])
    ]
    assert not atom_slice["is_present"].any()
    assert arrays["chains"][0]["res_num"] == 3


def test_parse_rejects_two_protein_chains():
    payload = _pdb(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 10.00           C  \n"
        "ATOM      2  CA  ALA B   1       1.000   0.000   0.000  1.00 10.00           C  \n"
    )
    try:
        parse_single_chain_pdb(payload, "T1")
    except ValueError as exc:
        assert "one protein chain" in str(exc)
    else:
        raise AssertionError("two protein chains should be rejected")
