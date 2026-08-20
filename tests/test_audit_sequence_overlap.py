import json

import pytest

from benchmarks.audit_sequence_overlap import (
    FastaRecord,
    audit_exact_overlap,
    build_report,
    read_fasta,
    write_fasta,
    write_ids,
)


def test_exact_overlap_keeps_source_provenance():
    targets = [FastaRecord("target_a", "ACDE"), FastaRecord("target_b", "FGHI")]
    training = [
        (
            "rcsb.fasta",
            [FastaRecord("1abc_A", "ACDE"), FastaRecord("2def_A", "KLMN")],
        ),
        ("afdb.fasta", [FastaRecord("AF-X", "ACDE")]),
    ]

    result = audit_exact_overlap(targets, training)

    assert result["target_records"] == 2
    assert result["training_records"] == 3
    assert result["exact_overlap_targets"] == 1
    assert result["exact_clean_targets"] == 1
    assert result["matches"][0]["target_id"] == "target_a"
    assert result["matches"][0]["training_matches"] == [
        {"source": "rcsb.fasta", "identifier": "1abc_A"},
        {"source": "afdb.fasta", "identifier": "AF-X"},
    ]


def test_report_uses_filenames_and_hashes_not_machine_paths(tmp_path):
    targets = tmp_path / "targets.fasta"
    training = tmp_path / "training.fasta"
    targets.write_text(">t1\nACDE\n>t2\nFGHI\n")
    training.write_text(">train\nACDE\n")

    report = build_report(targets, [training])

    assert report["scope"] == "exact_sequence_only_not_homology_clean"
    assert report["target_fasta"]["filename"] == "targets.fasta"
    assert report["training_fastas"][0]["filename"] == "training.fasta"
    assert "/" not in report["target_fasta"]["filename"]
    assert len(report["target_fasta"]["sha256"]) == 64


def test_filtered_fasta_refuses_overwrite(tmp_path):
    output = tmp_path / "clean.fasta"
    records = [FastaRecord("t1", "ACDE")]
    write_fasta(output, records)
    assert read_fasta(output) == records

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_fasta(output, records)


def test_filtered_ids_preserve_target_order(tmp_path):
    output = tmp_path / "clean.txt"
    records = [FastaRecord("t2", "FGHI"), FastaRecord("t1", "ACDE")]

    write_ids(output, records)

    assert output.read_text() == "t2\nt1\n"


def test_cli_writes_report_and_exact_clean_fasta(tmp_path, monkeypatch, capsys):
    from benchmarks import audit_sequence_overlap

    targets = tmp_path / "targets.fasta"
    training = tmp_path / "training.fasta"
    report = tmp_path / "report.json"
    clean = tmp_path / "clean.fasta"
    clean_ids = tmp_path / "clean.txt"
    targets.write_text(">seen\nACDE\n>clean\nFGHI\n")
    training.write_text(">train\nACDE\n")
    monkeypatch.setattr(
        "sys.argv",
        [
            "audit_sequence_overlap.py",
            "--targets",
            str(targets),
            "--training",
            str(training),
            "--out",
            str(report),
            "--write-exact-clean-fasta",
            str(clean),
            "--write-exact-clean-ids",
            str(clean_ids),
        ],
    )

    audit_sequence_overlap.main()

    payload = json.loads(report.read_text())
    assert payload["result"]["exact_overlap_targets"] == 1
    assert read_fasta(clean) == [FastaRecord("clean", "FGHI")]
    assert clean_ids.read_text() == "clean\n"
    assert "exact overlaps: 1/2" in capsys.readouterr().out
