from pathlib import Path

import pytest

from scripts.render_plddt_html import read_ca_trace, render_fragment


def test_render_plddt_fragment_is_dependency_free_and_embeds_scores(tmp_path: Path):
    pdb = tmp_path / "sample.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 42.00           C\n"
        "ATOM      2  CA  GLY A   2       3.800   0.000   0.000  1.00 91.00           C\n"
        "END\n"
    )
    coords, scores = read_ca_trace(pdb)
    fragment = render_fragment(pdb)
    assert coords == [[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]]
    assert scores == [42.0, 91.0]
    assert "https://" not in fragment
    assert '"scores":[42.0,91.0]' in fragment
    assert "drag to rotate" in fragment


def test_render_plddt_rejects_missing_trace(tmp_path: Path):
    pdb = tmp_path / "empty.pdb"
    pdb.write_text("END\n")
    with pytest.raises(ValueError, match="at least two CA"):
        read_ca_trace(pdb)
