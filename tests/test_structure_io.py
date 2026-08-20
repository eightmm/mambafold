"""Round-trip tests for prediction structure writers."""

import numpy as np
import pytest
from Bio.PDB import MMCIFParser, PDBParser

from mambafold.data.constants import AA_TO_ID, MAX_ATOMS_PER_RES
from mambafold.structure_io import write_mmcif, write_pdb


def test_pdb_and_mmcif_outputs_are_parseable(tmp_path):
    coords = np.zeros((2, MAX_ATOMS_PER_RES, 3), dtype=np.float32)
    coords[0, :5] = np.arange(15, dtype=np.float32).reshape(5, 3) * 0.1
    coords[1, :4] = 4.0 + np.arange(12, dtype=np.float32).reshape(4, 3) * 0.1
    res_type = np.asarray([AA_TO_ID["ALA"], AA_TO_ID["GLY"]], dtype=np.int64)
    atom_mask = np.zeros((2, MAX_ATOMS_PER_RES), dtype=bool)
    atom_mask[0, :5] = True
    atom_mask[1, :4] = True
    b_factors = np.full((2, MAX_ATOMS_PER_RES), 73.0, dtype=np.float32)
    chain_id = np.zeros(2, dtype=np.int64)
    pdb_path = tmp_path / "prediction.pdb"
    cif_path = tmp_path / "prediction.cif"

    write_pdb(coords, res_type, atom_mask, b_factors, chain_id, pdb_path)
    write_mmcif(
        coords,
        res_type,
        atom_mask,
        b_factors,
        chain_id,
        cif_path,
        entry_id="test prediction",
    )

    pdb_model = PDBParser(QUIET=True).get_structure("pdb", pdb_path)[0]
    cif_model = MMCIFParser(QUIET=True).get_structure("cif", cif_path)[0]
    pdb_atoms = list(pdb_model.get_atoms())
    cif_atoms = list(cif_model.get_atoms())

    assert len(pdb_atoms) == 9
    assert len(cif_atoms) == 9
    np.testing.assert_allclose(pdb_atoms[0].coord, coords[0, 0], atol=1e-3)
    np.testing.assert_allclose(cif_atoms[-1].coord, coords[1, 3], atol=1e-3)
    assert pdb_atoms[0].bfactor == 73.0
    assert cif_atoms[0].bfactor == 73.0


def test_pdb_writer_rejects_coordinates_that_break_fixed_width(tmp_path):
    coords = np.zeros((1, MAX_ATOMS_PER_RES, 3), dtype=np.float32)
    coords[0, 0, 0] = -1000.0
    res_type = np.asarray([AA_TO_ID["ALA"]], dtype=np.int64)
    atom_mask = np.zeros((1, MAX_ATOMS_PER_RES), dtype=bool)
    atom_mask[0, 0] = True
    b_factors = np.zeros((1, MAX_ATOMS_PER_RES), dtype=np.float32)
    chain_id = np.zeros(1, dtype=np.int64)
    path = tmp_path / "invalid.pdb"

    with pytest.raises(ValueError, match="do not fit PDB"):
        write_pdb(coords, res_type, atom_mask, b_factors, chain_id, path)

    assert not path.exists()
