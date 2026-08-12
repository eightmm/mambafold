from pathlib import Path

import numpy as np
import torch
from Bio.PDB import MMCIFParser, PDBParser

from projects.esm3.predict_fasta import sequence_example, write_cif, write_pdb


def test_structure_writers_emit_parseable_pdb_and_cif(tmp_path: Path):
    sequence = "ACDEFGHIKL"
    example = sequence_example(sequence, torch.zeros(len(sequence), 1536))
    coords = np.arange(example.atom_mask.numel() * 3, dtype=np.float32).reshape(
        *example.atom_mask.shape, 3
    )
    confidence = np.linspace(0.1, 1.0, len(sequence), dtype=np.float32)

    pdb_path = tmp_path / "prediction.pdb"
    cif_path = tmp_path / "prediction.cif"
    write_pdb(coords, example, confidence, pdb_path)
    write_cif(coords, example, confidence, cif_path, "test-protein")

    pdb_model = PDBParser(QUIET=True).get_structure("pdb", pdb_path)[0]
    cif_model = MMCIFParser(QUIET=True).get_structure("cif", cif_path)[0]
    assert len(list(pdb_model.get_residues())) == len(sequence)
    assert len(list(cif_model.get_residues())) == len(sequence)
    assert next(cif_model.get_atoms()).get_bfactor() == 10.0
