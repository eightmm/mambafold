"""Convert RCSB mmCIF to Boltz-style .npz for MambaFold inference.

Usage:
    python scripts/pdb_to_npz.py --pdb_id 10AF --out data/test_2025/10af.npz

Downloads the mmCIF from RCSB and converts to the .npz format expected by
RCSBDataset. Atoms are stored in canonical RESIDUE_ATOMS order per residue.
"""

import argparse
import sys
import urllib.request
from io import StringIO
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mambafold.data.constants import (
    AA_TO_ID, RESIDUE_ATOMS,
    BOLTZ_RESIDUES_DTYPE as RESIDUES_DTYPE,
    BOLTZ_ATOMS_DTYPE as ATOMS_DTYPE,
    BOLTZ_CHAINS_DTYPE as CHAINS_DTYPE,
)

MOL_TYPE_PROTEIN = 0


def download_cif(pdb_id: str) -> str:
    """Download mmCIF from RCSB."""
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
    print(f"Downloading {url} ...")
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read().decode("utf-8")


def parse_cif(cif_text: str):
    """Parse mmCIF with BioPython and return Structure."""
    from Bio.PDB.MMCIFParser import MMCIFParser
    parser = MMCIFParser(QUIET=True)
    return parser.get_structure("prot", StringIO(cif_text))


def convert(structure, pdb_id: str) -> dict:
    """Convert BioPython Structure to Boltz npz arrays.

    Only uses the first model and protein chains (standard amino acids).
    Atoms are stored in canonical RESIDUE_ATOMS order.
    """
    model = next(structure.get_models())

    all_res_records = []
    all_atom_records = []
    all_chain_records = []

    global_atom_idx = 0
    global_res_idx = 0

    for chain in model.get_chains():
        res_records_this_chain = []

        for res in chain.get_residues():
            het, seq, ins = res.get_id()
            if het.strip():            # skip HETATM
                continue
            res_name = res.get_resname().strip()
            if res_name not in AA_TO_ID or res_name == "UNK":
                continue

            canon_atoms = RESIDUE_ATOMS.get(res_name, [])
            n_canon = len(canon_atoms)
            atom_start = global_atom_idx

            for atom_name in canon_atoms:
                atom_rec = np.zeros(1, dtype=ATOMS_DTYPE)[0]
                if res.has_id(atom_name):
                    bio_atom = res[atom_name]
                    coord = bio_atom.get_coord().astype(np.float32)
                    atom_rec["coords"] = coord
                    atom_rec["is_present"] = True
                else:
                    atom_rec["coords"] = np.zeros(3, dtype=np.float32)
                    atom_rec["is_present"] = False
                all_atom_records.append(atom_rec)
                global_atom_idx += 1

            res_rec = np.zeros(1, dtype=RESIDUES_DTYPE)[0]
            res_rec["name"] = res_name
            res_rec["res_type"] = AA_TO_ID.get(res_name, 20)
            res_rec["res_idx"] = global_res_idx
            res_rec["atom_idx"] = atom_start
            res_rec["atom_num"] = n_canon
            res_rec["is_standard"] = True
            res_rec["is_present"] = True
            res_records_this_chain.append(res_rec)
            global_res_idx += 1

        if not res_records_this_chain:
            continue

        chain_rec = np.zeros(1, dtype=CHAINS_DTYPE)[0]
        chain_rec["name"] = chain.get_id()
        chain_rec["mol_type"] = MOL_TYPE_PROTEIN
        chain_rec["res_idx"] = res_records_this_chain[0]["res_idx"]
        chain_rec["res_num"] = len(res_records_this_chain)
        chain_rec["atom_idx"] = res_records_this_chain[0]["atom_idx"]
        chain_rec["atom_num"] = sum(r["atom_num"] for r in res_records_this_chain)
        all_chain_records.extend([chain_rec])
        all_res_records.extend(res_records_this_chain)

    if not all_chain_records:
        raise ValueError(f"No protein residues found in {pdb_id}")

    residues = np.array(all_res_records, dtype=RESIDUES_DTYPE)
    atoms    = np.array(all_atom_records, dtype=ATOMS_DTYPE)
    chains   = np.array(all_chain_records, dtype=CHAINS_DTYPE)

    # Count observed atoms
    obs = atoms["is_present"].sum()
    print(f"  Chains: {len(chains)}  Residues: {len(residues)}  "
          f"Atoms: {len(atoms)} ({obs} observed, {obs/len(atoms)*100:.1f}%)")

    return dict(
        residues=residues,
        atoms=atoms,
        chains=chains,
        bonds=np.zeros(0),
        connections=np.zeros(0),
        interfaces=np.zeros(0),
        mask=np.ones(len(residues), dtype=bool),
        coords=np.zeros((len(atoms), 3), dtype=np.float32),
        ensemble=np.zeros(0),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_id", required=True, help="PDB ID, e.g. 10AF")
    parser.add_argument("--out", required=True, help="Output .npz path")
    args = parser.parse_args()

    cif_text = download_cif(args.pdb_id)
    structure = parse_cif(cif_text)
    arrays = convert(structure, args.pdb_id)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
