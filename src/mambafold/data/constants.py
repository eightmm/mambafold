"""Canonical amino acid and atom constants for protein structure prediction."""

# 20 standard amino acids (3-letter → 1-letter → index)
AA_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

AA_1TO3 = {v: k for k, v in AA_3TO1.items()}

AA_TO_ID = {aa: i for i, aa in enumerate(sorted(AA_3TO1.keys()))}
AA_TO_ID["UNK"] = 20
NUM_AA_TYPES = 21

ID_TO_AA = {v: k for k, v in AA_TO_ID.items()}

# Maximum heavy atoms per residue (atom14 layout + OXT terminal oxygen)
MAX_ATOMS_PER_RES = 15

# Canonical heavy atom names per residue type (atom14-style, padded to 14 + OXT)
# Order: N, CA, C, O, CB, then side-chain specific, then OXT
BACKBONE_ATOMS = ["N", "CA", "C", "O"]
BACKBONE_ATOM_IDS = [0, 1, 2, 3]
CA_ATOM_ID = 1

# Per-residue heavy atom slot table
# Each residue maps to a list of atom names in canonical order (up to 14 slots)
# Slot index corresponds to atom_type encoding
RESIDUE_ATOMS = {
    "ALA": ["N", "CA", "C", "O", "CB"],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
    "CYS": ["N", "CA", "C", "O", "CB", "SG"],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
    "GLY": ["N", "CA", "C", "O"],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD"],
    "SER": ["N", "CA", "C", "O", "CB", "OG"],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2"],
    "TRP": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
    "UNK": ["N", "CA", "C", "O", "CB"],
}

# Build atom name → slot index mapping per residue
RESIDUE_ATOM_TO_SLOT = {
    res: {atom: i for i, atom in enumerate(atoms)}
    for res, atoms in RESIDUE_ATOMS.items()
}

# Number of heavy atoms per residue type
RESIDUE_ATOM_COUNT = {res: len(atoms) for res, atoms in RESIDUE_ATOMS.items()}

# Atom type vocabulary (unique atom names across all residues)
_all_atoms = sorted(set(a for atoms in RESIDUE_ATOMS.values() for a in atoms))
ATOM_NAME_TO_ID = {name: i for i, name in enumerate(_all_atoms)}
ATOM_NAME_TO_ID["PAD"] = len(_all_atoms)
NUM_ATOM_TYPES = len(ATOM_NAME_TO_ID)

# Coordinate normalization
COORD_SCALE = 10.0  # Angstrom -> normalized

# (residue, atom) pair vocabulary — unique chemical identity per atom slot
# Sorted by residue name then slot order for determinism.
RESIDUE_ATOM_PAIRS: list[tuple[str, str]] = [
    (res, atom)
    for res in sorted(RESIDUE_ATOMS.keys())
    for atom in RESIDUE_ATOMS[res]
]
PAIR_TO_ID: dict[tuple[str, str], int] = {p: i for i, p in enumerate(RESIDUE_ATOM_PAIRS)}
PAIR_PAD_ID: int = len(RESIDUE_ATOM_PAIRS)          # index for empty/padding slots
NUM_PAIR_TYPES: int = len(RESIDUE_ATOM_PAIRS) + 1   # +1 for PAD
