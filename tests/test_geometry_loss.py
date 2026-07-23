"""Tests for bond-length and Cα clash losses."""

import torch

from mambafold.data.constants import AA_TO_ID, COORD_SCALE
from mambafold.losses.geometry import (
    IDEAL,
    bond_length_loss,
    ca_clash_loss,
)


def _build_ideal_residue(res_name: str, base_z: float = 0.0) -> torch.Tensor:
    """Place N/CA/C/O/CB at canonical bond distances along a line; pad others with zeros.

    Coords returned in normalized units (Å / COORD_SCALE).
    """
    s = COORD_SCALE
    N = torch.tensor([0.0, 0.0, 0.0])
    CA = N + torch.tensor([IDEAL["N_CA"], 0.0, 0.0])
    C = CA + torch.tensor([IDEAL["CA_C"], 0.0, 0.0])
    oxygen = C + torch.tensor([0.0, IDEAL["C_O"], 0.0])
    CB = CA + torch.tensor([0.0, 0.0, IDEAL["CA_CB"]])
    atoms = torch.zeros(15, 3)
    atoms[0] = N
    atoms[1] = CA
    atoms[2] = C
    atoms[3] = oxygen
    if res_name != "GLY":
        atoms[4] = CB
    return atoms + torch.tensor([0.0, 0.0, base_z / s])


def test_bond_length_loss_near_zero_for_ideal_geometry():
    B, L = 1, 3
    res_names = ["ALA", "GLY", "ALA"]
    res_type = torch.tensor([[AA_TO_ID[n] for n in res_names]])
    atoms = torch.stack(
        [_build_ideal_residue(n, base_z=i * 10.0) for i, n in enumerate(res_names)]
    )  # [L, 15, 3]
    coords = atoms.unsqueeze(0)  # [1, L, 15, 3]
    atom_mask = torch.zeros(B, L, 15, dtype=torch.bool)
    atom_mask[:, :, :4] = True  # N, CA, C, O always valid
    atom_mask[:, 0, 4] = True  # CB for ALA0
    atom_mask[:, 2, 4] = True  # CB for ALA2
    res_mask = torch.ones(B, L, dtype=torch.bool)

    loss = bond_length_loss(coords, res_type, atom_mask, res_mask)
    # Within-residue bonds are exact; only the peptide C(i)-N(i+1) bonds (which we
    # spaced 10 Å apart) contribute error. Confirm the loss sees non-zero there.
    assert loss.item() > 0

    # Zero-out peptide contribution by using only L=1
    res_type1 = res_type[:, :1]
    coords1 = coords[:, :1]
    atom_mask1 = atom_mask[:, :1]
    res_mask1 = res_mask[:, :1]
    loss1 = bond_length_loss(coords1, res_type1, atom_mask1, res_mask1)
    assert loss1.item() < 1e-6


def test_bond_length_loss_skips_cb_for_glycine():
    """If a GLY residue has a garbage CB slot but its mask is off, GLY must be excluded."""
    B, L = 1, 1
    coords = torch.zeros(B, L, 15, 3)
    coords[0, 0, 0] = torch.tensor([0.0, 0.0, 0.0])  # N
    coords[0, 0, 1] = torch.tensor([IDEAL["N_CA"], 0.0, 0.0])  # CA
    coords[0, 0, 2] = coords[0, 0, 1] + torch.tensor([IDEAL["CA_C"], 0.0, 0.0])
    coords[0, 0, 3] = coords[0, 0, 2] + torch.tensor([0.0, IDEAL["C_O"], 0.0])
    coords[0, 0, 4] = torch.tensor([999.0, 0.0, 0.0])  # bogus CB
    atom_mask = torch.zeros(B, L, 15, dtype=torch.bool)
    atom_mask[:, :, :4] = True  # no CB
    res_type = torch.tensor([[AA_TO_ID["GLY"]]])
    res_mask = torch.ones(B, L, dtype=torch.bool)
    loss = bond_length_loss(coords, res_type, atom_mask, res_mask)
    assert loss.item() < 1e-6


def test_ca_clash_loss_zero_for_well_spaced_ca():
    """Cα spaced at 3.8 Å intervals along a line should produce zero clash."""
    B, L = 1, 5
    ca_spacing_norm = 3.8 / COORD_SCALE
    coords = torch.zeros(B, L, 15, 3)
    for i in range(L):
        coords[0, i, 1] = torch.tensor([i * ca_spacing_norm, 0.0, 0.0])
    res_mask = torch.ones(B, L, dtype=torch.bool)
    assert ca_clash_loss(coords, res_mask).item() == 0.0


def test_ca_clash_loss_penalises_overlap():
    """Cα at identical positions (modulo seq_sep window) must yield positive loss."""
    B, L = 1, 6
    coords = torch.zeros(B, L, 15, 3)  # all Cα at origin
    res_mask = torch.ones(B, L, dtype=torch.bool)
    assert ca_clash_loss(coords, res_mask).item() > 0.0
