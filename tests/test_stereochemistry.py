"""Tests for ground-truth-free stereochemical guidance energies."""

import torch

from mambafold.data.constants import (
    AA_TO_ID,
    COORD_SCALE,
    MAX_ATOMS_PER_RES,
    RESIDUE_ATOMS,
)
from mambafold.losses.stereochemistry import (
    _cpu_topology,
    all_atom_clash_surrogate_loss,
    all_atom_vdw_clash_loss,
    chemical_chirality_loss,
    covalent_geometry_loss,
    peptide_planarity_loss,
    stereochemical_energy_terms,
)


def _masks(residues: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    res_type = torch.tensor([[AA_TO_ID[name] for name in residues]])
    atom_mask = torch.zeros(1, len(residues), MAX_ATOMS_PER_RES, dtype=torch.bool)
    for index, residue in enumerate(residues):
        atom_mask[0, index, : len(RESIDUE_ATOMS[residue])] = True
    res_mask = torch.ones(1, len(residues), dtype=torch.bool)
    return res_type, atom_mask, res_mask


def test_reference_topology_contains_all_standard_specs():
    topology = _cpu_topology()

    assert int(topology.bond_mask.sum()) == 153
    assert int(topology.angle_mask.sum()) == 174
    assert topology.bond_i.shape[0] == len(AA_TO_ID)
    assert topology.atom_radius_A.shape == (len(AA_TO_ID), MAX_ATOMS_PER_RES)


def test_covalent_energy_has_finite_descent_direction():
    torch.manual_seed(17)
    res_type, atom_mask, res_mask = _masks(["ALA"])
    coords = (0.15 * torch.randn(1, 1, MAX_ATOMS_PER_RES, 3)).requires_grad_(True)

    before = covalent_geometry_loss(coords, res_type, atom_mask, res_mask)
    gradient = torch.autograd.grad(before, coords)[0]
    direction = gradient / gradient.norm().clamp(min=1e-8)
    after = covalent_geometry_loss(
        coords.detach() - 1e-3 * direction,
        res_type,
        atom_mask,
        res_mask,
    )

    assert torch.isfinite(gradient).all()
    assert after < before


def test_peptide_planarity_detects_twist_and_has_descent_direction():
    _, atom_mask, res_mask = _masks(["ALA", "ALA"])
    atom_mask.zero_()
    atom_mask[..., :4] = True
    atom_mask[:, :, 4] = True
    planar = torch.zeros(1, 2, MAX_ATOMS_PER_RES, 3)
    planar[0, 0, 1] = torch.tensor([-1.0, 0.0, 0.0])
    planar[0, 0, 2] = torch.tensor([0.0, 0.0, 0.0])
    planar[0, 0, 3] = torch.tensor([0.0, 1.0, 0.0])
    planar[0, 1, 0] = torch.tensor([1.0, 0.0, 0.0])
    planar[0, 1, 1] = torch.tensor([1.5, 1.0, 0.0])
    twisted = planar.clone()
    twisted[0, 1, 1, 2] = 1.0
    twisted.requires_grad_(True)

    planar_energy = peptide_planarity_loss(planar, atom_mask, res_mask)
    before = peptide_planarity_loss(twisted, atom_mask, res_mask)
    gradient = torch.autograd.grad(before, twisted)[0]
    after = peptide_planarity_loss(
        twisted.detach() - 1e-2 * gradient,
        atom_mask,
        res_mask,
    )

    assert planar_energy < 1e-7
    assert before > planar_energy
    assert after < before


def test_chirality_rejects_reflected_l_amino_acid():
    res_type, atom_mask, res_mask = _masks(["ALA"])
    atom_mask.zero_()
    atom_mask[..., :5] = True
    correct = torch.zeros(1, 1, MAX_ATOMS_PER_RES, 3)
    correct[0, 0, 0] = torch.tensor([1.0, 0.0, 0.0])
    correct[0, 0, 2] = torch.tensor([0.0, 1.0, 0.0])
    correct[0, 0, 4] = torch.tensor([0.0, 0.0, 1.0])
    reflected = correct.clone()
    reflected[0, 0, 4, 2] = -1.0

    correct_energy = chemical_chirality_loss(correct, res_type, atom_mask, res_mask)
    reflected_energy = chemical_chirality_loss(reflected, res_type, atom_mask, res_mask)

    assert correct_energy == 0
    assert reflected_energy > 0.5


def test_vdw_clash_excludes_bonded_pair_but_pushes_nonbonded_pair_apart():
    single_type, single_mask, single_res_mask = _masks(["ALA"])
    single_mask.zero_()
    single_mask[0, 0, :2] = True
    bonded = torch.zeros(1, 1, MAX_ATOMS_PER_RES, 3)
    assert all_atom_vdw_clash_loss(bonded, single_type, single_mask, single_res_mask).item() == 0.0

    res_type, atom_mask, res_mask = _masks(["ALA", "ALA"])
    atom_mask.zero_()
    atom_mask[:, :, 1] = True
    coords = torch.zeros(1, 2, MAX_ATOMS_PER_RES, 3)
    coords[0, 1, 1, 0] = 0.01
    coords.requires_grad_(True)
    seq = torch.tensor([[1, 3]])
    before = all_atom_vdw_clash_loss(
        coords,
        res_type,
        atom_mask,
        res_mask,
        res_seq_nums=seq,
        pair_chunk_size=1,
    )
    gradient = torch.autograd.grad(before, coords)[0]
    after = all_atom_vdw_clash_loss(
        coords.detach() - 1e-3 * gradient,
        res_type,
        atom_mask,
        res_mask,
        res_seq_nums=seq,
        pair_chunk_size=1,
    )

    assert before > 0
    assert torch.isfinite(gradient).all()
    assert after < before


def test_ost_surrogate_uses_reference_carbon_and_sulfur_floors():
    res_type, atom_mask, res_mask = _masks(["ALA", "ALA"])
    atom_mask.zero_()
    atom_mask[:, :, 1] = True  # C-alpha is carbon.
    seq = torch.tensor([[1, 3]])
    coords = torch.zeros(1, 2, MAX_ATOMS_PER_RES, 3)
    coords[0, 1, 1, 0] = 1.89 / COORD_SCALE
    carbon = all_atom_clash_surrogate_loss(coords, res_type, atom_mask, res_mask, res_seq_nums=seq)
    assert carbon["hard_clashes_per_atom"] == 0.5
    coords[0, 1, 1, 0] = 1.91 / COORD_SCALE
    carbon_clear = all_atom_clash_surrogate_loss(
        coords, res_type, atom_mask, res_mask, res_seq_nums=seq
    )
    assert carbon_clear["hard_clashes_per_atom"] == 0.0
    assert carbon_clear["loss"] > 0.0  # 0.10 A Huber safety margin.

    cys_type, cys_mask, cys_res_mask = _masks(["CYS", "CYS"])
    sg = RESIDUE_ATOMS["CYS"].index("SG")
    cys_mask.zero_()
    cys_mask[:, :, 1] = True
    cys_mask[:, :, sg] = True
    sulfur = torch.zeros(1, 2, MAX_ATOMS_PER_RES, 3)
    sulfur[0, 0, 1, 0] = -5.0 / COORD_SCALE
    sulfur[0, 1, 1, 0] = 6.04 / COORD_SCALE
    sulfur[0, 1, sg, 0] = 1.04 / COORD_SCALE
    sulfur_result = all_atom_clash_surrogate_loss(
        sulfur, cys_type, cys_mask, cys_res_mask, res_seq_nums=seq
    )
    assert sulfur_result["hard_clashes_per_atom"] == 0.0

    met_type, met_mask, met_res_mask = _masks(["MET", "MET"])
    sd = RESIDUE_ATOMS["MET"].index("SD")
    met_mask.zero_()
    met_mask[:, :, 1] = True
    met_mask[:, :, sd] = True
    methionine = torch.zeros(1, 2, MAX_ATOMS_PER_RES, 3)
    methionine[0, 0, 1, 0] = -5.0 / COORD_SCALE
    methionine[0, 1, 1, 0] = 6.5 / COORD_SCALE
    methionine[0, 1, sd, 0] = 1.5 / COORD_SCALE
    met_result = all_atom_clash_surrogate_loss(
        methionine, met_type, met_mask, met_res_mask, res_seq_nums=seq
    )
    assert met_result["hard_clashes_per_atom"] > 0.0


def test_ost_surrogate_excludes_bonds_but_keeps_one_three_pairs():
    res_type, atom_mask, res_mask = _masks(["ALA"])
    direct = torch.zeros(1, 1, MAX_ATOMS_PER_RES, 3)
    atom_mask.zero_()
    atom_mask[..., 0] = True
    atom_mask[..., 1] = True
    direct_result = all_atom_clash_surrogate_loss(direct, res_type, atom_mask, res_mask)
    assert direct_result["loss"] == 0.0

    atom_mask.zero_()
    atom_mask[..., 0] = True
    atom_mask[..., 2] = True  # N--C is a 1-3 pair, not a direct bond.
    one_three_result = all_atom_clash_surrogate_loss(direct, res_type, atom_mask, res_mask)
    assert one_three_result["loss"] > 0.0
    assert one_three_result["hard_clashes_per_atom"] == 0.5


def test_ost_surrogate_only_excludes_the_consecutive_peptide_bond():
    res_type, atom_mask, res_mask = _masks(["ALA", "ALA"])
    seq = torch.tensor([[8, 9]])
    coords = torch.zeros(1, 2, MAX_ATOMS_PER_RES, 3)
    atom_mask.zero_()
    atom_mask[:, :, 1] = True
    atom_mask[0, 0, 2] = True  # C(i)
    atom_mask[0, 1, 0] = True  # N(i+1)
    coords[0, 0, 1, 0] = -5.0 / COORD_SCALE
    coords[0, 1, 1, 0] = 5.0 / COORD_SCALE
    peptide = all_atom_clash_surrogate_loss(coords, res_type, atom_mask, res_mask, res_seq_nums=seq)
    assert peptide["loss"] == 0.0

    atom_mask.zero_()
    atom_mask[:, :, 1] = True
    atom_mask[0, 0, 3] = True  # O(i), a peptide-link 1-3 pair.
    atom_mask[0, 1, 0] = True
    one_three = all_atom_clash_surrogate_loss(
        coords, res_type, atom_mask, res_mask, res_seq_nums=seq
    )
    assert one_three["loss"] > 0.0


def test_ost_surrogate_chunked_matches_dense_value_and_gradient():
    torch.manual_seed(41)
    residues = ["ALA", "THR", "PHE", "GLY"]
    res_type, atom_mask, res_mask = _masks(residues)
    coords = (0.2 * torch.randn(1, len(residues), MAX_ATOMS_PER_RES, 3)).requires_grad_(True)
    seq = (2 * torch.arange(len(residues))).unsqueeze(0)
    dense = all_atom_clash_surrogate_loss(
        coords,
        res_type,
        atom_mask,
        res_mask,
        res_seq_nums=seq,
        pair_chunk_size=None,
    )["loss"]
    dense_gradient = torch.autograd.grad(dense, coords, retain_graph=True)[0]
    chunked = all_atom_clash_surrogate_loss(
        coords,
        res_type,
        atom_mask,
        res_mask,
        res_seq_nums=seq,
        pair_chunk_size=2,
    )["loss"]
    chunked_gradient = torch.autograd.grad(chunked, coords)[0]
    torch.testing.assert_close(chunked, dense, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(chunked_gradient, dense_gradient, rtol=2e-6, atol=2e-6)


def test_vdw_exact_overlap_has_a_finite_nonzero_descent_direction():
    res_type, atom_mask, res_mask = _masks(["ALA", "ALA"])
    atom_mask.zero_()
    atom_mask[:, :, 1] = True
    coords = torch.zeros(1, 2, MAX_ATOMS_PER_RES, 3, requires_grad=True)
    seq = torch.tensor([[1, 3]])

    before = all_atom_vdw_clash_loss(
        coords,
        res_type,
        atom_mask,
        res_mask,
        res_seq_nums=seq,
        pair_chunk_size=1,
    )
    gradient = torch.autograd.grad(before, coords)[0]
    after = all_atom_vdw_clash_loss(
        coords.detach() - 1e-3 * gradient,
        res_type,
        atom_mask,
        res_mask,
        res_seq_nums=seq,
        pair_chunk_size=1,
    )

    assert before > 0
    assert torch.isfinite(gradient).all()
    assert gradient.norm() > 0
    assert after < before


def test_vdw_default_extent_bound_keeps_long_sidechain_collision():
    res_type, atom_mask, res_mask = _masks(["LYS", "LYS"])
    nz = RESIDUE_ATOMS["LYS"].index("NZ")
    atom_mask.zero_()
    atom_mask[:, :, 1] = True
    atom_mask[:, :, nz] = True
    coords = torch.zeros(1, 2, MAX_ATOMS_PER_RES, 3)
    coords[0, 1, 1, 0] = 1.5  # C-alpha separation is 15 Angstrom.
    coords[0, :, nz, 0] = 0.75  # The two NZ atoms coincide midway.
    seq = torch.tensor([[1, 3]])

    bounded = all_atom_vdw_clash_loss(
        coords,
        res_type,
        atom_mask,
        res_mask,
        res_seq_nums=seq,
    )
    explicitly_truncated = all_atom_vdw_clash_loss(
        coords,
        res_type,
        atom_mask,
        res_mask,
        res_seq_nums=seq,
        residue_cutoff_A=14.0,
    )

    assert bounded > 0
    assert explicitly_truncated == 0


def test_chunked_vdw_matches_dense_value_and_gradient():
    torch.manual_seed(29)
    residues = ["ALA", "THR", "PHE", "GLY", "ILE", "ASP"]
    res_type, atom_mask, res_mask = _masks(residues)
    coords = (0.25 * torch.randn(1, len(residues), MAX_ATOMS_PER_RES, 3)).requires_grad_(True)
    chain_id = torch.tensor([[0, 0, 0, 1, 1, 1]])
    seq = torch.tensor([[1, 2, 4, 1, 2, 4]])

    dense = all_atom_vdw_clash_loss(
        coords,
        res_type,
        atom_mask,
        res_mask,
        chain_id,
        seq,
        residue_cutoff_A=100.0,
        pair_chunk_size=None,
    )
    dense_gradient = torch.autograd.grad(dense, coords, retain_graph=True)[0]
    chunked = all_atom_vdw_clash_loss(
        coords,
        res_type,
        atom_mask,
        res_mask,
        chain_id,
        seq,
        residue_cutoff_A=100.0,
        pair_chunk_size=2,
    )
    chunked_gradient = torch.autograd.grad(chunked, coords)[0]

    torch.testing.assert_close(chunked, dense, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        chunked_gradient,
        dense_gradient,
        rtol=2e-6,
        atol=2e-6,
    )


def test_chunked_vdw_long_candidate_stress_is_finite():
    torch.manual_seed(31)
    length = 256
    res_type, atom_mask, res_mask = _masks(["ALA"] * length)
    coords = (0.1 * torch.randn(1, length, MAX_ATOMS_PER_RES, 3)).requires_grad_(True)
    seq = (2 * torch.arange(length)).unsqueeze(0)

    energy = all_atom_vdw_clash_loss(
        coords,
        res_type,
        atom_mask,
        res_mask,
        res_seq_nums=seq,
        residue_cutoff_A=100.0,
        pair_chunk_size=64,
    )
    gradient = torch.autograd.grad(energy, coords)[0]

    assert energy > 0
    assert torch.isfinite(energy)
    assert torch.isfinite(gradient).all()
    assert gradient.norm() > 0


def test_complete_stereochemical_decomposition_is_finite():
    torch.manual_seed(23)
    residues = ["ALA", "THR", "PHE", "GLY"]
    res_type, atom_mask, res_mask = _masks(residues)
    coords = torch.randn(1, len(residues), MAX_ATOMS_PER_RES, 3) * 0.2
    chain_id = torch.zeros(1, len(residues), dtype=torch.long)
    seq = torch.arange(1, len(residues) + 1).unsqueeze(0)

    terms = stereochemical_energy_terms(
        coords,
        res_type,
        atom_mask,
        res_mask,
        chain_id,
        seq,
    )

    assert set(terms) == {
        "covalent",
        "peptide_planarity",
        "chirality",
        "sidechain_planarity",
        "ramachandran",
        "all_atom_clash",
    }
    assert all(value.ndim == 0 and torch.isfinite(value) for value in terms.values())
