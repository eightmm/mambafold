from __future__ import annotations

import torch

from mambafold.data.constants import (
    AA_TO_ID,
    ATOM_NAME_TO_ID,
    COORD_SCALE,
    MAX_ATOMS_PER_RES,
    RESIDUE_ATOMS,
)
from mambafold.losses.geometry import stereochemical_losses
from mambafold.train.config import parse_args


def _two_alanines() -> dict[str, torch.Tensor]:
    atoms = MAX_ATOMS_PER_RES
    # A non-degenerate local frame is enough because bond/angle targets come
    # from the supplied reference rather than an ideal-residue library.
    ala_A = torch.tensor(
        [
            [-1.45, 0.00, 0.00],  # N
            [0.00, 0.00, 0.00],  # CA
            [1.52, 0.00, 0.00],  # C
            [2.15, 1.05, 0.00],  # O
            [0.00, 1.53, 0.00],  # CB
        ]
    )
    true_A = torch.zeros(1, 2, atoms, 3)
    true_A[0, 0, :5] = ala_A
    true_A[0, 1, :5] = ala_A + torch.tensor([12.0, 0.0, 0.0])
    valid = torch.zeros(1, 2, atoms, dtype=torch.bool)
    valid[..., :5] = True
    atom_type = torch.full((1, 2, atoms), ATOM_NAME_TO_ID["PAD"], dtype=torch.long)
    for slot, atom_name in enumerate(RESIDUE_ATOMS["ALA"]):
        atom_type[..., slot] = ATOM_NAME_TO_ID[atom_name]
    return {
        "true": true_A / COORD_SCALE,
        "valid": valid,
        "atom_mask": valid.clone(),
        "res_mask": torch.ones(1, 2, dtype=torch.bool),
        "res_type": torch.full((1, 2), AA_TO_ID["ALA"], dtype=torch.long),
        "atom_type": atom_type,
        "res_seq_nums": torch.zeros(1, 2, dtype=torch.long),
        # Separate chains keep the rigidly translated residue out of the
        # peptide-bond loss while leaving it eligible for non-bonded clashes.
        "chain_id": torch.tensor([[0, 1]], dtype=torch.long),
    }


def _losses(pred: torch.Tensor, example: dict[str, torch.Tensor], chunk: int = 2):
    return stereochemical_losses(
        pred,
        example["true"],
        example["valid"],
        example["atom_mask"],
        example["res_mask"],
        example["res_type"],
        example["atom_type"],
        example["res_seq_nums"],
        example["chain_id"],
        clash_pair_chunk_size=chunk,
    )


def test_reference_geometry_has_zero_losses() -> None:
    example = _two_alanines()
    losses = _losses(example["true"].clone(), example)
    assert losses["bond"].item() == 0.0
    assert losses["angle"].item() == 0.0
    assert losses["clash"].item() == 0.0
    assert losses["clashes_per_1000_atoms"].item() == 0.0


def test_rigid_residue_overlap_is_a_clash_not_a_bond_or_angle_error() -> None:
    example = _two_alanines()
    pred = example["true"].clone()
    pred[:, 1] = pred[:, 0] + torch.tensor([0.2 / COORD_SCALE, 0.0, 0.0])
    losses = _losses(pred, example)
    assert losses["bond"].item() < 1e-10
    assert losses["angle"].item() < 1e-10
    assert losses["clash"].item() > 0.0
    assert losses["clashes_per_1000_atoms"].item() > 0.0


def test_clash_chunking_preserves_value_and_gradient() -> None:
    example = _two_alanines()
    base = example["true"].clone()
    base[:, 1] = base[:, 0] + torch.tensor([0.2 / COORD_SCALE, 0.0, 0.0])
    results = []
    gradients = []
    for chunk in (1, 32):
        pred = base.clone().requires_grad_(True)
        loss = _losses(pred, example, chunk=chunk)["clash"].sum()
        loss.backward()
        results.append(loss.detach())
        gradients.append(pred.grad.detach())
    torch.testing.assert_close(results[0], results[1])
    torch.testing.assert_close(gradients[0], gradients[1])
    assert torch.isfinite(gradients[0]).all()
    assert gradients[0].abs().sum() > 0


def test_only_direct_1_2_bonds_are_excluded_from_intra_residue_clashes() -> None:
    example = _two_alanines()
    example["res_mask"][:, 1] = False
    results = {}
    for label, slots in {"1-2": (0, 1), "1-3": (0, 2), "1-4": (0, 3)}.items():
        atom_mask = torch.zeros_like(example["atom_mask"])
        atom_mask[0, 0, list(slots)] = True
        example["atom_mask"] = atom_mask
        example["valid"] = atom_mask.clone()
        pred = example["true"].clone()
        pred[0, 0, slots[1]] = pred[0, 0, slots[0]]
        results[label] = _losses(pred, example)["clash"].item()

    assert results["1-2"] == 0.0
    assert results["1-3"] > 0.0
    assert results["1-4"] > 0.0


def test_sequential_peptide_bond_is_excluded_but_its_1_3_pair_is_not() -> None:
    example = _two_alanines()
    example["chain_id"][:] = 0
    example["res_seq_nums"][:] = torch.tensor([[1, 2]])

    direct_mask = torch.zeros_like(example["atom_mask"])
    direct_mask[0, 0, [1, 2]] = True  # CA(i), C(i)
    direct_mask[0, 1, [0, 1]] = True  # N(i+1), CA(i+1)
    example["atom_mask"] = direct_mask
    example["valid"] = direct_mask.clone()
    direct = example["true"].clone()
    direct[0, 0, 1] = torch.tensor([-10.0, 0.0, 0.0]) / COORD_SCALE
    direct[0, 0, 2] = 0.0
    direct[0, 1, 0] = 0.0
    direct[0, 1, 1] = torch.tensor([10.0, 0.0, 0.0]) / COORD_SCALE
    assert _losses(direct, example)["clash"].item() == 0.0

    one_three_mask = direct_mask.clone()
    one_three_mask[0, 0, 2] = False
    one_three_mask[0, 0, 3] = True  # O(i)-N(i+1) spans two covalent edges.
    example["atom_mask"] = one_three_mask
    example["valid"] = one_three_mask.clone()
    one_three = direct.clone()
    one_three[0, 0, 3] = 0.0
    assert _losses(one_three, example)["clash"].item() > 0.0


def test_clash_uses_canonical_atoms_even_when_reference_atom_is_unresolved() -> None:
    example = _two_alanines()
    atom_mask = torch.zeros_like(example["atom_mask"])
    atom_mask[..., 1] = True
    example["atom_mask"] = atom_mask
    example["valid"] = torch.zeros_like(atom_mask)
    pred = example["true"].clone()
    pred[0, 1, 1] = pred[0, 0, 1]
    assert _losses(pred, example)["clash"].item() > 0.0


def test_clash_hard_count_and_margin_have_distinct_thresholds() -> None:
    example = _two_alanines()
    atom_mask = torch.zeros_like(example["atom_mask"])
    atom_mask[..., 1] = True
    example["atom_mask"] = atom_mask
    example["valid"] = atom_mask.clone()

    inside = example["true"].clone()
    inside[0, 0, 1] = 0.0
    inside[0, 1, 1] = torch.tensor([1.89, 0.0, 0.0]) / COORD_SCALE
    inside_losses = _losses(inside, example)
    assert inside_losses["clashes_per_1000_atoms"].item() == 500.0

    margin_only = inside.clone()
    margin_only[0, 1, 1, 0] = 1.91 / COORD_SCALE
    margin_losses = _losses(margin_only, example)
    assert margin_losses["clashes_per_1000_atoms"].item() == 0.0
    assert margin_losses["clash"].item() > 0.0


def test_sulfur_pair_uses_openstructure_special_floor() -> None:
    example = _two_alanines()
    atom_mask = torch.zeros_like(example["atom_mask"])
    atom_mask[..., 1] = True
    example["atom_mask"] = atom_mask
    example["valid"] = atom_mask.clone()
    example["atom_type"][..., 1] = ATOM_NAME_TO_ID["SD"]
    pred = example["true"].clone()
    pred[0, 0, 1] = 0.0
    pred[0, 1, 1] = torch.tensor([1.5, 0.0, 0.0]) / COORD_SCALE
    losses = _losses(pred, example)
    assert losses["clash"].item() == 0.0
    assert losses["clashes_per_1000_atoms"].item() == 0.0


def test_exact_atom_overlap_has_a_finite_nonzero_descent_gradient() -> None:
    example = _two_alanines()
    atom_mask = torch.zeros_like(example["atom_mask"])
    atom_mask[..., 1] = True
    example["atom_mask"] = atom_mask
    example["valid"] = atom_mask.clone()
    pred = example["true"].clone()
    pred[0, 1, 1] = pred[0, 0, 1]
    pred.requires_grad_(True)
    _losses(pred, example)["clash"].sum().backward()
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0.0


def test_bond_and_angle_respond_to_local_distortion() -> None:
    example = _two_alanines()
    pred = example["true"].clone()
    pred[0, 0, 1, 1] += 0.5 / COORD_SCALE
    losses = _losses(pred, example)
    assert losses["bond"].item() > 0.0
    assert losses["angle"].item() > 0.0
    assert losses["bond_mae_A"].item() > 0.0
    assert losses["angle_mae_deg"].item() > 0.0


def test_geometry_config_defaults_off_and_accepts_finetune_weights() -> None:
    defaults, _ = parse_args([])
    assert (defaults.w_bond, defaults.w_angle, defaults.w_clash) == (0.0, 0.0, 0.0)
    configured, _ = parse_args(
        [
            "--w_bond",
            "0.5",
            "--w_angle",
            "0.25",
            "--w_clash",
            "2.0",
            "--clash_pair_chunk_size",
            "64",
        ]
    )
    assert (configured.w_bond, configured.w_angle, configured.w_clash) == (0.5, 0.25, 2.0)
    assert configured.clash_pair_chunk_size == 64
    assert defaults.clash_overlap_tolerance_A == 1.5
