"""Focused tests for geometry fine-tuning weights and diagnostics."""

import torch
from torch import nn

from mambafold.data.constants import AA_TO_ID, COORD_SCALE, MAX_ATOMS_PER_RES
from mambafold.data.types import ProteinBatch
from mambafold.train.engine import (
    _geometry_bundle,
    _geometry_time_weights,
    allatom_eval_step,
)


def _batch() -> ProteinBatch:
    batch_size, length, atoms = 1, 3, MAX_ATOMS_PER_RES
    res_mask = torch.ones(batch_size, length, dtype=torch.bool)
    atom_mask = torch.zeros(batch_size, length, atoms, dtype=torch.bool)
    atom_mask[:, :, 1] = True
    zeros_res = torch.zeros(batch_size, length, dtype=torch.long)
    coords = torch.zeros(batch_size, length, atoms, 3)
    coords[0, 1, 1, 0] = 10.0 / COORD_SCALE
    coords[0, 2, 1, 0] = 1.0 / COORD_SCALE
    return ProteinBatch(
        res_type=torch.full((batch_size, length), AA_TO_ID["ALA"], dtype=torch.long),
        res_seq_nums=torch.tensor([[1, 2, 3]]),
        atom_type=torch.zeros(batch_size, length, atoms, dtype=torch.long),
        pair_type=torch.zeros(batch_size, length, atoms, dtype=torch.long),
        res_mask=res_mask,
        atom_mask=atom_mask,
        valid_mask=atom_mask.clone(),
        ca_mask=res_mask.clone(),
        chain_id=zeros_res.clone(),
        entity_id=zeros_res.clone(),
        sym_id=zeros_res.clone(),
        is_nterm=torch.zeros_like(res_mask),
        is_cterm=torch.zeros_like(res_mask),
        x_clean=coords.clone(),
        x_t=coords.clone(),
        eps=torch.zeros_like(coords),
        t=torch.full((batch_size, 1, 1, 1), 0.8),
        esm=torch.zeros(batch_size, length, 1),
    )


class _ZeroVelocity(nn.Module):
    def forward(self, batch, return_aux=True):
        del return_aux
        return {
            "v_atom": torch.zeros_like(batch.x_t),
            "v_ca": torch.zeros_like(batch.x_t[..., 1, :]),
        }


def test_geometry_time_weight_cancels_reconstruction_jacobian_in_plateau():
    t = torch.tensor([0.7, 0.9]).reshape(2, 1, 1, 1)
    gate, weight = _geometry_time_weights(
        t,
        start=0.55,
        ramp_end=0.65,
        taper_start=0.95,
        end=0.98,
        jacobian_floor=0.1,
    )

    torch.testing.assert_close(gate, torch.ones_like(gate))
    torch.testing.assert_close(weight * (1.0 - t.flatten()), torch.ones_like(gate))


def test_geometry_time_weight_tapers_both_unsupported_ends():
    t = torch.tensor([0.5, 0.99]).reshape(2, 1, 1, 1)
    gate, weight = _geometry_time_weights(
        t,
        start=0.55,
        ramp_end=0.65,
        taper_start=0.95,
        end=0.98,
        jacobian_floor=0.1,
    )

    assert torch.equal(gate, torch.zeros_like(gate))
    assert torch.equal(weight, torch.zeros_like(weight))


def test_geometry_bundle_preserves_partial_gate_strength():
    batch = _batch()

    def objective_at(t: float) -> torch.Tensor:
        batch.t.fill_(t)
        return _geometry_bundle(
            batch.x_t,
            batch,
            compute_ost=True,
            compute_covalent=False,
            compute_planarity=False,
            ost_clash_mode="huber",
            ost_clash_margin_A=0.1,
            ost_clash_huber_A=0.25,
            ost_clash_softplus_tau_A=0.05,
            ost_clash_softplus_halo=6.0,
            ost_clash_pair_chunk_size=32,
            covalent_guard_tolerance_z=3.0,
            geo_t_start=0.55,
            geo_t_ramp_end=0.65,
            geo_t_taper_start=0.95,
            geo_t_end=0.98,
            geo_jacobian_floor=0.1,
            geo_max_examples_per_batch=1,
            use_time_weighting=True,
        )["ost_objective"]

    ramp_objective = objective_at(0.60)
    plateau_objective = objective_at(0.65)

    # gate(0.60)=0.5 and gate(0.65)=1.0.  For one selected example the
    # objective ratio retains that taper after inverse-Jacobian weighting:
    # (0.5 / 0.4) / (1.0 / 0.35) = 0.4375.
    torch.testing.assert_close(
        ramp_objective / plateau_objective,
        torch.tensor(0.4375),
    )


def test_eval_reports_real_geometry_instead_of_weight_zero_placeholders():
    metrics = allatom_eval_step(
        _ZeroVelocity(),
        _batch(),
        use_amp=False,
        max_lddt_atoms=32,
        max_clash_atoms=32,
        geo_max_examples_per_batch=1,
    )

    assert metrics["ost_hard_per_1k"] > 0.0
    assert metrics["ost_clash"] > 0.0
    assert metrics["clash"] > 0.0
