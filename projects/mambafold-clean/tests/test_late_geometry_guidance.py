from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mambafold.data.constants import AA_TO_ID, ATOM_NAME_TO_ID, COORD_SCALE, MAX_ATOMS_PER_RES
from mambafold.data.types import ProteinBatch, ProteinExample
from mambafold.sampling import LateGeometryGuidance, prepare_inference_batch, sample
from mambafold.sampling.geometry_guidance import (
    _reference_tables,
    guidance_energy,
    guide_clean_estimate,
)


def _example() -> ProteinExample:
    mask = torch.zeros(2, MAX_ATOMS_PER_RES, dtype=torch.bool)
    mask[:, :5] = True
    coords = torch.zeros(2, MAX_ATOMS_PER_RES, 3)
    coords[0, :5] = (
        torch.tensor(
            [[0.0, 0.0, 0.0], [1.45, 0.0, 0.0], [2.0, 1.3, 0.0], [1.7, 2.4, 0.0], [1.5, 0.0, 1.5]]
        )
        / COORD_SCALE
    )
    coords[1, :5] = coords[0, :5] + torch.tensor([0.3, 0.0, 0.0])
    atom_type = torch.zeros(2, MAX_ATOMS_PER_RES, dtype=torch.long)
    for slot, name in enumerate(("N", "CA", "C", "O", "CB")):
        atom_type[:, slot] = ATOM_NAME_TO_ID[name]
    return ProteinExample(
        res_type=torch.full((2,), AA_TO_ID["ALA"]),
        atom_type=atom_type,
        pair_type=torch.zeros(2, MAX_ATOMS_PER_RES, dtype=torch.long),
        coords=coords,
        atom_mask=mask,
        observed_mask=mask,
        res_seq_nums=torch.arange(2),
        seq_len=2,
    )


def _props(tmp_path: Path) -> Path:
    path = tmp_path / "stereo.txt"
    path.write_text(
        "Bond Residue Mean StdDev\n"
        "N-CA ALA 1.459 0.020\n"
        "CA-C ALA 1.525 0.026\n"
        "C-O ALA 1.229 0.019\n"
        "CA-CB ALA 1.520 0.021\n"
        "-\n"
        "Angle Residue Mean StdDev\n"
        "N-CA-C ALA 111.0 2.7\n"
        "N-CA-CB ALA 110.5 1.5\n"
        "-\n"
        "Non-bonded distance Minimum Dist Tolerance\n"
    )
    return path


def test_guidance_reduces_reference_free_energy_and_caps_movement(tmp_path: Path) -> None:
    batch = prepare_inference_batch([_example()], "cpu")
    config = LateGeometryGuidance(props_path=_props(tmp_path), every_n_steps=1, max_step_A=0.02)
    config.validate()
    tables = _reference_tables(str(config.props_path))
    before = guidance_energy(batch.x_clean, batch, config, tables)
    correction = guide_clean_estimate(batch.x_clean, batch, config, time=0.95)
    after = guidance_energy(batch.x_clean + correction, batch, config, tables)
    assert after.item() < before.item()
    assert (correction * COORD_SCALE).norm(dim=-1).max().item() <= 0.011
    assert torch.count_nonzero(correction[~batch.atom_mask]) == 0


def test_guidance_is_independent_of_batch_companions(tmp_path: Path) -> None:
    example = _example()
    config = LateGeometryGuidance(props_path=_props(tmp_path), max_step_A=0.02)
    single = prepare_inference_batch([example], "cpu")
    paired = prepare_inference_batch([example, example], "cpu")
    expected = guide_clean_estimate(single.x_clean, single, config, time=0.95)
    actual = guide_clean_estimate(paired.x_clean, paired, config, time=0.95)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[0], rtol=0, atol=0)


class _Velocity(torch.nn.Module):
    def forward(self, batch: ProteinBatch) -> dict[str, torch.Tensor]:
        return {"v_atom": -0.25 * batch.x_t, "trunk_latent": batch.res_type.float()}


def test_disabled_guidance_preserves_sde_sample(tmp_path: Path) -> None:
    batch = prepare_inference_batch([_example()], "cpu")
    base = sample(_Velocity(), batch, [7], n_steps=4, method="sde", return_trunk_latent=False)
    disabled = sample(
        _Velocity(),
        batch,
        [7],
        n_steps=4,
        method="sde",
        return_trunk_latent=False,
        geometry_guidance=LateGeometryGuidance(props_path=_props(tmp_path), max_step_A=0.0),
    )
    torch.testing.assert_close(base.final_aa, disabled.final_aa, rtol=0, atol=0)


def test_enabled_guidance_rejects_ode(tmp_path: Path) -> None:
    batch = prepare_inference_batch([_example()], "cpu")
    with pytest.raises(ValueError, match="requires SDE"):
        sample(
            _Velocity(),
            batch,
            [7],
            n_steps=2,
            method="ode",
            return_trunk_latent=False,
            geometry_guidance=LateGeometryGuidance(props_path=_props(tmp_path)),
        )
