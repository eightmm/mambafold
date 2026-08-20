"""Tests for inference-time sampling behavior."""

from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import mambafold.sampling.samplers as sampler_module
from benchmarks.run_inference import make_sampling_batch, prepare_static_batch
from mambafold.data.constants import AA_TO_ID
from mambafold.losses.geometry import bond_length_loss
from mambafold.losses.stereochemistry import all_atom_vdw_clash_loss
from mambafold.sampling.samplers import (
    GeometryGuidanceConfig,
    _cap_guidance_step,
    _geometry_guidance_gradient,
    _inference_autocast_dtype,
    _nonlocal_ca_segment_guidance,
    _nonlocal_ca_steric_energy,
    _project_adjacent_bond_axis,
    _smooth_residue_vectors,
    _steric_guidance_gradient,
    _vdw_guidance_gradient,
    sample,
)


@dataclass
class _Batch:
    x_t: torch.Tensor
    t: torch.Tensor
    x_self_cond: torch.Tensor | None = None


class _Model:
    def eval(self):
        return self

    def __call__(self, batch, *, return_aux=False):
        del return_aux
        batch_size, length = batch.x_t.shape[:2]
        return {
            "v_atom": 0.1 * batch.x_t,
            "conf": torch.full((batch_size, length), 0.75, device=batch.x_t.device),
        }


class _ZeroModel(_Model):
    def __call__(self, batch, *, return_aux=False):
        del return_aux
        batch_size, length = batch.x_t.shape[:2]
        return {
            "v_atom": torch.zeros_like(batch.x_t),
            "conf": torch.full((batch_size, length), 0.5, device=batch.x_t.device),
        }


@dataclass
class _GuidanceBatch:
    x_t: torch.Tensor
    t: torch.Tensor
    res_type: torch.Tensor
    res_seq_nums: torch.Tensor
    atom_mask: torch.Tensor
    res_mask: torch.Tensor
    chain_id: torch.Tensor
    x_self_cond: torch.Tensor | None = None


def _batch_fn(x, t_cur):
    return _Batch(
        x_t=x.unsqueeze(0),
        t=torch.tensor([[[[t_cur]]]], dtype=x.dtype, device=x.device),
    )


def _guidance_fixture(length=6, atoms=5):
    atom_mask = torch.ones(length, atoms, dtype=torch.bool)
    example = SimpleNamespace(atom_mask=atom_mask)

    def batch_fn(x, t_cur):
        return _GuidanceBatch(
            x_t=x.unsqueeze(0),
            t=torch.tensor([[[[t_cur]]]], dtype=x.dtype, device=x.device),
            res_type=torch.full((1, length), AA_TO_ID["ALA"], dtype=torch.long, device=x.device),
            res_seq_nums=torch.arange(length, device=x.device).unsqueeze(0),
            atom_mask=atom_mask.to(x.device).unsqueeze(0),
            res_mask=torch.ones(1, length, dtype=torch.bool, device=x.device),
            chain_id=torch.zeros(1, length, dtype=torch.long, device=x.device),
        )

    return example, batch_fn


def test_disabling_trajectory_preserves_sample_outputs():
    example = SimpleNamespace(atom_mask=torch.ones(3, 2, dtype=torch.bool))
    kwargs = {
        "n_steps": 4,
        "seed": 7,
        "device": "cpu",
        "sampler": "sde",
        "sde_log_timesteps": True,
    }

    with_trajectory = sample(_Model(), example, _batch_fn, record_trajectory=True, **kwargs)
    without_trajectory = sample(_Model(), example, _batch_fn, record_trajectory=False, **kwargs)
    zero_guidance = sample(
        _Model(),
        example,
        _batch_fn,
        record_trajectory=False,
        geometry_guidance=GeometryGuidanceConfig(scale=0.0),
        **kwargs,
    )
    zero_split_guidance = sample(
        _Model(),
        example,
        _batch_fn,
        record_trajectory=False,
        geometry_guidance=GeometryGuidanceConfig.self_avoidance(
            local_scale=0.0,
            steric_scale=0.0,
        ),
        **kwargs,
    )

    np.testing.assert_array_equal(with_trajectory[0], without_trajectory[0])
    np.testing.assert_array_equal(with_trajectory[1], without_trajectory[1])
    np.testing.assert_array_equal(with_trajectory[3], without_trajectory[3])
    np.testing.assert_array_equal(with_trajectory[4], without_trajectory[4])
    np.testing.assert_array_equal(without_trajectory[0], zero_guidance[0])
    np.testing.assert_array_equal(without_trajectory[1], zero_guidance[1])
    np.testing.assert_array_equal(without_trajectory[4], zero_guidance[4])
    np.testing.assert_array_equal(without_trajectory[0], zero_split_guidance[0])
    np.testing.assert_array_equal(without_trajectory[1], zero_split_guidance[1])
    assert with_trajectory[2].shape == (4, 3, 3)
    assert without_trajectory[2].shape == (0, 3, 3)


def test_inference_autocast_uses_fp16_only_when_cuda_lacks_bf16(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert _inference_autocast_dtype("cuda") == torch.float16
    assert _inference_autocast_dtype("cpu") == torch.bfloat16

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert _inference_autocast_dtype("cuda:0") == torch.bfloat16


def test_sampling_batch_reuses_static_target_tensors():
    length, atoms, esm_dim = 3, 2, 4
    example = SimpleNamespace(
        seq_len=length,
        res_type=torch.arange(length),
        res_seq_nums=torch.arange(length),
        atom_type=torch.zeros(length, atoms, dtype=torch.long),
        pair_type=torch.zeros(length, atoms, dtype=torch.long),
        coords=torch.zeros(length, atoms, 3),
        atom_mask=torch.ones(length, atoms, dtype=torch.bool),
        observed_mask=torch.ones(length, atoms, dtype=torch.bool),
        chain_id=torch.zeros(length, dtype=torch.long),
        entity_id=torch.zeros(length, dtype=torch.long),
        sym_id=torch.zeros(length, dtype=torch.long),
        is_nterm=torch.tensor([True, False, False]),
        is_cterm=torch.tensor([False, False, True]),
        esm=torch.zeros(length, esm_dim, dtype=torch.float16),
    )
    static_batch = prepare_static_batch(example, "cpu")
    coordinates = torch.randn(length, atoms, 3)

    sampling_batch = make_sampling_batch(static_batch, coordinates, 0.25)

    assert sampling_batch.res_type is static_batch.res_type
    assert sampling_batch.atom_mask is static_batch.atom_mask
    assert sampling_batch.esm is static_batch.esm
    assert sampling_batch.x_t.data_ptr() == coordinates.data_ptr()
    assert sampling_batch.t.item() == 0.25
    assert static_batch.t.item() == 0.0


def test_geometry_guidance_gradient_lowers_gt_free_bond_energy():
    example, batch_fn = _guidance_fixture()
    coords = torch.randn(6, 5, 3)
    batch = batch_fn(coords, 0.75)
    config = GeometryGuidanceConfig(
        scale=0.1,
        start=0.0,
        bond_weight=1.0,
        angle_weight=0.0,
        clash_weight=0.0,
    )

    grad = _geometry_guidance_gradient(coords, batch, config)

    def energy(value):
        return bond_length_loss(
            value.unsqueeze(0),
            batch.res_type,
            batch.atom_mask,
            batch.res_mask,
            chain_id=batch.chain_id,
            res_seq_nums=batch.res_seq_nums,
        )

    assert energy(coords - 1e-3 * grad) < energy(coords)
    assert not hasattr(example, "coords")  # guidance did not require ground truth


def test_independent_vdw_gradient_lowers_severe_overlap_energy():
    _, batch_fn = _guidance_fixture(length=3, atoms=15)
    atom_offsets = torch.zeros(15, 3)
    atom_offsets[:5] = torch.tensor(
        [
            [-0.10, 0.00, 0.00],
            [0.00, 0.00, 0.00],
            [0.10, 0.00, 0.00],
            [0.15, 0.05, 0.00],
            [0.00, 0.10, 0.00],
        ]
    )
    coords = atom_offsets.repeat(3, 1, 1)
    coords[1, :, 0] += 2.0
    batch = batch_fn(coords, 0.75)
    batch.atom_mask[..., 5:] = False
    config = GeometryGuidanceConfig(
        vdw_scale=0.1,
        vdw_start=0.0,
        vdw_every_n_steps=1,
        vdw_overlap_tolerance_A=1.5,
    )

    def energy(value):
        return all_atom_vdw_clash_loss(
            value.unsqueeze(0),
            batch.res_type,
            batch.atom_mask,
            batch.res_mask,
            chain_id=batch.chain_id,
            res_seq_nums=batch.res_seq_nums,
            overlap_tolerance_A=1.5,
        )

    grad = _vdw_guidance_gradient(coords, batch, config)

    assert torch.isfinite(grad).all()
    assert energy(coords - 1e-3 * grad) < energy(coords)


def test_independent_vdw_channel_uses_its_own_interval(monkeypatch):
    example, batch_fn = _guidance_fixture()
    calls = []

    def fake_vdw_gradient(clean_estimate, batch, config):
        del batch, config
        calls.append(clean_estimate.clone())
        gradient = torch.zeros_like(clean_estimate)
        gradient[0, :, 0] = 1.0
        gradient[-1, :, 0] = -1.0
        return gradient

    monkeypatch.setattr(sampler_module, "_vdw_guidance_gradient", fake_vdw_gradient)
    config = GeometryGuidanceConfig(
        scale=0.0,
        vdw_scale=0.1,
        vdw_start=0.0,
        vdw_every_n_steps=2,
    )

    result = sample(
        _ZeroModel(),
        example,
        batch_fn,
        n_steps=6,
        seed=17,
        device="cpu",
        sampler="sde",
        sde_log_timesteps=False,
        record_trajectory=False,
        geometry_guidance=config,
    )

    assert len(calls) == 3
    assert np.isfinite(result[1]).all()


@pytest.mark.parametrize("solver", ["ode", "sde"])
def test_geometry_guidance_runs_for_both_solvers(solver):
    example, batch_fn = _guidance_fixture()
    config = GeometryGuidanceConfig(
        scale=0.01,
        start=0.0,
        bond_weight=1.0,
        angle_weight=0.0,
        clash_weight=0.0,
    )
    common = {
        "n_steps": 4,
        "seed": 9,
        "device": "cpu",
        "sampler": solver,
        "sde_log_timesteps": False,
        "record_trajectory": False,
    }

    baseline = sample(_ZeroModel(), example, batch_fn, **common)
    guided = sample(_ZeroModel(), example, batch_fn, geometry_guidance=config, **common)

    assert np.isfinite(guided[1]).all()
    assert not np.array_equal(baseline[1], guided[1])


def test_stereochemical_guidance_preset_runs_in_sde_sampler():
    example, batch_fn = _guidance_fixture(atoms=15)
    config = GeometryGuidanceConfig.stereochemical(
        scale=0.005,
        start=0.0,
        every_n_steps=1,
    )

    result = sample(
        _ZeroModel(),
        example,
        batch_fn,
        n_steps=3,
        seed=13,
        device="cpu",
        sampler="sde",
        sde_log_timesteps=False,
        record_trajectory=False,
        geometry_guidance=config,
    )

    assert np.isfinite(result[1]).all()
    assert result[1].shape == (6, 15, 3)
    assert config.all_atom_clash_weight == 0.2


def test_steric_guidance_separates_exact_nonlocal_ca_overlap_coherently():
    length, atoms = 18, 5
    _, batch_fn = _guidance_fixture(length=length, atoms=atoms)
    coords = torch.zeros(length, atoms, 3)
    for residue in range(length):
        coords[residue, :, 0] = residue * 0.5
        coords[residue, :, 1] = torch.arange(atoms) * 0.01
    # Exact C-alpha superposition of sequence-distant residues exercises the
    # deterministic fallback direction rather than norm's zero derivative.
    coords[-1, :, 0] = coords[0, :, 0]
    batch = batch_fn(coords, 0.5)
    config = GeometryGuidanceConfig.self_avoidance(
        local_scale=0.0,
        steric_scale=0.1,
        steric_smoothing_radius=4,
    )

    before, _ = _nonlocal_ca_steric_energy(
        coords.unsqueeze(0),
        batch.res_mask,
        batch.chain_id,
        batch.res_seq_nums,
        min_dist_A=config.steric_ca_min_dist_A,
        seq_sep=config.steric_ca_seq_sep,
    )
    grad, severity = _steric_guidance_gradient(coords, batch, config)
    moved = coords - 1e-3 * grad
    after, _ = _nonlocal_ca_steric_energy(
        moved.unsqueeze(0),
        batch.res_mask,
        batch.chain_id,
        batch.res_seq_nums,
        min_dist_A=config.steric_ca_min_dist_A,
        seq_sep=config.steric_ca_seq_sep,
    )

    assert severity > 0
    assert torch.isfinite(grad).all()
    assert after < before
    # Every atom in a residue receives the same translation, preserving all
    # intra-residue vectors exactly for the pure coarse steric channel.
    torch.testing.assert_close(grad[:, 1:], grad[:, :1].expand_as(grad[:, 1:]))
    torch.testing.assert_close(
        moved[:, 1:] - moved[:, :1],
        coords[:, 1:] - coords[:, :1],
    )


def test_segment_guidance_separates_exact_interior_crossing():
    length, atoms = 18, 5
    coords = torch.zeros(1, length, atoms, 3)
    ca_mask = torch.zeros(1, length, dtype=torch.bool)
    ca_mask[0, [0, 1, 16, 17]] = True
    coords[0, 0, :, :2] = torch.tensor([-0.2, 0.0])
    coords[0, 1, :, :2] = torch.tensor([0.2, 0.0])
    coords[0, 16, :, :2] = torch.tensor([0.0, -0.2])
    coords[0, 17, :, :2] = torch.tensor([0.0, 0.2])
    chain = torch.zeros(1, length, dtype=torch.long)
    seq = torch.arange(1, length + 1).unsqueeze(0)

    gradient, before, severity = _nonlocal_ca_segment_guidance(
        coords,
        ca_mask,
        chain,
        seq,
        min_dist_A=2.5,
        max_edge_A=6.0,
        seq_sep=12,
        pair_chunk_size=1,
    )
    unfiltered = _nonlocal_ca_segment_guidance(
        coords,
        ca_mask,
        chain,
        seq,
        min_dist_A=2.5,
        max_edge_A=6.0,
        seq_sep=12,
        pair_chunk_size=1,
        spatial_prefilter=False,
    )
    moved = coords.clone()
    moved[:, :, 1, :] -= 1e-3 * gradient
    _, after, _ = _nonlocal_ca_segment_guidance(
        moved,
        ca_mask,
        chain,
        seq,
        min_dist_A=2.5,
        max_edge_A=6.0,
        seq_sep=12,
        pair_chunk_size=7,
    )

    assert before > 0
    assert severity > 0
    assert torch.isfinite(gradient).all()
    assert gradient.norm() > 0
    assert after < before
    for observed, expected in zip((gradient, before, severity), unfiltered):
        torch.testing.assert_close(observed, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        gradient.sum(dim=1),
        torch.zeros_like(gradient[:, 0]),
        atol=1e-6,
        rtol=0,
    )


def test_full_steric_pipeline_lowers_segment_energy_after_smoothing_and_projection():
    length, atoms = 18, 5
    _, batch_fn = _guidance_fixture(length=length, atoms=atoms)
    coords = torch.zeros(length, atoms, 3)
    for residue in range(length):
        coords[residue, :, 0] = 5.0 + residue
    coords[0, :, :2] = torch.tensor([-0.2, 0.0])
    coords[1, :, :2] = torch.tensor([0.2, 0.0])
    coords[16, :, :2] = torch.tensor([0.0, -0.2])
    coords[17, :, :2] = torch.tensor([0.0, 0.2])
    batch = batch_fn(coords, 0.5)
    config = replace(
        GeometryGuidanceConfig.self_avoidance(
            local_scale=0.0,
            steric_scale=1.0,
            steric_smoothing_radius=4,
        ),
        steric_ca_min_dist_A=0.01,
        steric_segment_weight=0.5,
        steric_segment_every_n_steps=2,
    )
    kwargs = {
        "min_dist_A": config.steric_segment_min_dist_A,
        "max_edge_A": config.steric_segment_max_edge_A,
        "seq_sep": config.steric_ca_seq_sep,
        "pair_chunk_size": config.steric_segment_pair_chunk_size,
    }

    _, before, _ = _nonlocal_ca_segment_guidance(
        coords.unsqueeze(0),
        batch.atom_mask[..., 1] & batch.res_mask,
        batch.chain_id,
        batch.res_seq_nums,
        **kwargs,
    )
    gradient, severity = _steric_guidance_gradient(coords, batch, config)
    moved = coords - 1e-3 * gradient
    _, after, _ = _nonlocal_ca_segment_guidance(
        moved.unsqueeze(0),
        batch.atom_mask[..., 1] & batch.res_mask,
        batch.chain_id,
        batch.res_seq_nums,
        **kwargs,
    )

    assert before > 0
    assert severity > 0
    assert torch.isfinite(gradient).all()
    assert after < before


def test_segment_guidance_respects_chain_continuity_and_is_chunk_invariant():
    torch.manual_seed(37)
    length = 40
    coords = torch.randn(1, length, 5, 3) * 0.2
    ca_mask = torch.ones(1, length, dtype=torch.bool)
    chain = torch.zeros(1, length, dtype=torch.long)
    seq = torch.arange(1, length + 1).unsqueeze(0)
    seq[0, 20:] += 5  # no segment may bridge this sequence gap

    small = _nonlocal_ca_segment_guidance(
        coords,
        ca_mask,
        chain,
        seq,
        min_dist_A=2.5,
        max_edge_A=6.0,
        seq_sep=12,
        pair_chunk_size=1,
    )
    large = _nonlocal_ca_segment_guidance(
        coords,
        ca_mask,
        chain,
        seq,
        min_dist_A=2.5,
        max_edge_A=6.0,
        seq_sep=12,
        pair_chunk_size=128,
    )

    for observed, expected in zip(small, large):
        torch.testing.assert_close(observed, expected, rtol=1e-6, atol=1e-6)


def test_segment_spatial_prefilter_matches_unfiltered_random_geometry():
    generator = torch.Generator().manual_seed(391)
    length = 44
    coords = torch.zeros(1, length, 5, 3)
    coords[:, :, 1] = torch.rand(1, length, 3, generator=generator) * 0.3 - 0.15
    ca_mask = torch.ones(1, length, dtype=torch.bool)
    chain = torch.zeros(1, length, dtype=torch.long)
    seq = torch.arange(1, length + 1).unsqueeze(0)
    kwargs = {
        "min_dist_A": 2.5,
        "max_edge_A": 6.0,
        "seq_sep": 12,
        "pair_chunk_size": 23,
    }

    filtered = _nonlocal_ca_segment_guidance(
        coords, ca_mask, chain, seq, spatial_prefilter=True, **kwargs
    )
    unfiltered = _nonlocal_ca_segment_guidance(
        coords, ca_mask, chain, seq, spatial_prefilter=False, **kwargs
    )

    for observed, expected in zip(filtered, unfiltered):
        torch.testing.assert_close(observed, expected, rtol=1e-6, atol=1e-6)


def test_segment_spatial_prefilter_matches_parallel_degenerate_and_gapped_edges():
    length = 48
    coords = torch.zeros(1, length, 5, 3)
    ca_mask = torch.zeros(1, length, dtype=torch.bool)
    chain = torch.zeros(1, length, dtype=torch.long)
    seq = torch.arange(1, length + 1).unsqueeze(0)

    # A degenerate segment near a parallel segment exercises zero-length
    # handling; a distant segment exercises the bounding-sphere rejection.
    ca_mask[0, [0, 1, 16, 17, 32, 33, 40, 41]] = True
    coords[0, 0:2, 1] = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    coords[0, 16:18, 1] = torch.tensor([[-0.2, 0.1, 0.0], [0.2, 0.1, 0.0]])
    coords[0, 32:34, 1] = torch.tensor([[5.0, 0.0, 0.0], [5.3, 0.0, 0.0]])
    coords[0, 40:42, 1] = torch.tensor([[-0.2, -0.2, 0.0], [0.2, -0.2, 0.0]])
    seq[0, 41] += 3  # Mask the last geometric edge through a numbering gap.
    kwargs = {
        "min_dist_A": 2.5,
        "max_edge_A": 6.0,
        "seq_sep": 12,
        "pair_chunk_size": 2,
    }

    filtered = _nonlocal_ca_segment_guidance(
        coords, ca_mask, chain, seq, spatial_prefilter=True, **kwargs
    )
    unfiltered = _nonlocal_ca_segment_guidance(
        coords, ca_mask, chain, seq, spatial_prefilter=False, **kwargs
    )

    assert filtered[1] > 0
    for observed, expected in zip(filtered, unfiltered):
        torch.testing.assert_close(observed, expected, rtol=1e-6, atol=1e-6)


def test_segment_spatial_prefilter_removes_far_pairs_without_changing_result(monkeypatch):
    length = 64
    coords = torch.zeros(1, length, 5, 3)
    coords[0, :, 1, 0] = torch.arange(length) * 0.38
    coords[0, 0, 1, :2] = torch.tensor([-0.2, 0.0])
    coords[0, 1, 1, :2] = torch.tensor([0.2, 0.0])
    # Keep the final edge short but make its predecessor implausibly long, so
    # it crosses the first edge without introducing an artificial bridge.
    coords[0, -2, 1, :2] = torch.tensor([0.0, -0.2])
    coords[0, -1, 1, :2] = torch.tensor([0.0, 0.2])
    ca_mask = torch.ones(1, length, dtype=torch.bool)
    chain = torch.zeros(1, length, dtype=torch.long)
    seq = torch.arange(1, length + 1).unsqueeze(0)
    kwargs = {
        "min_dist_A": 2.5,
        "max_edge_A": 6.0,
        "seq_sep": 12,
        "pair_chunk_size": 17,
    }

    original = sampler_module._closest_segment_parameters
    processed = {"count": 0}

    def counted_closest(p0, p1, q0, q1):
        processed["count"] += len(p0)
        return original(p0, p1, q0, q1)

    monkeypatch.setattr(sampler_module, "_closest_segment_parameters", counted_closest)
    unfiltered = _nonlocal_ca_segment_guidance(
        coords, ca_mask, chain, seq, spatial_prefilter=False, **kwargs
    )
    unfiltered_count = processed["count"]
    processed["count"] = 0
    filtered = _nonlocal_ca_segment_guidance(
        coords, ca_mask, chain, seq, spatial_prefilter=True, **kwargs
    )

    assert 0 < processed["count"] < unfiltered_count
    assert filtered[1] > 0
    for observed, expected in zip(filtered, unfiltered):
        torch.testing.assert_close(observed, expected, rtol=1e-6, atol=1e-6)


def test_segment_guidance_rejects_numbered_but_implausibly_long_edge():
    coords = torch.zeros(1, 16, 5, 3)
    coords[0, 1, :, 0] = 4.2  # 42 A in output units despite consecutive numbering
    ca_mask = torch.zeros(1, 16, dtype=torch.bool)
    ca_mask[0, [0, 1, 14, 15]] = True
    chain = torch.zeros(1, 16, dtype=torch.long)
    seq = torch.arange(1, 17).unsqueeze(0)

    gradient, energy, rms = _nonlocal_ca_segment_guidance(
        coords,
        ca_mask,
        chain,
        seq,
        min_dist_A=2.5,
        max_edge_A=6.0,
        seq_sep=12,
        pair_chunk_size=4,
    )

    assert energy == 0
    assert rms == 0
    assert torch.count_nonzero(gradient) == 0


def test_steric_force_smoothing_stops_at_chain_boundary_and_sequence_gap():
    vectors = torch.zeros(1, 10, 3)
    vectors[0, 4, 0] = 1.0
    mask = torch.ones(1, 10, dtype=torch.bool)
    chain_id = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])
    seq = torch.tensor([[1, 2, 3, 4, 5, 1, 2, 3, 4, 5]])

    smoothed = _smooth_residue_vectors(
        vectors,
        mask,
        chain_id,
        seq,
        radius=2,
        sigma=1.0,
    )

    assert smoothed[0, 2:5, 0].gt(0).all()
    assert torch.count_nonzero(smoothed[0, 5:]) == 0

    same_chain = torch.zeros_like(chain_id)
    gapped_seq = torch.tensor([[1, 2, 3, 4, 5, 10, 11, 12, 13, 14]])
    gap_smoothed = _smooth_residue_vectors(
        vectors,
        mask,
        same_chain,
        gapped_seq,
        radius=2,
        sigma=1.0,
    )
    assert torch.count_nonzero(gap_smoothed[0, 5:]) == 0


def test_self_avoidance_factory_accepts_explicit_start_and_ramp_end():
    config = GeometryGuidanceConfig.self_avoidance(
        local_scale=0.03,
        steric_scale=0.1,
        steric_start=0.6,
        steric_ramp_end=0.8,
    )
    config.validate()

    with pytest.raises(ValueError, match="steric start/ramp end"):
        replace(config, steric_ramp_end=0.5).validate()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("vdw_scale", -0.1, "VDW guidance scale"),
        ("vdw_start", 1.0, "VDW guidance start"),
        ("vdw_every_n_steps", 0, "VDW guidance interval"),
        ("vdw_overlap_tolerance_A", 0.0, "VDW overlap tolerance"),
        ("vdw_max_step_A", 0.0, "VDW maximum step"),
    ],
)
def test_independent_vdw_config_validation(field, value, message):
    with pytest.raises(ValueError, match=message):
        replace(GeometryGuidanceConfig(), **{field: value}).validate()


def test_steric_only_config_changes_sampling_and_stays_finite():
    example, batch_fn = _guidance_fixture(length=8, atoms=5)
    config = replace(
        GeometryGuidanceConfig.self_avoidance(
            local_scale=0.0,
            steric_scale=0.01,
            steric_start=0.0,
            steric_smoothing_radius=1,
        ),
        steric_ca_min_dist_A=100.0,
        steric_ca_seq_sep=2,
    )
    common = {
        "n_steps": 4,
        "seed": 19,
        "device": "cpu",
        "sampler": "sde",
        "sde_log_timesteps": False,
        "record_trajectory": False,
    }

    baseline = sample(_ZeroModel(), example, batch_fn, **common)
    guided = sample(_ZeroModel(), example, batch_fn, geometry_guidance=config, **common)

    assert np.isfinite(guided[1]).all()
    assert not np.array_equal(baseline[1], guided[1])
    assert config.all_atom_clash_weight == 0.0


def test_bond_axis_projection_and_global_step_cap_preserve_constraints():
    torch.manual_seed(29)
    length, atoms = 20, 5
    coords = torch.zeros(1, length, atoms, 3)
    position = torch.arange(length, dtype=torch.float32)
    coords[0, :, 0, 0] = position  # N
    coords[0, :, 1, 0] = position  # CA
    coords[0, :, 2, 0] = position  # C
    atom_mask = torch.ones(1, length, atoms, dtype=torch.bool)
    res_mask = torch.ones(1, length, dtype=torch.bool)
    chain_id = torch.zeros(1, length, dtype=torch.long)
    seq = torch.arange(length).unsqueeze(0)
    vectors = torch.randn(1, length, 3)

    projected = _project_adjacent_bond_axis(
        vectors,
        coords,
        atom_mask,
        res_mask,
        chain_id,
        seq,
        iterations=8,
    )
    before = (vectors[:, 1:, 0] - vectors[:, :-1, 0]).abs().max()
    after = (projected[:, 1:, 0] - projected[:, :-1, 0]).abs().max()
    assert after < 0.1 * before

    coherent = projected.squeeze(0).unsqueeze(1).expand(-1, atoms, -1)
    capped = _cap_guidance_step(coherent, dt=0.1, max_step_A=0.02)
    capped_residue = capped[:, 0]
    capped_after = (capped_residue[1:, 0] - capped_residue[:-1, 0]).abs().max()
    ratio = capped_after / after.clamp(min=1e-8)
    torch.testing.assert_close(
        capped[:, 1:],
        capped[:, :1].expand_as(capped[:, 1:]),
    )
    # The cap is one global scalar, so it cannot recreate an axial component.
    assert ratio <= 1.0 + 1e-6
