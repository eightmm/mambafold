"""CPU checks for compression/readout; real scan coverage lives in padding tests."""

from __future__ import annotations

import pytest
import torch

from mambafold.model.bimamba3 import Mamba3Layer, MambaStack
from mambafold.model.fold.latent_mamba import LatentMambaResidual


@pytest.fixture
def cpu_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    # The CUDA kernel is replaced only here: cumsum makes leaked padding and
    # broken reverse masks observable while testing the real surrounding blocks.
    def scan(self, x, mask=None):
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0)
        return x.cumsum(dim=1)

    monkeypatch.setattr(Mamba3Layer, "forward", scan)


def _branch(
    compression: int, loops: int = 1, mode: str = "plain",
) -> LatentMambaResidual:
    return LatentMambaResidual(
        64, compression, d_latent=64, n_layers=1, d_temb=16,
        d_state=64, mimo_rank=4, headdim=64,
        n_loops=loops,
        loop_mode=mode,
    )


def _activate(branch: LatentMambaResidual) -> None:
    with torch.no_grad():
        branch.output_proj.weight.normal_(std=0.03)
        branch.stack.layers[0].adaln.proj.bias.normal_(std=0.1)
        branch.stack.layers[0].adaln.proj.weight.normal_(std=0.01)


def test_hierarchy_preserves_fine_path_then_opens_and_ignores_padding(cpu_scan):
    from mambafold.model.fold.latent_mamba import HierarchicalMambaStack

    kwargs = dict(d_temb=16, stable=True)
    fine = MambaStack(64, 4, **kwargs)
    hierarchy = HierarchicalMambaStack(64, 4, **kwargs)
    missing, unexpected = hierarchy.load_state_dict(fine.state_dict(), strict=False)
    assert missing and all(name.startswith('coarse_') for name in missing) and not unexpected
    x, condition = torch.randn(2, 5, 64), torch.randn(2, 16)
    mask = torch.arange(5)[None] < torch.tensor([5, 1])[:, None]
    torch.testing.assert_close(hierarchy(x, mask, condition), fine(x, mask, condition),
                               rtol=0, atol=0)
    hierarchy(x, mask, condition).square().mean().backward()
    assert hierarchy.coarse_up.weight.grad.abs().sum() > 0
    optimizer = torch.optim.SGD(hierarchy.parameters(), lr=.01)
    optimizer.step()
    optimizer.zero_grad()
    actual = hierarchy(x, mask, condition)
    padded = torch.nn.functional.pad(x, (0, 0, 0, 7), value=float('nan'))
    padded_mask = torch.nn.functional.pad(mask, (0, 7))
    expected = hierarchy(padded, padded_mask, condition)[:, :5]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    actual.square().mean().backward()
    assert hierarchy.coarse_down.weight.grad.abs().sum() > 0


def test_frozen_teacher_readouts_match_real_u0_bypass_without_rng_or_hooks(cpu_scan):
    from benchmarks.adapt_shared_trunk import teacher_readouts
    from mambafold.train.trainer import _synthetic_batch, build_model

    model = build_model(dict(d_res=64, d_latent=64, n_trunk=5, d_temb=16,
                             n_latent_layers=4, n_atom_layers=0, n_atom_cross_layers=0,
                             use_plm=False, residue_refiner=True, latent_compression=4,
                             latent_loops=2, latent_supervision='lift'))
    with torch.no_grad():
        model.atom_decoder.out.weight.normal_(std=.01)
        model.latent_residual.output_proj.weight.normal_(std=.02)
    model.eval().requires_grad_(False)
    batch = _synthetic_batch(1, 8, 14, 64, torch.device('cpu'))
    rng = torch.random.get_rng_state().clone()
    anchor, final = teacher_readouts(model, batch)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert not model.latent_residual._forward_pre_hooks
    assert not model.atom_decoder._forward_pre_hooks
    with torch.no_grad():
        expected_final = model(batch)['v_atom']
        branch = model.latent_residual
        model.latent_residual = None
        expected_anchor = model(batch)['v_atom']
        model.latent_residual = branch
    torch.testing.assert_close(anchor, expected_anchor, rtol=0, atol=0)
    torch.testing.assert_close(final, expected_final, rtol=0, atol=0)
    assert not torch.equal(anchor, final)
    assert not anchor.requires_grad and not final.requires_grad
    model.train()
    with pytest.raises(ValueError, match='frozen eval'):
        teacher_readouts(model, batch)


def test_shared_trunk_migration_preserves_norms_plm_schedule_and_low_lr(cpu_scan):
    from benchmarks.compare_shared_trunk import migrate_ema
    from mambafold.train.optim import build_optimizer
    from mambafold.train.trainer import build_model

    config = dict(d_res=64, d_latent=64, n_trunk=5, d_temb=16, n_latent_layers=1,
                  n_atom_layers=0, n_atom_cross_layers=0, use_plm=False,
                  residue_refiner=True, latent_compression=4, latent_loops=2)
    source = build_model(config)
    with torch.no_grad():
        source.residue_trunk.layers[3].norm1.weight.fill_(2.7)
    target = build_model(dict(config, shared_trunk_layers=1))
    mapped, receipt = migrate_ema(source.state_dict(), target.state_dict(), 3, 1)
    target.load_state_dict(mapped, strict=True)
    assert not receipt['function_preserved']
    block = target.latent_residual.residue_stack.layers[0]
    assert not block.stable
    torch.testing.assert_close(block.norm1.weight,
                               source.residue_trunk.layers[3].norm1.weight)
    assert target.plm_inject_at == (1,)
    events = []
    handle = target.latent_residual.residue_stack.register_forward_pre_hook(
        lambda module, args, kwargs: events.append((args[2].clone(), kwargs['inject_at'])),
        with_kwargs=True)
    time = torch.randn(1, 16)
    target.latent_residual(torch.randn(1, 5, 64), torch.ones(1, 5, dtype=torch.bool), time)
    handle.remove()
    assert [event[1] for event in events] == [(0,), ()]
    assert all(torch.equal(event[0], time) for event in events)
    optimizer = build_optimizer(target, dict(optimizer='adamw', lr=.001, backbone_lr_scale=.1))
    roles = {name: group['parameter_role'] for group in optimizer.param_groups
             for name in group['param_names']}
    assert roles['latent_residual.residue_stack.layers.0.norm1.weight'] == 'backbone'
    assert roles['latent_residual.output_proj.weight'] == 'refiner'


def test_shared_hierarchy_reaches_lift_decoder_and_backward(cpu_scan):
    from mambafold.train.trainer import _synthetic_batch, build_model

    model = build_model(dict(d_res=64, d_latent=64, n_trunk=5, d_temb=16,
                             n_latent_layers=4, n_atom_layers=0, n_atom_cross_layers=0,
                             use_plm=False, residue_refiner=True, latent_compression=4,
                             latent_loops=2, shared_trunk_layers=1, latent_hierarchy=True,
                             latent_supervision='lift'))
    batch = _synthetic_batch(2, 5, 14, 8, torch.device('cpu'))
    output = model.train()(batch)
    assert len(output['v_atom_steps']) == 3
    assert output['latent_depths'].shape == (2, 2)
    sum(value.square().sum() for value in output['v_atom_steps']).backward()
    assert model.latent_residual.stack.coarse_up.weight.grad is not None
    assert model.latent_residual.residue_stack.layers[0].norm1.weight.grad is not None
    assert torch.isfinite(output['v_atom']).all()


@pytest.mark.parametrize('fine,coarse_width', [(4, None), (4, 64), (6, 128), (2, 32)])
def test_coarse_width_and_split_preserve_fine_path_then_open_without_padding_leaks(cpu_scan, fine, coarse_width):
    from mambafold.model.fold.latent_mamba import HierarchicalMambaStack

    kwargs = dict(d_temb=16, stable=True)
    torch.manual_seed(3)
    flat = MambaStack(64, fine, **kwargs)
    torch.manual_seed(3)
    hierarchy = HierarchicalMambaStack(64, fine, coarse_width=coarse_width, **kwargs)
    width = coarse_width or 64
    assert hierarchy.coarse_after == fine // 2
    assert hierarchy.coarse_down.weight.shape == (width, 256)
    assert hierarchy.coarse_up.weight.shape == (256, width)
    assert hierarchy.coarse_stack.layers[0].norm1.weight.shape == (width,)
    # The fine blocks are the first draws, exactly as in a flat stack of the same depth.
    assert all(torch.equal(value, hierarchy.state_dict()[name]) for name, value in flat.state_dict().items())
    x, condition = torch.randn(2, 9, 64), torch.randn(2, 16)
    mask = torch.arange(9)[None] < torch.tensor([9, 2])[:, None]
    torch.testing.assert_close(hierarchy(x, mask, condition), flat(x, mask, condition), rtol=0, atol=0)
    hierarchy(x, mask, condition).square().mean().backward()
    assert hierarchy.coarse_up.weight.grad.abs().sum() > 0
    torch.optim.SGD(hierarchy.parameters(), lr=.01).step()
    hierarchy.zero_grad()
    actual = hierarchy(x, mask, condition)
    assert not torch.equal(actual, flat(x, mask, condition))
    padded = torch.nn.functional.pad(x, (0, 0, 0, 7), value=float('nan'))
    expected = hierarchy(padded, torch.nn.functional.pad(mask, (0, 7)), condition)[:, :9]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_default_coarse_width_constructs_the_original_hierarchy_draws():
    from mambafold.model.fold.latent_mamba import HierarchicalMambaStack

    torch.manual_seed(5)
    default = HierarchicalMambaStack(64, 4, d_temb=16, stable=True).state_dict()
    torch.manual_seed(5)
    explicit = HierarchicalMambaStack(64, 4, coarse_width=64, d_temb=16, stable=True).state_dict()
    assert list(default) == list(explicit)
    assert all(torch.equal(default[name], explicit[name]) for name in default)


@pytest.mark.parametrize('changes', [dict(latent_compression=16, d_latent=128),
                                     dict(latent_hierarchy=True, n_latent_layers=6, latent_coarse_width=128),
                                     dict(latent_compression=8, d_latent=128, n_latent_layers=1)])
def test_latent_variants_reach_lift_decoder_and_backward(cpu_scan, changes):
    from mambafold.train.trainer import _synthetic_batch, build_model

    torch.manual_seed(11)
    model = build_model(dict(dict(d_res=64, d_latent=64, n_trunk=5, d_temb=16, n_latent_layers=6,
                                  n_atom_layers=0, n_atom_cross_layers=0, use_plm=False, residue_refiner=True,
                                  latent_compression=4, latent_loops=2, shared_trunk_layers=1,
                                  latent_supervision='lift'), **changes))
    latent = model.latent_residual
    assert latent.compression == changes.get('latent_compression', 4)
    assert latent.stack.n_layers == changes.get('n_latent_layers', 6)
    assert latent.compress_proj.weight.shape == (latent.residue_proj.out_features,
                                                 latent.compression * latent.residue_proj.out_features)
    if changes.get('latent_hierarchy'):
        assert latent.stack.coarse_after == 3 and latent.stack.coarse_down.out_features == 128
    with torch.no_grad():
        # The zero-initialized atom readout would otherwise block every upstream gradient.
        model.atom_decoder.out.weight.normal_(std=.01)
    batch = _synthetic_batch(2, 37, 14, 8, torch.device('cpu'))
    output = model.train()(batch)
    assert len(output['v_atom_steps']) == 3 and torch.isfinite(output['v_atom']).all()
    sum(value.square().sum() for value in output['v_atom_steps']).backward()
    assert latent.output_proj.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('changes, message', [
    (dict(latent_coarse_width=128), 'latent_coarse_width'),
    (dict(latent_hierarchy=True, latent_coarse_width=-1), 'latent_coarse_width'),
    (dict(latent_hierarchy=True, n_latent_layers=1), 'at least two fine blocks'),
    (dict(latent_compression=12), 'latent_compression'),
    (dict(latent_hierarchy=True, latent_compression=16), 'r4 compression'),
])
def test_latent_variant_construction_rejects_undeclared_layouts(changes, message):
    from mambafold.train.trainer import build_model

    with pytest.raises(ValueError, match=message):
        build_model(dict(dict(d_res=64, d_latent=64, n_trunk=5, d_temb=16, n_latent_layers=4, n_atom_layers=0,
                              n_atom_cross_layers=0, use_plm=False, residue_refiner=True, latent_compression=4,
                              latent_loops=2, shared_trunk_layers=1, latent_supervision='lift'), **changes))


@pytest.mark.parametrize('changes, admitted', [
    (dict(latent_compression=16), True), (dict(latent_hierarchy=True, n_latent_layers=6), True),
    (dict(latent_hierarchy=True, latent_coarse_width=1024), True), (dict(latent_coarse_width=1024), False),
    (dict(latent_hierarchy=True, latent_coarse_width=1000), False), (dict(latent_compression=12), False),
])
def test_config_admits_only_declared_latent_layouts(tmp_path, changes, admitted):
    import yaml

    from mambafold.train.config import parse_args

    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(dict(dict(d_res=1024, n_trunk=15, shared_trunk_layers=6, residue_refiner=True,
                                             latent_compression=4, d_latent=512, n_latent_layers=4,
                                             latent_supervision='lift'), **changes)))
    if admitted:
        args, _ = parse_args(['--config', str(path), '--out_dir', str(tmp_path / 'out')])
        assert all(getattr(args, key) == value for key, value in changes.items())
        assert parse_args(['--config', str(path), '--out_dir', str(tmp_path / 'o')])[0].latent_coarse_width == \
            changes.get('latent_coarse_width', 0)
    else:
        with pytest.raises(SystemExit):
            parse_args(['--config', str(path), '--out_dir', str(tmp_path / 'out')])


# Digests of the seed-20261023 CPU construction by the model source that trained the reused flat_fine6, fixed2
# and hierarchy runs (out/seed-replicate-scratch-wsd-20261027T063211Z/code). Any constructor change for an
# existing layout changes these bytes.
PRODUCTION = dict(d_res=1024, n_trunk=15, shared_trunk_layers=6, latent_hierarchy=False, latent_compression=4,
                  residue_refiner=True, d_latent=512, n_latent_layers=6, latent_loops=2,
                  latent_loop_mode='conditioned', latent_position='post', latent_supervision='lift',
                  atom_recurrence=False, d_res_pos=64, d_res_type=32, use_plm=True, d_plm=2560,
                  self_conditioning=False, bimamba_share=False, d_atom=128, n_atom_layers=4, atom_mixer='mlp',
                  atom_d_state=64, n_atom_cross_layers=1, n_backbone_streams=5, atom_mimo_rank=2,
                  d_plm_proj=1024, d_ca_emb=128, d_temb=256, d_state=64, mimo_rank=4, headdim=64, expand=2)
PINNED = {'flat_fine6': ({}, 269123226, 'eee0e38906fd4c96cf88ecfd71ba61e67138edd5ade9fb7d3a99fa090a045f8d'),
          'fixed2': (dict(atom_recycling='carry', recycle_policy='fixed2'), 269123226,
                     'eee0e38906fd4c96cf88ecfd71ba61e67138edd5ade9fb7d3a99fa090a045f8d'),
          'hierarchy': (dict(latent_hierarchy=True, n_latent_layers=4), 271223962,
                        '81e9b976c0b053d93074d7b9f9aee1ec71cefaf63dc65e03fa64a2e7fc78046c')}


@pytest.mark.parametrize('name', sorted(PINNED))
def test_existing_production_layouts_build_their_pinned_initial_tensors(name):
    from benchmarks.compare_full_refinement import tensor_digest
    from mambafold.train.trainer import build_model, seed_all

    changes, parameters, digest = PINNED[name]
    seed_all(20261023)
    model = build_model(dict(PRODUCTION, **changes), 'cpu')
    assert sum(p.numel() for p in model.parameters()) == parameters
    assert tensor_digest(model.state_dict().items()) == digest


@pytest.mark.parametrize('shared,hierarchy', [(0, False), (6, False), (6, True)])
def test_production_split_u0_equals_actual_bypass_and_plm_schedule(cpu_scan, shared, hierarchy):
    from mambafold.train.trainer import _synthetic_batch, build_model

    model = build_model(dict(d_res=64, d_latent=64, n_trunk=15, d_temb=16,
                             n_latent_layers=4, n_atom_layers=0, n_atom_cross_layers=0,
                             use_plm=False, residue_refiner=True, latent_compression=4,
                             latent_loops=2, shared_trunk_layers=shared,
                             latent_hierarchy=hierarchy, latent_supervision='lift'))
    branch = model.latent_residual
    assert model.residue_trunk.n_layers == (3 if shared else 15)
    if shared:
        assert branch.residue_stack.n_layers == 6 and model.plm_inject_at == ()
    with torch.no_grad():
        # The fresh coordinate head is zero; open it so equality tests u0 routing.
        model.atom_decoder.out.weight.normal_(std=.01)
        branch.output_proj.weight.normal_(std=.01)
        for layer in model.residue_trunk.layers:
            layer.adaln.proj.bias.normal_(std=.1)
        for layer in branch.residue_stack.layers:
            layer.adaln.proj.bias.normal_(std=.1)
    events = []
    handle = branch.residue_stack.register_forward_pre_hook(
        lambda module, args, kwargs: events.append(kwargs.get('inject_at')),
        with_kwargs=True)
    batch = _synthetic_batch(2, 5, 14, 8, torch.device('cpu'))
    output = model.train()(batch)
    handle.remove()
    if shared:
        assert events == [(2,), (1,)]
    try:
        model.latent_residual = None
        with torch.no_grad():
            bypass = model.eval()(batch)['v_atom']
    finally:
        model.latent_residual = branch
    torch.testing.assert_close(output['v_atom_steps'][0], bypass, rtol=0, atol=0)
    assert not torch.allclose(output['v_atom_steps'][-1], bypass)
    if shared:
        from benchmarks.compare_direct_shared_trunk import validate_student_anchor

        batch.x_clean = torch.randn_like(batch.x_clean)
        batch.eps = torch.randn_like(batch.eps)
        model.train()
        output = model(batch)
        proof = validate_student_anchor(model, batch, output, dict(use_rigid_align=True))
        assert proof['u0_matches_actual_K0'] and proof['detached_anchor_gradient_verified']
        assert model.training and model.latent_residual is branch


def test_attribution_routes_restore_state_and_distinguish_local_readout(cpu_scan) -> None:
    from types import SimpleNamespace

    from benchmarks.probe_latent_contribution import route

    branch = _branch(4, loops=2, mode="conditioned").eval()
    _activate(branch)
    model = SimpleNamespace(latent_residual=branch)
    x = torch.randn(1, 8, 64)
    mask = torch.ones(1, 8, dtype=torch.bool)
    temb = torch.randn(1, 16)
    expected = branch(x, mask, temb)
    prelude = branch(x, mask, temb, return_states=True)[0]
    with route(model, "bypass"):
        torch.testing.assert_close(branch(x, mask, temb), x)
    with route(model, "compression_readout"):
        torch.testing.assert_close(branch(x, mask, temb), prelude)
    h = branch.input_norm(x)
    local = branch.residue_proj(h) + branch.offset_embed(torch.arange(8) % 4)
    expected_local = x + branch.output_proj(torch.nn.functional.silu(branch.output_norm(local)))
    with route(model, "local_only"):
        torch.testing.assert_close(branch(x, mask, temb), expected_local)
    with route(model, "K1"):
        one_pass = branch(x, mask, temb)
    with route(model, "reset_history"):
        torch.testing.assert_close(branch(x, mask, temb), one_pass)
    with pytest.raises(RuntimeError), route(model, "K8"):
        raise RuntimeError("restore even on failure")
    assert branch.n_loops == 2 and "forward" not in branch.__dict__
    torch.testing.assert_close(branch(x, mask, temb), expected)


@pytest.mark.parametrize("compression", [4, 8, 16])
def test_partial_group_pool_ignores_padding_and_empty_rows(compression: int) -> None:
    branch = _branch(compression)
    with torch.no_grad():
        branch.compress_proj.weight.copy_(torch.eye(64).repeat(1, compression))
        branch.compress_proj.bias.fill_(7)
    length = compression + 1
    values = torch.arange(length, dtype=torch.float32)[None, :, None].expand(2, -1, 64)
    mask = torch.zeros(2, length, dtype=torch.bool)
    mask[0] = True
    z, zmask = branch._compress(values, mask)
    assert z.shape == (2, 2, 64)
    assert zmask.tolist() == [[True, True], [False, False]]
    torch.testing.assert_close(z[0, 0], torch.full((64,), (compression - 1) / 2 + 7))
    torch.testing.assert_close(z[0, 1], torch.full((64,), compression + 7.0))
    assert torch.count_nonzero(z[1]) == 0


@pytest.mark.parametrize("compression", [4, 8, 16])
def test_compression_initializes_as_masked_mean_with_matching_input_gradient(compression):
    branch = _branch(compression)
    length = compression + 3
    h = torch.randn(3, length, 64, requires_grad=True)
    mask = torch.arange(length)[None] < torch.tensor([length, 1, 0])[:, None]
    # Non-finite padding must neither enter the projection nor affect gradients.
    padded = h.masked_fill(~mask.unsqueeze(-1), float("nan"))
    actual, latent_mask = branch._compress(padded, mask)
    weights = branch.compress_proj.weight.reshape(64, compression, 64)
    for offset in range(1, compression):
        torch.testing.assert_close(weights[:, offset], weights[:, 0], rtol=0, atol=0)
    expected = []
    for start in range(0, length, compression):
        valid = mask[:, start:start + compression]
        mean = h[:, start:start + compression].masked_fill(~valid.unsqueeze(-1), 0)
        mean = mean.sum(1) / valid.sum(1).clamp_min(1).unsqueeze(-1)
        projected = torch.nn.functional.linear(mean, weights[:, 0], branch.compress_proj.bias)
        expected.append(projected.masked_fill(~valid.any(1).unsqueeze(-1), 0))
    expected = torch.stack(expected, dim=1)
    torch.testing.assert_close(actual, expected)
    assert latent_mask.tolist() == [[True, True], [True, False], [False, False]]
    actual_grad, = torch.autograd.grad(actual.square().sum(), h, retain_graph=True)
    expected_grad, = torch.autograd.grad(expected.square().sum(), h)
    torch.testing.assert_close(actual_grad, expected_grad)
    assert torch.count_nonzero(actual_grad[~mask]) == 0


@pytest.mark.parametrize("compression", [4, 8])
def test_compression_can_learn_group_order_with_muon(compression):
    from mambafold.train.optim import build_optimizer

    torch.manual_seed(29)
    branch = _branch(compression)
    optimizer = build_optimizer(branch, {"optimizer": "muon", "lr": 1e-3, "weight_decay": 0})
    h = torch.randn(2, compression, 64)
    mask = torch.ones(2, compression, dtype=torch.bool)
    permutation = torch.arange(compression).roll(1)
    before, _ = branch._compress(h, mask)
    shuffled, _ = branch._compress(h[:, permutation], mask)
    torch.testing.assert_close(before, shuffled)
    before.square().mean().backward()
    optimizer.step()
    after, _ = branch._compress(h, mask)
    shuffled, _ = branch._compress(h[:, permutation], mask)
    assert torch.isfinite(after).all()
    assert not torch.allclose(after, shuffled, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("compression", [4, 8])
@pytest.mark.parametrize("loops", [1, 3])
@pytest.mark.parametrize("mode", ["plain", "conditioned"])
def test_zero_readout_preserves_path_and_can_open(
    cpu_scan, compression: int, loops: int, mode: str,
) -> None:
    branch = _branch(compression, loops, mode)
    x = torch.randn(2, 11, 64, requires_grad=True)
    mask = torch.ones(2, 11, dtype=torch.bool)
    temb = torch.randn(2, 16)
    out = branch(x, mask, temb)
    torch.testing.assert_close(out, x, rtol=0, atol=0)
    out.square().mean().backward()
    assert branch.output_proj.weight.grad.abs().sum() > 0
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for name, p in branch.named_parameters() if ".ssm." not in name
    )


@pytest.mark.parametrize("compression", [4, 8])
@pytest.mark.parametrize("loops", [1, 3])
@pytest.mark.parametrize("mode", ["plain", "conditioned"])
def test_active_branch_padding_equivalence_and_backward(
    cpu_scan, compression: int, loops: int, mode: str,
) -> None:
    torch.manual_seed(4)
    branch = _branch(compression, loops, mode)
    _activate(branch)
    x = torch.randn(4, 19, 64, requires_grad=True)
    lengths = [13, 1, 9, 0]
    mask = torch.arange(19)[None] < torch.tensor(lengths)[:, None]
    temb = torch.randn(4, 16)
    together = branch(x, mask, temb)
    assert torch.count_nonzero(together[~mask]) == 0
    assert not torch.allclose(together[0, :13], x[0, :13])
    for row, length in enumerate(lengths[:-1]):
        alone = branch(x[row:row + 1, :length], mask[row:row + 1, :length], temb[row:row + 1])
        torch.testing.assert_close(together[row, :length], alone[0], rtol=1e-5, atol=1e-5)
    together.square().mean().backward()
    assert torch.count_nonzero(x.grad[~mask]) == 0
    assert branch.compress_proj.weight.grad.abs().sum() > 0
    assert branch.stack.layers[0].adaln.proj.weight.grad.abs().sum() > 0
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for name, p in branch.named_parameters() if ".ssm." not in name
    )


@pytest.mark.parametrize("compression", [4, 8])
def test_midpoint_preserves_checkpoint_and_injection_indices(cpu_scan, compression: int) -> None:
    from mambafold.train.trainer import build_model

    config = dict(
        d_res=64, n_trunk=3, d_temb=16, d_latent=64, n_latent_layers=1,
        n_atom_layers=0, n_atom_cross_layers=0, use_plm=False,
    )
    baseline = build_model(config)
    enabled = build_model(dict(config, latent_compression=compression))
    incompatible = enabled.load_state_dict(baseline.state_dict(), strict=False)
    assert incompatible.unexpected_keys == []
    assert set(incompatible.missing_keys) == {
        key for key in enabled.state_dict() if key.startswith("latent_residual.")
    }
    assert incompatible.missing_keys
    x, mask, temb = torch.randn(1, 13, 64), torch.ones(1, 13, dtype=torch.bool), torch.randn(1, 16)
    inject = torch.randn_like(x)
    events = []
    handles = [layer.register_forward_hook(
        lambda module, args, out, index=i: events.append(index)
    ) for i, layer in enumerate(enabled.residue_trunk.layers)]
    handles.append(enabled.latent_residual.register_forward_hook(
        lambda *args: events.append("latent")
    ))
    kwargs = dict(inject=inject, inject_at=(1, 2))
    expected = baseline.residue_trunk(x, mask, temb, **kwargs)
    actual = enabled.residue_trunk(
        x, mask, temb, midpoint_residual=enabled.latent_residual, **kwargs,
    )
    for handle in handles:
        handle.remove()
    assert events == [0, "latent", 1, 2]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="Missing key"):
        enabled.load_state_dict(baseline.state_dict(), strict=True)


def test_midpoint_rejects_stack_without_two_blocks(cpu_scan) -> None:
    stack = MambaStack(64, 0)
    with pytest.raises(ValueError, match="at least two"):
        stack(torch.randn(1, 4, 64), torch.ones(1, 4, dtype=torch.bool), torch.randn(1, 16),
              midpoint_residual=_branch(4))


@pytest.mark.parametrize("compression", [4, 8])
@pytest.mark.parametrize("mode", ["plain", "conditioned"])
def test_loops_reuse_weights_and_backpropagate_through_every_pass(
    cpu_scan, compression: int, mode: str,
) -> None:
    torch.manual_seed(11)
    single = _branch(compression, mode=mode)
    _activate(single)
    looped = _branch(compression, loops=3, mode=mode)
    looped.load_state_dict(single.state_dict(), strict=True)
    assert looped.state_dict().keys() == single.state_dict().keys()
    assert sum(p.numel() for p in looped.parameters()) == sum(
        p.numel() for p in single.parameters()
    )
    x = torch.randn(1, 13, 64, requires_grad=True)
    mask = torch.ones(1, 13, dtype=torch.bool)
    temb = torch.randn(1, 16)
    passes, compress_calls, readout_calls = [], [], []

    def record_pass(module, args, output):
        output.retain_grad()
        passes.append((args, output))

    handles = [
        looped.stack.register_forward_hook(record_pass),
        looped.compress_proj.register_forward_hook(lambda *args: compress_calls.append(1)),
        looped.output_proj.register_forward_hook(lambda *args: readout_calls.append(1)),
    ]
    actual = looped(x, mask, temb)
    for handle in handles:
        handle.remove()
    assert len(passes) == 3
    assert len(compress_calls) == len(readout_calls) == 1
    if mode == "plain":
        assert all(args[2] is temb for args, output in passes)
        assert passes[1][0][0] is passes[0][1]
        assert passes[2][0][0] is passes[1][1]
    else:
        assert not torch.allclose(passes[0][0][2], passes[1][0][2])
    assert not torch.allclose(actual, single(x, mask, temb))
    actual.square().mean().backward()
    assert all(out.grad is not None and out.grad.abs().sum() > 0 for args, out in passes)


@pytest.mark.parametrize("loops", [0, -1, 1.5, True])
def test_latent_loop_count_requires_positive_integer(loops) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        _branch(4, loops)


@pytest.mark.parametrize("compression", [4, 8])
@pytest.mark.parametrize("temb_dtype", [torch.float32, torch.bfloat16])
def test_conditioned_depth_grid_stays_separate_from_fm_time(
    cpu_scan, compression: int, temb_dtype: torch.dtype,
) -> None:
    torch.manual_seed(12)
    branch = _branch(compression, loops=3, mode="conditioned")
    _activate(branch)
    x, mask = torch.randn(2, 13, 64), torch.ones(2, 13, dtype=torch.bool)
    temb = torch.randn(2, 16).to(temb_dtype)
    original_temb = temb.clone()
    coordinates, conditions, compress_calls = [], [], []
    handles = [
        branch.depth_embed.register_forward_pre_hook(
            lambda module, args: coordinates.append(args[0].detach().clone())
        ),
        branch.stack.register_forward_pre_hook(
            lambda module, args: conditions.append(args[2].detach().clone())
        ),
        branch.compress_proj.register_forward_hook(lambda *args: compress_calls.append(1)),
    ]
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=temb_dtype == torch.bfloat16):
        result = branch(x, mask, temb)
    for handle in handles:
        handle.remove()
    torch.testing.assert_close(temb, original_temb, rtol=0, atol=0)
    assert len(compress_calls) == 1
    assert len(coordinates) == len(conditions) == 3
    assert all(coordinate.dtype == torch.float32 for coordinate in coordinates)
    torch.testing.assert_close(
        torch.stack(coordinates)[:, 0, 0], torch.tensor([1 / 3, 2 / 3, 1.0]),
    )
    assert not torch.allclose(conditions[0], conditions[1])
    result.square().mean().backward()
    assert branch.depth_embed.mlp[-1].weight.grad.abs().sum() > 0
    extended = _branch(compression, loops=7, mode="conditioned")
    extended.load_state_dict(branch.state_dict(), strict=True)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=temb_dtype == torch.bfloat16):
        assert torch.isfinite(extended(x, mask, temb)).all()


def test_conditioned_mode_declares_only_new_depth_keys() -> None:
    plain = _branch(4)
    assert not any(key.startswith("depth_embed.") for key in plain.state_dict())
    conditioned = _branch(4, mode="conditioned")
    incompatible = conditioned.load_state_dict(plain.state_dict(), strict=False)
    assert incompatible.unexpected_keys == []
    assert incompatible.missing_keys
    assert all(key.startswith("depth_embed.") for key in incompatible.missing_keys)
    with pytest.raises(ValueError, match="plain or conditioned"):
        _branch(4, mode="invalid")


@pytest.mark.parametrize("compression", [4, 8])
def test_trajectory_shares_readout_caches_compression_and_keeps_rollout_gradients(
    cpu_scan, compression: int,
) -> None:
    branch = _branch(compression, loops=2, mode="conditioned")
    _activate(branch)
    x = torch.randn(2, 13, 64, requires_grad=True)
    mask = torch.arange(13)[None] < torch.tensor([13, 7])[:, None]
    temb = torch.randn(2, 16)
    depths = branch.depth_grid(2, x.device, randomize=True)
    assert depths.dtype == torch.float32
    assert torch.all((0 < depths[:, 0]) & (depths[:, 0] < depths[:, 1]))
    assert torch.equal(depths[:, -1], torch.ones(2))
    passes, compressed, projected = [], [], []

    def record(module, args, out):
        out.retain_grad()
        passes.append(out)

    handles = [branch.stack.register_forward_hook(record),
               branch.compress_proj.register_forward_hook(lambda *args: compressed.append(1)),
               branch.residue_proj.register_forward_hook(lambda *args: projected.append(1))]
    states = branch(x, mask, temb, depths=depths, return_states=True)
    for handle in handles:
        handle.remove()
    assert len(states) == 3 and len(passes) == 2
    assert len(compressed) == len(projected) == 1
    for state in states:
        assert torch.count_nonzero(state[~mask]) == 0
    expected = branch(x, mask, temb, depths=depths)
    torch.testing.assert_close(states[-1], expected, rtol=0, atol=0)
    states[-1].square().mean().backward()
    assert all(state.grad.abs().sum() > 0 for state in passes)


@pytest.mark.parametrize("compression", [4, 8, 16])
def test_residue_refiner_feedback_padding_and_gradient(cpu_scan, compression):
    from mambafold.model.fold.latent_mamba import ResidueLatentRefiner

    torch.manual_seed(31)
    branch = ResidueLatentRefiner(
        64, compression, 64, 1, n_residue_layers=1, n_loops=2,
        d_temb=16, d_state=64, mimo_rank=4, headdim=64,
    )
    _activate(branch)
    with torch.no_grad():
        branch.residue_stack.layers[0].adaln.proj.weight.normal_(std=.05)
    x = torch.randn(2, 13, 64, requires_grad=True)
    mask = torch.arange(13)[None] < torch.tensor([13, 7])[:, None]
    x_masked = x.masked_fill(~mask.unsqueeze(-1), 0)
    temb = torch.randn(2, 16)
    calls = []
    handle = branch.input_norm.register_forward_pre_hook(
        lambda module, args: calls.append(args[0])
    )
    states = branch(x_masked, mask, temb, return_states=True)
    handle.remove()
    assert len(states) == 3 and len(calls) == 2
    assert calls[1] is states[1]
    assert not torch.allclose(states[1], states[2])
    states[1].retain_grad()
    states[-1].square().mean().backward()
    assert states[1].grad.abs().sum() > 0
    assert branch.compress_proj.weight.grad.abs().sum() > 0
    assert branch.residue_stack.layers[0].adaln.proj.weight.grad.abs().sum() > 0
    for row, length in enumerate([13, 7]):
        alone = branch(x_masked[row:row+1, :length], mask[row:row+1, :length], temb[row:row+1])
        torch.testing.assert_close(states[-1][row, :length], alone[0], atol=1e-5, rtol=1e-5)
    assert torch.count_nonzero(states[-1][~mask]) == 0


def test_stable_block_bounds_large_conditioning_and_scan_gain(monkeypatch):
    from mambafold.model.bimamba3 import BiMamba3Block

    monkeypatch.setattr(Mamba3Layer, 'forward', lambda self, x, mask=None: x * 1e12)
    block = BiMamba3Block(64, d_temb=16, stable=True)
    with torch.no_grad():
        block.adaln.proj.weight.normal_(std=10)
    x = torch.randn(2, 8, 64, requires_grad=True)
    y = block(x, torch.ones(2, 8, dtype=torch.bool), torch.randn(2, 16) * 1e9)
    assert torch.isfinite(y).all()
    assert (y - x).square().mean(-1).sqrt().max() <= 2.001
    y.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert block.mamba_f.ssm.is_outproj_norm


def test_refiner_causal_controls_restore_weights_and_history(cpu_scan):
    from types import SimpleNamespace

    from benchmarks.validate_residue_refiner import route
    from mambafold.model.fold.latent_mamba import ResidueLatentRefiner

    branch = ResidueLatentRefiner(
        64, 4, 64, 1, n_residue_layers=1, n_loops=2,
        d_temb=16, d_state=64, mimo_rank=4, headdim=64,
    ).eval()
    _activate(branch)
    with torch.no_grad():
        branch.residue_stack.layers[0].adaln.proj.weight.normal_(std=.03)
    model = SimpleNamespace(latent_residual=branch)
    x, mask, temb = torch.randn(1, 13, 64), torch.ones(1, 13, dtype=torch.bool), torch.randn(1, 16)
    expected = branch(x, mask, temb)
    with route(model, 'K1'):
        one_pass = branch(x, mask, temb)
    with route(model, 'reset_history'):
        torch.testing.assert_close(branch(x, mask, temb), one_pass, rtol=0, atol=0)
    with route(model, 'zero_latent'):
        assert not torch.allclose(branch(x, mask, temb), expected)
    with pytest.raises(RuntimeError), route(model, 'reset_history'):
        raise RuntimeError('cleanup')
    torch.testing.assert_close(branch(x, mask, temb), expected, rtol=0, atol=0)


def test_runtime_metadata_records_mixed_native_normalization():
    from mambafold.model.bimamba3 import mamba3_runtime_info

    legacy = Mamba3Layer(64, outproj_norm=False)
    stable = Mamba3Layer(64, outproj_norm=True)
    assert mamba3_runtime_info(legacy)['outproj_norm'] is False
    assert mamba3_runtime_info(stable)['outproj_norm'] is True
    assert mamba3_runtime_info(torch.nn.ModuleList([legacy, stable]))['outproj_norm'] == 'mixed'


@pytest.mark.parametrize('compression', [4, 8])
def test_safe_refiner_initialization_preserves_full_backbone(cpu_scan, compression):
    from mambafold.train.trainer import build_model, initialize_residue_refiner

    config = dict(d_res=128, n_trunk=3, d_temb=16, d_latent=64, n_latent_layers=1,
                  n_atom_layers=0, n_atom_cross_layers=0, use_plm=False)
    source = build_model(config).eval()
    with torch.no_grad():
        source.atom_decoder.out.weight.normal_(std=.03)
        for layer in source.residue_trunk.layers:
            layer.adaln.proj.weight.normal_(std=.03)
    target_config = dict(config, residue_refiner=True, latent_compression=compression,
                         latent_supervision='lift')
    target = build_model(target_config).eval()
    receipt = initialize_residue_refiner(target, source.state_dict(), config, target_config)
    assert receipt['plm_inject_at'] == [1, 2]
    assert all(key.startswith('latent_residual.') for key in receipt['initialized'])
    x, mask, temb = torch.randn(2, 13, 128), torch.ones(2, 13, dtype=torch.bool), torch.randn(2, 16)
    branch = target.latent_residual
    assert branch.residue_stack.layers[0].norm1.weight.shape == (64,)
    for loops in (1, 2, 3):
        branch.n_loops = loops
        states = branch(x, mask, temb, return_states=True)
        assert len(states) == loops + 1
        for state in states:
            torch.testing.assert_close(state, x, rtol=0, atol=0)
    # A changed trunk cannot be smuggled through a partial state-dict load.
    wrong_config = dict(target_config, n_trunk=2)
    wrong = build_model(wrong_config)
    before = wrong.trunk_proj.weight.detach().clone()
    with pytest.raises(ValueError, match='backbone configuration'):
        initialize_residue_refiner(wrong, source.state_dict(), config, wrong_config)
    torch.testing.assert_close(wrong.trunk_proj.weight, before, rtol=0, atol=0)


@pytest.mark.parametrize(
    'corruption', ['missing', 'shape', 'normalization', 'injection', 'used_readout'],
)
def test_refiner_initialization_rejects_function_changing_migration(corruption):
    from mambafold.train.trainer import build_model, initialize_residue_refiner

    cfg = dict(d_res=64, n_trunk=2, d_latent=64, n_latent_layers=1,
               n_atom_layers=0, n_atom_cross_layers=0, use_plm=False)
    source = build_model(cfg).state_dict()
    target_cfg = dict(cfg, residue_refiner=True, latent_compression=4)
    target = build_model(target_cfg)
    if corruption == 'missing':
        source.pop('trunk_proj.weight')
    elif corruption == 'shape':
        source['trunk_proj.weight'] = source['trunk_proj.weight'][:-1]
    elif corruption == 'normalization':
        target.residue_trunk.layers[0].stable = True
    elif corruption == 'injection':
        target.plm_inject_at = (1,)
    else:
        with torch.no_grad():
            target.latent_residual.output_proj.weight.fill_(.1)
    before = target.trunk_proj.weight.detach().clone()
    with pytest.raises(ValueError):
        initialize_residue_refiner(target, source, cfg, target_cfg)
    torch.testing.assert_close(target.trunk_proj.weight, before, rtol=0, atol=0)
