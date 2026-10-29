from __future__ import annotations

import math
import os
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from mambafold.data.collate import ProteinCollator
from mambafold.data.constants import CA_ATOM_ID, MAX_ATOMS_PER_RES, PAIR_PAD_ID
from mambafold.data.types import ProteinBatch, ProteinExample
from mambafold.model.fold import MambaFoldAllAtom
from mambafold.sampling import prepare_inference_batch, sample
from mambafold.train.config import parse_args
from mambafold.train.engine import allatom_loss_surface


def _batch(batch_size: int = 2, length: int = 2) -> ProteinBatch:
    atoms = MAX_ATOMS_PER_RES
    atom_mask = torch.zeros(batch_size, length, atoms, dtype=torch.bool)
    atom_mask[..., :5] = True
    clean = torch.randn(batch_size, length, atoms, 3) * 0.1
    eps = torch.randn_like(clean)
    t = torch.linspace(0.25, 0.75, batch_size).reshape(batch_size, 1, 1, 1)
    zeros_l = torch.zeros(batch_size, length, dtype=torch.long)
    pair_type = torch.full((batch_size, length, atoms), PAIR_PAD_ID, dtype=torch.long)
    pair_type[atom_mask] = 0
    return ProteinBatch(
        res_type=zeros_l,
        res_seq_nums=torch.arange(length).expand(batch_size, -1),
        atom_type=torch.zeros(batch_size, length, atoms, dtype=torch.long),
        pair_type=pair_type,
        res_mask=torch.ones(batch_size, length, dtype=torch.bool),
        atom_mask=atom_mask,
        valid_mask=atom_mask,
        ca_mask=atom_mask[..., CA_ATOM_ID],
        chain_id=zeros_l,
        entity_id=zeros_l,
        sym_id=zeros_l,
        is_nterm=torch.zeros(batch_size, length, dtype=torch.bool),
        is_cterm=torch.zeros(batch_size, length, dtype=torch.bool),
        x_clean=clean,
        x_t=t * clean + (1.0 - t) * eps,
        eps=eps,
        t=t,
        esm=None,
    )


def _tiny_model(*, use_plm: bool = False, d_plm: int = 8) -> MambaFoldAllAtom:
    # Zero SSM layers keeps the smoke test CPU-only while exercising every
    # embedding, pooling, conditioning, and folding-output parameter.
    return MambaFoldAllAtom(
        d_res=32,
        n_trunk=0,
        d_res_type=8,
        d_res_pos=8,
        d_plm=d_plm,
        d_plm_proj=8,
        d_ca_emb=16,
        use_plm=use_plm,
        mimo_rank=1,
        d_state=16,
        expand=1,
        headdim=8,
        d_atom=16,
        n_atom_layers=0,
        # The cross-residue backbone mixer is an SSM too, so it has to go with
        # the rest of them for this to stay CPU-only. It is exercised on GPU
        # instead — nothing here covers it.
        n_atom_cross_layers=0,
        self_conditioning=True,
    )


def test_folding_model_has_only_core_outputs_and_all_parameters_receive_gradients() -> None:
    torch.manual_seed(5)
    model = _tiny_model()
    assert not any(
        token in name
        for name, _ in model.named_parameters()
        for token in ("conf", "pcb", "distogram", "contact", "aux_pair")
    )

    batch = _batch()
    output = model(batch)
    assert set(output) == {"v_atom", "trunk_latent"}
    loss, _ = allatom_loss_surface(
        output,
        batch,
        use_rigid_align=False,
        w_lddt_atom=0.0,
    )
    loss.backward()

    missing = [name for name, parameter in model.named_parameters() if parameter.grad is None]
    assert missing == []
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def test_latent_training_probe_uses_production_loss_accumulation_and_ema(monkeypatch) -> None:
    from functools import wraps

    from benchmarks import probe_latent_training as probe
    from mambafold.train.ema import EMA

    model = _tiny_model()
    ema = EMA(model, decay=0.5)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    before = model.atom_decoder.out.weight.detach().clone()
    batches = [_batch(), _batch()]
    calls = []
    production_loss = probe.allatom_forward_and_loss

    @wraps(production_loss)
    def record(model, batch, **kwargs):
        calls.append(kwargs)
        return production_loss(model, batch, **kwargs)

    monkeypatch.setattr(probe, "allatom_forward_and_loss", record)
    metrics = probe.training_update(model, optimizer, ema, batches, dict(
        grad_clip=2, w_lddt_atom=0, use_amp=False, use_rigid_align=False,
    ))
    assert len(calls) == 2 and all(call["w_lddt_atom"] == 0 for call in calls)
    assert math.isfinite(metrics["loss"]) and math.isfinite(metrics["grad_norm"])
    after = model.atom_decoder.out.weight.detach()
    assert not torch.equal(before, after)
    torch.testing.assert_close(ema.shadow.atom_decoder.out.weight, (before + after) / 2)


@pytest.mark.parametrize("profile_update,selected,median,mean_max", [
    (3, [2, 5], 36, 36), (0, [2, 3, 4, 5], 35, 35.5),
])
def test_pipeline_summary_excludes_shutdown_and_uses_slowest_rank_interval(
    tmp_path, profile_update, selected, median, mean_max,
):
    import json

    from benchmarks.probe_training_pipeline import summarize

    for rank, durations in enumerate(([10, 20, 30, 40, 50, 5000], [12, 18, 30, 40, 52, 6000])):
        report = dict(
            world_size=2, completed=True, source_unchanged=True,
            warmup_updates=1, profile_update=profile_update, backend="gloo", device="cpu",
            source_sha256="same", peak_allocated_gib=0,
            steps=[dict(step=i, updated=True, wall_ms=duration, valid_residues=100,
                        padded_residues=128, lengths=[8], stages_ms={"forward": 2})
                   for i, duration in enumerate(durations, 1)],
        )
        (tmp_path / f"rank{rank}.json").write_text(json.dumps(report))
    result = summarize(tmp_path)
    assert [r["step"] for r in result["measured_steps"]] == selected
    assert result["mean_ms"] == pytest.approx(35)
    assert result["mean_step_max_ms"] == pytest.approx(mean_max)
    assert result["median_ms"] == pytest.approx(median)
    assert result["valid_residues_per_second"] == pytest.approx(200 / 0.035)


@pytest.mark.parametrize("fail", [False, True])
def test_training_gc_collects_new_autograd_cycles_and_thaws_on_exit(fail):
    import subprocess
    import sys
    import textwrap

    # The standalone entrypoint owns the whole permanent generation. Isolate
    # it from pytest's interpreter roots and check that freezing really ran.
    code = textwrap.dedent("""
        import gc
        import weakref
        import torch
        from mambafold.train.trainer import freeze_training_setup

        fail = FAIL_VALUE
        previous = gc.get_freeze_count()
        enabled = gc.isenabled()
        model = torch.nn.Linear(4, 4)
        try:
            with freeze_training_setup() as count:
                assert count > previous
                assert gc.isenabled() == enabled
                tensor = model(torch.ones(4))
                tensor.sum().backward()
                assert torch.isfinite(model.weight.grad).all()
                reference = weakref.ref(tensor)
                cycle = [tensor]
                cycle.append(cycle)
                del tensor, cycle
                gc.collect()
                assert reference() is None
                if fail:
                    raise RuntimeError("training failed")
        except RuntimeError as exc:
            assert fail and str(exc) == "training failed"
        assert gc.get_freeze_count() == 0
        assert gc.isenabled() == enabled
    """).replace("FAIL_VALUE", repr(fail))
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_batched_ema_preserves_scalar_updates_and_buffers(device):
    from mambafold.train.ema import EMA

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA allocation required")
    model = torch.nn.ModuleList([
        torch.nn.Linear(9, 4, dtype=dtype, device=device)
        for dtype in (torch.float32, torch.float64)
    ])
    model.register_buffer("counter", torch.tensor(2, device=device))
    ema = EMA(model, decay=0.97)
    reference = deepcopy(ema.shadow)
    for _ in range(3):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p))
            model.counter.add_(1)
            for shadow, current in zip(reference.parameters(), model.parameters()):
                shadow.mul_(ema.decay).add_(current, alpha=1 - ema.decay)
        ema.update(model)
        for actual, expected in zip(ema.shadow.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert ema.shadow.counter == 2


@pytest.mark.parametrize("repeat", [1, 5])
def test_shared_adaln_projection_preserves_outputs_and_gradients(repeat):
    from mambafold.model.bimamba3 import AdaLNZero

    torch.manual_seed(37)
    reference = AdaLNZero(8, 4).double()
    torch.nn.init.normal_(reference.proj.weight)
    actual = deepcopy(reference)
    time = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    other_time = time.detach().clone().requires_grad_()
    inputs = [torch.randn(2 * repeat, 3, 8, dtype=torch.float64) for _ in range(2)]
    params = actual.project(other_time)
    first, second = [], []
    for branch, x in enumerate(inputs):
        h, gate = reference(x, time, branch, repeat)
        first.append(h * gate)
        h, gate = actual.modulate(x, params, branch, repeat)
        second.append(h * gate)
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    sum(x.square().sum() for x in first).backward()
    sum(x.square().sum() for x in second).backward()
    torch.testing.assert_close(time.grad, other_time.grad)
    for a, b in zip(reference.parameters(), actual.parameters()):
        torch.testing.assert_close(a.grad, b.grad)


@pytest.mark.parametrize("pinned", [False, True])
def test_batch_padding_preserves_fields_and_loader_pinning(pinned):
    from dataclasses import fields

    if pinned and not torch.cuda.is_available():
        pytest.skip("Pinned allocation requires a CUDA driver")
    batch = _batch(length=3)
    batch.esm = torch.randn(1, 3, 8)
    batch.x_self_cond = torch.randn_like(batch.x_clean)
    if pinned:
        # Exercise DataLoader's dispatch for custom classes, not just the method.
        from torch.utils.data import DataLoader
        batch = next(iter(DataLoader([batch], batch_size=None, pin_memory=True)))
    padded = batch.pad_to_length(7)
    assert padded.t is batch.t
    assert padded.pad_to_length(7) is padded
    for field in fields(batch):
        before, after = getattr(batch, field.name), getattr(padded, field.name)
        assert after.is_pinned() == pinned
        if field.name != "t":
            torch.testing.assert_close(after[:, :3], before, rtol=0, atol=0)
            fill = PAIR_PAD_ID if field.name == "pair_type" else 0
            assert torch.all(after[:, 3:] == fill)
    if pinned:
        moved = padded.to(torch.device("cuda"))
        for field in fields(padded):
            torch.testing.assert_close(
                getattr(moved, field.name).cpu(), getattr(padded, field.name),
            )


def _batch_protocol_worker(rank, rendezvous):
    from mambafold.train.distributed import any_rank_true, distributed_batch_info

    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        cases = [
            ((None, 64), (True, 64)),
            ((32, 64), (False, 64)),
            ((None, None), (True, 0)),
            ((64, None), (True, 64)),
        ]
        for lengths, expected in cases:
            assert distributed_batch_info(lengths[rank], "cpu") == expected
        # Loss checks agree before backward; no additional post-loop skip
        # collective is necessary even when only one rank has a nonfinite loss.
        assert any_rank_true(rank == 1, "cpu")
        assert not any_rank_true(False, "cpu")
    finally:
        dist.destroy_process_group()


def test_two_rank_batch_protocol_agrees_on_missing_lengths_and_nonfinite(tmp_path, monkeypatch):
    import torch.multiprocessing as mp

    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    mp.spawn(_batch_protocol_worker, args=(str(tmp_path / "batch-rendezvous"),), nprocs=2)


def test_muon_assigns_all_hidden_linear_weights_once_and_excludes_embeddings_and_head() -> None:
    from mambafold.train.optim import parameter_groups
    from mambafold.train.trainer import build_model

    with torch.device("meta"):
        model = build_model(dict(
            d_res=64, n_trunk=2, d_latent=64, n_latent_layers=1, latent_compression=4,
            latent_loops=2, d_atom=32, atom_mixer="mlp", n_atom_layers=1,
            n_atom_cross_layers=1, use_plm=True, d_plm=32, d_plm_proj=32,
            self_conditioning=True, bimamba_share=True,
        ), "meta")
    model.res_type_embed.weight.requires_grad_(False)
    # Exclusions must win when an embedding and a projection share a parameter.
    model.embedding_alias = torch.nn.Linear(64, 4, bias=False)
    model.embedding_alias.weight = model.latent_residual.offset_embed.weight
    groups = parameter_groups(model)
    names = {kind: {name for name, _ in pairs} for kind, pairs in groups.items()}
    assigned = [id(p) for pairs in groups.values() for _, p in pairs]
    assert len(assigned) == len(set(assigned))
    assert set(assigned) == {id(p) for p in model.parameters() if p.requires_grad}
    assert {
        "residue_trunk.layers.0.mamba_f.ssm.in_proj.weight",
        "residue_trunk.layers.0.mamba_f.ssm.out_proj.weight",
        "residue_trunk.layers.0.ffn.w1.weight",
        "residue_trunk.layers.0.adaln.proj.weight",
        "atom_encoder.mamba.slot_mix.0.weight",
        "atom_decoder.ctx_proj.1.weight", "atom_decoder.out_mod.weight",
        "plm_proj.weight", "self_cond_proj.weight", "time_embed.mlp.0.weight",
        "latent_residual.output_proj.weight", "latent_residual.depth_embed.mlp.2.weight",
    } <= names["muon"]
    assert {
        "atom_decoder.out.weight", "atom_decoder.out.bias",
        "latent_residual.offset_embed.weight",
        "residue_trunk.layers.0.mamba_f.ssm.dt_bias",
        "residue_trunk.layers.0.mamba_f.ssm.mimo_x",
        "residue_trunk.layers.0.norm1.weight",
    } <= names["adamw"]
    assert "res_type_embed.weight" not in names["adamw"]


@pytest.mark.parametrize("kind", ["muon", "adamw"])
def test_backbone_lr_scale_updates_all_pretrained_roles_and_roundtrips(kind):
    from mambafold.train.optim import build_optimizer, load_optimizer_state
    from mambafold.train.trainer import linear_warmup_lr

    def network():
        net = torch.nn.Module()
        net.residue_refiner = True
        net.encoder = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
        net.latent_residual = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
        net.atom_decoder = torch.nn.Module()
        net.atom_decoder.out = torch.nn.Linear(4, 3)
        return net

    torch.manual_seed(8)
    model = network()
    reference = deepcopy(model)
    cfg = dict(optimizer=kind, lr=1e-4, muon_lr=3e-3, muon_adjust_lr_fn="original",
               backbone_lr_scale=0.1)
    opt = build_optimizer(model, cfg)
    full = build_optimizer(reference, dict(cfg, backbone_lr_scale=1))
    assigned = [id(p) for g in opt.param_groups for p in g["params"]]
    assert len(set(assigned)) == len(assigned) == len(list(model.parameters()))
    for group in opt.param_groups:
        is_refiner = group["parameter_role"] == "refiner"
        assert all(name.startswith("latent_residual.") == is_refiner
                   for name in group["param_names"])
        peak = cfg["muon_lr"] if group["optimizer_kind"] == "muon" else cfg["lr"]
        assert group["lr"] == pytest.approx(peak * (1 if is_refiner else 0.1))
    initial = deepcopy(model.state_dict())
    for net, optimizer in ((model, opt), (reference, full)):
        for parameter in net.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
    for name, parameter in model.named_parameters():
        factor = 1 if name.startswith("latent_residual.") else 0.1
        torch.testing.assert_close(
            parameter - initial[name], (reference.state_dict()[name] - initial[name]) * factor,
            atol=5e-8, rtol=0.01,
        )
    scheduler = linear_warmup_lr(opt, 2, cfg["lr"], total_steps=8, cooldown_steps=2)
    for _ in range(3):
        opt.step()
        scheduler.step()
    restored = deepcopy(model)
    restored_opt = build_optimizer(restored, cfg)
    restored_scheduler = linear_warmup_lr(restored_opt, 2, cfg["lr"],
                                            total_steps=8, cooldown_steps=2)
    load_optimizer_state(restored_opt, deepcopy(opt.state_dict()))
    restored_scheduler.load_state_dict(scheduler.state_dict())
    for _ in range(4):
        for net in (model, restored):
            for parameter in net.parameters():
                parameter.grad = torch.ones_like(parameter)
        opt.step()
        restored_opt.step()
        scheduler.step()
        restored_scheduler.step()
        assert scheduler.get_last_lr() == restored_scheduler.get_last_lr()
        for left, right in zip(model.parameters(), restored.parameters(), strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    with pytest.raises(ValueError, match="reset_optimizer"):
        load_optimizer_state(build_optimizer(network(), dict(cfg, backbone_lr_scale=0.2)),
                             opt.state_dict())


def test_backbone_lr_scale_config_requires_refiner_and_finite_positive_fraction(tmp_path):
    config = tmp_path / "refiner.yaml"
    recipe = "residue_refiner: true\nlatent_compression: 4\n"
    config.write_text(recipe + "backbone_lr_scale: 0.1\n")
    assert parse_args(["--config", str(config)])[0].backbone_lr_scale == 0.1
    for value in ("0", "-0.1", "1.1", ".nan", ".inf", "true", "invalid"):
        config.write_text(recipe + f"backbone_lr_scale: {value}\n")
        with pytest.raises(SystemExit):
            parse_args(["--config", str(config)])
    with pytest.raises(SystemExit):
        parse_args(["--backbone_lr_scale", "0.1"])


@pytest.mark.parametrize(
    "kind,muon_lr", [("adamw", None), ("muon", None), ("muon", 1e-2), ("muon", 3e-3)],
)
def test_optimizer_checkpoint_continues_same_updates_schedule_and_ema(
    tmp_path, kind, muon_lr,
) -> None:
    from mambafold.train.ema import EMA
    from mambafold.train.optim import build_optimizer, load_optimizer_state
    from mambafold.train.trainer import linear_warmup_lr, save_checkpoint

    torch.manual_seed(9)
    model = _tiny_model()
    cfg = dict(optimizer=kind, lr=1e-3, weight_decay=0.01)
    if muon_lr is not None:
        cfg.update(lr=1e-4, muon_lr=muon_lr, muon_adjust_lr_fn="original")
    optimizer = build_optimizer(model, cfg)
    scheduler = linear_warmup_lr(optimizer, 3, cfg["lr"], total_steps=8, cooldown_steps=3)
    peak_lrs = [muon_lr if group.get("optimizer_kind") == "muon" and muon_lr is not None
                else cfg["lr"] for group in optimizer.param_groups]
    assert scheduler.get_last_lr() == pytest.approx([lr * 1e-6 / cfg["lr"] for lr in peak_lrs])
    if kind == "muon":
        assert optimizer.optimizers["muon"].param_groups[0]["adjust_lr_fn"] == (
            "original" if muon_lr is not None else "match_rms_adamw"
        )
    ema = EMA(model, decay=0.9)
    batch = _batch()

    def update(net, opt, sched, average):
        opt.zero_grad(set_to_none=True)
        # Accumulate twice, as in production. Both optimizer groups update once.
        for _ in range(2):
            loss, _ = allatom_loss_surface(net(batch), batch, use_rigid_align=False, w_lddt_atom=0)
            (loss / 2).backward()
        assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 2)
        opt.step()
        sched.step()
        average.update(net)

    for _ in range(3):
        update(model, optimizer, scheduler, ema)
    assert scheduler.get_last_lr() == pytest.approx(peak_lrs)
    save_checkpoint(tmp_path, 3, model, ema, optimizer, scheduler, cfg)
    state = torch.load(tmp_path / "ckpt_latest.pt", weights_only=False)
    from mambafold.model.bimamba3 import mamba3_runtime_info

    assert state["mamba3_runtime"] == mamba3_runtime_info()
    restored = _tiny_model()
    restored.load_state_dict(state["model"])
    restored_optimizer = build_optimizer(restored, cfg)
    restored_scheduler = linear_warmup_lr(
        restored_optimizer, 3, cfg["lr"], total_steps=8, cooldown_steps=3,
    )
    load_optimizer_state(restored_optimizer, state["optimizer"])
    restored_scheduler.load_state_dict(state["scheduler"])
    restored_ema = EMA(restored, decay=0.9)
    restored_ema.load_state_dict(state["ema"])
    for _ in range(5):
        update(model, optimizer, scheduler, ema)
        update(restored, restored_optimizer, restored_scheduler, restored_ema)
        assert scheduler.get_last_lr() == restored_scheduler.get_last_lr()
        for original, resumed in zip(model.parameters(), restored.parameters()):
            torch.testing.assert_close(original, resumed, rtol=0, atol=0)
        for original, resumed in zip(ema.shadow.parameters(), restored_ema.shadow.parameters()):
            torch.testing.assert_close(original, resumed, rtol=0, atol=0)
    assert scheduler.get_last_lr() == pytest.approx([lr * 1e-6 / cfg["lr"] for lr in peak_lrs])
    # A zero-initialized readout must open, then train its upstream projections.
    assert torch.count_nonzero(model.atom_decoder.out.weight) > 0
    assert torch.count_nonzero(model.self_cond_proj.weight) == 0  # no self-conditioning input
    assert model.trunk_proj.weight.grad.abs().sum() > 0


def test_mamba_kernel_symbols_hit_cache_without_collapsing_static_options():
    from functools import lru_cache

    from tvm import tir

    from mambafold.model.bimamba3 import _stable_symbolic_factory

    @lru_cache(maxsize=16)
    def compile_kernel(*args, **kwargs):
        return object()

    factory = _stable_symbolic_factory(compile_kernel)
    first = factory(tir.Var("B", "int32"), tir.Var("S", "int32"), 64, chunk_size=8)
    assert factory(tir.Var("B", "int32"), tir.Var("S", "int32"), 64, chunk_size=8) is first
    assert compile_kernel.cache_info().misses == 1
    assert factory(tir.Var("B", "int32"), tir.Var("S", "int32"), 64, chunk_size=16) is not first
    assert factory(tir.Var("B", "int64"), tir.Var("S", "int32"), 64, chunk_size=8) is not first
    assert compile_kernel.cache_info().misses == 3


def test_mamba3_rejects_old_decay_and_has_finite_heavy_tail_gradients(monkeypatch):
    import types

    from mambafold.model import bimamba3

    module = bimamba3.importlib.import_module("mamba_ssm.modules.mamba3")
    x = torch.tensor([-10000.0, -2.0, 0.0, 2.0, 10000.0], requires_grad=True)
    y = module.heavy_tail_activation(x)
    torch.testing.assert_close(y, torch.tensor([1 / 10001, 1 / 3, 1, 3, 10001]))
    y.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert (x.grad > 0).all()
    monkeypatch.setattr(bimamba3.importlib, "import_module", lambda _: types.SimpleNamespace())
    with pytest.raises(RuntimeError, match="heavy-tail decay"):
        bimamba3._load_mamba3_class()


def test_muon_resume_rejects_optimizer_switch_and_parameter_reordering() -> None:
    from mambafold.train.optim import build_optimizer, load_optimizer_state

    model = _tiny_model()
    adamw = build_optimizer(model, {})
    muon = build_optimizer(model, {"optimizer": "muon"})
    for target, source in ((adamw, muon), (muon, adamw)):
        with pytest.raises(ValueError, match="reset_optimizer"):
            load_optimizer_state(target, source.state_dict())
    state = deepcopy(muon.state_dict())
    state["layout"][0]["names"].reverse()
    with pytest.raises(ValueError, match="layout changed"):
        load_optimizer_state(muon, state)


def test_optimizer_config_and_legacy_adamw_state(tmp_path) -> None:
    from mambafold.train.optim import build_optimizer, load_optimizer_state

    config = tmp_path / "muon.yaml"
    config.write_text("optimizer: muon\n")
    assert parse_args(["--config", str(config)])[0].optimizer == "muon"
    assert parse_args(["--config", str(config), "--optimizer", "adamw"])[0].optimizer == "adamw"
    config.write_text("optimizer: muon\nmuon_lr: 0.01\nmuon_adjust_lr_fn: original\n")
    args, _ = parse_args(["--config", str(config)])
    assert args.muon_lr == 1e-2 and args.muon_adjust_lr_fn == "original"
    assert parse_args(["--config", str(config), "--muon_lr", "0.02"])[0].muon_lr == 2e-2
    for extra in ("muon_lr: 0", "muon_lr: -.inf", "muon_lr: .nan", "muon_lr: invalid",
                  "muon_adjust_lr_fn: invalid"):
        config.write_text("optimizer: muon\n" + extra + "\n")
        with pytest.raises(SystemExit):
            parse_args(["--config", str(config)])
    config.write_text("optimizer: invalid\n")
    with pytest.raises(SystemExit):
        parse_args(["--config", str(config)])
    model = _tiny_model()
    legacy = torch.optim.AdamW(model.parameters(), lr=1e-4)
    load_optimizer_state(build_optimizer(model, {}), legacy.state_dict())


@pytest.mark.parametrize("capability,shared_memory,expected", [
    ((8, 9), 101376, (16, 8)),
    ((8, 0), 166912, (16, 8)),
    ((9, 0), 232448, (32, 16)),
    ((10, 0), 232448, (32, 16)),
    ((12, 0), 101376, (16, 8)),
])
def test_mamba_chunk_selection_respects_shared_memory_budget(
    monkeypatch, capability, shared_memory, expected,
) -> None:
    from types import SimpleNamespace

    from mambafold.model.bimamba3 import _default_chunk_size

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: SimpleNamespace(
        shared_memory_per_block_optin=shared_memory,
    ))
    assert (_default_chunk_size(2), _default_chunk_size(4)) == expected
    assert _default_chunk_size(1) == 64


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
@pytest.mark.parametrize("supervision", ["final", "lift"])
@pytest.mark.parametrize("optimizer_kind", ["adamw", "muon"])
@pytest.mark.parametrize("layout", ["plain", "refiner", "shared_trunk"])
def test_two_ddp_iterations_need_no_unused_parameter_detection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    supervision: str,
    optimizer_kind: str,
    layout: str,
) -> None:
    from mambafold.train.optim import build_optimizer

    # shared_trunk is the flat_fine6 layout: one prelude block, then a trunk
    # suffix reused on each of the two recurrent passes.
    residue_refiner = layout != "plain"
    shared = int(layout == "shared_trunk")

    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    rendezvous = tmp_path / "ddp-rendezvous"
    try:
        dist.init_process_group(
            "gloo",
            init_method=f"file://{rendezvous}",
            rank=0,
            world_size=1,
        )
    except RuntimeError as exc:
        if "Operation not permitted" in str(exc) or "Cannot resolve" in str(exc):
            pytest.skip(f"sandbox does not permit a local Gloo socket: {exc}")
        raise
    try:
        raw = _tiny_model()
        if supervision == "lift" or residue_refiner:
            from mambafold.model.bimamba3 import Mamba3Layer
            from mambafold.train.trainer import build_model

            def cpu_scan(self, x, mask):
                # Keep the substituted scan parameters in the graph so DDP
                # tests the surrounding recurrent/readout graph, not this stub.
                zero = sum(p.reshape(-1)[0] * 0 for p in self.parameters())
                return x.cumsum(dim=1) + zero

            monkeypatch.setattr(Mamba3Layer, "forward", cpu_scan)
            raw = build_model(dict(
                d_res=64, n_trunk=3 if shared else 1, shared_trunk_layers=shared,
                d_latent=64, n_latent_layers=1,
                n_atom_layers=0, n_atom_cross_layers=0, latent_compression=4,
                latent_supervision=supervision, residue_refiner=residue_refiner,
            ))
        model = DDP(raw, find_unused_parameters=False)
        optimizer = build_optimizer(model, dict(
            optimizer=optimizer_kind, lr=1e-3,
            backbone_lr_scale=0.1 if residue_refiner else 1.0,
        ))
        batch = _batch()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss, _ = allatom_loss_surface(
                output,
                batch,
                use_rigid_align=False,
                w_lddt_atom=0.0,
            )
            loss.backward()
            optimizer.step()
    finally:
        dist.destroy_process_group()


def _example(length: int) -> ProteinExample:
    atoms = MAX_ATOMS_PER_RES
    atom_mask = torch.zeros(length, atoms, dtype=torch.bool)
    atom_mask[:, :5] = True
    pair_type = torch.full((length, atoms), PAIR_PAD_ID, dtype=torch.long)
    pair_type[atom_mask] = 0
    return ProteinExample(
        res_type=torch.zeros(length, dtype=torch.long),
        atom_type=torch.zeros(length, atoms, dtype=torch.long),
        pair_type=pair_type,
        coords=torch.zeros(length, atoms, 3),
        atom_mask=atom_mask,
        observed_mask=atom_mask.clone(),
        res_seq_nums=torch.arange(length),
        seq_len=length,
    )


class _DummyVelocity(torch.nn.Module):
    def forward(self, batch: ProteinBatch) -> dict[str, torch.Tensor]:
        return {
            "v_atom": -0.25 * batch.x_t,
            "trunk_latent": batch.res_type.float().unsqueeze(-1),
        }


@pytest.mark.parametrize("method", ["ode", "sde"])
def test_batch_one_sampler_is_the_same_batch_first_path(method: str) -> None:
    model = _DummyVelocity()
    short, long = _example(2), _example(3)
    batched = sample(
        model,
        prepare_inference_batch([short, long], "cpu"),
        [3, 7],
        n_steps=3,
        method=method,
    )
    single = sample(
        model,
        prepare_inference_batch([short], "cpu"),
        [3],
        n_steps=3,
        method=method,
    )

    torch.testing.assert_close(batched.final_aa[0, :2], single.final_aa[0])
    torch.testing.assert_close(batched.final_ca[0, :2], single.final_ca[0])
    assert batched.residue_mask[0].tolist() == [True, True, False]
    assert not hasattr(batched, "confidence")


def test_noise_copies_share_one_plm_row_and_model_broadcasts_it() -> None:
    torch.manual_seed(17)
    example = _example(2)
    example.esm = torch.randn(2, 8)
    batch = ProteinCollator(
        augment=False,
        copies_per_protein=2,
        max_length=2,
    )([example])
    assert batch is not None
    assert batch.batch_size == 2
    assert batch.esm is not None and batch.esm.shape == (1, 2, 8)
    assert not torch.equal(batch.eps[0], batch.eps[1])

    model = _tiny_model(use_plm=True, d_plm=8)
    output = model(batch)
    assert output["v_atom"].shape[0] == 2
    output["v_atom"].sum().backward()
    assert model.plm_proj.weight.grad is not None


def test_yaml_config_rejects_unknown_keys(tmp_path: Path) -> None:
    config = tmp_path / "typo.yaml"
    config.write_text("w_confidence_typo: 1.0\n")
    with pytest.raises(SystemExit):
        parse_args(["--config", str(config), "--out_dir", str(tmp_path / "out")])


def test_active_latent_recipe_defaults_to_lift_with_explicit_final_override() -> None:
    recipe = Path(__file__).resolve().parents[1] / "configs" / "latent_mamba.yaml"
    args, _ = parse_args(["--config", str(recipe)])
    assert (args.latent_supervision, args.latent_compression, args.latent_loops) == ("lift", 4, 2)
    args, _ = parse_args(["--config", str(recipe), "--latent_supervision", "final"])
    assert args.latent_supervision == "final"


@pytest.mark.parametrize("compression", [0, 4, 8])
@pytest.mark.parametrize("loops", [1, 2])
@pytest.mark.parametrize("mode", ["plain", "conditioned"])
@pytest.mark.parametrize("position", ["midpoint", "post"])
def test_latent_config_cli_overrides_yaml_and_reaches_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, compression: int, loops: int, mode: str,
    position: str,
) -> None:
    from mambafold.model.bimamba3 import Mamba3Layer
    from mambafold.train.trainer import build_model

    def cpu_scan(self, x, mask=None):
        return x.masked_fill(~mask.unsqueeze(-1), 0).cumsum(dim=1)

    monkeypatch.setattr(Mamba3Layer, "forward", cpu_scan)
    config = tmp_path / "latent.yaml"
    config.write_text(
        "latent_compression: 4\nn_trunk: 2\nd_res: 64\nd_latent: 64\n"
        "n_latent_layers: 1\nn_atom_layers: 0\nn_atom_cross_layers: 0\nlatent_loops: 3\n"
        "latent_loop_mode: conditioned\n"
    )
    args, _ = parse_args([
        "--config", str(config), "--latent_compression", str(compression),
        "--latent_loops", str(loops),
        "--latent_loop_mode", mode, "--latent_position", position,
    ])
    model = build_model(vars(args))
    assert model.latent_position == position
    if compression:
        assert model.latent_residual.compression == compression
        assert model.latent_residual.stack.n_layers == 1
        assert model.latent_residual.n_loops == loops
        assert model.latent_residual.loop_mode == mode
        assert model.latent_residual.compress_proj.out_features == 64
        restored = build_model(vars(args))
        restored.load_state_dict(model.state_dict(), strict=True)
    else:
        assert model.latent_residual is None
        assert not any(key.startswith("latent_residual.") for key in model.state_dict())
    output = model(_batch(length=13))
    assert output["v_atom"].shape == (2, 13, MAX_ATOMS_PER_RES, 3)
    assert output["trunk_latent"].shape == (2, 13, 64)
    (output["trunk_latent"].square().mean() + output["v_atom"].square().mean()).backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for name, p in model.named_parameters() if ".ssm." not in name
    )


@pytest.mark.parametrize("settings", [
    "latent_compression: 2",
    "latent_compression: 4\nd_latent: 33", "n_latent_layers: 0", "d_latent: 0",
    "latent_loops: 0", "latent_loops: -2", "latent_loops: 1.5", "latent_loops: true",
    "latent_loop_mode: invalid", "latent_loop_mode: true",
    "latent_position: side", "latent_position: true",
    "latent_compression: 4\nn_trunk: 1\nlatent_position: midpoint",
    "latent_supervision: invalid", "latent_supervision: true", "latent_supervision: lift",
    "latent_supervision: lift\nlatent_compression: 4\nlatent_loop_mode: plain",
    "latent_supervision: lift\nlatent_compression: 4\nlatent_position: midpoint",
])
def test_yaml_rejects_invalid_latent_settings(tmp_path: Path, settings: str) -> None:
    config = tmp_path / "invalid_latent.yaml"
    config.write_text(settings + "\n")
    with pytest.raises(SystemExit):
        parse_args(["--config", str(config)])


def test_post_latent_runs_after_whole_trunk_and_starts_as_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mambafold.model.bimamba3 import Mamba3Layer
    from mambafold.train.trainer import build_model

    def cpu_scan(self, x, mask=None):
        return x.masked_fill(~mask.unsqueeze(-1), 0).cumsum(dim=1)

    monkeypatch.setattr(Mamba3Layer, "forward", cpu_scan)
    # One trunk block is enough after the trunk; midpoint still needs two.
    config = tmp_path / "post.yaml"
    config.write_text(
        "latent_compression: 4\nlatent_position: post\nn_trunk: 1\nd_res: 64\n"
        "d_latent: 64\nn_latent_layers: 1\nn_atom_layers: 0\nn_atom_cross_layers: 0\n"
        "latent_loops: 2\nlatent_loop_mode: conditioned\n"
    )
    args, _ = parse_args(["--config", str(config)])
    torch.manual_seed(0)
    enabled = build_model(vars(args))
    baseline = build_model(dict(vars(args), latent_compression=0))
    midpoint = build_model(dict(vars(args), n_trunk=2, latent_position="midpoint"))
    post = build_model(dict(vars(args), n_trunk=2))
    assert set(midpoint.state_dict()) == set(post.state_dict())
    incompatible = enabled.load_state_dict(baseline.state_dict(), strict=False)
    assert incompatible.unexpected_keys == []
    assert set(incompatible.missing_keys) == {
        key for key in enabled.state_dict() if key.startswith("latent_residual.")
    }
    events = []
    handles = [enabled.residue_trunk.layers[0].register_forward_hook(
        lambda *args: events.append("trunk")
    ), enabled.latent_residual.register_forward_hook(lambda *args: events.append("latent"))]
    batch = _batch(length=13)
    expected = baseline(batch)
    actual = enabled(batch)
    for handle in handles:
        handle.remove()
    assert events == ["trunk", "latent"]
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    with torch.no_grad():
        enabled.latent_residual.output_proj.weight.normal_(std=0.03)
    assert not torch.equal(enabled(batch)["trunk_latent"], expected["trunk_latent"])


@pytest.mark.parametrize("compression", [4, 8])
@pytest.mark.parametrize("residue_refiner", [False, True])
def test_lift_config_training_and_sampling_use_one_shared_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, compression: int, residue_refiner: bool,
) -> None:
    from mambafold.model.bimamba3 import AdaLNZero, Mamba3Layer
    from mambafold.train.engine import allatom_forward_and_loss
    from mambafold.train.trainer import build_model

    monkeypatch.setattr(Mamba3Layer, "forward", lambda self, x, mask: x.cumsum(dim=1))
    config = tmp_path / "lift.yaml"
    config.write_text(
        f"latent_compression: {compression}\nlatent_supervision: lift\n"
        f"residue_refiner: {str(residue_refiner).lower()}\n"
        "d_res: 64\nn_trunk: 1\nd_latent: 64\nn_latent_layers: 1\n"
        "n_atom_layers: 0\nn_atom_cross_layers: 0\n"
    )
    args, _ = parse_args(["--config", str(config)])
    assert (args.latent_position, args.latent_loop_mode, args.latent_loops) == (
        "post", "conditioned", 2,
    )
    model = build_model(vars(args))
    with torch.no_grad():
        model.latent_residual.output_proj.weight.normal_(std=0.03)
        model.atom_decoder.out.weight.normal_(std=0.03)
        for module in model.modules():
            if isinstance(module, AdaLNZero):
                module.proj.bias.normal_(std=0.1)
                module.proj.weight.normal_(std=0.01)
    final = build_model(dict(vars(args), latent_supervision="final"))
    final.load_state_dict(model.state_dict(), strict=True)
    batch = _batch(length=13).pad_to_length(16)
    calls = []
    handles = [module.register_forward_hook(
        lambda *args, label=label: calls.append(label)
    ) for label, module in (
        ("encoder", model.atom_encoder), ("trunk", model.residue_trunk),
        ("decoder", model.atom_decoder),
    )]
    torch.manual_seed(91)
    final_out = final(batch)
    torch.manual_seed(91)
    lift_out = model(batch)
    assert calls.count("encoder") == calls.count("trunk") == 1
    assert calls.count("decoder") == 3
    if residue_refiner:
        assert len(model.latent_residual.residue_stack.layers) == 3
    assert len(lift_out["v_atom_steps"]) == 3
    torch.testing.assert_close(lift_out["v_atom"], final_out["v_atom"], rtol=0, atol=0)
    for velocity in lift_out["v_atom_steps"]:
        assert torch.count_nonzero(velocity[~batch.atom_mask]) == 0
    loss, metrics = allatom_forward_and_loss(model, batch, use_amp=False, w_lddt_atom=0)
    assert "fm_trajectory" in metrics and "fm_prelude" in metrics
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for name, p in model.named_parameters() if ".ssm." not in name)
    assert model.latent_residual.depth_embed.mlp[-1].weight.grad.abs().sum() > 0
    assert model.latent_residual.compress_proj.weight.grad.abs().sum() > 0

    # Neither eval nor a no-grad self-conditioning pass decodes intermediates.
    for evaluation in (False, True):
        model.train(not evaluation)
        calls.clear()
        with torch.no_grad():
            out = model(batch)
        assert set(out) == {"v_atom", "trunk_latent"}
        assert calls.count("decoder") == 1
    model.eval()
    calls.clear()
    model(batch)
    assert calls.count("decoder") == 1
    for handle in handles:
        handle.remove()
    restored = build_model(vars(args)).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored(batch)["v_atom"], model(batch)["v_atom"])


def test_mamba_resource_probe_supplies_required_time_condition(monkeypatch: pytest.MonkeyPatch):
    from benchmarks import probe_mamba3_dstate as probe
    from mambafold.model.bimamba3 import Mamba3Layer

    # Run the real probe/block call path on CPU. CUDA kernels and resource
    # counters are substituted; this verifies API wiring, not GPU capacity.
    def cpu_factory(factory):
        def create(*args, **kwargs):
            kwargs["device"] = "cpu"
            return factory(*args, **kwargs)
        return create

    monkeypatch.setattr(torch, "randn", cpu_factory(torch.randn))
    monkeypatch.setattr(torch, "ones", cpu_factory(torch.ones))
    monkeypatch.setattr(probe.BiMamba3Block, "cuda", lambda self: self)
    monkeypatch.setattr(Mamba3Layer, "forward", lambda self, x, mask: x.cumsum(dim=1))
    for name in ("empty_cache", "reset_peak_memory_stats", "synchronize"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    result = probe.try_case(64, 64, 4, 64, 2, 17, torch.float32)
    assert result["status"] == "ok", result.get("error")
    assert result["forward"] == result["backward"] == "ok"


def test_length_bucketed_sampler_draws_uniformly_and_groups_by_length() -> None:
    from mambafold.data.length_sampler import LengthBucketedDistributedBatchSampler

    # 200 short chains and 200 long ones. The weighting this sampler used to
    # apply unconditionally, w = clip((L/200)**0.5, 1.0, 1.5), scores 1.0 at
    # L=100 and 1.5 at L=900, so it would put the long half at 0.60. Uniform is
    # 0.50, and 4000 draws give a standard error of 0.008 — the two are ~12
    # sigma apart, so this band separates them without being flaky.
    index_lengths = {i: (100 if i < 200 else 900) for i in range(400)}
    sampler = LengthBucketedDistributedBatchSampler(
        index_lengths=index_lengths, batch_size=4, rank=0, world_size=1, seed=0,
        num_samples_per_rank=4000,
    )
    drawn = [i for batch in sampler for i in batch]
    long_share = sum(i >= 200 for i in drawn) / len(drawn)
    assert 0.47 < long_share < 0.53, f"draw is length-weighted: {long_share:.3f} long"

    # Batches are cut from length-sorted megabatches, so exactly one batch per
    # megabatch can straddle the short/long boundary. Everything else must be
    # length-pure, which is the property the collator's padding depends on.
    mixed = sum(1 for batch in sampler if len({index_lengths[i] for i in batch}) > 1)
    assert mixed <= math.ceil(len(sampler) / sampler.megabatch_mult), (
        f"{mixed} of {len(sampler)} batches mix lengths; bucketing is not grouping"
    )


def test_chain_index_cache_key_ignores_how_the_path_was_spelled(tmp_path: Path) -> None:
    """The prebuilt index must be findable by the loader that consumes it.

    `pipeline/13_prebuild_loader_caches.py` constructs its dataset from absolute
    paths (`ROOT / source["data_dir"]`) while the training loader passes the
    config's relative string through unchanged. The cache key hashes the
    directory and the file list, so the two spellings produced different keys:
    the stage spent an hour building an index training could never find, and the
    only symptom was that training rebuilt it silently.
    """
    from mambafold.data.dataset import RCSBDataset
    from mambafold.data.length_cache import _DEFAULT_CACHE_DIR, _cache_path

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name in ("a.npz", "b.npz"):
        (corpus / name).touch()

    def key(data_dir: str) -> str:
        dataset = RCSBDataset(
            data_dir=data_dir, max_length=1024, extract_monomer_chains=False
        )
        return _cache_path(dataset, Path(_DEFAULT_CACHE_DIR)).name

    relative = os.path.relpath(corpus, Path.cwd())
    assert key(str(corpus)) == key(relative), (
        "absolute and relative spellings of one corpus hash differently, so a "
        "prebuilt cache cannot be found by the training loader"
    )


def test_chain_cache_fingerprint_matches_pathlib_for_links_and_retargeting(tmp_path):
    import hashlib
    import json

    from mambafold.data.dataset import RCSBDataset
    from mambafold.data.length_cache import _cache_path, _dataset_cfg

    corpus = tmp_path / "corpus"
    external = tmp_path / "external"
    corpus.mkdir()
    external.mkdir()
    (external / "one.npz").touch()
    (external / "two.npz").touch()
    (corpus / "plain.npz").touch()
    link = corpus / "alias.npz"
    link.symlink_to(external / "one.npz")
    (corpus / "chain.npz").symlink_to("alias.npz")
    (corpus / "dangling.npz").symlink_to("missing/sub/file.npz")
    (external / "branch").mkdir()
    (corpus / "parent.npz").symlink_to(external / "branch" / "..")
    (tmp_path / "dir-alias").symlink_to(corpus, target_is_directory=True)
    dataset = RCSBDataset(str(tmp_path / "dir-alias"), max_length=1024)

    def reference():
        cfg = _dataset_cfg(dataset)
        digest = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode())
        digest.update(str(len(dataset.files)).encode())
        root = Path(cfg["data_dir"])
        identities = []
        for file in dataset.files:
            canonical = file.resolve()
            try:
                identities.append(str(canonical.relative_to(root)))
            except ValueError:
                identities.append(str(canonical))
        digest.update("\n".join(identities).encode())
        return digest.hexdigest()[:12]

    before = _cache_path(dataset, tmp_path)
    assert before.stem.endswith(reference())
    link.unlink()
    link.symlink_to(external / "two.npz")
    after = _cache_path(dataset, tmp_path)
    assert after.stem.endswith(reference())
    assert before != after
    link.unlink()
    link.symlink_to("chain.npz")
    with pytest.raises((RuntimeError, OSError)):
        _cache_path(dataset, tmp_path)


class _StreamProbe(torch.utils.data.Dataset):
    """Reports the draws a DataLoader worker makes from its seeded RNGs."""

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> tuple[float, float, float]:
        import random

        import numpy as np

        return torch.rand(()).item(), float(np.random.rand()), random.random()


def _train_script():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts/train.py"
    spec = importlib.util.spec_from_file_location("train_script_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rank_draws(train, seed: int, rank: int | None):
    """Seed as main() does, then record host draws, worker draws and sampler slices."""
    import random

    import numpy as np

    from mambafold.data.length_sampler import LengthBucketedDistributedBatchSampler
    from mambafold.train.trainer import capture_rng_state, seed_all

    seed_all(seed)
    torch.randn(7)  # model construction consumes the shared stream identically
    if rank is not None:
        train.seed_rank_streams(seed, rank)
    state = capture_rng_state()
    loader = torch.utils.data.DataLoader(_StreamProbe(), batch_size=None, num_workers=1)
    workers = [tuple(item) for item in loader]
    host = (torch.rand(3).tolist(), float(np.random.rand()), random.random())
    sampler = LengthBucketedDistributedBatchSampler(
        index_lengths={i: 50 + i for i in range(64)}, batch_size=1,
        rank=rank or 0, world_size=4, seed=seed,
    )
    return state, workers, host, list(sampler)


def test_rank_streams_leave_rank_zero_unchanged_and_separate_other_ranks() -> None:
    """DDP ranks need their own crop/t/noise/rotation streams; one process must not change.

    Before seed_rank_streams every rank was seeded identically and the loader
    has no generator, so each rank's workers got the same base seed and drew
    the same corruption for their different proteins.
    """
    import numpy as np

    train = _train_script()
    seed = 20261023
    assert train.rank_stream_seed(seed, 0) == seed
    derived = [train.rank_stream_seed(seed, rank) for rank in range(1, 8)]
    assert len(set(derived)) == 7 and all(0 <= value < 2**32 for value in derived)
    assert seed + 1 not in derived and derived == [train.rank_stream_seed(seed, r) for r in range(1, 8)]
    with pytest.raises(ValueError):
        train.rank_stream_seed(seed, -1)

    baseline = _rank_draws(train, seed, None)
    ranks = [_rank_draws(train, seed, rank) for rank in range(4)]
    # Rank 0 (and so any world_size 1 run) keeps the exact pre-change RNG state and draws.
    state, base_state = ranks[0][0], baseline[0]
    assert torch.equal(state["torch"], base_state["torch"])
    assert state["python"] == base_state["python"]
    assert all(np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b
               for a, b in zip(state["numpy"], base_state["numpy"]))
    assert ranks[0][1:3] == baseline[1:3]
    # Every rank draws different worker (crop/t/noise/SE3) and host/CUDA streams.
    assert len({tuple(r[1]) for r in ranks}) == 4
    assert len({repr(r[2]) for r in ranks}) == 4
    # The batch sampler's global draw stays shared: rank r takes slot r of the
    # same length-sorted global batch a one-rank batch-4 sampler would yield.
    from mambafold.data.length_sampler import LengthBucketedDistributedBatchSampler

    global_batches = list(LengthBucketedDistributedBatchSampler(
        index_lengths={i: 50 + i for i in range(64)}, batch_size=4, rank=0, world_size=1, seed=seed,
    ))
    assert len(global_batches) == len(ranks[0][3]) == 16
    for step, members in enumerate(global_batches):
        assert [r[3][step] for r in ranks] == [[index] for index in members]


def test_single_rank_accumulation_over_one_protein_is_runnable() -> None:
    """A one-GPU run must be able to start.

    `accum_same_protein` repeats indices through the batch sampler, and the
    batch sampler only existed when `batch_size * world_size > 1`. On one rank
    at batch_size 1 that product is 1, so the sampler was absent and the loader
    raised — the configuration this project trains could not be run on a single
    GPU at all, which is exactly the configuration anyone debugging would reach
    for. Measured as a hard failure of the one-rank stage of the DDP smoke.
    """
    from mambafold.data.loader import LengthBucketedDistributedBatchSampler, RepeatBatchSampler

    base = LengthBucketedDistributedBatchSampler(
        index_lengths={i: 100 + i for i in range(32)},
        batch_size=1, rank=0, world_size=1, seed=0,
    )
    repeated = RepeatBatchSampler(base, 2)
    batches = list(repeated)
    assert len(batches) == 2 * len(base)
    # Consecutive pairs must be the same indices: that identity is what makes
    # accumulation over a protein equal to one pass with twice the copies.
    assert all(batches[i] == batches[i + 1] for i in range(0, len(batches) - 1, 2))


@pytest.mark.parametrize("settings", [
    'residue_refiner: "true"\n',
    'residue_refiner: true\n',
    'residue_refiner: true\nlatent_compression: 4\nlatent_loop_mode: plain\n',
    'residue_refiner: true\nlatent_compression: 4\nlatent_position: midpoint\n',
])
def test_residue_refiner_rejects_invalid_yaml_contract(tmp_path, settings):
    path = tmp_path / 'invalid-refiner.yaml'
    path.write_text(settings)
    with pytest.raises(SystemExit):
        parse_args(['--config', str(path)])


@pytest.mark.parametrize('extra', [[], ['--resume', 'old.pt'], ['--reset_optimizer'],
                                   ['--start_step', '1']])
def test_refiner_initialization_is_separate_from_resume(tmp_path, extra):
    config = tmp_path / 'refiner.yaml'
    config.write_text('residue_refiner: true\nlatent_compression: 4\n')
    argv = ['--config', str(config), '--initialize_refiner_from', 'source.pt', *extra]
    if extra:
        with pytest.raises(SystemExit):
            parse_args(argv)
    else:
        args, _ = parse_args(argv)
        assert args.initialize_refiner_from == 'source.pt' and not args.resume


def test_refiner_checkpoint_initialization_uses_ema_and_fresh_optimizer(tmp_path):
    from mambafold.model.bimamba3 import mamba3_runtime_info
    from mambafold.train.ema import EMA
    from mambafold.train.optim import build_optimizer
    from mambafold.train.trainer import build_model, initialize_refiner_from_checkpoint

    cfg = dict(d_res=64, n_trunk=2, d_latent=64, n_latent_layers=1,
               n_atom_layers=0, n_atom_cross_layers=0, use_plm=False)
    source = build_model(cfg)
    shadow = {key: value.clone() for key, value in source.state_dict().items()}
    shadow['trunk_proj.weight'].fill_(.125)
    path = tmp_path / 'source.pt'
    torch.save(dict(model=source.state_dict(), ema=shadow, args=cfg, step=20000,
                    mamba3_runtime=mamba3_runtime_info(),
                    optimizer='deliberately incompatible: must never restore'), path)
    target_cfg = dict(cfg, residue_refiner=True, latent_compression=4)
    target = build_model(target_cfg)
    receipt = initialize_refiner_from_checkpoint(target, path, target_cfg)
    torch.testing.assert_close(
        target.trunk_proj.weight, shadow['trunk_proj.weight'], rtol=0, atol=0,
    )
    assert receipt['weight_source'] == 'ema' and receipt['checkpoint_step'] == 20000
    assert torch.count_nonzero(target.latent_residual.output_proj.weight) == 0
    ema = EMA(target, decay=.999)
    torch.testing.assert_close(ema.state_dict()['trunk_proj.weight'], target.trunk_proj.weight)
    optimizer = build_optimizer(target, dict(optimizer='muon', lr=1e-4, muon_lr=3e-3))
    assert all(not child.state for child in optimizer.optimizers.values())


@pytest.mark.parametrize('runtime', [None, {'decay': 'softplus'}, {'module_sha256': 'different'}])
def test_refiner_initialization_rejects_missing_or_changed_runtime(tmp_path, runtime):
    from mambafold.train.trainer import build_model, initialize_refiner_from_checkpoint

    cfg = dict(d_res=64, n_trunk=2, d_latent=64, n_latent_layers=1,
               n_atom_layers=0, n_atom_cross_layers=0, use_plm=False)
    source = build_model(cfg)
    path = tmp_path / 'different-runtime.pt'
    torch.save(dict(ema=source.state_dict(), args=cfg, step=1, mamba3_runtime=runtime), path)
    target_cfg = dict(cfg, residue_refiner=True, latent_compression=4)
    target = build_model(target_cfg)
    before = target.trunk_proj.weight.detach().clone()
    with pytest.raises(ValueError, match='same recorded Mamba3 runtime'):
        initialize_refiner_from_checkpoint(target, path, target_cfg)
    torch.testing.assert_close(target.trunk_proj.weight, before, rtol=0, atol=0)
