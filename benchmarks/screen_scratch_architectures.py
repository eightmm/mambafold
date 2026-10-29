"""Ground-truth architecture screens with fresh model weights and a 10k WSD cap.

The preflight performs backward passes without optimizer updates. Formal training
constructs another seeded model, starts its data stream over, and never loads a
checkpoint. Cached PLM features remain common conditioning inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from benchmarks import compare_full_refinement as audit  # noqa: E402
from benchmarks import compare_shared_trunk as screen  # noqa: E402
from benchmarks.screen_architectures import (  # noqa: E402
    VARIANTS,
    validate_anchor,
    verify_metrics,
)
from mambafold.train import engine, optim, trainer  # noqa: E402

TOTAL_STEPS = 10000
PREFLIGHT_BATCHES = 8


def expected_config(arm):
    if arm not in VARIANTS:
        raise ValueError("undeclared architecture")
    return dict(
        n_trunk=15, shared_trunk_layers=0, latent_loops=2,
        latent_compression=4, n_latent_layers=4, latent_hierarchy=False,
        latent_loop_sampling=False, residue_refiner=True, atom_recurrence=False,
        d_res=1024, d_latent=512, d_atom=128, d_temb=256, d_ca_emb=128,
        d_res_type=32, d_res_pos=64, d_plm=2560, d_plm_proj=1024,
        d_state=64, mimo_rank=4, expand=2, headdim=64, bimamba_share=False,
        n_atom_layers=4, atom_mixer="mlp", atom_d_state=64, atom_mimo_rank=2,
        n_atom_cross_layers=1, n_backbone_streams=5, use_plm=True,
        latent_loop_mode="conditioned", latent_position="post", latent_supervision="lift",
        self_conditioning=False, self_condition_prob=0.0,
        w_fm=1.0, w_lddt_atom=1.0, w_bond=0.0, w_angle=0.0, w_clash=0.0,
        lddt_cutoff_A=15.0, lddt_pair_chunk_size=512,
        alpha_mode="const", use_rigid_align=True, grad_clip=2.0,
        batch_size=1, copies_per_protein=4, grad_accum_steps=4,
        accum_same_protein=True, seed=20261023, optimizer="muon",
        muon_lr=0.003, lr=0.0001, min_lr=0.000001,
        muon_adjust_lr_fn="original", backbone_lr_scale=1.0, weight_decay=0.0,
        warmup_steps=500, lr_cooldown_steps=2000, total_steps=TOTAL_STEPS,
        ema_decay=0.999, max_length=1024, length_bin=64, length_bucketing=True,
        t_schedule="logit_normal", t_uniform_weight=0.1, prewarm_kernels=True,
    ) | VARIANTS[arm]


def reject_initialization(cfg):
    forbidden = (
        "resume", "initialize_refiner_from", "initialize_model_from_ema",
        "reset_optimizer", "start_step", "expected_resume_step", "expected_missing_resume_keys",
    )
    if any(cfg.get(key) not in (None, False, 0, [], "") for key in forbidden):
        raise ValueError("scratch training forbids checkpoint initialization, resume and offsets")


def validate_config(cfg, arm):
    if not isinstance(cfg, dict):
        raise ValueError("training config must be a mapping")
    reject_initialization(cfg)
    expected = expected_config(arm)
    if cfg.get("recycle_policy", "legacy") != expected.get("recycle_policy", "legacy"):
        raise ValueError("recycle policy differs from the declared scratch architecture")
    if cfg.get("atom_recycling", "none") != expected.get("atom_recycling", "none"):
        raise ValueError("atom recycling differs from the declared scratch architecture")
    wrong = [key for key, value in expected.items() if cfg.get(key, 0 if key.startswith("w_")
                                                               else None) != value]
    if wrong or engine.LIFT_PRELUDE_WEIGHT != 0.01:
        raise ValueError("configuration differs from scratch WSD screen: " + ", ".join(wrong))
    if type(cfg.get("num_workers")) is not int or cfg["num_workers"] < 1:
        raise ValueError("scratch screen requires a worker-backed input stream")
    interval = cfg.get("log_interval")
    if type(interval) is not int or interval < 1 or TOTAL_STEPS % interval:
        raise ValueError("log_interval must divide the complete 10k budget")


def actual_training_loops(cfg, step):
    """Resolve the executed policy without changing the nominal model layout."""
    policy = cfg.get("recycle_policy", "legacy")
    if policy == "sampled16_tail2":
        return trainer.sample_recycle_loops(step, cfg["seed"])
    if policy == "fixed2":
        return 2
    if policy == "fixed4":
        return 4
    if policy != "legacy":
        raise ValueError("unknown recycle training policy")
    return (trainer.sample_refinement_loops(step, cfg["seed"])
            if cfg["latent_loop_sampling"] else cfg["latent_loops"])


def schedule_multiplier(step):
    if type(step) is not int or not 0 <= step <= TOTAL_STEPS:
        raise ValueError("schedule index must be within the declared 10k horizon")
    floor = 0.01
    if step < 500:
        return floor + (1 - floor) * step / 500
    if step < 8000:
        return 1.0
    return floor + (1 - floor) * (1 + math.cos(math.pi * (step - 8000) / 2000)) / 2


def verify_group_lrs(optimizer, step):
    expected_kinds = {"muon", "adamw"}
    rows = []
    for group in optimizer.param_groups:
        kind = group.get("optimizer_kind")
        peak = {"muon": 0.003, "adamw": 0.0001}.get(kind)
        if peak is None or group.get("lr_scale", 1.0) != 1 or group.get("parameter_role"):
            raise ValueError("scratch optimizer has an undeclared kind or pretrained LR role")
        expected = peak * schedule_multiplier(step)
        if not math.isclose(group["lr"], expected, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError(f"actual {kind} LR differs at schedule index {step}")
        rows.append(dict(kind=kind, lr=group["lr"]))
    if {row["kind"] for row in rows} != expected_kinds or len(rows) != 2:
        raise ValueError("expected exactly one Muon and one AdamW group")
    return rows


def verify_schedule(scheduler):
    boundaries = (0, 1, 499, 500, 7999, 8000, 9000, 9999, 10000)
    if scheduler.last_epoch != 0 or len(scheduler.lr_lambdas) != 2:
        raise ValueError("scheduler did not start fresh at index zero")
    for step in boundaries:
        if any(not math.isclose(fn(step), schedule_multiplier(step), abs_tol=1e-15)
               for fn in scheduler.lr_lambdas):
            raise ValueError("existing scheduler differs from declared WSD boundaries")
    verify_group_lrs(scheduler.optimizer, 0)
    return {str(step): dict(multiplier=schedule_multiplier(step),
                           muon_lr=0.003 * schedule_multiplier(step),
                           adamw_lr=0.0001 * schedule_multiplier(step)) for step in boundaries}


@contextmanager
def forbid_checkpoint_loads():
    original = torch.load

    def forbidden(*unused, **kwargs):
        raise ValueError("scratch execution forbids all checkpoint loads")

    torch.load = forbidden
    try:
        yield
    finally:
        torch.load = original


def initial_snapshot(model, cfg, out):
    from mambafold.model.bimamba3 import mamba3_runtime_info

    if any(not p.requires_grad for p in model.parameters()):
        raise ValueError("every scratch model parameter must be trainable")
    digest = audit.tensor_digest(model.state_dict().items())
    runtime = mamba3_runtime_info(model)
    state = dict(checkpoint_version=2, args=dict(cfg), model=model.state_dict(),
                 ema=model.state_dict(), step=0, mamba3_runtime=runtime,
                 initialization="fresh seeded model constructor; no donor checkpoint")
    torch.save(state, out / "initial_ema.pt")
    receipt = dict(initialization="scratch", seed=cfg["seed"], source_checkpoint=None,
                   checkpoint_loads=0, teacher_models=0, distilled_examples=0,
                   optimizer="fresh", ema="fresh copy of random model", data_cursor="fresh",
                   parameters=sum(p.numel() for p in model.parameters()),
                   trainable_parameters=sum(p.numel() for p in model.parameters()
                                            if p.requires_grad),
                   target_sha256=digest, initial_ema_sha256=digest,
                   initial_snapshot_sha256=screen.sha256(out / "initial_ema.pt"),
                   shared_trunk_layers=model.shared_trunk_layers,
                   prelude_layers=model.residue_trunk.n_layers,
                   hierarchy=bool(cfg["latent_hierarchy"]), mamba3_runtime=runtime)
    audit.write_json(out / "initialization.json", receipt)
    return receipt


def _read_config(args):
    cfg = yaml.safe_load(args.config.read_text())
    validate_config(cfg, args.arm)
    from mambafold.train.config import parse_args

    parsed, _ = parse_args(["--config", str(args.config), "--out_dir", str(args.out), "--no_wandb"])
    validate_config(vars(parsed), args.arm)
    return cfg, parsed


def read_native_receipt(path, cfg, arm, config_path):
    payload = path.read_bytes()
    receipt = json.loads(payload)
    expected = dict(arm=arm, seed=cfg["seed"], config_sha256=screen.sha256(config_path),
                    native_backward_microbatches=PREFLIGHT_BATCHES,
                    declared_training_updates=TOTAL_STEPS, schedule_horizon=TOTAL_STEPS,
                    checkpoint_loads=0, teacher_models=0, optimizer_updates=0)
    digest = receipt.get("model_sha256", "")
    if (any(type(receipt.get(key)) is not type(value) or receipt[key] != value
            for key, value in expected.items())
            or receipt.get("model_and_ema_unchanged") is not True
            or not isinstance(digest, str) or len(digest) != 64
            or any(letter not in "0123456789abcdef" for letter in digest)):
        raise ValueError("native receipt does not bind the exact zero-update scratch configuration")
    return dict(receipt, receipt_sha256=hashlib.sha256(payload).hexdigest())


def bind_initial_to_native(initial, native, path, out):
    if initial["target_sha256"] != native["model_sha256"]:
        raise ValueError("fresh training initialization differs from admitted native random model")
    initial.update(native_receipt=str(path), native_receipt_sha256=native["receipt_sha256"],
                   native_initial_model_sha256=native["model_sha256"],
                   native_initialization_match=True)
    audit.write_json(out / "initialization.json", initial)


def _gradient_rows(model):
    rows = {}
    for prefix in ("atom_encoder.", "residue_trunk.", "latent_residual.", "atom_decoder."):
        parameters = [p for name, p in model.named_parameters() if name.startswith(prefix)]
        gradients = [p.grad for p in parameters if p.grad is not None]
        if not parameters or not gradients or any(not torch.isfinite(g).all() for g in gradients):
            raise ValueError(f"missing/nonfinite scratch gradient group: {prefix}")
        rows[prefix] = dict(tensors=len(parameters), connected=len(gradients),
                            norm=math.sqrt(sum(float(g.float().square().sum()) for g in gradients)))
    return rows


def preflight(args):
    """Eight native backwards; the fresh model, EMA and optimizer stay unmodified."""
    from mambafold.data.loader import build_dataloaders
    from mambafold.train.distributed import enable_cuda_perf_flags
    from mambafold.train.ema import EMA

    _, parsed = _read_config(args)
    cfg = vars(parsed)
    args.out.mkdir(parents=True, exist_ok=False)
    enable_cuda_perf_flags()
    with forbid_checkpoint_loads():
        loader, _, _ = build_dataloaders(parsed, False)
        loader.generator = torch.Generator().manual_seed(cfg["seed"])
        trainer.seed_all(cfg["seed"])
        model = trainer.build_model(cfg, "cuda").train()
        trainer.seed_all(cfg["seed"])
        initial = initial_snapshot(model, cfg, args.out)
        ema = EMA(model, decay=cfg["ema_decay"])
        optimizer = optim.build_optimizer(model, cfg)
        scheduler = trainer.linear_warmup_lr(
            optimizer, 500, cfg["lr"], cfg["min_lr"], total_steps=TOTAL_STEPS, cooldown_steps=2000)
        boundaries = verify_schedule(scheduler)
        rows = []
        iterator = iter(loader)
        for index in range(PREFLIGHT_BATCHES):
            batch = next(iterator)
            identity = audit.batch_digest(batch)
            batch = batch.to("cuda")
            model.refinement_training_loops = actual_training_loops(
                cfg, index // cfg["grad_accum_steps"])
            loops = model.refinement_training_loops or model.latent_residual.n_loops
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(batch)
            anchor = validate_anchor(model, batch, output, cfg)
            loss, metrics = engine.allatom_loss_surface(
                output, batch, alpha_mode=cfg["alpha_mode"], use_rigid_align=cfg["use_rigid_align"],
                w_fm=cfg["w_fm"], w_lddt_atom=cfg["w_lddt_atom"],
                lddt_cutoff_A=cfg["lddt_cutoff_A"],
                lddt_pair_chunk_size=cfg["lddt_pair_chunk_size"])
            verify_metrics([dict(metrics, loss=float(loss.detach()), update_applied=1.0)],
                           dict(cfg, latent_loops=loops, latent_loop_sampling=False), actual_loops=loops)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite native scratch loss")
            loss.backward()
            gradients = _gradient_rows(model)
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"]))
            if not math.isfinite(norm) or norm >= 1e4:
                raise ValueError("native scratch gradient guard would skip an update")
            rows.append(dict(microbatch=index + 1, input_sha256=identity, K=loops,
                             loss=float(loss.detach()), grad_norm=norm, gradients=gradients,
                             ground_truth_metrics=metrics,
                             **anchor))
            del output, loss, batch
        if (audit.tensor_digest(model.state_dict().items()) != initial["target_sha256"]
                or audit.tensor_digest(ema.state_dict().items()) != initial["target_sha256"]
                or any(child.state for child in optimizer.optimizers.values())):
            raise ValueError("zero-update preflight changed model/EMA/optimizer state")
        optimizer.zero_grad(set_to_none=True)
        model.refinement_training_loops = None
    audit.write_json(args.out / "native_verified.json", dict(
        arm=args.arm, seed=cfg["seed"], config_sha256=screen.sha256(args.config),
        model_sha256=initial["target_sha256"], optimizer_updates=0,
        declared_training_updates=TOTAL_STEPS, schedule_horizon=TOTAL_STEPS,
        native_backward_microbatches=PREFLIGHT_BATCHES, checkpoint_loads=0,
        model_and_ema_unchanged=True, teacher_models=0, schedule_boundaries=boundaries,
        rows=rows, device=torch.cuda.get_device_name(),
        limits="Zero output head initialization can give zero backbone gradients before the "
        "first optimizer update. This is not peak-LR or optimizer-kernel admission."))


def train(args):
    from mambafold.data.types import ProteinBatch
    from mambafold.train.ema import EMA

    cfg, _ = _read_config(args)
    native = read_native_receipt(args.native_receipt, cfg, args.arm, args.config)
    args.out.mkdir(parents=True, exist_ok=False)
    originals = dict(build=trainer.build_model, prewarm=trainer.prewarm_sequence_kernels,
                     schedule=trainer.linear_warmup_lr, optimizer=optim.build_optimizer,
                     forward=engine.allatom_forward_and_loss,
                     transfer=ProteinBatch.to, clip=torch.nn.utils.clip_grad_norm_,
                     ema=EMA.update, argv=sys.argv[:])
    micro_depths, lr_rows = Counter(), []
    model_ref, schedule_boundaries = None, None

    def build(config, device):
        nonlocal model_ref
        reject_initialization(config)
        model_ref = originals["build"](config, device)
        # Architecture-dependent constructor draws must not shift corruption/LiFT RNG.
        trainer.seed_all(config["seed"])
        return model_ref

    def optimizer(model, config):
        result = originals["optimizer"](model, config)
        initial = initial_snapshot(model, config, args.out)
        bind_initial_to_native(initial, native, args.native_receipt, args.out)
        original_step = result.step

        def step(instance, *values, **kwargs):
            index = len(lr_rows)
            rates = verify_group_lrs(instance, index)
            value = original_step(*values, **kwargs)
            row = dict(update=index + 1, schedule_index=index, groups=rates)
            lr_rows.append(row)
            with (args.out / "audit/used_lrs.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            return value

        result.step = MethodType(step, result)
        return result

    def schedule(*values, **kwargs):
        nonlocal schedule_boundaries
        result = originals["schedule"](*values, **kwargs)
        schedule_boundaries = verify_schedule(result)
        audit.write_json(args.out / "schedule_verified.json", dict(
            total_updates=TOTAL_STEPS, warmup_updates=500, stable_updates=7500,
            cooldown_updates=2000, boundaries=schedule_boundaries,
            indexing="Update n uses schedule index n-1; logged rates follow scheduler.step."))
        return result

    def prewarm(*values, **kwargs):
        warmed = originals["prewarm"](*values, **kwargs)
        expected = math.ceil(kwargs["max_length"] / kwargs["length_bin"])
        if len(warmed) != expected:
            raise ValueError("not every declared training shape passed native prewarm")
        return warmed

    def forward(model, *values, **kwargs):
        loss, metrics = originals["forward"](model, *values, **kwargs)
        raw = getattr(model, "module", model)
        loops = raw.refinement_training_loops or raw.latent_residual.n_loops
        verify_metrics([dict(metrics, update_applied=1.0)],
                       dict(cfg, latent_loops=loops, latent_loop_sampling=False), actual_loops=loops)
        micro_depths[loops] += 1
        return loss, metrics

    trainer.build_model, trainer.prewarm_sequence_kernels = build, prewarm
    trainer.linear_warmup_lr, optim.build_optimizer = schedule, optimizer
    engine.allatom_forward_and_loss = forward
    try:
        with forbid_checkpoint_loads():
            audit.train(SimpleNamespace(config=args.config, out=args.out))
    finally:
        trainer.build_model = originals["build"]
        trainer.prewarm_sequence_kernels = originals["prewarm"]
        trainer.linear_warmup_lr = originals["schedule"]
        optim.build_optimizer = originals["optimizer"]
        engine.allatom_forward_and_loss = originals["forward"]
        ProteinBatch.to, torch.nn.utils.clip_grad_norm_ = originals["transfer"], originals["clip"]
        EMA.update, sys.argv = originals["ema"], originals["argv"]
    initial = json.loads((args.out / "audit/initial.json").read_text())
    complete = json.loads((args.out / "audit/complete.json").read_text())
    screen.verify_budget(initial, complete, TOTAL_STEPS, cfg["grad_accum_steps"])
    expected = Counter(actual_training_loops(cfg, step) for step in range(TOTAL_STEPS))
    counts = Counter({int(k): v for k, v in complete["depth_counts"].items()})
    if counts != expected or micro_depths != Counter({k: v * 4 for k, v in expected.items()}):
        raise ValueError("actual step/microbatch K budgets differ from the declared policy")
    rows = [json.loads(line)
            for line in (args.out / "audit/metrics.jsonl").read_text().splitlines()]
    verify_metrics(rows, cfg)
    previous = 0
    for row in rows:
        for kind, peak in (("muon", 0.003), ("adamw", 0.0001)):
            mean = peak * sum(schedule_multiplier(s) for s in range(previous + 1, row["step"] + 1))
            mean /= row["step"] - previous
            if not math.isclose(row[f"lr_{kind}"], mean, rel_tol=1e-10, abs_tol=1e-14):
                raise ValueError("logged next-step mean LR differs from the WSD schedule")
        previous = row["step"]
    if len(lr_rows) != TOTAL_STEPS or previous != TOTAL_STEPS or schedule_boundaries is None:
        raise ValueError("incomplete actual LR audit")
    final = args.out / f"ckpt_{TOTAL_STEPS:07d}.pt"
    state = torch.load(final, map_location="cpu", mmap=True, weights_only=False)
    if (state["step"] != TOTAL_STEPS or (args.out / ".requeue_requested").exists()
            or state["scheduler"]["last_epoch"] != TOTAL_STEPS):
        raise ValueError("final checkpoint does not certify the exact fresh 10k budget")
    audit.write_json(args.out / "ground_truth_verified.json", dict(
        arm=args.arm, initialization="scratch", checkpoint_loads_during_training=0,
        teacher_models=0, distilled_examples=0, optimizer_updates=TOTAL_STEPS,
        preflight_optimizer_updates=0, microbatches=TOTAL_STEPS * 4, depth_counts=dict(counts),
        microbatch_endpoint_checks=dict(micro_depths), prelude_weight=0.01,
        all_parameters_trainable=True, actual_used_lr_checks=len(lr_rows),
        native_receipt_sha256=native["receipt_sha256"],
        native_initial_model_sha256=native["model_sha256"], native_initialization_match=True,
        final_schedule_index=state["scheduler"]["last_epoch"],
        initial_model_sha256=initial["model_sha256"],
        final_ema_sha256=audit.tensor_digest(state["ema"].items()),
        final_checkpoint_sha256=screen.sha256(final), device=torch.cuda.get_device_name()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    for name in ("preflight", "train"):
        stage = sub.add_parser(name)
        stage.add_argument("--arm", choices=tuple(VARIANTS), required=True)
        stage.add_argument("--config", type=Path, required=True)
        stage.add_argument("--out", type=Path, required=True)
        if name == "train":
            stage.add_argument("--native-receipt", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("native/training stages require a Slurm GPU allocation")
    torch.set_num_threads(1)
    (preflight if args.stage == "preflight" else train)(args)


if __name__ == "__main__":
    main()
