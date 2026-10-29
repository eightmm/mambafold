"""Audited architecture-only screens from the same production10k EMA."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from benchmarks import compare_shared_trunk as screen  # noqa: E402
from benchmarks.compare_full_refinement import write_json  # noqa: E402
from mambafold.train import engine  # noqa: E402

VARIANTS = {
    "control": {},
    "control_r8": dict(latent_compression=8),
    "shared": dict(shared_trunk_layers=6),
    "hierarchy": dict(shared_trunk_layers=6, latent_hierarchy=True),
    "shared9p3": dict(shared_trunk_layers=3),
    "shared7p4": dict(shared_trunk_layers=4),
    "shared_k1": dict(shared_trunk_layers=6, n_trunk=9, latent_loops=1),
    "shared_k3": dict(shared_trunk_layers=6, n_trunk=21, latent_loops=3),
    "shared_k4": dict(shared_trunk_layers=6, n_trunk=27, latent_loops=4),
    "shared_r8": dict(shared_trunk_layers=6, latent_compression=8),
    "shared_fine2": dict(shared_trunk_layers=6, n_latent_layers=2),
    "shared_sampled": dict(shared_trunk_layers=6, latent_loop_sampling=True),
}
ARCHITECTURE_KEYS = {
    "n_trunk",
    "shared_trunk_layers",
    "latent_hierarchy",
    "latent_compression",
    "latent_loops",
    "n_latent_layers",
    "latent_loop_sampling",
}


def validate_config(cfg, arm):
    if arm not in VARIANTS:
        raise ValueError("undeclared architecture")
    expected = dict(
        n_trunk=15,
        shared_trunk_layers=0,
        latent_loops=2,
        latent_compression=4,
        n_latent_layers=4,
        latent_hierarchy=False,
        latent_loop_sampling=False,
        residue_refiner=True,
        atom_recurrence=False,
        latent_supervision="lift",
        w_fm=1.0,
        w_lddt_atom=1.0,
        grad_accum_steps=4,
        copies_per_protein=4,
        batch_size=1,
        seed=20261023,
        optimizer="muon",
        muon_lr=0.003,
        lr=0.0001,
        backbone_lr_scale=0.1,
        warmup_steps=500,
        lr_cooldown_steps=0,
        ema_decay=0.999,
    )
    expected.update(VARIANTS[arm])
    if (
        any(cfg.get(k) != v for k, v in expected.items())
        or cfg.get("resume")
        or cfg.get("initialize_refiner_from")
        or cfg.get("total_steps") not in (8, 10000)
        or any(cfg.get(k, 0) for k in ("w_bond", "w_angle", "w_clash"))
        or engine.LIFT_PRELUDE_WEIGHT != 0.01
    ):
        raise ValueError("configuration differs from declared matched architecture screen")


def migrate(source, target, prelude, shared):
    """Copy exact donors; only r8 slot tiling and explicit layer deletion differ."""
    result, mapping, operations, fresh = {}, {}, {}, []
    for name, expected in target.items():
        donor = name
        if shared and name.startswith("latent_residual.residue_stack.layers."):
            index, suffix = name.removeprefix("latent_residual.residue_stack.layers.").split(".", 1)
            if int(index) >= shared:
                raise ValueError("invalid shared block index")
            donor = f"residue_trunk.layers.{prelude + int(index)}.{suffix}"
        if name.startswith(screen.COARSE_PREFIX):
            if name.startswith(screen.COARSE_PREFIX + "up.") and torch.count_nonzero(expected):
                raise ValueError("coarse restoration must start at zero")
            result[name] = expected
            fresh.append(name)
            continue
        if donor not in source:
            raise ValueError(f"missing donor: {name}")
        value = source[donor]
        if value.dtype != expected.dtype:
            raise ValueError(f"donor dtype differs: {name}")
        if value.shape != expected.shape:
            if (
                name == "latent_residual.compress_proj.weight"
                and value.ndim == 2
                and expected.shape == (value.shape[0], 2 * value.shape[1])
            ):
                value = value.repeat(1, 2)
            elif (
                name == "latent_residual.offset_embed.weight"
                and value.ndim == 2
                and expected.shape == (2 * value.shape[0], value.shape[1])
            ):
                value = value.repeat(2, 1)
            else:
                raise ValueError(f"incompatible donor shape: {name}")
            operations[name] = "r4-to-r8 repeated slot donor; grouping changes function"
        result[name], mapping[name] = value, donor
    excluded = sorted(set(source) - set(mapping.values()))
    for name in excluded:
        fine = re.match(r"latent_residual\.stack\.layers\.([23])\.", name)
        if not (
            fine
            or (
                shared
                and name.startswith(("residue_trunk.layers.", "latent_residual.residue_stack."))
            )
        ):
            raise ValueError(f"unrelated source component would be discarded: {name}")
    return result, dict(
        mapping=mapping,
        operations=operations,
        fresh=fresh,
        excluded_source=excluded,
        function_preserved=not bool(shared or operations or excluded or fresh),
    )


def verify_metrics(rows, cfg, *, actual_loops=None):
    if not rows:
        raise ValueError("missing actual training diagnostics")
    for row in rows:
        if any(k.startswith("kd_") for k in row):
            raise ValueError("teacher loss is forbidden")
        objective = row["fm_trajectory"] + 0.01 * row["fm_prelude"]
        if (
            not math.isclose(row["fm_objective"], objective, rel_tol=1e-5, abs_tol=1e-6)
            or not math.isclose(
                row["loss"], objective + row["lddt_atom_weighted"], rel_tol=1e-5, abs_tol=1e-6
            )
            or row["update_applied"] != 1.0
        ):
            raise ValueError("executed objective/update differs from GT LiFT")
        if ((actual_loops is not None or not cfg["latent_loop_sampling"]
                and cfg.get("recycle_policy", "legacy") != "sampled16_tail2")
                and not math.isclose(
                    row["fm_atom"], row[f"fm_endpoint_k{actual_loops if actual_loops is not None else 4 if cfg.get('recycle_policy') == 'fixed4' else cfg['latent_loops']}"],
                    rel_tol=1e-5, abs_tol=1e-6)):
            raise ValueError("endpoint does not match actual configured K")


def validate_anchor(model, batch, output, cfg):
    steps = output["v_atom_steps"]
    loops = model.refinement_training_loops or model.latent_residual.n_loops
    if len(steps) != loops + 1 or output["latent_depths"].shape != (batch.t.shape[0], loops):
        raise ValueError("incorrect recurrent readout count")
    branch, training = model.latent_residual, model.training
    try:
        model.latent_residual = None
        model.eval()
        with (
            torch.no_grad(),
            torch.autocast("cuda", dtype=torch.bfloat16, enabled=batch.device.type == "cuda"),
        ):
            bypass = model(batch)["v_atom"]
        torch.testing.assert_close(steps[0].detach(), bypass, rtol=1e-4, atol=1e-5)
    finally:
        model.latent_residual = branch
        model.train(training)
    supervised = output.get("latent_supervised_steps", tuple(range(1, loops + 1)))
    expected_supervised = (tuple(range(max(1, loops - 1), loops + 1))
                           if cfg.get("recycle_policy") == "sampled16_tail2"
                           else tuple(range(1, loops + 1)))
    if supervised != expected_supervised:
        raise ValueError("incorrect supervised recurrent steps")
    leaves = tuple(v.detach().float().requires_grad_(index == 0 or index in supervised)
                   for index, v in enumerate(steps))
    loss, _ = engine.allatom_loss_surface(
        dict(
            v_atom=leaves[-1], v_atom_steps=leaves, latent_depths=output["latent_depths"].detach(),
            **({"latent_supervised_steps": supervised} if "latent_supervised_steps" in output else {})
        ),
        batch,
        use_rigid_align=cfg["use_rigid_align"],
        w_lddt_atom=0,
    )
    differentiable = tuple(index for index, leaf in enumerate(leaves) if leaf.requires_grad)
    values = torch.autograd.grad(loss, tuple(leaves[index] for index in differentiable), allow_unused=True)
    gradients = dict(zip(differentiable, values, strict=True))
    with torch.no_grad():
        clean = batch.x_clean.float()
        if cfg["use_rigid_align"]:
            clean = engine.weighted_rigid_align(
                clean,
                engine._recon_atom(batch.x_t.float(), batch.t, leaves[-1].detach()),
                batch.valid_mask,
            )
    aux, _ = engine.allatom_loss_surface(
        dict(v_atom=leaves[0]), replace(batch, x_clean=clean), use_rigid_align=False, w_lddt_atom=0
    )
    (expected,) = torch.autograd.grad(0.01 * aux, leaves[0])
    torch.testing.assert_close(gradients[0], expected, rtol=1e-4, atol=1e-7)
    for index in range(len(leaves)):
        gradient = gradients.get(index)
        if index == 0 or index in supervised:
            if (gradient is None or not torch.isfinite(gradient).all()
                    or not gradient[batch.valid_mask].abs().sum() > 0):
                raise ValueError("missing/nonfinite supervised readout gradients")
        elif gradient is not None and torch.count_nonzero(gradient):
            raise ValueError("unsupervised prefix readout received a gradient")
    return dict(
        prelude_layers=model.residue_trunk.n_layers,
        core_passes=loops,
        u0_matches_actual_K0=True,
        detached_anchor_gradient_verified=True,
    )


def train(args):
    cfg = yaml.safe_load(args.config.read_text())
    validate_config(cfg, args.arm)
    args.architecture_changes = ("n_trunk", "n_latent_layers", "latent_compression", "latent_loops")
    args.migration = migrate
    # The existing audit omits this snapshot for sampled K; store the same
    # unadapted EMA explicitly, with the fully parsed inference configuration.
    if cfg["latent_loop_sampling"]:
        from mambafold.train.config import parse_args

        full, _ = parse_args(["--config", str(args.config), "--no_wandb"])

        def migration(*values):
            weights, receipt = migrate(*values)
            torch.save(dict(args=vars(full), ema=weights, step=0), args.out / "initial_ema.pt")
            return weights, receipt

        args.migration = migration
    original_forward = engine.allatom_forward_and_loss
    micro_depths = Counter()

    def forward(model, *values, **kwargs):
        loss, metrics = original_forward(model, *values, **kwargs)
        raw = getattr(model, "module", model)
        loops = raw.refinement_training_loops or raw.latent_residual.n_loops
        verify_metrics(
            [dict(metrics, update_applied=1.0)],
            dict(cfg, latent_loops=loops, latent_loop_sampling=False),
        )
        micro_depths[loops] += 1
        return loss, metrics

    engine.allatom_forward_and_loss = forward
    try:
        screen.train(args)
    finally:
        engine.allatom_forward_and_loss = original_forward
    rows = [json.loads(v) for v in (args.out / "audit/metrics.jsonl").read_text().splitlines()]
    verify_metrics(rows, cfg)
    complete = json.loads((args.out / "audit/complete.json").read_text())
    counts = {int(k): v for k, v in complete["depth_counts"].items()}
    if cfg["latent_loop_sampling"]:
        from mambafold.train.trainer import sample_refinement_loops

        expected = {
            k: sum(sample_refinement_loops(s, cfg["seed"]) == k for s in range(cfg["total_steps"]))
            for k in (1, 2, 3)
        }
    else:
        expected = {cfg["latent_loops"]: cfg["total_steps"]}
    if counts != {k: v for k, v in expected.items() if v}:
        raise ValueError("executed loop budgets differ from declared training policy")
    if micro_depths != Counter({k: v * cfg["grad_accum_steps"] for k, v in counts.items()}):
        raise ValueError("microbatch loop selection differs within optimizer accumulation")
    write_json(
        args.out / "ground_truth_verified.json",
        dict(
            teacher_models=0,
            optimizer_updates=cfg["total_steps"],
            depth_counts=counts,
            ground_truth_metrics_verified=True,
            microbatch_endpoint_checks=dict(micro_depths),
            device=torch.cuda.get_device_name(),
            prelude_weight=0.01,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("train")
    p.add_argument("--arm", choices=tuple(VARIANTS), required=True)
    for name in ("config", "source", "out"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p = sub.add_parser("probe")
    for name in ("checkpoint", "out", "audit"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--expected-step", type=int, required=True)
    p.add_argument("--batches", type=int, default=16)
    p = sub.add_parser("infer")
    for name in ("checkpoint", "out", "panel"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--depths", type=int, nargs="+", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        parser.error("GPU stages require a Slurm GPU allocation")
    if args.stage == "train":
        train(args)
    elif args.stage == "probe":
        args.validate_output = validate_anchor
        screen.probe(args)
    else:
        screen.infer(args)
