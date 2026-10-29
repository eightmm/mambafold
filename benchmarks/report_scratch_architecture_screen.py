"""Verify and score fresh-model WSD trials without selecting on mixed-device costs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from benchmarks import compare_full_refinement as audit  # noqa: E402
from benchmarks import compare_shared_trunk as screen  # noqa: E402
from benchmarks import input_equivalence as inputs  # noqa: E402
from benchmarks import screen_scratch_architectures as scratch  # noqa: E402
from benchmarks.screen_architectures import (  # noqa: E402
    ARCHITECTURE_KEYS,
    VARIANTS,
    verify_metrics,
)

STEPS = 10000
MICROBATCHES = 40000
GROUPS = ("atom_encoder.", "residue_trunk.", "latent_residual.", "atom_decoder.")
METRICS = ("lddt", "bb_lddt", "tm_score", "rmsd", "oligo_gdtts", "oligo_gdtha")
IGNORED_CONFIG = {"config", "out_dir", "no_wandb"}
MODEL_FIELDS = (
    "d_res",
    "n_trunk",
    "latent_compression",
    "d_latent",
    "n_latent_layers",
    "latent_loops",
    "latent_loop_mode",
    "latent_position",
    "latent_supervision",
    "residue_refiner",
    "atom_recurrence",
    "shared_trunk_layers",
    "latent_hierarchy",
    "d_res_type",
    "d_res_pos",
    "d_plm",
    "d_plm_proj",
    "d_temb",
    "d_ca_emb",
    "use_plm",
    "mimo_rank",
    "d_state",
    "expand",
    "headdim",
    "self_conditioning",
    "bimamba_share",
    "d_atom",
    "n_atom_layers",
    "atom_mixer",
    "atom_d_state",
    "atom_mimo_rank",
    "n_atom_cross_layers",
    "n_backbone_streams",
    "latent_loop_sampling",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _decode(text):
    def invalid(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    def number(value):
        result = float(value)
        require(math.isfinite(result), "nonfinite JSON number")
        return result

    return json.loads(text, object_pairs_hook=_unique, parse_constant=invalid, parse_float=number)


def read(path):
    return _decode(Path(path).read_text())


def lines(path):
    return [_decode(line) for line in Path(path).read_text().splitlines()]


def valid_sha(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def finite(value, *, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (not positive or value > 0)


def exact(actual, expected, message):
    require(
        all(
            k in actual and type(actual[k]) is type(v) and actual[k] == v
            for k, v in expected.items()
        ),
        message,
    )


def verify_protocol(protocol):
    exact(
        protocol,
        dict(
            formal_applied_update_cap_per_arm=STEPS,
            preflight_optimizer_updates=0,
            microbatches_per_formal_run=MICROBATCHES,
            teacher_models=0,
            all_parameters_trainable=True,
            common_device_final_confirmation_required=True,
        ),
        "protocol differs from the authorized scratch budget",
    )
    require(set(protocol.get("arms", {})) == set(VARIANTS), "protocol needs all twelve fresh arms")
    for arm, entry in protocol["arms"].items():
        expected = scratch.expected_config(arm)
        require(
            set(entry.get("model_fields", {})) == set(MODEL_FIELDS),
            "protocol needs all 34 architecture fields",
        )
        exact(
            entry["model_fields"],
            {key: expected[key] for key in MODEL_FIELDS},
            "protocol architecture differs",
        )
        exact(entry, dict(nominal_K=expected["latent_loops"]), "protocol nominal depth differs")
    exact(
        protocol["WSD"],
        dict(
            warmup_steps=500,
            stable_steps=7500,
            cosine_decay_steps=2000,
            scheduler_horizon=STEPS,
            adamw_peak=0.0001,
            adamw_floor=0.000001,
            muon_peak=0.003,
            muon_floor=0.00003,
            backbone_lr_scale=1.0,
            actual_pre_step_rates_required=True,
        ),
        "protocol WSD differs",
    )


def verify_sources(run):
    manifest = read(run / "source_manifest.json")
    modules = (
        Path(__file__),
        Path(scratch.__file__),
        Path(screen.__file__),
        Path(audit.__file__),
        ROOT / "benchmarks/score_openstructure.py",
        Path(inputs.__file__),
    )
    for module in modules:
        relative = str(module.resolve().relative_to(run.resolve()))
        require(
            manifest.get(relative) == screen.sha256(module),
            f"unsealed scorer dependency: {relative}",
        )
    for name, digest in manifest.items():
        path = run / name
        require(
            not Path(name).is_absolute() and ".." not in Path(name).parts,
            "source manifest path escapes the run",
        )
        require(
            valid_sha(digest) and screen.sha256(path) == digest, f"changed frozen source: {name}"
        )
    return manifest


def panel_targets(protocol):
    panel = Path(protocol["panel"])
    require(screen.sha256(panel / "panel.json") == protocol["panel_sha256"], "changed panel")
    targets = read(panel / "panel.json")["targets"]
    names = [target["target"] for target in targets]
    require(len(names) == len(set(names)) == 12, "expected twelve distinct panel targets")
    declared = protocol["targets"]
    require(len(declared) == 12, "incomplete protocol panel")
    for target, frozen in zip(targets, declared, strict=True):
        require(
            isinstance(frozen, dict)
            and all(target.get(k) == v for k, v in frozen.items())
            and {"target", "length", "sequence_sha256", "reference_sha256", "packed_sha256"}
            <= frozen.keys(),
            "protocol target differs from sealed panel",
        )
        name = target["target"]
        require(
            type(target["length"]) is int
            and target["length"] == len(target["sequence"])
            and target["length"] > 0,
            "invalid panel sequence length",
        )
        require(
            hashlib.sha256(target["sequence"].encode()).hexdigest() == target["sequence_sha256"],
            "changed panel sequence",
        )
        require(
            screen.sha256(panel / f"references/{name}.pdb") == target["reference_sha256"]
            and screen.sha256(panel / f"inputs/{name}.npz") == target["packed_sha256"],
            "changed panel reference or packed input",
        )
    return {target["target"]: target for target in targets}


def verify_native(native, cfg, arm, config_sha, model_sha, delivered):
    exact(
        native,
        dict(
            arm=arm,
            seed=cfg["seed"],
            config_sha256=config_sha,
            native_backward_microbatches=8,
            declared_training_updates=STEPS,
            schedule_horizon=STEPS,
            checkpoint_loads=0,
            teacher_models=0,
            optimizer_updates=0,
            model_and_ema_unchanged=True,
        ),
        "native receipt does not certify zero-update scratch admission",
    )
    require(
        valid_sha(model_sha) and native.get("model_sha256") == model_sha,
        "native random model identity differs",
    )
    rows = native.get("rows", [])
    require(len(rows) == 8, "native admission needs eight backwards")
    for index, row in enumerate(rows):
        loops = scratch.actual_training_loops(cfg, index // 4)
        exact(
            row,
            dict(
                microbatch=index + 1,
                K=loops,
                core_passes=loops,
                prelude_layers=cfg["n_trunk"] - cfg["shared_trunk_layers"] * cfg["latent_loops"],
                u0_matches_actual_K0=True,
                detached_anchor_gradient_verified=True,
            ),
            "native routing or backward row differs",
        )
        require(
            row.get("input_sha256") == delivered[index],
            "native inputs differ from formal first eight inputs",
        )
        require(
            finite(row.get("loss"))
            and finite(row.get("grad_norm"))
            and 0 <= row["grad_norm"] < 1e4,
            "invalid native loss/gradient",
        )
        gradients = row.get("gradients", {})
        require(set(gradients) == set(GROUPS), "missing native gradient group")
        for group in gradients.values():
            require(
                type(group.get("tensors")) is int
                and type(group.get("connected")) is int
                and 0 < group["connected"] <= group["tensors"]
                and finite(group.get("norm"))
                and group["norm"] >= 0,
                "missing/nonfinite native gradient group",
            )
        verify_metrics(
            [dict(row["ground_truth_metrics"], loss=row["loss"], update_applied=1.0)],
            dict(cfg, latent_loops=loops, latent_loop_sampling=False),
            actual_loops=loops,
        )


def verify_lrs(used, metrics, schedule, cfg):
    exact(
        schedule,
        dict(total_updates=STEPS, warmup_updates=500, stable_updates=7500, cooldown_updates=2000),
        "wrong WSD horizon",
    )
    boundaries = (0, 1, 499, 500, 7999, 8000, 9000, 9999, 10000)
    require(
        set(schedule.get("boundaries", {})) == {str(s) for s in boundaries},
        "missing WSD boundary evidence",
    )
    for step in boundaries:
        row = schedule["boundaries"][str(step)]
        factor = scratch.schedule_multiplier(step)
        for key, value in dict(
            multiplier=factor, muon_lr=0.003 * factor, adamw_lr=0.0001 * factor
        ).items():
            require(
                finite(row.get(key))
                and math.isclose(row[key], value, rel_tol=1e-12, abs_tol=1e-15),
                "wrong WSD boundary",
            )
    require(len(used) == STEPS, "missing actual optimizer-step LR evidence")
    for index, row in enumerate(used):
        exact(row, dict(update=index + 1, schedule_index=index), "reordered optimizer LR audit")
        groups = row.get("groups", [])
        require(
            len(groups) == 2 and {g.get("kind") for g in groups} == {"muon", "adamw"},
            "wrong optimizer LR groups",
        )
        for group in groups:
            expected = {"muon": 0.003, "adamw": 0.0001}[
                group["kind"]
            ] * scratch.schedule_multiplier(index)
            require(
                finite(group.get("lr"))
                and math.isclose(group["lr"], expected, rel_tol=1e-12, abs_tol=1e-15),
                "actual used LR differs from WSD",
            )
    require(
        [r.get("step") for r in metrics]
        == list(range(cfg["log_interval"], STEPS + 1, cfg["log_interval"])),
        "missing/reordered training metric intervals",
    )
    previous = 0
    for row in metrics:
        require(row.get("applied_updates") == row["step"], "metric update count differs")
        for kind, peak in (("muon", 0.003), ("adamw", 0.0001)):
            expected = peak * statistics.fmean(
                scratch.schedule_multiplier(s) for s in range(previous + 1, row["step"] + 1)
            )
            require(
                finite(row.get(f"lr_{kind}"))
                and math.isclose(row[f"lr_{kind}"], expected, rel_tol=1e-10, abs_tol=1e-14),
                "logged next-step LR interval differs",
            )
        previous = row["step"]


def verify_training(cfg, arm, initial, complete, receipt, native, gt, delivered, config_sha):
    scratch.validate_config(cfg, arm)
    exact(cfg, scratch.expected_config(arm), "typed scratch configuration differs")
    exact(complete, inputs.BUDGET, "incomplete applied training budget")
    count = initial.get("parameters")
    require(
        type(count) is int and count > 0 and initial.get("trainable_parameters") == count,
        "scratch parameters were frozen",
    )
    exact(
        receipt,
        dict(
            initialization="scratch",
            seed=cfg["seed"],
            source_checkpoint=None,
            checkpoint_loads=0,
            teacher_models=0,
            distilled_examples=0,
            optimizer="fresh",
            ema="fresh copy of random model",
            data_cursor="fresh",
            parameters=count,
            trainable_parameters=count,
            native_initialization_match=True,
        ),
        "scratch initialization provenance differs",
    )
    digest = initial.get("model_sha256")
    require(
        valid_sha(digest)
        and receipt.get("target_sha256")
        == receipt.get("initial_ema_sha256")
        == receipt.get("native_initial_model_sha256")
        == digest,
        "initial model binding differs",
    )
    require(len(delivered) == MICROBATCHES, "incomplete delivered input audit")
    verify_native(native, cfg, arm, config_sha, digest, delivered)
    for key in ("group_sha256", "group_changed"):
        require(set(complete.get(key, {})) == set(GROUPS), "missing completed model group")
    require(set(initial.get("group_sha256", {})) == set(GROUPS), "missing initial model group")
    for group in GROUPS:
        require(
            complete["group_changed"][group] is True
            and valid_sha(initial["group_sha256"][group])
            and valid_sha(complete["group_sha256"][group])
            and initial["group_sha256"][group] != complete["group_sha256"][group],
            "model group did not change",
        )
    exact(
        gt,
        dict(
            arm=arm,
            initialization="scratch",
            checkpoint_loads_during_training=0,
            teacher_models=0,
            distilled_examples=0,
            optimizer_updates=STEPS,
            preflight_optimizer_updates=0,
            microbatches=MICROBATCHES,
            prelude_weight=0.01,
            all_parameters_trainable=True,
            actual_used_lr_checks=STEPS,
            native_initialization_match=True,
            final_schedule_index=STEPS,
        ),
        "GT/scratch/LR completion proof differs",
    )
    require(
        gt.get("initial_model_sha256") == gt.get("native_initial_model_sha256") == digest
        and valid_sha(gt.get("final_ema_sha256"))
        and valid_sha(gt.get("final_checkpoint_sha256")),
        "GT checkpoint identity binding differs",
    )
    expected = Counter(scratch.actual_training_loops(cfg, s) for s in range(STEPS))
    require(
        {str(k): v for k, v in expected.items()}
        == complete.get("depth_counts")
        == gt.get("depth_counts")
        and gt.get("microbatch_endpoint_checks") == {str(k): v * 4 for k, v in expected.items()},
        "executed training K counts differ",
    )


def checkpoint_identity(path, cfg, step, expected_model=None, expected_ema=None, runtime=None):
    import torch

    state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    require(type(state.get("step")) is int and state["step"] == step, "checkpoint step differs")
    saved = state.get("args", {})
    require(saved == cfg, "checkpoint full configuration differs")
    require(
        runtime is None or state.get("mamba3_runtime") == runtime,
        "checkpoint native runtime identity differs",
    )
    ema = audit.tensor_digest(state["ema"].items())
    require(expected_ema is None or ema == expected_ema, "checkpoint EMA identity differs")
    if expected_model is not None:
        require(
            audit.tensor_digest(state["model"].items()) == expected_model,
            "checkpoint initial student identity differs",
        )
    if step:
        require(
            state.get("scheduler", {}).get("last_epoch") == step,
            "checkpoint schedule index differs",
        )
    return ema


def verify_inference(directory, step, depths, checkpoint_sha, targets, hierarchy, *, atom_recycling="none",
                     n_steps=500):
    require(type(n_steps) is int and n_steps > 0, "n_steps must be a positive integer")
    measurements = read(directory / "sampling.json")
    expected = {(target, depth) for target in targets for depth in depths}
    observed = [(row.get("target"), row.get("K")) for row in measurements]
    require(
        len(observed) == len(expected) and set(observed) == expected,
        "missing, duplicate or unexpected inference rows",
    )
    for depth in depths:
        pairs = directory / f"K{depth}/pairs"
        manifest = read(pairs / "manifest.json")
        exact(
            manifest,
            dict(
                expected=len(targets),
                checkpoint_sha256=checkpoint_sha,
                weight_source="ema",
                sampler=dict(method="sde", n_steps=n_steps, tau=0.01, seed=0),
            ),
            "inference checkpoint/sampler manifest differs",
        )
        selected = [row for row in measurements if row["K"] == depth]
        require(manifest.get("rows") == selected, "sampling/manifest rows differ")
        for row in selected:
            require(
                type(row.get("checkpoint_step")) is int and row["checkpoint_step"] == step,
                "inference checkpoint step differs",
            )
            calls = screen.expected_call_counts(depth, atom_recycling=atom_recycling,
                                               network=n_steps + 1, hierarchy=hierarchy)
            require(row.get("observed_calls") == calls, "incorrect exact measured stack calls")
            if atom_recycling != "none":
                exact(row, dict(atom_recycling=atom_recycling), "atom recycling inference policy differs")
            require(
                finite(row.get("seconds"), positive=True)
                and finite(row.get("peak_allocated_gib"), positive=True),
                "invalid inference cost",
            )
            name = row["target"]
            require(
                screen.sha256(pairs / f"{name}_seed0_pred.pdb") == row.get("prediction_sha256")
                and screen.sha256(pairs / f"{name}_seed0_gt.pdb")
                == row.get("reference_sha256")
                == targets[name]["reference_sha256"],
                "changed prediction/reference or wrong panel reference",
            )
    return measurements


def verify_summary(summary, targets):
    rows = summary.get("rows", [])
    observed = [row.get("target_id") for row in rows]
    expected = {target + "_seed0" for target in targets}
    require(
        summary.get("evaluation_complete") is True
        and summary.get("successful_targets") == len(expected)
        and len(observed) == len(expected)
        and set(observed) == expected,
        "incomplete canonical summary",
    )
    for name in METRICS:
        values = [row.get(name) for row in rows]
        require(all(finite(value) for value in values), "nonfinite canonical metric")
        actual = summary.get("metrics", {}).get(name, {})
        require(
            type(actual.get("n")) is int and actual["n"] == len(values),
            "canonical statistic coverage differs",
        )
        for key, value in dict(
            mean=statistics.fmean(values),
            median=statistics.median(values),
            min=min(values),
            max=max(values),
        ).items():
            require(
                finite(actual.get(key))
                and math.isclose(actual[key], value, rel_tol=1e-12, abs_tol=1e-15),
                "canonical summary arithmetic differs",
            )
    return {row["target_id"].removesuffix("_seed0"): row["lddt"] for row in rows}


def verify_arm(run, arm, protocol, source_manifest, targets):
    require(arm in protocol["arms"] and arm in VARIANTS, "undeclared scoring arm")
    entry = protocol["arms"][arm]
    training, preflight = Path(entry["training"]), Path(entry["preflight"])
    config_path = Path(entry["config"])
    config_sha = screen.sha256(config_path)
    require(
        source_manifest.get(str(config_path.resolve().relative_to(run.resolve()))) == config_sha,
        "configuration is not sealed",
    )
    cfg = read(training / "config.json")
    declared = yaml.safe_load(config_path.read_text())
    require(
        all(cfg.get(k) == v for k, v in declared.items() if k not in IGNORED_CONFIG),
        "actual training configuration differs from source YAML",
    )
    require(
        all(type(cfg.get(k)) is type(v) and cfg[k] == v for k, v in entry["model_fields"].items())
        and len(entry["model_fields"]) == len(MODEL_FIELDS)
        and entry["nominal_K"] == cfg["latent_loops"],
        "declared architecture differs",
    )
    initial, complete = (
        read(training / "audit/initial.json"),
        read(training / "audit/complete.json"),
    )
    receipt, gt = (
        read(training / "initialization.json"),
        read(training / "ground_truth_verified.json"),
    )
    native_path = preflight / "native_verified.json"
    native = read(native_path)
    native_sha = screen.sha256(native_path)
    require(
        receipt.get("native_receipt_sha256") == gt.get("native_receipt_sha256") == native_sha
        and Path(receipt["native_receipt"]).resolve() == native_path.resolve(),
        "native receipt binding differs",
    )
    delivered, stream = inputs.read_inputs(training / "audit/inputs.jsonl", MICROBATCHES)
    require(stream == complete.get("input_stream_sha256"), "delivered input stream differs")
    verify_training(cfg, arm, initial, complete, receipt, native, gt, delivered, config_sha)
    metrics = lines(training / "audit/metrics.jsonl")
    verify_metrics(metrics, cfg)
    schedule = read(training / "schedule_verified.json")
    require(
        native.get("schedule_boundaries") == schedule.get("boundaries"),
        "native/formal WSD boundaries differ",
    )
    verify_lrs(
        lines(training / "audit/used_lrs.jsonl"),
        metrics,
        schedule,
        cfg,
    )
    native_initial = read(preflight / "initialization.json")
    native_cfg = read_snapshot_config(preflight)
    native_keys = set(native_cfg) | set(cfg)
    require(
        all(
            native_cfg.get(k) == cfg.get(k)
            for k in native_keys
            if k not in IGNORED_CONFIG | {"requested_num_workers"}
        ),
        "native checkpoint full configuration differs from formal training",
    )
    for home, saved, expected_cfg in (
        (training, receipt, cfg),
        (preflight, native_initial, native_cfg),
    ):
        require(
            saved.get("initialization") == "scratch"
            and saved.get("target_sha256")
            == saved.get("initial_ema_sha256")
            == initial["model_sha256"],
            "fresh snapshot model binding differs",
        )
        require(
            screen.sha256(home / "initial_ema.pt") == saved.get("initial_snapshot_sha256"),
            "initial checkpoint bytes differ",
        )
        checkpoint_identity(
            home / "initial_ema.pt",
            expected_cfg,
            0,
            initial["model_sha256"],
            initial["model_sha256"],
            runtime=receipt["mamba3_runtime"],
        )
    final = training / "ckpt_0010000.pt"
    require(screen.sha256(final) == gt["final_checkpoint_sha256"], "final checkpoint bytes differ")
    checkpoint_identity(
        final, cfg, STEPS, expected_ema=gt["final_ema_sha256"], runtime=receipt["mamba3_runtime"]
    )
    probe_path = Path(entry["u0"]) / f"{arm}-10000/u0.json"
    probe = read(probe_path)
    require(
        probe.get("checkpoint_sha256") == gt["final_checkpoint_sha256"]
        and probe.get("initial_model_sha256") == gt["final_ema_sha256"]
        and probe.get("checkpoint_step") == STEPS
        and probe.get("updates") == 0
        and probe.get("weight_source") == "checkpoint EMA; no probe updates",
        "final u0 probe EMA/checkpoint binding differs",
    )
    rows = probe.get("rows", [])
    require(
        len(rows) == 16
        and [row.get("microbatch") for row in rows] == list(range(1, 17))
        and tuple(row.get("input_sha256") for row in rows) == delivered[:16]
        and all(
            row.get("u0_matches_actual_K0") is True
            and row.get("detached_anchor_gradient_verified") is True
            for row in rows
        ),
        "final probe input/routing coverage differs",
    )
    for row in rows:
        probe_loops = 4 if cfg.get("recycle_policy") == "fixed4" else cfg["latent_loops"]
        exact(
            row,
            dict(
                core_passes=probe_loops,
                readouts=probe_loops + 1,
                prelude_layers=cfg["n_trunk"] - cfg["shared_trunk_layers"] * cfg["latent_loops"],
            ),
            "final probe architecture differs",
        )
        require(
            all(math.isfinite(v) for v in row.values() if isinstance(v, float)),
            "nonfinite final probe diagnostic",
        )
    require(not (training / ".requeue_requested").exists(), "interrupted formal training")
    complete = dict(complete, probe_input_sha256=list(delivered[:16]))
    evidence = [
        config_path,
        training / "config.json",
        training / "audit/initial.json",
        training / "audit/complete.json",
        training / "initialization.json",
        training / "ground_truth_verified.json",
        native_path,
        probe_path,
        preflight / "initialization.json",
        training / "audit/inputs.jsonl",
        training / "audit/metrics.jsonl",
        training / "audit/used_lrs.jsonl",
        training / "schedule_verified.json",
    ]
    return cfg, complete, metrics, {str(path): screen.sha256(path) for path in evidence}


def read_snapshot_config(preflight):
    import torch

    state = torch.load(
        preflight / "initial_ema.pt", map_location="cpu", mmap=True, weights_only=False
    )
    return state["args"]


def score(run, arm):
    run = Path(run)
    protocol = read(run / "protocol.json")
    verify_protocol(protocol)
    manifest = verify_sources(run)
    targets = panel_targets(protocol)
    cfg, complete, metrics, evidence = verify_arm(run, arm, protocol, manifest, targets)
    entry = protocol["arms"][arm]
    training = Path(entry["training"])
    work = run / "score-work" / arm
    destination = work / "evaluation/trained"
    destination.mkdir(parents=True, exist_ok=True)
    audit.write_json(work / "protocol.json", protocol)
    for step in (5000, 8000, STEPS):
        source = Path(entry["evaluation"]) / f"{arm}-{step}"
        depths = sorted({1, 2, cfg["latent_loops"]}) if step == STEPS else [cfg["latent_loops"]]
        checkpoint_sha = screen.sha256(training / f"ckpt_{step:07d}.pt")
        checkpoint_identity(
            training / f"ckpt_{step:07d}.pt",
            cfg,
            step,
            runtime=read(training / "initialization.json")["mamba3_runtime"],
        )
        verify_inference(source, step, depths, checkpoint_sha, targets, cfg["latent_hierarchy"])
        link = destination / source.name
        if link.exists() or link.is_symlink():
            require(
                link.is_symlink() and link.resolve() == source.resolve(), "wrong scoring workspace"
            )
        else:
            link.symlink_to(source.resolve(), target_is_directory=True)
        evidence[str(source / "sampling.json")] = screen.sha256(source / "sampling.json")
        for depth in depths:
            path = source / f"K{depth}/pairs/manifest.json"
            evidence[str(path)] = screen.sha256(path)
    require(
        {p.name for p in destination.iterdir()} == {f"{arm}-{s}" for s in (5000, 8000, STEPS)},
        "unexpected scoring directories",
    )
    scores, rows = screen.score_all(work, "trained")
    for row in rows:
        directory = Path(entry["evaluation"]) / row["label"]
        summary_path = directory / f"K{row['K']}/scores/summary.json"
        summary = read(summary_path)
        verify_summary(summary, targets)
        row["gpu"] = read(directory / "hardware.json")["gpu"]
        evidence[str(summary_path)] = screen.sha256(summary_path)
    result = dict(
        arm=arm,
        scores=scores,
        rows=rows,
        training=complete,
        config=cfg,
        steady_loss_median=statistics.median(r["loss"] for r in metrics if r["step"] >= 2000),
        verification=evidence,
        scorer_source_sha256=screen.sha256(__file__),
    )
    audit.write_json(run / f"scores/{arm}.json", result)
    return result


def verify_score_rows(protocol, arm, value, targets):
    entry = protocol["arms"][arm]
    training = Path(entry["training"])
    require(
        value["config"] == read(training / "config.json"), "reported actual configuration differs"
    )
    complete = read(training / "audit/complete.json")
    require(
        all(value["training"].get(k) == v for k, v in complete.items()),
        "reported training receipt differs",
    )
    expected_scores, expected_rows = {}, {}
    for step in (5000, 8000, STEPS):
        directory = Path(entry["evaluation"]) / f"{arm}-{step}"
        depths = sorted({1, 2, entry["nominal_K"]}) if step == STEPS else [entry["nominal_K"]]
        first = read(directory / f"K{depths[0]}/pairs/manifest.json")
        measured = verify_inference(
            directory,
            step,
            depths,
            first["checkpoint_sha256"],
            targets,
            value["config"]["latent_hierarchy"],
        )
        gpu = read(directory / "hardware.json")["gpu"]
        for depth in depths:
            summary = read(directory / f"K{depth}/scores/summary.json")
            vector = verify_summary(summary, targets)
            label = directory.name
            expected_scores[f"{label}/K{depth}"] = vector
            selected = [row for row in measured if row["K"] == depth]
            expected_rows[label, depth] = dict(
                label=label,
                K=depth,
                lddt=statistics.fmean(vector.values()),
                bb_lddt=summary["metrics"]["bb_lddt"]["mean"],
                mean_seconds=statistics.fmean(r["seconds"] for r in selected),
                peak_allocated_gib=max(r["peak_allocated_gib"] for r in selected),
                gpu=gpu,
            )
    require(value["scores"] == expected_scores, "reported canonical score vectors differ")
    observed = {(row["label"], row["K"]): row for row in value["rows"]}
    require(
        len(observed) == len(value["rows"]) and observed == expected_rows,
        "reported canonical quality/cost rows differ",
    )


def aggregate(protocol, results, input_comparison):
    require(set(results) == set(protocol["arms"]) == set(VARIANTS), "incomplete twelve-arm results")
    control = results["control"]

    def common(cfg):
        return {
            k: v
            for k, v in cfg.items()
            if k not in ARCHITECTURE_KEYS | IGNORED_CONFIG and not k.startswith("wandb_")
        }

    require(
        all(common(value["config"]) == common(control["config"]) for value in results.values()),
        "common full training configuration differs",
    )
    rankings = []
    baseline = control["scores"]["control-10000/K2"]
    for arm, value in results.items():
        require(value.get("arm") == arm, "score arm identity differs")
        nominal = value["config"]["latent_loops"]
        require(nominal == protocol["arms"][arm]["nominal_K"], "nominal policy differs")
        key = f"{arm}-10000/K{nominal}"
        rows = [r for r in value["rows"] if r["label"] == f"{arm}-10000" and r["K"] == nominal]
        require(len(rows) == 1, "missing nominal ranking row")
        row = rows[0]
        require(
            all(
                finite(row.get(k))
                for k in ("lddt", "bb_lddt", "mean_seconds", "peak_allocated_gib")
            )
            and row["mean_seconds"] > 0
            and row["peak_allocated_gib"] > 0,
            "invalid nominal metrics/cost",
        )
        require(
            math.isclose(
                row["lddt"], statistics.fmean(value["scores"][key].values()), abs_tol=1e-15
            ),
            "nominal score mean differs",
        )
        rankings.append(
            dict(
                arm=arm,
                nominal_K=nominal,
                **{
                    k: row[k]
                    for k in ("lddt", "bb_lddt", "mean_seconds", "peak_allocated_gib", "gpu")
                },
                versus_control=screen.paired_summary(baseline, value["scores"][key]),
                K2_minus_K1=screen.paired_summary(
                    value["scores"][f"{arm}-10000/K1"], value["scores"][f"{arm}-10000/K2"]
                ),
                train_step_ms=value["training"]["steady_mean_step_ms"],
                train_peak_gib=value["training"]["peak_allocated_gib"],
            )
        )
    rankings.sort(key=lambda row: (-row["lddt"], row["arm"]))
    zero = all(row["lddt"] == 0 for row in rankings)
    return dict(
        rankings=rankings,
        training={k: v["training"] for k, v in results.items()},
        input_comparison=input_comparison,
        scope=protocol["limits"],
        architecture_selected=False,
        all_zero_quality=zero,
        decision=(
            "Inconclusive: all nominal all-atom scores are zero. "
            if zero
            else "Exploratory fresh-model10k WSD quality ranking. "
        )
        + "Common-device final confirmation and parent review remain required; "
        "mixed raw inference times do not select an architecture.",
    )


def report(run, certificate=None, certificate_sha256=None):
    run = Path(run)
    protocol = read(run / "protocol.json")
    verify_protocol(protocol)
    verify_sources(run)
    targets = panel_targets(protocol)
    require(
        (certificate is None) == (certificate_sha256 is None),
        "input equivalence requires both certificate and immutable seal",
    )
    results = {arm: read(run / f"scores/{arm}.json") for arm in protocol["arms"]}
    certified = (
        inputs.load_certificate(run, certificate, certificate_sha256) if certificate else None
    )
    streams, prefixes, bindings = {}, {}, {}
    for arm, value in results.items():
        require(
            value.get("scorer_source_sha256") == screen.sha256(__file__), "different scorer source"
        )
        require(value.get("verification"), "missing score evidence seals")
        for path, digest in value["verification"].items():
            require(screen.sha256(path) == digest, "changed score evidence")
        verify_score_rows(protocol, arm, value, targets)
        home = Path(protocol["arms"][arm]["training"])
        values, digest = inputs.read_inputs(home / "audit/inputs.jsonl", MICROBATCHES)
        exact(value["training"], inputs.BUDGET, "incomplete reported training budget")
        require(
            value["training"].get("input_stream_sha256") == digest
            and tuple(value["training"].get("probe_input_sha256", ())) == values[:16],
            "reported input/probe identity differs",
        )
        native = read(Path(protocol["arms"][arm]["preflight"]) / "native_verified.json")
        require(
            tuple(r["input_sha256"] for r in native["rows"]) == values[:8],
            "native/formal input prefix differs",
        )
        if certified:
            require(
                digest in certified[1] and values == certified[1][digest],
                "uncertified actual inputs",
            )
        streams[arm], prefixes[arm], bindings[arm] = values, values[:16], digest
    if certified:
        cert = certified[0]
        comparison = dict(
            method="certified_coordinate_equivalence",
            certificate=str(Path(certificate).resolve()),
            certificate_sha256=certificate_sha256,
            microbatches_per_arm=MICROBATCHES,
            atol=cert["atol"],
            rtol=0,
            absolute_bound_A=cert["absolute_bound_A"],
            coordinate_statistics=cert["coordinate_statistics"],
            arm_streams=bindings,
            limits="Existing sealed CPU replay contract; only x_clean/x_t rounding is bounded; "
            "all noncoordinate inputs stay exact. No CUDA trajectory equivalence claim.",
        )
    else:
        require(
            len(set(streams.values())) == len(set(prefixes.values())) == 1,
            "actual ordered delivered inputs differ; no implicit coordinate tolerance",
        )
        comparison = dict(
            method="byte_exact", microbatches_per_arm=MICROBATCHES, arm_streams=bindings
        )
    result = aggregate(protocol, results, comparison)
    audit.write_json(run / "comparison.json", result)
    text = [
        "# Fresh-model10k WSD architecture screen",
        "",
        result["decision"],
        "",
        "| Arm | nominal K | all-atom lDDT | backbone lDDT | seconds | allocated GiB | GPU |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in result["rankings"]:
        text.append(
            f"|{row['arm']}|{row['nominal_K']}|{row['lddt']:.5f}|{row['bb_lddt']:.5f}|"
            f"{row['mean_seconds']:.2f}|{row['peak_allocated_gib']:.3f}|{row['gpu']}|"
        )
    (run / "comparison.md").write_text("\n".join(text + ["", result["scope"], ""]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("score", "report"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--arm", choices=tuple(VARIANTS))
    parser.add_argument("--input-equivalence-certificate", type=Path)
    parser.add_argument("--certificate-sha256")
    args = parser.parse_args()
    if args.stage == "score":
        if not args.arm or args.input_equivalence_certificate or args.certificate_sha256:
            parser.error("score requires --arm; certificate options apply only to report")
        score(args.run, args.arm)
    else:
        if args.arm:
            parser.error("--arm applies only to score")
        report(args.run, args.input_equivalence_certificate, args.certificate_sha256)


if __name__ == "__main__":
    main()
