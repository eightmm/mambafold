"""The long flat_fine6 run reuses the 10k screen under a declared 30k WSD horizon, by value."""

import ast
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from benchmarks import evaluate_recycle_checkpoint_curve as curve
from benchmarks import input_equivalence as inputs
from benchmarks import train_flat_fine6_long as trial

trial.bind()
scratch, proof = trial.scratch, trial.proof
TARGETS = dict(a=dict(entry_id="e1"), b=dict(entry_id="e1"), c=dict(entry_id="e2"))
INITIAL = "4" * 64
TRAINER_DEFAULTS = dict(atom_recycling="none", recycle_policy="legacy")
EXTRA = dict(num_workers=3, prefetch_factor=2, log_interval=100, ckpt_interval=5000, keep_last_checkpoints=1,
             resume=None, initialize_refiner_from=None, initialize_model_from_ema=False,
             keep_checkpoint_steps=[5000, 10000])
STAMP = "20990101T000000Z"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def template(home):
    return trial.expected(trial.REF_ARM, trial.TEN_K) | EXTRA | dict(
        config=str(home / f"configs/{trial.REF_ARM}.yaml"), out_dir=str(home / f"training/{trial.REF_ARM}"),
        wandb_name=f"seed-replicate-{trial.REF_ARM}", wandb_tags=["scratch", "WSD", "max10k"])


def schedule_optimizer():
    groups = [dict(params=[torch.nn.Parameter(torch.zeros(1))], lr=0.003, optimizer_kind="muon"),
              dict(params=[torch.nn.Parameter(torch.zeros(1))], lr=0.0001, optimizer_kind="adamw")]
    return torch.optim.SGD(groups, lr=0.0001)


# ── horizon ─────────────────────────────────────────────────────────────


def test_ten_k_horizon_is_exactly_the_screen_schedule():
    assert trial.TEN_K.boundaries == (0, 1, 499, 500, 7999, 8000, 9000, 9999, 10000)
    assert trial.TEN_K.stable == 7500
    assert all(trial.TEN_K.multiplier(step) == scratch.schedule_multiplier(step) for step in range(10001))
    with pytest.raises(ValueError):
        trial.TEN_K.multiplier(10001)


def test_long_horizon_matches_the_trainer_schedule_at_every_index():
    assert (trial.LONG.stable, trial.LONG.decay_start) == (25500, 26000)
    assert trial.LONG.boundaries == (0, 1, 499, 500, 25999, 26000, 28000, 29999, 30000)
    scheduler = trial.trainer.linear_warmup_lr(schedule_optimizer(), 500, 1e-4, 1e-6, total_steps=30000,
                                               cooldown_steps=4000)
    for step in range(30001):
        assert all(math.isclose(fn(step), trial.LONG.multiplier(step), abs_tol=1e-15) for fn in scheduler.lr_lambdas)
    assert math.isclose(trial.LONG.multiplier(30000), 0.01) and trial.LONG.multiplier(25999) == 1.0
    with pytest.raises(ValueError):
        trial.LONG.multiplier(30001)


def screen_bindings():
    return (scratch.TOTAL_STEPS, scratch.schedule_multiplier, scratch.verify_schedule, scratch.expected_config,
            scratch.audit, proof.STEPS, proof.MICROBATCHES, inputs.MICROBATCHES, inputs.BUDGET,
            scratch.VARIANTS, proof.VARIANTS, trial.trainer.linear_warmup_lr)


def test_horizon_registry_and_preflight_schedule_restore_every_binding_after_failure():
    before = screen_bindings()
    with pytest.raises(RuntimeError):
        with trial.registry(trial.ARM), trial.horizon(trial.LONG), trial.preflight_schedule(trial.LONG):
            assert (scratch.TOTAL_STEPS, proof.STEPS, proof.MICROBATCHES, inputs.MICROBATCHES) == (
                30000, 30000, 120000, 120000)
            assert inputs.BUDGET == dict(updates=30000, attempts=30000, skipped_updates=0, microbatches=120000,
                                         finite_losses=120000)
            assert scratch.expected_config(trial.ARM)["lr_cooldown_steps"] == 4000
            assert set(scratch.VARIANTS) == set(proof.VARIANTS) == {trial.ARM}
            raise RuntimeError("restore")
    assert all(first is second for first, second in zip(before, screen_bindings(), strict=True))
    with trial.horizon(trial.TEN_K):
        assert inputs.BUDGET == before[8] and scratch.TOTAL_STEPS == 10000


def test_long_configuration_is_admitted_only_under_the_long_horizon(tmp_path):
    base = template(tmp_path / "old")
    trial.verify_template(base)
    config = trial.long_config(base, tmp_path / "new")
    assert {key: config[key] for key in trial.CONFIG_CHANGES} == trial.CONFIG_CHANGES
    assert config["wandb_tags"] == ["scratch", "WSD", "max30k"] and config["seed"] == trial.SEED
    assert {key for key in config if config[key] != base.get(key)} == {*trial.CONFIG_CHANGES, *trial.PATH_KEYS}
    with trial.registry(trial.ARM), trial.horizon(trial.LONG):
        scratch.validate_config(config, trial.ARM)
        scratch.validate_config(dict(config, **TRAINER_DEFAULTS), trial.ARM)
        for bad in (dict(config, total_steps=10000), dict(config, lr_cooldown_steps=2000),
                    dict(config, shared_trunk_layers=0), dict(config, recycle_policy="fixed2"),
                    dict(config, log_interval=7000)):
            with pytest.raises(ValueError):
                scratch.validate_config(bad, trial.ARM)
    with trial.registry(trial.ARM), trial.horizon(trial.TEN_K), pytest.raises(ValueError):
        scratch.validate_config(config, trial.ARM)


@pytest.mark.parametrize("change", [dict(seed=20261024), dict(total_steps=20000), dict(ema_decay=0.99),
                                    dict(muon_lr=0.002), dict(copies_per_protein=2), dict(max_length=512),
                                    dict(shared_trunk_layers=0), dict(atom_recycling="carry"),
                                    dict(recycle_policy="fixed2"), dict(resume="ckpt_latest.pt"),
                                    dict(min_lr=1)])
def test_template_must_be_the_seed_20261023_flat_recipe_by_value(tmp_path, change):
    base = template(tmp_path)
    trial.verify_template(dict(base, **TRAINER_DEFAULTS))
    with pytest.raises(ValueError):
        trial.verify_template(dict(base, **change))


def test_preflight_schedule_substitutes_only_the_declared_cooldown():
    with trial.registry(trial.ARM), trial.horizon(trial.LONG), trial.preflight_schedule(trial.LONG):
        scheduler = trial.trainer.linear_warmup_lr(schedule_optimizer(), 500, 1e-4, 1e-6, total_steps=30000,
                                                   cooldown_steps=2000)
        boundaries = scratch.verify_schedule(scheduler)
        assert set(boundaries) == {str(step) for step in trial.LONG.boundaries}
        assert math.isclose(boundaries["28000"]["muon_lr"], 0.003 * trial.LONG.multiplier(28000))
        for kwargs in (dict(total_steps=10000, cooldown_steps=2000), dict(total_steps=30000, cooldown_steps=4000)):
            with pytest.raises(ValueError):
                trial.trainer.linear_warmup_lr(schedule_optimizer(), 500, 1e-4, 1e-6, **kwargs)
    stale = trial.trainer.linear_warmup_lr(schedule_optimizer(), 500, 1e-4, 1e-6, total_steps=30000,
                                           cooldown_steps=2000)
    with trial.horizon(trial.LONG), pytest.raises(ValueError):
        scratch.verify_schedule(stale)


def test_schedule_labels_are_rewritten_only_for_the_declared_horizon(tmp_path):
    labels = dict(total_updates=30000, warmup_updates=500, stable_updates=7500, cooldown_updates=2000, boundaries={})
    with trial.horizon(trial.LONG):
        scratch.audit.write_json(tmp_path / "schedule_verified.json", labels)
        scratch.audit.write_json(tmp_path / "other.json", labels)
        with pytest.raises(ValueError):
            scratch.audit.write_json(tmp_path / "x/schedule_verified.json", dict(labels, total_updates=10000))
        assert scratch.audit.tensor_digest is trial.base.audit.tensor_digest
    written = json.loads((tmp_path / "schedule_verified.json").read_text())
    assert (written["stable_updates"], written["cooldown_updates"]) == (25500, 4000)
    assert json.loads((tmp_path / "other.json").read_text()) == labels
    with trial.horizon(trial.TEN_K):
        scratch.audit.write_json(tmp_path / "y/schedule_verified.json", dict(labels, total_updates=10000))
    assert json.loads((tmp_path / "y/schedule_verified.json").read_text())["stable_updates"] == 7500


def lr_evidence(h, interval=100):
    used = [dict(update=index + 1, schedule_index=index, groups=[
        dict(kind=kind, lr=peak * h.multiplier(index)) for kind, peak in trial.PEAKS.items()]) for index in range(h.total)]
    metrics, previous = [], 0
    for step in range(interval, h.total + 1, interval):
        metrics.append(dict(step=step, applied_updates=step, **{
            f"lr_{kind}": peak * statistics.fmean(h.multiplier(s) for s in range(previous + 1, step + 1))
            for kind, peak in trial.PEAKS.items()}))
        previous = step
    schedule = dict(total_updates=h.total, warmup_updates=h.warmup, stable_updates=h.stable,
                    cooldown_updates=h.cooldown, boundaries={str(step): dict(
                        multiplier=h.multiplier(step), muon_lr=0.003 * h.multiplier(step),
                        adamw_lr=0.0001 * h.multiplier(step)) for step in h.boundaries})
    return used, metrics, schedule, dict(log_interval=interval)


def tamper(evidence, fault):
    used, metrics, schedule, cfg = (json.loads(json.dumps(value)) for value in evidence)
    if fault == "used":
        used[-1]["groups"][0]["lr"] *= 1.001
    elif fault == "missing":
        used.pop()
    elif fault == "label":
        schedule["cooldown_updates"] = 4000
    elif fault == "boundary":
        schedule["boundaries"]["1"]["muon_lr"] *= 2
    elif fault == "logged":
        metrics[-1]["lr_adamw"] *= 1.01
    return used, metrics, schedule, cfg


@pytest.mark.parametrize("fault", ["used", "missing", "label", "boundary", "logged"])
def test_lr_audit_port_agrees_with_the_screen_on_10k_and_rejects_tampering(fault):
    ten = lr_evidence(trial.TEN_K)
    proof.verify_lrs(*ten)
    trial.verify_lrs(trial.TEN_K, *ten)
    trial.verify_lrs(trial.LONG, *lr_evidence(trial.LONG))
    with pytest.raises(ValueError):
        trial.verify_lrs(trial.LONG, *ten)
    with pytest.raises(ValueError):
        proof.verify_lrs(*tamper(ten, fault))
    with pytest.raises(ValueError):
        trial.verify_lrs(trial.TEN_K, *tamper(ten, fault))


# ── admission pieces ────────────────────────────────────────────────────


def checkpoint_home(tmp_path, steps, latest):
    home = tmp_path / "home"
    home.mkdir()
    for step in steps:
        (home / f"ckpt_{step:07d}.pt").write_text(f"weights {step}")
    (home / "ckpt_latest.pt").symlink_to(f"ckpt_{latest:07d}.pt")
    return home


@pytest.mark.parametrize("steps, latest, final_ok, admitted", [
    ((10000, 20000, 30000), 30000, True, True), ((10000, 20000, 25000, 30000), 30000, True, False),
    ((10000, 30000), 30000, True, False), ((10000, 20000, 30000), 20000, True, False),
    ((10000, 20000, 30000), 30000, False, False)])
def test_exactly_the_milestone_checkpoints_are_retained_and_bound(tmp_path, monkeypatch, steps, latest, final_ok,
                                                                  admitted):
    home = checkpoint_home(tmp_path, steps, latest)
    calls = []

    def identity(path, cfg, step, runtime=None):
        calls.append((path.name, step, runtime))
        return f"{step % 16:x}" * 64

    monkeypatch.setattr(proof, "checkpoint_identity", identity)
    gt = dict(final_checkpoint_sha256=trial.sha(home / "ckpt_0030000.pt") if final_ok else "f" * 64,
              final_ema_sha256=f"{30000 % 16:x}" * 64)
    if not admitted:
        with pytest.raises(ValueError):
            trial.verify_checkpoints(home, {}, trial.LONG, gt, "native")
        return
    rows = trial.verify_checkpoints(home, {}, trial.LONG, gt, "native")
    assert set(rows) == {"10000", "20000", "30000"} and rows["20000"]["path"] == "ckpt_0020000.pt"
    assert calls == [(f"ckpt_{step:07d}.pt", step, "native") for step in (10000, 20000, 30000)]


def test_data_prefix_is_recorded_byte_exactly_and_never_gated():
    reference = tuple(f"{index:064x}" for index in range(10))
    same = trial.prefix_comparison(reference + ("f" * 64,), reference)
    assert same["identical"] and same["matching_prefix_microbatches"] == 10 and same["gated"] is False
    assert same["first_divergent_microbatch"] is None
    changed = list(reference)
    changed[5] = "e" * 64
    diverged = trial.prefix_comparison(tuple(changed), reference)
    assert not diverged["identical"] and diverged["first_divergent_microbatch"] == 6
    assert diverged["matching_prefix_microbatches"] == 5 and diverged["prefix_stream_sha256"] != same[
        "prefix_stream_sha256"]
    with pytest.raises(ValueError):
        trial.prefix_comparison(reference[:9], reference)


def test_native_gate_requires_the_reference_initial_tensors_and_first_inputs():
    values = tuple(f"{index}" * 64 for index in range(9))
    protocol = dict(reference=dict(initial_model_sha256=INITIAL))
    native = dict(arm=trial.ARM, seed=trial.SEED, model_sha256=INITIAL,
                  rows=[dict(input_sha256=value) for value in values[:8]])
    trial.check_native(native, protocol, values)
    for bad in (dict(native, model_sha256="f" * 64), dict(native, seed=20261024), dict(native, arm=trial.REF_ARM),
                dict(native, rows=native["rows"][::-1]), dict(native, rows=native["rows"][:7])):
        with pytest.raises(ValueError):
            trial.check_native(bad, protocol, values)


@pytest.mark.parametrize("row, admitted", [
    ({}, True), (TRAINER_DEFAULTS, True), (dict(TRAINER_DEFAULTS, atom_recycling="carry"), False),
    (dict(recycle_policy="fixed2"), False)])
def test_inference_rows_carrying_trainer_defaults_are_admitted_by_value(monkeypatch, row, admitted):
    context = dict(run=Path("/run"), targets=TARGETS)
    rows = [dict(row, peak_reserved_gib=1.5)]
    monkeypatch.setattr(proof, "verify_inference", lambda *values, **kwargs: rows)
    if admitted:
        assert trial.verify_inference(context, 10000, "1" * 64) == rows
    else:
        with pytest.raises(ValueError):
            trial.verify_inference(context, 10000, "1" * 64)


# ── report ──────────────────────────────────────────────────────────────


def quality(offset):
    return {target: dict(lddt=.1 + offset + index * .01, bb_lddt=.5 + offset, tm_score=.4 + offset)
            for index, target in enumerate(TARGETS)}


def reference_value():
    arms = {arm: dict(per_target=quality(offset)) for arm, offset in zip(trial.FLAT_ARMS, (.02, -.01, 0.))}
    for row in arms.values():
        row["runs"] = trial.seedrep.run_quality(TARGETS, row["per_target"])
    spread = {metric: {kind: trial.seedrep.spread(arms[arm]["runs"][metric][kind] for arm in trial.FLAT_ARMS)
                       for kind in trial.KINDS} for metric in trial.QUALITY}
    return dict(arms=arms, spread=spread)


def test_report_marks_only_the_final_checkpoint_post_cooldown_and_pairs_against_seed_20261023():
    reference = reference_value()
    vectors = {"10000": quality(-.05), "20000": quality(-.03), "30000": quality(.05)}
    stats = trial.summarize(TARGETS, vectors, reference)
    rows = stats["checkpoints"]
    assert [rows[key]["post_cooldown"] for key in ("10000", "20000", "30000")] == [False, False, True]
    assert rows["10000"]["phase"].startswith("pre-cooldown") and rows["30000"]["phase"].startswith("post-cooldown")
    final = rows["30000"]
    lddt = final["versus_three_seed_10k"]["lddt"]["chain_mean"]
    mean = statistics.fmean(trial.seedrep.run_quality(TARGETS, reference["arms"][arm]["per_target"])["lddt"]
                            ["chain_mean"] for arm in trial.FLAT_ARMS)
    assert math.isclose(lddt["delta"], final["lddt"]["chain_mean"] - mean)
    assert math.isclose(lddt["delta_over_sd"], lddt["delta"] / reference["spread"]["lddt"]["chain_mean"]["sd"])
    assert lddt["within_reference_range"] is False
    paired = final["paired_minus_seed20261023_10k"]["tm_score"]
    assert math.isclose(paired["chain_mean_delta"], .05 - .02) and (paired["wins"], paired["losses"]) == (3, 0)
    assert set(stats["progression"]) == {"20000_minus_10000", "30000_minus_20000", "30000_minus_10000"}
    assert math.isclose(stats["progression"]["30000_minus_10000"]["bb_lddt"]["chain_mean_delta"], .1)
    with pytest.raises(ValueError):
        trial.summarize(TARGETS, {"30000": quality(0.)}, reference)
    assert trial.CAVEAT in trial.LIMITS and "Only the 30000-update checkpoint is post-cooldown" in trial.CAVEAT


# ── scripts, CLI and reused evaluator ───────────────────────────────────


GPU_PARTITIONS = ("6000ada", "test", "heavy")


def test_stage_scripts_run_frozen_code_from_the_submit_directory(tmp_path):
    for name, text in trial.scripts(tmp_path / "run").items():
        assert text.startswith("#!/usr/bin/env bash\nset -euo pipefail\n"), name
        assert 'cd "$SLURM_SUBMIT_DIR"' in text and 'PY="$SLURM_SUBMIT_DIR/.venv/bin/python"' in text
        assert 'export PYTHONPATH="$CODE:$CODE/src"' in text and '"$PY" "$RUN/verify_frozen.py"' in text
        assert ("CUDA_VISIBLE_DEVICES" in text) == (name in ("cpu_admit.sh", "admit_training.sh", "score.sh"))
    text = trial.scripts(tmp_path / "run")["cpu_admit.sh"]
    assert all(test in text for test in trial.TESTS) and "--junitxml" in text and " reference " in text


def test_submission_chain_orders_partitions_dependencies_and_never_sizes_gpu_jobs(tmp_path):
    text = trial.submit_chain(tmp_path / "run", tmp_path)
    lines = [line for line in text.splitlines() if "sbatch" in line and not line.startswith("#")]
    stages = ("cpu_admit.sh", "train.sh", "admit_training.sh", "infer.sh", "score.sh")
    assert [re.search(r'"\$RUN/(\w+\.sh)"', line).group(1) for line in lines] == list(stages)
    assert [re.search(r"-p (\S+)", line).group(1) for line in lines] == [
        "cpu_only", "6000ada", "cpu_only", "6000ada", "cpu_only"]
    previous = None
    for line in lines:
        assert " -A lcdd " in line
        gpu = re.search(r"-p (\S+)", line).group(1) in GPU_PARTITIONS
        assert ("--gres=gpu:1" in line) == gpu
        if gpu:
            assert not re.search(r"(\s-c\s|--cpus-per-task|--mem)", line)
        if previous:
            assert f"--dependency=afterok:${previous}" in line and "--kill-on-invalid-dep=yes" in line
        else:
            assert "--dependency" not in line
        previous = re.match(r"(\w+)=", line).group(1)
    assert "-q verylong --gres=gpu:1 --time=1-00:00:00" in lines[1] and "-q short" in lines[3]
    assert 'test ! -e "$RUN/jobs.json"' in text


def test_cli_stages_require_a_slurm_allocation(tmp_path):
    env = {key: value for key, value in os.environ.items() if not key.startswith("SLURM_")}
    for stage in ("reference", "train", "admit-training", "infer", "score", "report", "dry-run-admission"):
        done = subprocess.run([sys.executable, str(Path(trial.__file__)), stage, "--run", str(tmp_path)], env=env,
                              capture_output=True, text=True)
        assert done.returncode == 2 and "Slurm allocation" in done.stderr, stage


def test_reference_script_only_reads_admitted_results_with_the_reused_evaluator():
    tree = ast.parse(trial.REFERENCE_SCRIPT)
    calls = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)}
    assert {"validate_run", "admitted_inference", "verify_canonical", "geometry_diagnostics", "unchanged",
            "write_new"} <= calls
    assert not calls & {"sample", "infer", "score", "report", "train", "probe", "preflight", "load"}
    assert "compare_seed_replicates" in trial.REFERENCE_SCRIPT and trial.REFERENCE_VERIFIER.endswith(
        "compare_seed_replicates.py")


# ── torch-free prepare on a synthetic world ─────────────────────────────


def sha(path):
    return inputs.sha256(path)


@pytest.fixture
def world(tmp_path):
    training, evaluation, panel_run, ost = (tmp_path / name for name in ("training", "eval", "panel", "ost"))
    ost.write_text("canonical ost")
    recipe = template(training)
    config = training / f"configs/{trial.REF_ARM}.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(yaml.safe_dump(recipe, sort_keys=False))
    home = training / f"training/{trial.REF_ARM}"
    stream = "5" * 64
    write(home / "initialization.json", dict(target_sha256=INITIAL))
    write(home / "audit/complete.json", dict(input_stream_sha256=stream))
    (home / "audit/inputs.jsonl").write_text('{"micro": 1, "sha256": "' + "6" * 64 + '"}\n')
    write(training / "source_manifest.json", {f"configs/{trial.REF_ARM}.yaml": sha(config)})
    write(panel_run / "panel/panel.json", dict(targets=[dict(target=name, **row) for name, row in TARGETS.items()]))
    write(panel_run / "panel_admitted.json", dict(admitted=True))
    fields = {key: recipe[key] for key in proof.MODEL_FIELDS}
    write(evaluation / "protocol.json", dict(
        training_run=dict(run=str(training), configs={trial.REF_ARM: sha(config)}),
        arms={trial.REF_ARM: dict(training=str(home), config=str(config), preflight=str(
            training / f"preflight/{trial.REF_ARM}"), u0=str(training / "u0"), model_fields=fields)},
        reference10k=dict(run=str(panel_run)), panel=str(panel_run / "panel"),
        panel_sha256=sha(panel_run / "panel/panel.json"), ost=str(ost)))
    for arm in trial.FLAT_ARMS:
        write(evaluation / f"train_admitted/{arm}.json", dict(complete=dict(input_stream_sha256=stream),
              final_checkpoint_sha256="1" * 64, final_ema_sha256="2" * 64))
        write(evaluation / f"scores/{arm}.json", dict(admitted=True, arm=arm))
    write(evaluation / "comparison.json", dict(sampler=dict(trial.SAMPLER), architecture_selected=False, identities={
        trial.REF_ARM: dict(input_stream_sha256=stream, initial_model_sha256=INITIAL)}))
    write(evaluation / "inference_admitted.json", dict(admitted=True))
    write(evaluation / "cpu_admitted.json", dict(admitted=True))
    (evaluation / "code/benchmarks").mkdir(parents=True)
    shutil.copy2(curve.ROOT / trial.REFERENCE_VERIFIER, evaluation / "code" / trial.REFERENCE_VERIFIER)
    source = {f"code/{name}": sha(curve.ROOT / name) for name in map(str, curve.frozen_files(curve.ROOT))
              if name.startswith("src/")}
    write(evaluation / "source_manifest.json", source)
    return tmp_path


def prepared(world):
    receipt = trial.prepare(world / "out", eval_run=world / "eval", stamp=STAMP)
    return Path(receipt["run"]), receipt


def test_prepare_is_torch_free_sealed_and_binds_the_reference_recipe(world):
    run, receipt = prepared(world)
    assert run.name == f"flat-fine6-long-30k-{STAMP}" and receipt["arm"] == trial.ARM
    manifest = json.loads((run / "source_manifest.json").read_text())
    assert not set(manifest) & set(trial.UNSEALED)
    assert {"protocol.json", "verify_frozen.py", "write_cpu_admission.py", "submit_chain.sh", "train.sh",
            f"configs/{trial.ARM}.yaml", "code/benchmarks/train_flat_fine6_long.py"} <= set(manifest)
    assert (run / "logs").is_dir() and not (run / "jobs.json").exists()
    protocol = json.loads((run / "protocol.json").read_text())
    assert protocol["reference"]["initial_model_sha256"] == INITIAL and protocol["checkpoint_phases"] == trial.PHASES
    trial.validate_protocol(run, protocol)
    done = subprocess.run([sys.executable, str(run / "verify_frozen.py")], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    code = run / "code/benchmarks/train_flat_fine6_long.py"
    tree = ast.parse(code.read_text())
    top = {alias.name.split(".")[0] for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
           for alias in node.names} | {node.module.split(".")[0] for node in tree.body
                                       if isinstance(node, ast.ImportFrom) and node.module}
    assert "torch" not in top and "mambafold" not in top
    with pytest.raises(FileExistsError):
        trial.prepare(world / "out", eval_run=world / "eval", stamp=STAMP)


@pytest.mark.parametrize("fault", ["config", "reference_file", "home", "template", "model_source", "protocol"])
def test_prepared_run_detects_tampering(world, fault):
    if fault == "model_source":
        manifest = json.loads((world / "eval/source_manifest.json").read_text())
        manifest[next(iter(manifest))] = "0" * 64
        (world / "eval/source_manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError):
            prepared(world)
        return
    run, _ = prepared(world)
    protocol = json.loads((run / "protocol.json").read_text())
    if fault == "config":
        config = yaml.safe_load((run / f"configs/{trial.ARM}.yaml").read_text())
        (run / f"configs/{trial.ARM}.yaml").write_text(yaml.safe_dump(dict(config, ema_decay=0.99)))
    elif fault == "reference_file":
        write(world / "eval/comparison.json", dict(architecture_selected=True))
    elif fault == "home":
        write(world / f"training/training/{trial.REF_ARM}/initialization.json", dict(target_sha256="9" * 64))
    elif fault == "template":
        path = world / f"training/configs/{trial.REF_ARM}.yaml"
        path.write_text(path.read_text() + "\n# changed\n")
    else:
        protocol["horizon"]["cooldown_updates"] = 2000
        (run / "protocol.json").write_text(json.dumps(protocol))
    with pytest.raises((ValueError, KeyError)):
        trial.validate_protocol(run, protocol)
    assert subprocess.run([sys.executable, str(run / "verify_frozen.py")], capture_output=True).returncode != 0
