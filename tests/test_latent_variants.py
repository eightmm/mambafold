"""The latent-variant screen binds six scratch arms to the reused, hash-bound flat_fine6 seed-20261023 point."""

import ast
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from benchmarks import compare_latent_variants as trial
from benchmarks import evaluate_recycle_checkpoint_curve as curve

trial.bind()
B = trial.BASELINE
TARGETS = dict(a=dict(entry_id="e1"), b=dict(entry_id="e1"), c=dict(entry_id="e2"))
INITIAL, STREAM = "4" * 64, "5" * 64
EXTRA = dict(num_workers=3, prefetch_factor=2, log_interval=100, ckpt_interval=5000, resume=None,
             initialize_refiner_from=None, initialize_model_from_ema=False, keep_checkpoint_steps=[5000, 10000])
WSD = dict(warmup_steps=500, stable_steps=7500, cosine_decay_steps=2000, scheduler_horizon=10000)
STAMP = "20990101T000000Z"


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value))


def probe(code, **changes):
    return dict(code=str(code), initial_model_sha256=INITIAL, perturbed_model_sha256="1" * 64, batch_sha256="2" * 64,
                loss=1.053164005279541, gradient_norms={key: 1. for key in trial.seedrep.GROUPS},
                parameters=trial.BASELINE_PARAMETERS, **changes)


def outputs(offset=0.):
    velocity = np.linspace(-.8, .8, 4 * 14 * 3, dtype=np.float32).reshape(1, 4, 14, 3)
    return dict(eval_velocity=velocity + np.float32(offset), train_velocities=np.stack([velocity] * 3),
                depths=np.ones((1, 2), dtype=np.float32))


def historical_equivalence(training, bitwise=True):
    """The reused training run's sealed current-code probe and the admission that found it bitwise equal."""
    for label in ("old", "new"):
        folder = training / f"preflight/equivalence/{label}"
        folder.mkdir(parents=True, exist_ok=True)
        write(folder / "probe.json", probe(training.resolve() / "code"))
        np.savez(folder / "outputs.npz", **outputs())
    evidence = {name: curve.sha(training / name) for name in trial.EQUIVALENCE_REFERENCE}
    (training / "preflight_verified.json").unlink(missing_ok=True)
    write(training / "preflight_verified.json", dict(admitted=True, evidence_sha256=evidence, equivalence=dict(
        admitted=True, arrays={name: dict(bitwise_equal=bitwise) for name in trial.EQUIVALENCE_ARRAYS})))
    return dict(preflight_admission_sha256=curve.sha(training / "preflight_verified.json"), preflight=evidence)


def baseline_template(training):
    value = {key: item for key, item in trial.expected("r4_w1024").items() if key != "latent_coarse_width"}
    value.update({key: item for key, item in trial.BASELINE_LATENT.items() if key != "latent_coarse_width"})
    return dict(value, **EXTRA, config=str(training / f"configs/{B}.yaml"), out_dir=str(training / f"training/{B}"),
                wandb_name=f"seed-replicate-{B}")


@pytest.fixture
def world(tmp_path):
    training, evaluation, full4, ost = (tmp_path / name for name in ("train", "eval", "full4", "ost"))
    ost.write_text("canonical ost")
    template = baseline_template(training)
    (training / "configs").mkdir(parents=True)
    (training / f"configs/{B}.yaml").write_text(yaml.safe_dump(template, sort_keys=False))
    home = training / f"training/{B}"
    (home / "audit").mkdir(parents=True)
    (home / "ckpt_0010000.pt").write_bytes(b"checkpoint")
    (home / "audit/inputs.jsonl").write_text("")
    write(home / "initialization.json", dict(target_sha256=INITIAL, parameters=trial.BASELINE_PARAMETERS))
    write(home / "audit/complete.json", dict(input_stream_sha256=STREAM))
    write(training / "source_manifest.json", curve.seal(training, [f"configs/{B}.yaml"]))
    sealed = historical_equivalence(training)
    (evaluation / "code/benchmarks").mkdir(parents=True)
    shutil.copy2(curve.ROOT / trial.BASELINE_VERIFIER, evaluation / "code" / trial.BASELINE_VERIFIER)
    write(full4 / "panel_admitted.json", dict(panel_sha256="6" * 64))
    write(evaluation / "protocol.json", dict(training_run=dict(run=str(training), **sealed), sampler=dict(trial.SAMPLER),
        inference_depth=2, panel=str(full4 / "panel"), panel_sha256="6" * 64, ost=str(ost), WSD=WSD,
        reference10k=dict(run=str(full4), panel_admission_sha256=curve.sha(full4 / "panel_admitted.json")),
        arms={B: dict(training=str(home), model_fields={key: template[key] for key in trial.proof.MODEL_FIELDS})}))
    write(evaluation / "comparison.json", dict(architecture_selected=False, architectures=dict(flat_fine6=dict(
        new_code_seeds=dict(lddt=dict(chain_mean=dict(n=3, sd=.0115)))))))
    for name in ("cpu_admitted.json", "inference_admitted.json", f"scores/{B}.json", f"train_admitted/{B}.json"):
        write(evaluation / name, dict(admitted=True))
    write(evaluation / "source_manifest.json", curve.seal(evaluation, ["code/" + trial.BASELINE_VERIFIER]))
    run = Path(trial.prepare(tmp_path / "out", eval_run=evaluation, stamp=STAMP)["run"])
    return SimpleNamespace(run=run, protocol=trial.read(run / "protocol.json"), training=training,
                           evaluation=evaluation, template=template)


def test_prepare_is_torch_free_sealed_and_matches_the_run_layout(world, tmp_path):
    code = ("import sys; sys.modules['torch'] = None; from pathlib import Path; "
            "from benchmarks import compare_latent_variants as trial; "
            f"print(trial.prepare(Path({str(tmp_path / 'login')!r}), eval_run=Path({str(world.evaluation)!r}), "
            f"stamp={STAMP!r})['run'])")
    completed = subprocess.run([sys.executable, "-c", code], cwd=curve.ROOT, capture_output=True, text=True,
                               check=True)
    assert Path(completed.stdout.strip()).name == f"latent-variants-scratch-wsd-{STAMP}"
    run = world.run
    manifest = trial.read(run / "source_manifest.json")
    assert not set(manifest) & set(trial.UNSEALED)
    assert all(curve.sha(run / name) == digest for name, digest in manifest.items())
    assert "code/benchmarks/compare_latent_variants.py" in manifest and "code/src/mambafold/model/fold/latent_mamba.py" \
        in manifest and all(f"configs/{arm}.yaml" in manifest for arm in trial.ARMS)
    for name in ("cpu_admit.sh", "preflight.sh", "train.sh", "infer.sh", "score.sh"):
        text = (run / name).read_text()
        assert text.startswith("#!/usr/bin/env bash\nset -euo pipefail\ncd \"$SLURM_SUBMIT_DIR\"\n")
        assert 'PY="$SLURM_SUBMIT_DIR/.venv/bin/python"' in text and 'export PYTHONPATH="$CODE:$CODE/src"' in text
        assert '"$PY" "$RUN/verify_frozen.py"' in text and "--mem" not in text and "--cpus" not in text
    assert f'ARMS=({" ".join(trial.ARMS)})' in (run / "train.sh").read_text()
    assert all(name in (run / "cpu_admit.sh").read_text() for name in trial.TESTS)
    assert 'export CUDA_VISIBLE_DEVICES=""' in (run / "score.sh").read_text()
    for arm, entry in world.protocol["arms"].items():
        config = yaml.safe_load(Path(entry["config"]).read_text())
        differing = {key for key in set(config) | set(world.template) if config.get(key) != world.template.get(key)}
        assert differing <= set(trial.LATENT_KEYS) | trial.TEMPLATE_KEYS and config["latent_coarse_width"] == (
            1024 if arm == "hier_wide" else 0) and config["seed"] == 20261023
        assert entry["model_fields"]["latent_coarse_width"] == config["latent_coarse_width"]
    assert world.protocol["model_source_changes_since_baseline"]


def test_protocol_admits_the_declared_arms_with_trainer_defaults(world):
    configs = trial.validate_protocol(world.run, world.protocol)
    assert list(configs) == list(trial.ARMS)
    with trial.registry():
        for arm, config in configs.items():
            trial.scratch.validate_config(config, arm)
            with pytest.raises(ValueError, match="atom recycling"):
                trial.scratch.validate_config(dict(config, atom_recycling="carry"), arm)
    trial.seedrep.verify_flat_construction(configs["hier_wide"], dict(recycle_policy="legacy"))


def edit_yaml(path, **changes):
    value = yaml.safe_load(Path(path).read_text())
    Path(path).write_text(yaml.safe_dump(dict(value, **changes), sort_keys=False))


@pytest.mark.parametrize("fault", ["arm_yaml", "flop", "parameters", "matched", "fields", "sampler", "panel",
                                   "eval_file", "template", "arm_set", "path", "equivalence_outputs",
                                   "equivalence_binding"])
def test_protocol_rejects_changed_arms_recipe_or_baseline(world, fault):
    protocol = json.loads(json.dumps(world.protocol))
    if fault == "arm_yaml":
        edit_yaml(protocol["arms"]["r8_w1024"]["config"], d_latent=512)
    elif fault == "flop":
        protocol["arms"]["r4_w1024"]["latent_flop_ratio"] = 1.0
    elif fault == "parameters":
        protocol["arms"]["fine_deep"]["expected_parameters"] += 1
    elif fault == "matched":
        protocol["arms"]["r8_w1024"]["parameter_matched"] = True
    elif fault == "fields":
        protocol["arms"]["hier_wide"]["model_fields"].pop("latent_coarse_width")
    elif fault == "sampler":
        protocol["sampler"]["n_steps"] = 50
    elif fault == "panel":
        protocol["panel_sha256"] = "7" * 64
    elif fault == "eval_file":
        comparison = trial.read(world.evaluation / "comparison.json")
        comparison["architectures"]["flat_fine6"]["new_code_seeds"]["lddt"]["chain_mean"]["sd"] = .005
        write(world.evaluation / "comparison.json", comparison)
    elif fault == "template":
        edit_yaml(world.training / f"configs/{B}.yaml", lr=2e-4)
    elif fault == "arm_set":
        protocol["arms"].pop("fine_deep")
    elif fault == "equivalence_outputs":
        np.savez(world.training / trial.EQUIVALENCE_REFERENCE[1], **outputs(1e-3))
    elif fault == "equivalence_binding":
        protocol["equivalence_reference"]["loss"] = 1.
    else:
        protocol["arms"]["r16_w2048"]["training"] = str(world.run / "training/other")
    with pytest.raises(ValueError):
        trial.validate_protocol(world.run, protocol)


@pytest.mark.parametrize("fault", ["recipe", "equivalence_reference"])
def test_frozen_verifier_accepts_the_run_and_rejects_a_changed_recipe_or_reference(world, fault):
    subprocess.run([sys.executable, str(world.run / "verify_frozen.py")], check=True, capture_output=True)
    if fault == "recipe":
        edit_yaml(world.run / "configs/r4_w1024.yaml", d_latent=512)
    else:
        np.savez(world.training / trial.EQUIVALENCE_REFERENCE[1], **outputs(1e-3))
    assert subprocess.run([sys.executable, str(world.run / "verify_frozen.py")], capture_output=True).returncode


def test_equivalence_reference_is_the_sealed_bitwise_admitted_historical_probe(world):
    bound = world.protocol["equivalence_reference"]
    assert bound == trial.equivalence_binding(world.protocol["baseline"])
    assert bound["probe_sha256"] == curve.sha(world.training / trial.EQUIVALENCE_REFERENCE[0])
    assert Path(bound["code"]) == world.training.resolve() / "code" != world.run / "code"
    assert bound["outputs"].endswith("preflight/equivalence/new/outputs.npz")
    assert "equivalence_reference" in world.protocol["baseline_reproduction_gate"]
    sealed = historical_equivalence(world.training, bitwise=False)
    evaluation = trial.read(world.evaluation / "protocol.json")
    evaluation["training_run"].update(sealed)
    write(world.evaluation / "protocol.json", evaluation)
    with pytest.raises(ValueError, match="bitwise"):
        trial.equivalence_binding(world.protocol["baseline"])
    evaluation["training_run"]["preflight_admission_sha256"] = "0" * 64
    write(world.evaluation / "protocol.json", evaluation)
    with pytest.raises(ValueError, match="sealed by the reused evaluation"):
        trial.equivalence_binding(world.protocol["baseline"])


def test_cost_columns_and_parameter_matching_are_declared_exactly():
    assert {arm: trial.flop_ratio(arch) for arm, arch in trial.ARMS.items()} == dict(
        r4_w1024=4.0, r8_w1024=2.0, r16_w2048=4.0, r8_w1024_pm=1 / 3, hier_wide=4 / 3, fine_deep=2.0)
    assert trial.flop_ratio(trial.BASELINE_LATENT) == 1.0
    matched = {arm for arm in trial.ARMS if abs(trial.parameter_ratio(arm) - 1) <= trial.PARAMETER_TOLERANCE}
    assert matched == set(trial.PARAMETER_MATCHED) == {"r8_w1024_pm"}
    assert all(trial.ARMS[arm][key] == trial.BASELINE_LATENT[key] for arm in trial.ARMS
               for key in ("latent_hierarchy", "latent_coarse_width") if arm != "hier_wide")


def test_declared_parameter_counts_match_cpu_construction(world):
    from mambafold.train.trainer import build_model

    for arm, entry in world.protocol["arms"].items():
        model = build_model(yaml.safe_load(Path(entry["config"]).read_text()), "cpu")
        assert sum(p.numel() for p in model.parameters()) == trial.PARAMETERS[arm], arm
        del model


def probes(binding, new=0., warmup=0.):
    """Sealed reference plus a warm-up and a second current-code probe offset from it."""
    reference = dict(probe=trial.read(binding["probe"]), arrays=trial.probe_arrays(binding["outputs"]))
    code = Path(binding["run"]).parent / "latent/code"
    return (dict(reference=reference["probe"], warmup=probe(code), new=probe(code)),
            dict(reference=reference["arrays"], warmup=outputs(warmup), new=outputs(new)))


def rows(binding=None, baseline=INITIAL):
    names = [B, *trial.ARMS]
    parameters = dict(trial.PARAMETERS, **{B: trial.BASELINE_PARAMETERS})
    rows = {name: dict(parameters=parameters[name], trainable_parameters=parameters[name],
                       initial_model_sha256=f"{index + 10:064x}", parameter_signature_sha256=f"{index + 10:064x}")
            for index, name in enumerate(names)}
    rows[B]["initial_model_sha256"] = baseline
    memory = {name: dict(feasible=True, peak_allocated_gib=20., peak_reserved_gib=22., device_total_gib=47.5,
                         warm_update_ms=1300., losses_finite=True) for name in names}
    equivalence = trial.admit_equivalence(binding, INITIAL, *probes(binding)) if binding else dict(admitted=True)
    evidence = {f"preflight/equivalence/{label}/{name}": "0" * 64 for label in trial.EQUIVALENCE_LABELS
                for name in trial.EQUIVALENCE_FILES}
    return dict(rows=rows, memory=memory, equivalence=equivalence, evidence_sha256=evidence)


def test_preflight_records_feasibility_and_requires_the_reused_initial_tensors(world):
    binding = world.protocol["equivalence_reference"]
    assert set(trial.verify_preflight(rows(binding), world.protocol).values()) == {True}
    value = rows(binding)
    value["memory"]["r16_w2048"] = dict(feasible=False, error="OutOfMemoryError: CUDA out of memory")
    assert trial.verify_preflight(value, world.protocol)["r16_w2048"] is False


@pytest.mark.parametrize("fault", ["baseline", "parameters", "duplicate", "equivalence", "baseline_oom",
                                   "incomplete", "reserved", "silent_oom", "coverage", "tolerance", "reference",
                                   "bitwise_record", "warmup_record", "evidence"])
def test_preflight_rejects_non_reproducing_or_unmeasured_construction(world, monkeypatch, fault):
    value = rows(world.protocol["equivalence_reference"], "9" * 64 if fault == "baseline" else INITIAL)
    if fault == "parameters":
        value["rows"]["r8_w1024"]["parameters"] += 1
    elif fault == "duplicate":
        value["rows"]["fine_deep"]["initial_model_sha256"] = INITIAL
    elif fault == "equivalence":
        value["equivalence"]["admitted"] = False
    elif fault == "baseline_oom":
        value["memory"][B] = dict(feasible=False, error="OutOfMemoryError")
    elif fault == "incomplete":
        value["memory"]["hier_wide"].pop("warm_update_ms")
    elif fault == "reserved":
        value["memory"]["r4_w1024"]["peak_reserved_gib"] = 48.
    elif fault == "silent_oom":
        value["memory"]["r16_w2048"] = dict(feasible=False)
    elif fault == "coverage":
        value["memory"].pop("fine_deep")
    elif fault == "tolerance":
        monkeypatch.setitem(trial.PARAMETERS, "r8_w1024_pm", 300_000_000)
        value["rows"]["r8_w1024_pm"].update(parameters=300_000_000, trainable_parameters=300_000_000)
    elif fault == "reference":
        value["equivalence"]["reference"]["probe_sha256"] = "0" * 64
    elif fault == "bitwise_record":
        value["equivalence"]["arrays"]["depths"].pop("bitwise_equal")
    elif fault == "warmup_record":
        value["equivalence"].pop("warmup")
    else:
        value["evidence_sha256"].pop("preflight/equivalence/warmup/outputs.npz")
    with pytest.raises(ValueError):
        trial.verify_preflight(value, world.protocol)


def test_preflight_equivalence_gates_the_second_probe_and_only_records_the_warmup(world):
    binding = world.protocol["equivalence_reference"]
    # Sub-tolerance drift admits without bitwise equality; a far-off warm-up is recorded and never gates.
    value = trial.admit_equivalence(binding, INITIAL, *probes(binding, new=1e-4, warmup=.5))
    assert value["admitted"] is True and value["reference"] == trial.reference_record(binding)
    assert value["arrays"]["eval_velocity"]["bitwise_equal"] is False
    assert value["arrays"]["depths"]["bitwise_equal"] is True
    assert value["loss"]["old"] == binding["loss"]
    warmup = value["warmup"]["arrays"]["eval_velocity"]
    assert warmup["bitwise_equal"] is False and warmup["max_abs_difference"] == pytest.approx(.5)
    assert value["warmup"]["identity_equal"]["initial_model_sha256"] is True
    exact = trial.admit_equivalence(binding, INITIAL, *probes(binding))
    assert all(row["bitwise_equal"] for row in exact["arrays"].values())
    assert all(row["bitwise_equal"] for row in exact["warmup"]["arrays"].values())
    with pytest.raises(ValueError, match="eval_velocity differs"):
        trial.admit_equivalence(binding, INITIAL, *probes(binding, new=.5))
    probe_rows, arrays = probes(binding)
    with pytest.raises(ValueError, match="sealed historical probe"):
        trial.admit_equivalence(binding, INITIAL, dict(probe_rows, new=probe_rows["reference"]), arrays)
    with pytest.raises(ValueError, match="sealed historical probe"):
        trial.admit_equivalence(dict(binding, code="/elsewhere"), INITIAL, probe_rows, arrays)


def test_native_gate_checks_fresh_weights_and_the_baseline_input_prefix():
    gate = rows()
    first = tuple(f"{index:064x}" for index in range(8))
    native = dict(arm="r8_w1024", seed=20261023, model_sha256=gate["rows"]["r8_w1024"]["initial_model_sha256"],
                  rows=[dict(input_sha256=value) for value in first])
    trial.check_native(native, gate, "r8_w1024", first)
    for changed in (dict(model_sha256=INITIAL), dict(rows=native["rows"][::-1]), dict(seed=1), dict(arm="r4_w1024")):
        with pytest.raises(ValueError):
            trial.check_native(dict(native, **changed), gate, "r8_w1024", first)


def test_arms_without_their_own_training_admission_are_excluded_not_dropped(tmp_path):
    for arm in ("r4_w1024", "fine_deep"):
        write(tmp_path / f"train_admitted/{arm}.json", dict(admitted=True))
    (tmp_path / "training/r16_w2048").mkdir(parents=True)
    present, excluded = trial.split_arms(tmp_path)
    assert present == ["r4_w1024", "fine_deep"] and set(excluded) == set(trial.ARMS) - set(present)
    assert excluded["r16_w2048"] == dict(reason="no training admission", training_home_exists=True,
                                         final_checkpoint_exists=False)
    (tmp_path / "train_admitted/hier_wide.json").symlink_to(tmp_path / "train_admitted/r4_w1024.json")
    with pytest.raises(ValueError, match="regular"):
        trial.split_arms(tmp_path)


def quality(offset, bump=0.):
    return {name: dict(lddt=.15 + offset + (bump if name == "a" else 0), bb_lddt=.57 + offset, tm_score=.52)
            for name in TARGETS}


def test_report_statistics_are_paired_to_the_baseline_and_descriptive_only():
    vectors = {B: quality(0.), "r8_w1024": quality(.03), "fine_deep": quality(0., bump=.01)}
    stats = trial.summarize(TARGETS, vectors, .0115)
    assert set(stats["paired_minus_baseline"]) == {"r8_w1024", "fine_deep"}
    wide, deep = stats["paired_minus_baseline"]["r8_w1024"], stats["paired_minus_baseline"]["fine_deep"]
    assert wide["aa_chain_mean_delta"] == pytest.approx(.03) and wide["within_seed_noise"] is False
    assert wide["aa_delta_over_seed_sd"] == pytest.approx(.03 / .0115)
    assert (wide["effects"]["lddt"]["wins"], wide["effects"]["lddt"]["ties"], wide["effects"]["lddt"]["losses"]) \
        == (3, 0, 0)
    assert deep["within_seed_noise"] is True and deep["effects"]["lddt"]["wins"] == 1 \
        and deep["effects"]["lddt"]["ties"] == 2 and deep["effects"]["tm_score"]["ties"] == 3
    assert deep["effects"]["lddt"]["equal_entry_mean_delta"] == pytest.approx(statistics.fmean([.005, 0.]))
    assert stats["runs"][B]["lddt"]["chain_mean"] == pytest.approx(.15)


@pytest.mark.parametrize("fault", ["baseline", "alone", "coverage", "nonfinite", "undeclared"])
def test_report_statistics_require_the_baseline_and_finite_complete_vectors(fault):
    vectors = {B: quality(0.), "r4_w1024": quality(.01)}
    if fault == "baseline":
        vectors.pop(B)
    elif fault == "alone":
        vectors.pop("r4_w1024")
    elif fault == "coverage":
        vectors["r4_w1024"].pop("c")
    elif fault == "nonfinite":
        vectors["r4_w1024"]["a"]["lddt"] = float("nan")
    else:
        vectors["other"] = quality(0.)
    with pytest.raises(ValueError):
        trial.summarize(TARGETS, vectors, .0115)


def test_registry_restores_shared_verifier_globals_after_failure():
    before = trial.scratch.VARIANTS, trial.proof.VARIANTS, trial.proof.MODEL_FIELDS
    with pytest.raises(RuntimeError), trial.registry():
        assert trial.proof.MODEL_FIELDS[-1] == "latent_coarse_width"
        assert set(trial.scratch.VARIANTS) == set(trial.ARMS)
        raise RuntimeError
    assert (trial.scratch.VARIANTS, trial.proof.VARIANTS, trial.proof.MODEL_FIELDS) == before


def test_reference_script_only_reads_admitted_results_and_writes_one_receipt():
    tree = ast.parse(trial.REFERENCE_SCRIPT)
    calls = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)}
    assert "write_new" in calls and not calls & {"infer", "score", "train", "preflight", "admit_training", "report"}
    assert "torch" not in trial.REFERENCE_SCRIPT
