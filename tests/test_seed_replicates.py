"""Seed replicates bind five new scratch runs to the reused seed-20261023 flat_fine6 and fixed2 points."""

import ast
import json
import math
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from benchmarks import compare_seed_replicates as trial
from benchmarks import evaluate_recycle_checkpoint_curve as curve

trial.bind()
TARGETS = dict(a=dict(entry_id="e1"), b=dict(entry_id="e1"), c=dict(entry_id="e2"))
HARDWARE = dict(name=trial.base.TRAIN_GPU, uuid="gpu-0", visible_devices="0", slurm_job="1", pid=1,
                torch="2.11", cuda="13.0")
WSD = dict(curve.WSD, adamw_peak=1e-4, adamw_floor=1e-6, muon_peak=3e-3, muon_floor=3e-5,
           backbone_lr_scale=1.0, actual_pre_step_rates_required=True)
INITIAL, STREAM = "4" * 64, "5" * 64
EXTRA = dict(num_workers=3, prefetch_factor=2, log_interval=100, ckpt_interval=5000, resume=None,
             initialize_refiner_from=None, initialize_model_from_ema=False, keep_checkpoint_steps=[5000, 10000])
STAMP = "20990101T000000Z"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def quality(offset, spread=0.):
    return {target: dict(lddt=.1 + offset + index * spread, bb_lddt=.5 + offset, tm_score=.4 + offset)
            for index, target in enumerate(TARGETS)}


def template(arm, home, name):
    return trial.expected(arm) | EXTRA | dict(seed=20261023, config=str(home / f"configs/{name}.yaml"),
                                              out_dir=str(home / f"training/{name}"), wandb_name=name)


def training_home(home, arm):
    path = home / f"training/{arm}"
    write(path / "initialization.json", dict(target_sha256=INITIAL, mamba3_runtime=dict(version="native")))
    write(path / "audit/complete.json", dict(input_stream_sha256=STREAM))
    write(path / "ckpt_0010000.pt", "weights")
    return path


@pytest.fixture
def world(tmp_path):
    full4, pair, old, ost = (tmp_path / name for name in ("full4", "pair", "old", "ost"))
    for name in curve.frozen_files(curve.ROOT):
        (full4 / "code" / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(curve.ROOT / name, full4 / "code" / name)
    ost.write_text("canonical ost")
    panel = dict(targets=[dict(target=name, **row) for name, row in TARGETS.items()])
    flat = template(trial.REPRODUCTION_ARM, old, "flat_fine6")
    fixed = template("fixed2_s20261024", pair, "fixed2")
    (old / "configs").mkdir(parents=True)
    (old / "configs/flat_fine6.yaml").write_text(yaml.safe_dump(flat, sort_keys=False))
    (pair / "configs").mkdir(parents=True)
    (pair / "configs/fixed2.yaml").write_text(yaml.safe_dump(fixed, sort_keys=False))
    old_home, pair_home = training_home(old, "flat_fine6"), training_home(pair, "fixed2")
    pair_receipt = dict(config=dict(fixed, requested_num_workers=3), complete=dict(input_stream_sha256=STREAM),
                        final_checkpoint_sha256="1" * 64, final_ema_sha256="2" * 64, hardware=HARDWARE)
    write(pair / "train_admitted/fixed2.json", pair_receipt)
    write(pair / "protocol.json", dict(arms=dict(fixed2=dict(
        model_fields={key: fixed[key] for key in trial.RECYCLE_FIELDS}))))
    write(pair / "source_manifest.json", curve.seal(pair, ["configs/fixed2.yaml", "protocol.json"]))
    # old flat_fine6 run with its own frozen evaluator
    (old / "code/benchmarks").mkdir(parents=True)
    shutil.copy2(curve.ROOT / trial.FLAT_VERIFIER, old / "code" / trial.FLAT_VERIFIER)
    write(old / "protocol.json", dict(sampler=dict(trial.SAMPLER), WSD=WSD, ost=str(ost), arms=dict(flat_fine6=dict(
        training=str(old_home), model_fields={key: flat[key] for key in trial.BASE_FIELDS}))))
    write(old / "panel/panel.json", panel)
    write(old / "panel_admitted.json", dict(panel_sha256=curve.sha(old / "panel/panel.json")))
    old_receipt = dict(config=dict(flat, requested_num_workers=3), complete=dict(input_stream_sha256=STREAM,
        steady_mean_step_ms=1294.), final_checkpoint_sha256="3" * 64, final_ema_sha256="6" * 64, hardware=HARDWARE)
    write(old / "train_admitted/flat_fine6.json", old_receipt)
    scorer = dict(version="2.9.1", executable=str(ost), compare_args=["--lddt"])
    old_vectors = quality(.04)
    write(old / "scores/flat_fine6.json", dict(metrics=old_vectors, scorer=scorer))
    overall = dict(coverage=3, **{m: statistics.fmean(v[m] for v in old_vectors.values()) for m in trial.QUALITY})
    costs = dict(parameters=trial.PARAMETER_COUNT, training=dict(steady_mean_step_ms=1294.,
        all_update_weighted_mean_step_ms=1292., peak_allocated_gib=18.4, peak_reserved_gib=20.),
        inference=dict(coverage=3, mean_seconds=34.8, peak_allocated_gib=1.6))
    write(old / "comparison.json", dict(overall=dict(flat_fine6=overall), costs=dict(flat_fine6=costs),
                                        inference_hardware=HARDWARE))
    write(old / "inference_admitted.json", dict(admitted=True))
    write(old / "source_manifest.json", curve.seal(old, ["protocol.json", "configs/flat_fine6.yaml", "panel/panel.json",
                                                         "code/" + trial.FLAT_VERIFIER]))
    # three-way run that evaluated the reused fixed2 model
    original = dict(benchmark_role="three-way", sampler=dict(curve.SAMPLER, K=2), inference_depths=[1, 2, 4, 8, 16],
        WSD=WSD, ost=str(ost), arms=dict(fixed4=dict(training=str(full4 / "training/fixed4"))),
        external_pair=dict(run=str(pair), source_manifest_sha256=curve.sha(pair / "source_manifest.json"),
                           expected_initial_model_sha256=INITIAL))
    write(full4 / "protocol.json", original)
    write(full4 / "panel/panel.json", panel)
    write(full4 / "panel_admitted.json", dict(panel_sha256=curve.sha(full4 / "panel/panel.json"), coverage={}))
    write(full4 / "source_manifest.json", curve.seal(full4))
    binding = {key: curve.sha(full4 / name) for key, name in curve.REFERENCE_FILES.items()
               if key in ("source_manifest_sha256", "protocol_sha256", "panel_admission_sha256")}
    write(full4 / "comparison.json", dict(benchmark_role="three-way", architecture_selected=False, **binding))
    write(full4 / "inference_admitted.json", dict(admitted=True))
    write(full4 / "external_training_admitted.json", dict(admitted=True))
    run = Path(trial.prepare(tmp_path / "out", flat_reference=old, full4_reference=full4, stamp=STAMP)["run"])
    protocol = trial.read(run / "protocol.json")
    flat_value = dict(admitted=True, arm="flat_fine6", seed=20261023, K=2,
        **{key: protocol["reference_flat_fine6"][key] for key in trial.FLAT_FILES},
        verifier_sha256=curve.sha(old / "code" / trial.FLAT_VERIFIER), ost_executable_sha256=curve.sha(ost),
        scorer=scorer, hardware=HARDWARE, per_target=old_vectors, overall=overall, costs=costs,
        training_receipt=old_receipt, training=str(old_home), checkpoint_sha256="3" * 64, ema_sha256="6" * 64,
        config=old_receipt["config"], runtime=dict(version="native"), initial_model_sha256=INITIAL,
        input_stream_sha256=STREAM)
    write(trial.flat_reference_path(run), flat_value)
    row = dict(arm="fixed2", K=2, coverage=3, steady_train_ms=1441., all_update_train_ms=1441.5,
               train_allocated_gib=21.5, train_reserved_gib=24.1, inference_seconds=41.9,
               inference_allocated_gib=1.6, inference_reserved_gib=1.7)
    reference = dict(scorer=scorer, hardware=HARDWARE, models=dict(fixed2=dict(K=2, training=str(pair_home), row=row,
        per_target=quality(0.), geometry=dict(totals={}), training_receipt=pair_receipt, checkpoint_sha256="1" * 64,
        ema_sha256="2" * 64, config=pair_receipt["config"], runtime=dict(version="native"))))
    return SimpleNamespace(tmp=tmp_path, run=run, protocol=protocol, old=old, pair=pair, full4=full4, ost=ost,
                           reference=reference, flat_value=flat_value)


def context(world):
    return dict(run=world.run, protocol=world.protocol, targets=TARGETS,
                configs=trial.validate_protocol(world.run, world.protocol))


def test_prepare_is_torch_free_sealed_and_matches_the_run_layout(world):
    code = ("import sys;sys.path[:0]=[sys.argv[1]];from benchmarks import compare_seed_replicates as t;"
            "t.prepare(sys.argv[2],flat_reference=sys.argv[3],full4_reference=sys.argv[4],stamp='20990102T000000Z');"
            "assert 'torch' not in sys.modules")
    out = world.tmp / "login"
    subprocess.run([sys.executable, "-c", code, str(trial.ROOT), str(out), str(world.old), str(world.full4)],
                   check=True)
    run = out / "seed-replicate-scratch-wsd-20990102T000000Z"
    manifest = json.loads((run / "source_manifest.json").read_text())
    for name in ("protocol.json", "verify_frozen.py", "write_cpu_admission.py", "cpu_admit.sh", "preflight.sh",
                 "train.sh", "infer.sh", "score.sh", "code/benchmarks/compare_seed_replicates.py",
                 "code/tests/test_seed_replicates.py", *(f"configs/{arm}.yaml" for arm in trial.ARMS)):
        assert manifest[name] == curve.sha(run / name)
    assert not set(manifest) & set(trial.UNSEALED) and not any("__pycache__" in name for name in manifest)
    for name, stage in (("cpu_admit.sh", "reference"), ("preflight.sh", "preflight"), ("train.sh", "train"),
                        ("infer.sh", "infer"), ("score.sh", "score"), ("score.sh", "report")):
        text = (run / name).read_text()
        assert text.startswith('#!/usr/bin/env bash\nset -euo pipefail\ncd "$SLURM_SUBMIT_DIR"\n')
        assert '"$SLURM_SUBMIT_DIR/.venv/bin/python"' in text and 'PYTHONPATH="$CODE:$CODE/src"' in text
        assert f'compare_seed_replicates.py" {stage} --run "$RUN"' in text and "verify_frozen.py" in text
    assert all(test in (run / "cpu_admit.sh").read_text() for test in trial.TESTS)
    assert "ARMS=(" + " ".join(trial.ARMS) + ")" in (run / "train.sh").read_text()
    protocol = json.loads((run / "protocol.json").read_text())
    assert {arm: (entry["architecture"], entry["seed"]) for arm, entry in protocol["arms"].items()} == {
        "flat_fine6_s20261023": ("flat_fine6", 20261023), "flat_fine6_s20261024": ("flat_fine6", 20261024),
        "flat_fine6_s20261025": ("flat_fine6", 20261025), "fixed2_s20261024": ("fixed2", 20261024),
        "fixed2_s20261025": ("fixed2", 20261025)}
    assert protocol["reproduction"]["expected_initial_model_sha256"] == INITIAL
    for arm in trial.ARMS:
        config = yaml.safe_load((run / f"configs/{arm}.yaml").read_text())
        assert config["keep_checkpoint_steps"] == [5000, 10000] and config["seed"] == trial.SEED_OF[arm]
    subprocess.run([sys.executable, str(run / "verify_frozen.py")], check=True)
    with pytest.raises(FileExistsError):
        trial.prepare(out, flat_reference=world.old, full4_reference=world.full4, stamp="20990102T000000Z")
    for path in (world.old / "comparison.json", world.pair / "configs/fixed2.yaml"):
        saved = path.read_bytes()
        path.write_text("{}")
        assert subprocess.run([sys.executable, str(run / "verify_frozen.py")], capture_output=True).returncode
        path.write_bytes(saved)
    assert not subprocess.run([sys.executable, str(run / "verify_frozen.py")], capture_output=True).returncode


def test_protocol_pairs_architectures_by_seed_and_keeps_the_reused_recipe(world):
    configs = trial.validate_protocol(world.run, world.protocol)
    assert set(configs) == set(trial.ARMS)
    drop = trial.TEMPLATE_KEYS | {"seed", *trial.ARCH_KEYS}
    assert len({json.dumps({k: v for k, v in c.items() if k not in drop}, sort_keys=True)
                for c in configs.values()}) == 1
    for arm, config in configs.items():
        assert config["seed"] == trial.SEED_OF[arm]
        if trial.ARCHITECTURE_OF[arm] == "fixed2":
            assert (config["atom_recycling"], config["recycle_policy"]) == ("carry", "fixed2")
            assert len(world.protocol["arms"][arm]["model_fields"]) == 36
        else:
            assert not set(config) & set(trial.ARCH_KEYS) and len(world.protocol["arms"][arm]["model_fields"]) == 34
    assert world.protocol["reused_fixed2"]["training"] == str(world.pair / "training/fixed2")
    assert "limits" in world.protocol and "cross-code" in world.protocol["limits"]


def edit_yaml(path, **changes):
    value = yaml.safe_load(path.read_text())
    path.write_text(yaml.safe_dump(dict(value, **changes), sort_keys=False))


@pytest.mark.parametrize("fault", ["seed", "seed_type", "arm", "architecture", "fields", "lr", "recycling", "n_steps",
                                   "budget", "panel", "reproduction", "reused_initial", "template", "flat_hash",
                                   "flat_file", "flat_stream", "flat_wsd", "reference10k"])
def test_protocol_rejects_changed_seeds_arms_recipe_or_reference(world, fault):
    value, run = world.protocol, world.run
    if fault == "seed":
        edit_yaml(run / "configs/fixed2_s20261025.yaml", seed=20261024)
    elif fault == "seed_type":
        value["arms"]["fixed2_s20261024"]["seed"] = 20261024.
    elif fault == "arm":
        value["arms"].pop("flat_fine6_s20261025")
    elif fault == "architecture":
        value["arms"]["fixed2_s20261024"]["architecture"] = "flat_fine6"
    elif fault == "fields":
        value["arms"]["fixed2_s20261024"]["model_fields"].pop("recycle_policy")
    elif fault == "lr":
        edit_yaml(run / "configs/flat_fine6_s20261024.yaml", muon_lr=.004)
    elif fault == "recycling":
        edit_yaml(run / "configs/flat_fine6_s20261024.yaml", atom_recycling="carry", recycle_policy="fixed2")
    elif fault == "n_steps":
        value["sampler"]["n_steps"] = 200
    elif fault == "budget":
        value["formal_applied_update_cap_per_arm"] = 5000
    elif fault == "panel":
        value["panel"] = str(world.old / "panel")
    elif fault == "reproduction":
        value["reproduction"]["expected_initial_model_sha256"] = "f" * 64
    elif fault == "reused_initial":
        write(world.pair / "training/fixed2/initialization.json", dict(target_sha256="f" * 64))
    elif fault == "template":
        edit_yaml(world.pair / "configs/fixed2.yaml", lr=2e-4)
    elif fault == "flat_hash":
        value["reference_flat_fine6"]["comparison_sha256"] = "f" * 64
    elif fault == "flat_file":
        write(world.old / "scores/flat_fine6.json", dict(tampered=True))
    elif fault == "flat_stream":
        write(world.old / "training/flat_fine6/audit/complete.json", dict(input_stream_sha256="f" * 64))
    elif fault == "flat_wsd":
        protocol = trial.read(world.old / "protocol.json")
        write(world.old / "protocol.json", dict(protocol, WSD=dict(WSD, stable_steps=4000)))
    else:
        value["reference10k"]["comparison_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        trial.validate_protocol(run, value)


def geometry(monkeypatch):
    monkeypatch.setattr(trial.tail, "geometry_diagnostics", lambda directory, targets, vectors, scorer, depth: dict(
        totals=dict(model_clashes=4, model_bad_bonds=2, model_bad_angles=1), unit="counts"))


def test_reused_points_are_admitted_only_with_their_sealed_identities(world, monkeypatch):
    geometry(monkeypatch)
    value = trial.admitted_old_flat(world.run, world.protocol, TARGETS)
    assert value["per_target"] == quality(.04) and value["geometry"]["totals"]["model_clashes"] == 4
    fixed = trial.reused_fixed2(context(world), world.reference)
    assert fixed["config"]["seed"] == 20261023 and fixed["hardware"] == HARDWARE


@pytest.mark.parametrize("fault", ["vector", "scorer", "receipt", "ema", "hardware", "verifier", "seed", "home",
                                   "targets", "panel", "costs"])
def test_old_flat_binding_detects_tampering(world, monkeypatch, fault):
    geometry(monkeypatch)
    path, value, targets = trial.flat_reference_path(world.run), dict(world.flat_value), TARGETS
    if fault == "vector":
        value["per_target"] = quality(.05)
    elif fault == "scorer":
        value["scorer"] = dict(value["scorer"], version="2.10")
    elif fault == "receipt":
        value["training_receipt"] = dict(value["training_receipt"], final_ema_sha256="0" * 64)
    elif fault == "ema":
        value["ema_sha256"] = "0" * 64
    elif fault == "hardware":
        value["hardware"] = dict(HARDWARE, uuid="gpu-9")
    elif fault == "verifier":
        value["verifier_sha256"] = "f" * 64
    elif fault == "seed":
        value["seed"] = 20261024
    elif fault == "home":
        value["training"] = str(world.pair / "training/fixed2")
    elif fault == "targets":
        targets = dict(TARGETS, d=dict(entry_id="e3"))
    elif fault == "panel":
        write(world.old / "panel/panel.json", dict(targets=[]))
    else:
        value["costs"] = dict(value["costs"], parameters=1)
    write(path, value)
    with pytest.raises(ValueError):
        trial.admitted_old_flat(world.run, world.protocol, targets)


@pytest.mark.parametrize("fault", ["depth", "seed", "policy", "stream", "initial", "recipe", "home"])
def test_reused_fixed2_binding_detects_tampering(world, fault):
    reference = json.loads(json.dumps(world.reference))
    row = reference["models"]["fixed2"]
    if fault == "depth":
        row["K"] = 4
    elif fault == "seed":
        row["config"]["seed"] = 20261024
    elif fault == "policy":
        row["config"]["recycle_policy"] = "sampled16_tail2"
    elif fault == "stream":
        row["training_receipt"]["complete"]["input_stream_sha256"] = "f" * 64
    elif fault == "initial":
        write(world.pair / "training/fixed2/initialization.json", dict(target_sha256="f" * 64))
    elif fault == "recipe":
        row["config"]["lr"] = 2e-4
    else:
        row["training"] = str(world.old / "training/flat_fine6")
    with pytest.raises(ValueError):
        trial.reused_fixed2(context(world), reference)


def synthetic_vectors():
    fixed = {trial.REUSED_FIXED2: 0., "fixed2_s20261024": .01, "fixed2_s20261025": .02}
    flat = {trial.OLD_FLAT: .05, "flat_fine6_s20261023": .03, "flat_fine6_s20261024": .04, "flat_fine6_s20261025": .07}
    return {name: quality(offset) for name, offset in (fixed | flat).items()}


def test_report_statistics_are_paired_by_seed_and_descriptive_only():
    vectors = synthetic_vectors()
    vectors["fixed2_s20261024"]["c"]["bb_lddt"] = vectors["flat_fine6_s20261024"]["c"]["bb_lddt"]
    value = trial.summarize(TARGETS, vectors)
    flat = value["architectures"]["flat_fine6"]["new_code_seeds"]["lddt"]["chain_mean"]
    assert flat["n"] == 3 and math.isclose(flat["mean"], .1 + (.03 + .04 + .07) / 3)
    assert math.isclose(flat["sd"], statistics.stdev([.13, .14, .17]))
    fixed = value["architectures"]["fixed2"]["new_and_reused_seeds"]["tm_score"]["chain_mean"]
    assert fixed["values"] == pytest.approx([.4, .41, .42]) and value["architectures"]["fixed2"]["new_code_seeds"][
        "lddt"]["chain_mean"]["n"] == 2
    assert value["architectures"]["flat_fine6"]["old_code_seed20261023"]["run"] == trial.OLD_FLAT
    pairs = value["paired_flat_fine6_minus_fixed2"]
    assert pairs["20261024"]["effects"]["tm_score"]["chain_mean_delta"] == pytest.approx(.03)
    assert pairs["20261025"]["effects"]["lddt"]["equal_entry_mean_delta"] == pytest.approx(.05)
    assert (pairs["20261024"]["effects"]["bb_lddt"]["wins"], pairs["20261024"]["effects"]["bb_lddt"]["ties"],
            pairs["20261024"]["effects"]["bb_lddt"]["losses"]) == (2, 1, 0)
    assert pairs["20261023_oldcode_flat"]["cross_code"] is True and pairs["20261024"]["cross_code"] is False
    assert pairs["20261023_oldcode_flat"]["effects"]["bb_lddt"]["chain_mean_delta"] == pytest.approx(.05)
    summary = value["paired_delta_summaries"]["seeds_20261024_20261025"]["tm_score"]["chain_mean"]
    assert summary["values"] == pytest.approx([.03, .05]) and summary["sd"] == pytest.approx(statistics.stdev([.03, .05]))
    noise = value["seed_noise"]
    pooled = math.sqrt((2 * statistics.variance([.13, .14, .17]) + 2 * statistics.variance([.1, .11, .12])) / 4)
    assert noise["pooled_within_architecture_sd"] == pytest.approx(pooled)
    assert noise["gap_over_pooled_sd"] == pytest.approx(noise["mean_paired_gap_seeds_20261024_20261025"] / pooled)
    reproduction = value["reproduction_new_minus_old"]["effects"]["lddt"]
    assert reproduction["chain_mean_delta"] == pytest.approx(-.02) and reproduction["losses"] == 3


@pytest.mark.parametrize("fault", ["missing", "coverage", "nonfinite"])
def test_report_statistics_require_every_finite_point(fault):
    vectors = synthetic_vectors()
    if fault == "missing":
        vectors.pop(trial.REUSED_FIXED2)
    elif fault == "coverage":
        vectors["fixed2_s20261025"].pop("c")
    else:
        vectors["flat_fine6_s20261024"]["a"]["lddt"] = math.nan
    with pytest.raises(ValueError):
        trial.summarize(TARGETS, vectors)


def preflight_value(world, *, identical=True):
    rows = {}
    for seed in trial.SEEDS:
        for index, arch in enumerate(trial.ARCHITECTURES):
            digest = INITIAL if seed == 20261023 else f"{seed % 10}" * 63 + ("0" if identical else str(index))
            rows.setdefault(str(seed), {})[arch] = dict(initial_model_sha256=digest, parameters=trial.PARAMETER_COUNT,
                trainable_parameters=trial.PARAMETER_COUNT, parameter_signature_sha256="9" * 64)
    pairs = {str(seed): dict(identical_parameter_sets=True, identical_initial_tensors=len({row["initial_model_sha256"]
             for row in rows[str(seed)].values()}) == 1, initial_model_sha256={arch: row["initial_model_sha256"]
             for arch, row in rows[str(seed)].items()}) for seed in trial.SEEDS}
    return dict(rows=rows, pairs=pairs, equivalence=dict(admitted=True))


@pytest.mark.parametrize("identical", [True, False])
def test_preflight_records_pairing_and_requires_the_old_initial_tensors(world, identical):
    pairs = trial.verify_preflight(preflight_value(world, identical=identical), world.protocol)
    assert pairs["20261023"]["identical_initial_tensors"] is True
    assert pairs["20261024"]["identical_initial_tensors"] is identical


@pytest.mark.parametrize("fault", ["old_initial", "signature", "parameters", "seed_blind", "pairs", "equivalence",
                                   "coverage"])
def test_preflight_rejects_non_reproducing_construction(world, fault):
    value = preflight_value(world)
    rows = value["rows"]
    if fault == "old_initial":
        rows["20261023"]["flat_fine6"]["initial_model_sha256"] = "f" * 64
    elif fault == "signature":
        rows["20261025"]["fixed2"]["parameter_signature_sha256"] = "8" * 64
    elif fault == "parameters":
        rows["20261024"]["flat_fine6"]["parameters"] = 271223962
    elif fault == "seed_blind":
        rows["20261025"]["fixed2"]["initial_model_sha256"] = rows["20261024"]["fixed2"]["initial_model_sha256"]
    elif fault == "pairs":
        value["pairs"]["20261024"]["identical_initial_tensors"] = False
    elif fault == "equivalence":
        value["equivalence"] = dict(admitted=False)
    else:
        rows.pop("20261025")
    with pytest.raises(ValueError):
        trial.verify_preflight(value, world.protocol)


def probe(code="old"):
    return dict(code=code, initial_model_sha256=INITIAL, perturbed_model_sha256="7" * 64, batch_sha256="8" * 64,
                parameters=trial.PARAMETER_COUNT, loss=1.5, gradient_norms=dict.fromkeys(trial.GROUPS, 2.))


def arrays(scale=1.):
    generator = np.random.default_rng(0)
    return dict(eval_velocity=generator.normal(size=(1, 4, 14, 3)) * scale,
                train_velocities=generator.normal(size=(2, 1, 4, 14, 3)), depths=np.array([[.5, 1.]]))


def test_old_and_current_flat_fine6_equivalence_accepts_bf16_level_agreement():
    new = arrays()
    new["eval_velocity"] = new["eval_velocity"] * (1 + 1e-4)
    value = trial.verify_equivalence(probe(), probe("new"), dict(old=arrays(), new=new), INITIAL)
    assert value["arrays"]["depths"]["bitwise_equal"] and not value["arrays"]["eval_velocity"]["bitwise_equal"]


@pytest.mark.parametrize("fault", ["velocity", "shape", "loss", "gradient", "zero_gradient", "initial", "batch",
                                   "same_code", "parameters", "trivial"])
def test_equivalence_rejects_any_semantic_difference(fault):
    old, new, outputs = probe(), probe("new"), dict(old=arrays(), new=arrays())
    if fault == "velocity":
        outputs["new"]["train_velocities"][1] += .1
    elif fault == "shape":
        outputs["new"]["depths"] = np.array([[1.]])
    elif fault == "loss":
        new["loss"] = 1.6
    elif fault == "gradient":
        new["gradient_norms"] = dict(new["gradient_norms"], **{"latent_residual.": 2.1})
    elif fault == "zero_gradient":
        old["gradient_norms"] = new["gradient_norms"] = dict(old["gradient_norms"], **{"atom_decoder.": 0.})
    elif fault == "initial":
        old["initial_model_sha256"] = new["initial_model_sha256"] = "f" * 64
    elif fault == "batch":
        new["batch_sha256"] = "9" * 64
    elif fault == "same_code":
        new["code"] = "old"
    elif fault == "parameters":
        old["parameters"] = new["parameters"] = 271223962
    else:
        outputs = dict(old=arrays(0.), new=arrays(0.))
    with pytest.raises(ValueError):
        trial.verify_equivalence(old, new, outputs, INITIAL)


def test_native_gate_checks_fresh_weights_and_the_old_input_prefix(world):
    gate = preflight_value(world)
    rows = [dict(input_sha256=f"{index}" * 64) for index in range(8)]
    native = dict(arm=trial.REPRODUCTION_ARM, seed=20261023, model_sha256=INITIAL, rows=rows)
    trial.check_native(native, gate, trial.REPRODUCTION_ARM, [row["input_sha256"] for row in rows])
    trial.check_native(dict(native, arm="fixed2_s20261024", seed=20261024,
                            model_sha256=gate["rows"]["20261024"]["fixed2"]["initial_model_sha256"]), gate,
                       "fixed2_s20261024")
    for value, old in ((dict(native, model_sha256="f" * 64), None), (native, ["f" * 64] * 8),
                       (dict(native, seed=20261024), None), (dict(native, arm="flat_fine6_s20261024"), None)):
        with pytest.raises(ValueError):
            trial.check_native(value, gate, trial.REPRODUCTION_ARM, old)


def test_bindings_restore_every_shared_registry_and_callable_after_failure():
    before = (trial.base.ARMS, trial.base.registry, trial.base.verify_actual_training, trial.base.admitted_training,
              trial.base.admitted_inference, trial.base.unchanged, trial.base.PARAMETERS, trial.proof.MODEL_FIELDS,
              trial.scratch.VARIANTS, trial.proof.VARIANTS)
    with pytest.raises(RuntimeError):
        with trial.bindings():
            assert set(trial.base.ARMS) == set(trial.ARMS) and set(trial.scratch.VARIANTS) == set(trial.ARMS)
            assert trial.base.expected("fixed2_s20261025")["seed"] == 20261025
            assert trial.base.expected("flat_fine6_s20261024").get("atom_recycling") is None
            raise RuntimeError("restore")
    after = (trial.base.ARMS, trial.base.registry, trial.base.verify_actual_training, trial.base.admitted_training,
             trial.base.admitted_inference, trial.base.unchanged, trial.base.PARAMETERS, trial.proof.MODEL_FIELDS,
             trial.scratch.VARIANTS, trial.proof.VARIANTS)
    assert all(first is second for first, second in zip(before, after, strict=True))


def test_reference_and_equivalence_scripts_only_read_admitted_results_or_probe_without_updates():
    calls = {node.func.attr for node in ast.walk(ast.parse(trial.OLD_FLAT_SCRIPT)) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)}
    assert {"validate_run", "admitted_inference", "admitted_training", "verify_canonical", "unchanged"} <= calls
    assert not calls & {"sample", "infer", "score", "report", "train"}
    calls = {node.func.attr for node in ast.walk(ast.parse(trial.EQUIVALENCE_SCRIPT)) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)}
    assert {"build_model", "backward", "allatom_loss_surface"} <= calls
    assert not calls & {"step", "load", "load_state_dict", "save"}


# ── value-based flat_fine6 construction ─────────────────────────────────

# Exactly what the trainer writes into every flat_fine6 config.json (training/flat_fine6_s20261023/config.json).
TRAINER_DEFAULTS = dict(atom_recycling="none", recycle_policy="legacy")


@pytest.mark.parametrize("config, native, admitted", [
    (TRAINER_DEFAULTS, {}, True), ({}, {}, True), (TRAINER_DEFAULTS, TRAINER_DEFAULTS, True),
    (dict(TRAINER_DEFAULTS, recycle_policy="fixed2"), {}, False),
    (dict(TRAINER_DEFAULTS, atom_recycling="carry"), {}, False),
    (dict(TRAINER_DEFAULTS, atom_recycling="reset"), {}, False),
    (dict(atom_recycling="carry", recycle_policy="fixed2"), {}, False),
    (TRAINER_DEFAULTS, dict(atom_recycling="carry"), False), (TRAINER_DEFAULTS, dict(recycle_policy="fixed2"), False),
    (dict(TRAINER_DEFAULTS, atom_recycling=None), {}, False)])
def test_flat_construction_is_checked_by_value_not_key_presence(config, native, admitted):
    if admitted:
        trial.verify_flat_construction(config, native)
    else:
        with pytest.raises(ValueError, match="non-recycling"):
            trial.verify_flat_construction(config, native)


@pytest.mark.parametrize("config_change, native_change, admitted", [
    ({}, {}, True), (None, {}, True), ({}, TRAINER_DEFAULTS, True),
    (dict(recycle_policy="fixed2", atom_recycling="carry"), {}, False), (dict(atom_recycling="carry"), {}, False),
    ({}, dict(recycle_policy="fixed2", atom_recycling="carry"), False)])
def test_actual_training_admits_trainer_written_flat_defaults(world, monkeypatch, config_change, native_change,
                                                              admitted):
    arm = "flat_fine6_s20261024"
    entry = world.protocol["arms"][arm]
    gate = preflight_value(world)
    config = dict(trial.expected(arm)) if config_change is None else trial.expected(arm) | TRAINER_DEFAULTS | config_change
    monkeypatch.setattr(trial, "_BASE_VERIFY", lambda context, name: (config, dict(input_stream_sha256=STREAM), [], {}))
    monkeypatch.setattr(trial, "admitted_preflight", lambda context: gate)
    write(Path(entry["training"]) / "initialization.json",
          dict(target_sha256=gate["rows"]["20261024"]["flat_fine6"]["initial_model_sha256"]))
    write(Path(entry["preflight"]) / "native_verified.json", dict(arm=arm, seed=20261024, **native_change))
    write(trial.preflight_path(world.run), dict(admitted=True))
    if admitted:
        _, _, _, evidence = trial.verify_actual_training(context(world), arm)
        assert evidence["preflight_verified.json"] == curve.sha(trial.preflight_path(world.run))
    else:
        with pytest.raises(ValueError, match="non-recycling"):
            trial.verify_actual_training(context(world), arm)


@pytest.mark.parametrize("row, admitted", [({}, True), (TRAINER_DEFAULTS, True), (dict(atom_recycling="carry"), False),
                                           (dict(recycle_policy="fixed2"), False)])
def test_flat_inference_rows_are_checked_by_value(tmp_path, monkeypatch, row, admitted):
    monkeypatch.setattr(trial.proof, "verify_inference", lambda *args, **kwargs: [dict(row, peak_reserved_gib=1.)])
    value = dict(run=tmp_path, targets=TARGETS)
    if admitted:
        trial.verify_inference(value, "flat_fine6_s20261024", "1" * 64)
    else:
        with pytest.raises(ValueError):
            trial.verify_inference(value, "flat_fine6_s20261024", "1" * 64)


# ── follow-up admission of a finished training run ──────────────────────

JOB = "7"


def finished(original):
    """Turn a prepared fixture run into one whose preflight ran and whose training array was submitted."""
    write(original / "cpu_admitted.json", dict(admitted=True, source_manifest_sha256=curve.sha(
        original / "source_manifest.json")))
    write(original / "preflight_verified.json", dict(admitted=True))
    write(original / "reference10k_admitted.json", dict(admitted=True))
    write(original / "jobs.json", dict(train_array=JOB))
    for arm in trial.ARMS:
        write(original / f"preflight/{arm}/native_verified.json", dict(arm=arm))
        write(original / f"preflight/{arm}/initialization.json", dict(arm=arm))
        (original / f"preflight/{arm}/initial_ema.pt").write_bytes(b"initial")
    write(original / "preflight/equivalence/old/probe.json", dict(code="old"))


def tree(path):
    return {str(item.relative_to(path)): curve.sha(item) for item in sorted(Path(path).rglob("*")) if item.is_file()}


@pytest.fixture
def followup(world):
    finished(world.run)
    before = tree(world.run)
    run = Path(trial.prepare_followup(world.tmp / "eval", world.run, stamp=STAMP)["run"])
    assert tree(world.run) == before
    return SimpleNamespace(run=run, protocol=trial.read(run / "protocol.json"), original=world.run)


def test_followup_prepare_is_torch_free_sealed_and_binds_the_training_run(world):
    finished(world.run)
    before = tree(world.run)
    code = ("import sys;sys.path[:0]=[sys.argv[1]];from benchmarks import compare_seed_replicates as t;"
            "t.prepare_followup(sys.argv[2],sys.argv[3],stamp='20990103T000000Z');assert 'torch' not in sys.modules")
    out = world.tmp / "eval"
    subprocess.run([sys.executable, "-c", code, str(trial.ROOT), str(out), str(world.run)], check=True)
    assert tree(world.run) == before
    run = out / "seed-replicate-eval-20990103T000000Z"
    manifest = json.loads((run / "source_manifest.json").read_text())
    scripts = ("cpu_admit.sh", "admit_training.sh", "infer.sh", "score.sh")
    for name in ("protocol.json", "verify_frozen.py", "write_cpu_admission.py", *scripts,
                 "code/benchmarks/compare_seed_replicates.py", "code/tests/test_seed_replicates.py"):
        assert manifest[name] == curve.sha(run / name)
    assert not (run / "train.sh").exists() and not (run / "preflight.sh").exists() and not (run / "configs").exists()
    for name, stage in (("cpu_admit.sh", "reference"), ("admit_training.sh", "admit-training"), ("infer.sh", "infer"),
                        ("score.sh", "score"), ("score.sh", "report")):
        text = (run / name).read_text()
        assert f'compare_seed_replicates.py" {stage} --run "$RUN"' in text and "verify_frozen.py" in text
    assert 'CUDA_VISIBLE_DEVICES=""' in (run / "admit_training.sh").read_text()
    protocol, source = trial.read(run / "protocol.json"), trial.read(world.run / "protocol.json")
    assert trial.shared_protocol(protocol) == trial.shared_protocol(source) and protocol["arms"] == source["arms"]
    bound = protocol["training_run"]
    assert bound["run"] == str(world.run) and bound["training_job"] == JOB and set(bound["configs"]) == set(trial.ARMS)
    assert bound["protocol_sha256"] == curve.sha(world.run / "protocol.json")
    assert not any(name.endswith(".pt") for name in bound["preflight"]) and len(bound["preflight"]) == 11
    assert trial.validate_protocol(run, protocol) == trial.validate_protocol(world.run, source)
    subprocess.run([sys.executable, str(run / "verify_frozen.py")], check=True)
    for path in (world.run / "preflight/flat_fine6_s20261024/native_verified.json", world.run / "jobs.json"):
        saved = path.read_bytes()
        path.write_text("{}")
        assert subprocess.run([sys.executable, str(run / "verify_frozen.py")], capture_output=True).returncode
        path.write_bytes(saved)
    assert not subprocess.run([sys.executable, str(run / "verify_frozen.py")], capture_output=True).returncode
    with pytest.raises(ValueError):
        trial.prepare_followup(world.tmp / "chain", run, stamp=STAMP)


@pytest.mark.parametrize("fault", ["config", "preflight_added", "preflight_changed", "jobs", "cpu", "role", "shared",
                                   "path", "binding"])
def test_followup_protocol_rejects_a_changed_training_run(followup, fault):
    value, original = json.loads(json.dumps(followup.protocol)), followup.original
    if fault == "config":
        edit_yaml(original / "configs/fixed2_s20261025.yaml", lr=2e-4)
    elif fault == "preflight_added":
        write(original / "preflight/fixed2_s20261024/extra.json", {})
    elif fault == "preflight_changed":
        write(original / "preflight/equivalence/old/probe.json", dict(code="new"))
    elif fault == "jobs":
        write(original / "jobs.json", dict(train_array="8"))
    elif fault == "cpu":
        write(original / "cpu_admitted.json", dict(admitted=False))
    elif fault == "role":
        value["followup_role"] = "retraining"
    elif fault == "shared":
        value["limits"] = "changed"
    elif fault == "path":
        value["arms"]["flat_fine6_s20261024"]["training"] = str(followup.run / "training/flat_fine6_s20261024")
    else:
        value["training_run"].pop("preflight")
    with pytest.raises(ValueError):
        trial.validate_protocol(followup.run, value)


def traceback_log(original, *, frame=296, last=trial.FLAT_FAILURE, done=True):
    code = original / "code/benchmarks"
    return "\n".join([*(["Done. Total steps: 10000"] if done else []), "[loader] train", "Traceback (most recent call last):",
        f'  File "{code / "compare_seed_replicates.py"}", line 1254, in main', "    train(context, args.arm)",
        f'  File "{code / "compare_scratch_hierarchy_fine6.py"}", line {frame}, in train',
        "    config, complete, metrics, evidence = verify_actual_training(context, arm)",
        f'  File "{code / "compare_seed_replicates.py"}", line 924, in verify_actual_training', "    require(...)",
        f'  File "{code / "compare_seed_replicates.py"}", line 87, in require', "    raise ValueError(message)", last, ""])


def ended_context(tmp_path):
    original = tmp_path / "orig"
    (original / "logs").mkdir(parents=True)
    return original, dict(run=tmp_path / "new", protocol=dict(training_run=dict(training_job=JOB)),
                          training_context=dict(run=original))


@pytest.mark.parametrize("case", ["known", "fixed2", "running", "pending_range", "no_log", "other_error",
                                  "other_frame", "not_done", "two_tracebacks", "completed", "receipt_symlink"])
def test_training_must_have_ended_completed_or_failed_only_at_the_flat_check(tmp_path, case):
    original, value = ended_context(tmp_path)
    arm = "fixed2_s20261024" if case == "fixed2" else "flat_fine6_s20261023"
    task = f"{JOB}_{list(trial.ARMS).index(arm)}"
    text = traceback_log(original, frame=297 if case == "other_frame" else 296, done=case != "not_done",
                         last="ValueError: final checkpoint bytes differ" if case == "other_error" else trial.FLAT_FAILURE)
    if case == "two_tracebacks":
        text = "Traceback (most recent call last):\n" + text
    if case != "no_log":
        (original / f"logs/train-{task}.out").write_text(text)
    queued = {"running": {f"{JOB}_4": "RUNNING"}, "pending_range": {f"{JOB}_[3-4]": "PENDING"}}.get(case, {})
    if case == "completed":
        write(original / f"train_admitted/{arm}.json", dict(admitted=True, arm=arm))
    elif case == "receipt_symlink":
        (original / "train_admitted").mkdir()
        (original / f"train_admitted/{arm}.json").symlink_to(original / f"logs/train-{task}.out")
    if case in ("known", "completed"):
        job, receipt = trial.training_ended(value, arm, queued)
        assert job == dict(task=task, log=f"logs/train-{task}.out", log_sha256=curve.sha(original / job["log"]),
                           termination=trial.FAILED_AT_FLAT_CHECK if case == "known" else trial.COMPLETED_TRAINING)
        assert (receipt is None) == (case == "known")
    else:
        with pytest.raises(ValueError):
            trial.training_ended(value, arm, queued)


def test_squeue_rows_select_only_the_training_array_including_pending_ranges():
    rows = trial.parse_queue("7|7_4|RUNNING\n7|7_[2-3]|PENDING\n7|7_0|COMPLETING\n70|70_1|RUNNING\n9|9|RUNNING\n\n",
                             "7")
    assert rows == {"7_4": "RUNNING", "7_[2-3]": "PENDING"} and trial.parse_queue("70|70_1|RUNNING\n", "7") == {}


def unrecorded(**changes):
    return dict(dict(name=trial.base.TRAIN_GPU, uuid=None, slurm_job="7_0", source=trial.UNRECORDED_GPU), **changes)


@pytest.mark.parametrize("value, admitted", [(unrecorded(), True), (HARDWARE, True), (unrecorded(slurm_job=""), False),
                                             (unrecorded(source="guess"), False), (unrecorded(name="A5000"), False),
                                             (dict(HARDWARE, uuid=""), False)])
def test_training_gpu_may_be_unrecorded_only_with_slurm_provenance(value, admitted):
    if admitted:
        trial.validate_training_hardware(value)
    else:
        with pytest.raises(ValueError):
            trial.validate_training_hardware(value)
    assert trial.same_gpu(None, "gpu-0") is None and trial.same_gpu("gpu-0", "gpu-0") is True


def binding(run):
    return dict(source_manifest_sha256=curve.sha(run / "source_manifest.json"),
                protocol_sha256=curve.sha(run / "protocol.json"), panel_admission_sha256="9" * 64)


def admission_world(followup, arm):
    original = followup.original
    protocol = trial.read(original / "protocol.json")
    source = dict(run=original, protocol=protocol, targets=TARGETS, **binding(original))
    value = dict(run=followup.run, protocol=followup.protocol, targets=TARGETS, training_context=source,
                 **binding(followup.run))
    entry = protocol["arms"][arm]
    home, preflight = Path(entry["training"]), Path(entry["preflight"])
    config = TRAINER_DEFAULTS | yaml.safe_load(Path(entry["config"]).read_text()) | dict(requested_num_workers=3)
    write(home / "config.json", config)
    write(home / "audit/complete.json", dict(trial.inputs.BUDGET, input_stream_sha256=STREAM, steady_mean_step_ms=1280.,
                                             peak_allocated_gib=18.4, peak_reserved_gib=19.9))
    (home / "ckpt_0010000.pt").write_bytes(b"final")
    (home / "initial_ema.pt").write_bytes(b"initial")
    write(home / "ground_truth_verified.json", dict(arm=arm, optimizer_updates=10000, microbatches=40000,
        final_schedule_index=10000, checkpoint_loads_during_training=0, device=trial.base.TRAIN_GPU,
        final_checkpoint_sha256=curve.sha(home / "ckpt_0010000.pt"), final_ema_sha256="e" * 64))
    task = f"{JOB}_{list(trial.ARMS).index(arm)}"
    (original / f"logs/train-{task}.out").parent.mkdir(exist_ok=True)
    (original / f"logs/train-{task}.out").write_text(traceback_log(original))
    evidence = {trial.base.relative(original, path): curve.sha(path) for path in (
        home / "config.json", home / "audit/complete.json", home / "ground_truth_verified.json",
        preflight / "native_verified.json")}
    metrics = [dict(step=5000, perf_train_step_ms_max=1000.), dict(step=10000, perf_train_step_ms_max=1200.)]
    complete = dict(trial.read(home / "audit/complete.json"), probe_input_sha256=["1" * 64])
    return value, (config, complete, metrics, evidence), home


@pytest.mark.parametrize("fault", [None, "checkpoint", "config", "binding", "log", "budget"])
def test_followup_training_admission_round_trips_and_detects_tampering(followup, monkeypatch, fault):
    arm = "flat_fine6_s20261024"
    value, verified, home = admission_world(followup, arm)
    monkeypatch.setattr(trial, "verify_actual_training", lambda context, name: verified)
    with pytest.raises(ValueError, match="still has queued"):
        trial.training_admission(value, arm, {f"{JOB}_4": "RUNNING"})
    receipt = trial.training_admission(value, arm, {})
    assert receipt["hardware"] == dict(name=trial.base.TRAIN_GPU, uuid=None, slurm_job=f"{JOB}_1",
                                       source=trial.UNRECORDED_GPU) and receipt["original_training_admission_sha256"] is None
    assert receipt["training_job"]["termination"] == trial.FAILED_AT_FLAT_CHECK
    assert receipt["all_update_weighted_mean_step_ms"] == pytest.approx(1100.)
    assert receipt["training_binding"] == trial.base.context_binding(value["training_context"])
    assert receipt["evidence_sha256"]["training/flat_fine6_s20261024/ckpt_0010000.pt"] == curve.sha(home / "ckpt_0010000.pt")
    trial.write_new(trial.base.training_receipt_path(followup.run, arm), receipt)
    assert not (followup.original / "train_admitted").exists()
    if fault is None:
        assert trial.followup_receipt(value, arm) == receipt
        return
    if fault == "checkpoint":
        (home / "ckpt_0010000.pt").write_bytes(b"other")
    elif fault == "config":
        write(home / "config.json", dict(trial.read(home / "config.json"), recycle_policy="fixed2"))
    elif fault == "binding":
        value = dict(value, training_context=dict(value["training_context"], protocol_sha256="f" * 64))
    elif fault == "log":
        (followup.original / f"logs/train-{JOB}_1.out").write_text("edited")
    else:
        write(home / "audit/complete.json", dict(trial.read(home / "audit/complete.json"), skipped_updates=1))
    with pytest.raises(ValueError):
        trial.followup_receipt(value, arm)


def test_followup_training_admission_requires_matching_original_receipt(followup, monkeypatch):
    arm = "fixed2_s20261024"
    value, verified, home = admission_world(followup, arm)
    monkeypatch.setattr(trial, "verify_actual_training", lambda context, name: verified)
    config, complete, metrics, evidence = verified
    evidence = dict(evidence, **{trial.base.relative(followup.original, path): curve.sha(path) for path in (
        Path(followup.protocol["arms"][arm]["preflight"]) / "initial_ema.pt", home / "initial_ema.pt",
        home / "ckpt_0010000.pt")})
    original = dict(admitted=True, arm=arm, **trial.base.context_binding(value["training_context"]), hardware=HARDWARE,
                    parameters=trial.PARAMETER_COUNT, nominal_K=2, config=config, complete=complete,
                    all_update_weighted_mean_step_ms=1100., final_checkpoint_sha256=curve.sha(home / "ckpt_0010000.pt"),
                    final_ema_sha256="e" * 64, evidence_sha256=evidence)
    path = trial.base.training_receipt_path(followup.original, arm)
    write(path, original)
    receipt = trial.training_admission(value, arm, {})
    assert receipt["hardware"] == HARDWARE and receipt["original_training_admission_sha256"] == curve.sha(path)
    assert receipt["training_job"]["termination"] == trial.COMPLETED_TRAINING
    trial.write_new(trial.base.training_receipt_path(followup.run, arm), receipt)
    assert trial.followup_receipt(value, arm) == receipt
    write(path, dict(original, final_ema_sha256="0" * 64))
    with pytest.raises(ValueError, match="re-verified"):
        trial.training_admission(value, arm, {})
    with pytest.raises(ValueError, match="original training admission changed"):
        trial.followup_receipt(value, arm)
