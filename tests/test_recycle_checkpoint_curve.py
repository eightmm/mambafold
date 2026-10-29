"""Own5000 checkpoints are evaluated once and joined to hash-bound reused10000 rows."""

import ast
import hashlib
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import evaluate_recycle_checkpoint_curve as curve

curve.bind()
TARGETS = dict(a=dict(entry_id="e1"), b=dict(entry_id="e1"), c=dict(entry_id="e2"))
HARDWARE = dict(name=curve.base.TRAIN_GPU, uuid="gpu-0", visible_devices="0", slurm_job="1", pid=1,
                torch="2.11", cuda="13.0")
WSD = dict(curve.WSD, adamw_peak=1e-4, adamw_floor=1e-6, muon_peak=3e-3, muon_floor=3e-5,
           backbone_lr_scale=1.0, actual_pre_step_rates_required=True)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def quality(offset):
    return {target: dict(lddt=.1 + offset + index / 100, bb_lddt=.2 + offset, tm_score=.3 + offset)
            for index, target in enumerate(TARGETS)}


@pytest.fixture
def reference(tmp_path):
    old, pair, ost = tmp_path / "full4", tmp_path / "pair", tmp_path / "ost"
    for name in curve.frozen_files(curve.ROOT):
        (old / "code" / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(curve.ROOT / name, old / "code" / name)
    ost.write_text("canonical ost")
    original = dict(benchmark_role="three-way", sampler=dict(curve.SAMPLER, K=2), inference_depths=[1, 2, 4, 8, 16],
        WSD=WSD, ost=str(ost), arms=dict(fixed4=dict(training=str(old / "training/fixed4"))),
        external_pair=dict(run=str(pair)))
    write(old / "protocol.json", original)
    write(old / "panel/panel.json", dict(targets=[]))
    write(old / "panel_admitted.json", dict(panel_sha256=curve.sha(old / "panel/panel.json")))
    write(old / "source_manifest.json", curve.seal(old))
    homes, models, rows, per_target, geometry = curve.training_homes(old, original), {}, [], {}, {}
    for index, arm in enumerate(curve.MODELS):
        home, depth = homes[arm], curve.DEPTHS[arm]
        for name in ("ckpt_0005000.pt", "ckpt_0010000.pt"):
            write(home / name, name)
        write(home / "initialization.json", dict(mamba3_runtime=dict(version="native")))
        receipt = dict(arm=arm, config=dict(seed=1, recycle_policy=arm), final_checkpoint_sha256=f"{index}" * 64,
                       final_ema_sha256=f"{index + 5}" * 64, complete={}, evidence_sha256={})
        receipt_path = home.parents[1] / f"train_admitted/{arm}.json"
        write(receipt_path, receipt)
        values = quality(index / 10)
        row = dict(arm=arm, K=depth, coverage=3, **{metric: statistics.fmean(value[metric] for value in values.values())
            for metric in curve.QUALITY}, inference_seconds=40., inference_allocated_gib=1., inference_reserved_gib=2.)
        totals = dict(model_clashes=3 + index, model_bad_bonds=2, model_bad_angles=1)
        rows.append(row)
        per_target[arm], geometry[arm] = {str(depth): values}, {str(depth): dict(totals=totals)}
        models[arm] = dict(K=depth, training=str(home), row=row, per_target=values, geometry=geometry[arm][str(depth)],
            training_receipt=receipt, training_receipt_sha256=curve.sha(receipt_path),
            checkpoint_sha256=receipt["final_checkpoint_sha256"], ema_sha256=receipt["final_ema_sha256"],
            config=receipt["config"], runtime=dict(version="native"))
    binding = {key: curve.sha(old / name) for key, name in curve.REFERENCE_FILES.items()
               if key in ("source_manifest_sha256", "protocol_sha256", "panel_admission_sha256")}
    write(old / "comparison.json", dict(benchmark_role="three-way", architecture_selected=False, **binding, rows=rows,
                                        per_target=per_target, geometry=geometry, inference_hardware=HARDWARE))
    write(old / "inference_admitted.json", dict(admitted=True))
    write(old / "external_training_admitted.json", dict(admitted=True))
    run = Path(curve.prepare(tmp_path / "out", old, stamp="20990101T000000Z")["run"])
    protocol = curve.read(run / "protocol.json")
    bound = {key: value for key, value in protocol["reference10k"].items() if key not in ("run", "code")}
    write(curve.reference_path(run), dict(admitted=True, **bound, scorer=dict(version="2.9.1"), hardware=HARDWARE,
        verifier_sha256=curve.sha(old / "code/benchmarks/compare_scratch_recycle_full4.py"),
        ost_executable_sha256=curve.sha(ost), models=models))
    return SimpleNamespace(tmp=tmp_path, old=old, run=run, protocol=protocol, original=original, homes=homes)


def test_prepare_is_torch_free_sealed_and_matches_the_run_layout(reference):
    code = ("import sys;sys.path[:0]=[sys.argv[1]];from benchmarks import evaluate_recycle_checkpoint_curve as c;"
            "c.prepare(sys.argv[2],sys.argv[3],stamp='20990102T000000Z');assert 'torch' not in sys.modules")
    out = reference.tmp / "login"
    subprocess.run([sys.executable, "-c", code, str(curve.ROOT), str(out), str(reference.old)], check=True)
    run = out / "recycle-checkpoint-curve-20990102T000000Z"
    manifest = json.loads((run / "source_manifest.json").read_text())
    for name in ("protocol.json", "verify_frozen.py", "write_cpu_admission.py", "cpu_admit.sh", "infer.sh", "score.sh",
                 "code/benchmarks/evaluate_recycle_checkpoint_curve.py", "code/tests/test_recycle_checkpoint_curve.py"):
        assert manifest[name] == curve.sha(run / name)
    assert not set(manifest) & set(curve.UNSEALED) and not any("__pycache__" in name for name in manifest)
    for name, stage in (("cpu_admit.sh", "reference"), ("infer.sh", "infer"), ("score.sh", "report")):
        text = (run / name).read_text()
        assert text.startswith('#!/usr/bin/env bash\nset -euo pipefail\ncd "$SLURM_SUBMIT_DIR"\n')
        assert '"$SLURM_SUBMIT_DIR/.venv/bin/python"' in text and 'PYTHONPATH="$CODE:$CODE/src"' in text
        assert f'evaluate_recycle_checkpoint_curve.py" {stage} --run "$RUN"' in text
    assert 'evaluate_recycle_checkpoint_curve.py" score --run "$RUN"' in (run / "score.sh").read_text()
    assert all(test in (run / "cpu_admit.sh").read_text() for test in curve.TESTS)
    protocol = json.loads((run / "protocol.json").read_text())
    assert protocol["expected_new_predictions"] == 45 and protocol["checkpoint_steps"] == [5000, 10000]
    assert {Path(entry["checkpoint"]).name for entry in protocol["models"].values()} == {"ckpt_0005000.pt"}
    subprocess.run([sys.executable, str(run / "verify_frozen.py")], check=True)
    with pytest.raises(FileExistsError):
        curve.prepare(out, reference.old, stamp="20990102T000000Z")
    write(reference.old / "comparison.json", dict(tampered=True))
    assert subprocess.run([sys.executable, str(run / "verify_frozen.py")], capture_output=True).returncode


def test_prepared_protocol_binds_the_10k_reference(reference):
    assert curve.validate_protocol(reference.run, reference.protocol) == reference.original
    assert reference.protocol["models"]["fixed4"]["training"] == str(reference.old / "training/fixed4")
    assert reference.protocol["models"]["fixed2"]["training"] == str(reference.tmp / "pair/training/fixed2")


@pytest.mark.parametrize("fault", ["steps", "float_depth", "depth", "n_steps", "seed_type", "checkpoint", "training",
                                   "panel", "ost", "wsd", "reference_hash", "reference_file", "reference_wsd"])
def test_protocol_rejects_changed_checkpoints_sampler_or_reference(reference, fault):
    value = reference.protocol
    if fault == "steps":
        value["checkpoint_steps"] = [5000, 8000]
    elif fault == "float_depth":
        value["inference_depths"]["fixed4"] = 4.
    elif fault == "depth":
        value["inference_depths"]["fixed4"] = 2
    elif fault == "n_steps":
        value["sampler"]["n_steps"] = 200
    elif fault == "seed_type":
        value["sampler"]["seed"] = False
    elif fault == "checkpoint":
        value["models"]["fixed2"]["checkpoint"] = value["models"]["fixed2"]["checkpoint"].replace("5000", "10000")
    elif fault == "training":
        value["arms"] = dict(fixed2={})
    elif fault == "panel":
        value["panel"] = str(reference.tmp / "another-panel")
    elif fault == "ost":
        value["ost"] = "/another-ost"
    elif fault == "wsd":
        value["WSD"] = dict(value["WSD"], stable_steps=4000)
    elif fault == "reference_hash":
        value["reference10k"]["comparison_sha256"] = "f" * 64
    elif fault == "reference_file":
        (reference.old / "inference_admitted.json").write_text("{}")
    else:
        write(reference.old / "protocol.json", dict(reference.original, WSD=dict(WSD, cosine_decay_steps=0)))
        value["reference10k"]["protocol_sha256"] = curve.sha(reference.old / "protocol.json")
    with pytest.raises(ValueError):
        curve.validate_protocol(reference.run, value)


def test_reference_admission_binds_sealed_rows_and_training_receipts(reference):
    value = curve.admitted_reference(reference.run, reference.protocol, TARGETS)
    assert {arm: row["K"] for arm, row in value["models"].items()} == curve.DEPTHS


@pytest.mark.parametrize("fault", ["receipt", "vector", "row", "depth", "home", "hardware", "verifier", "targets"])
def test_reference_admission_detects_tampering(reference, fault):
    path, targets = curve.reference_path(reference.run), TARGETS
    value = json.loads(path.read_text())
    if fault == "receipt":
        write(reference.homes["fixed4"].parents[1] / "train_admitted/fixed4.json", dict(tampered=True))
    elif fault == "vector":
        value["models"]["fixed2"]["per_target"]["a"]["lddt"] += 1e-6
    elif fault == "row":
        value["models"]["sampled16_tail2"]["row"]["lddt"] += 1e-6
    elif fault == "depth":
        value["models"]["fixed4"]["K"] = 2
    elif fault == "home":
        value["models"]["fixed2"]["training"] = str(reference.homes["sampled16_tail2"])
    elif fault == "hardware":
        value["hardware"] = dict(HARDWARE, uuid="gpu-1")
    elif fault == "verifier":
        value["verifier_sha256"] = "f" * 64
    else:
        targets = dict(TARGETS, d=dict(entry_id="e3"))
    write(path, value)
    with pytest.raises(ValueError):
        curve.admitted_reference(reference.run, reference.protocol, targets)


def test_reference_checker_only_reads_admitted_results_without_sampling():
    tree = ast.parse(curve.REFERENCE10K_SCRIPT)
    calls = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)}
    assert {"validate_run", "admitted_inference", "all_training", "verify_canonical", "geometry_diagnostics"} <= calls
    assert not calls & {"sample", "infer", "score", "report", "train"}


def progress_home(tmp_path, fault=None):
    home = tmp_path / "pair/training/fixed2"
    digests = [hashlib.sha256(str(index).encode()).hexdigest() for index in range(40000)]
    (home / "audit").mkdir(parents=True)
    (home / "audit/inputs.jsonl").write_text("".join(json.dumps(dict(micro=index + 1, sha256=value)) + "\n"
                                                     for index, value in enumerate(digests)))
    prefix = hashlib.sha256(b"".join(bytes.fromhex(value) for value in digests[:20000])).hexdigest()
    metric = dict(step=5000, applied_updates=4999 if fault == "applied" else 5000, update_applied=1.0,
                  input_stream_sha256=digests[0] if fault == "prefix" else prefix)
    (home / "audit/metrics.jsonl").write_text(json.dumps(dict(step=4900, applied_updates=4900)) + "\n"
                                              + json.dumps(metric) + "\n")
    lrs = [dict(update=index + 1, schedule_index=index, groups=[dict(kind="muon", lr=3e-3), dict(kind="adamw", lr=1e-4)])
           for index in range(10000)]
    if fault == "lr":
        lrs[4999]["groups"][0]["lr"] = 1e-3
    elif fault == "index":
        lrs[4999]["schedule_index"] = 5000
    (home / "audit/used_lrs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in lrs))
    stream = hashlib.sha256(b"".join(bytes.fromhex(value) for value in digests)).hexdigest()
    seals = {f"training/fixed2/audit/{name}": curve.sha(home / f"audit/{name}")
             for name in ("metrics.jsonl", "used_lrs.jsonl", "inputs.jsonl")}
    if fault == "unsealed":
        seals["training/fixed2/audit/metrics.jsonl"] = "f" * 64
    return home, dict(complete=dict(microbatches=40000, updates=10000, input_stream_sha256=stream), evidence_sha256=seals)


def test_progress_proves_5000_stable_phase_updates_from_sealed_audits(tmp_path):
    home, receipt = progress_home(tmp_path)
    value = curve.verify_progress(home, receipt, WSD)
    assert value["applied_updates"] == 5000 and value["microbatches"] == 20000 and value["schedule_index"] == 4999
    assert value["used_lrs"] == dict(muon=3e-3, adamw=1e-4)


@pytest.mark.parametrize("fault", ["applied", "prefix", "lr", "index", "unsealed"])
def test_progress_rejects_wrong_count_inputs_schedule_or_unsealed_audits(tmp_path, fault):
    home, receipt = progress_home(tmp_path, fault)
    with pytest.raises(ValueError):
        curve.verify_progress(home, receipt, WSD)


@pytest.mark.parametrize("step,depth,ok", [(5000, 2, True), (10000, 2, False), (5000, 4, False)])
def test_inference_accepts_only_own5000_rows_at_the_declared_depth(tmp_path, step, depth, ok):
    directory = curve.evaluation(tmp_path, "fixed2")
    pairs = directory / f"K{depth}/pairs"
    pairs.mkdir(parents=True)
    (pairs / "a_seed0_pred.pdb").write_text("prediction")
    (pairs / "a_seed0_gt.pdb").write_text("reference")
    row = dict(target="a", K=depth, checkpoint_step=step, seconds=40., peak_allocated_gib=1., peak_reserved_gib=2.,
        prediction_sha256=curve.sha(pairs / "a_seed0_pred.pdb"), reference_sha256=curve.sha(pairs / "a_seed0_gt.pdb"),
        observed_calls=curve.screen.expected_call_counts(depth, atom_recycling="carry", network=501),
        atom_recycling="carry", recycle_policy="fixed2")
    write(directory / "sampling.json", [row])
    write(pairs / "manifest.json", dict(expected=1, checkpoint_sha256="a" * 64, weight_source="ema",
                                        sampler=dict(method="sde", n_steps=500, tau=.01, seed=0), rows=[row]))
    context = dict(run=tmp_path, targets=dict(a=dict(reference_sha256=row["reference_sha256"])))
    if ok:
        assert curve.verify_inference(context, "fixed2", "a" * 64) == [row]
    else:
        with pytest.raises(ValueError):
            curve.verify_inference(context, "fixed2", "a" * 64)


def test_paths_are_per_model_and_reject_undeclared_models(tmp_path):
    assert curve.evaluation(tmp_path, "fixed4") == tmp_path / "evaluation/step5000/fixed4"
    assert curve.score_path(tmp_path, "fixed4") == tmp_path / "scores/fixed4-step5000-K4.json"
    with pytest.raises(ValueError):
        curve.evaluation(tmp_path, "fixed8")


def test_curve_effects_report_paired_deltas_tallies_and_gap_changes():
    base = {target: dict(lddt=.1, bb_lddt=.2, tm_score=.3) for target in TARGETS}
    def shifted(**deltas):
        return {target: dict(row, lddt=row["lddt"] + deltas.get(target, 0.)) for target, row in base.items()}
    vectors = {"5000": dict(fixed2=shifted(), sampled16_tail2=shifted(a=-.2, b=-.2, c=-.2), fixed4=shifted(c=.4)),
               "10000": dict(fixed2=shifted(a=.3, c=-.1), sampled16_tail2=shifted(a=.1, b=.1, c=-.1), fixed4=shifted())}
    totals = {step: {arm: dict(model_clashes=10 * index + int(step == "10000"), model_bad_bonds=5, model_bad_angles=2)
                     for index, arm in enumerate(curve.MODELS)} for step in vectors}
    effects = curve.curve_effects(TARGETS, vectors, totals)
    fixed2 = effects["step10000_minus_5000"]["fixed2"]["lddt"]
    assert fixed2["chain_mean_delta"] == pytest.approx(.2 / 3)
    assert fixed2["equal_entry_mean_delta"] == pytest.approx((.15 - .1) / 2)
    assert (fixed2["wins"], fixed2["ties"], fixed2["losses"]) == (1, 1, 1)
    assert effects["step10000_minus_5000"]["fixed2"]["tm_score"]["ties"] == 3
    gaps = effects["between_model_gaps"]
    assert gaps["5000"]["fixed2_K2_minus_sampled16_tail2_K2"]["lddt"]["chain_mean_delta"] == pytest.approx(.2)
    assert gaps["10000"]["fixed2_K2_minus_sampled16_tail2_K2"]["lddt"]["chain_mean_delta"] == pytest.approx(.1 / 3)
    assert gaps["5000"]["fixed4_K4_minus_fixed2_K2"]["lddt"]["equal_entry_mean_delta"] == pytest.approx(.2)
    assert gaps["10000"]["fixed4_K4_minus_fixed2_K2"]["lddt"]["wins"] == 1
    change = effects["gap_change_10000_minus_5000"]
    assert change["fixed2_K2_minus_sampled16_tail2_K2"]["lddt"]["chain_mean_delta"] == pytest.approx(.1 / 3 - .2)
    assert change["fixed4_K4_minus_fixed2_K2"]["lddt"]["equal_entry_mean_delta"] == pytest.approx((.0 - .05) / 2 - .2)
    assert effects["geometry_totals_10000_minus_5000"]["fixed4"] == dict(model_clashes=1, model_bad_bonds=0,
                                                                          model_bad_angles=0)
