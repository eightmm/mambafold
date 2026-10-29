"""Sealed fixed-checkpoint scratch hierarchy versus flat fine-six comparison."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from benchmarks import compare_full_refinement as audit  # noqa: E402
from benchmarks import compare_shared_trunk as screen  # noqa: E402
from benchmarks import input_equivalence as inputs  # noqa: E402
from benchmarks import report_scratch_architecture_screen as proof  # noqa: E402
from benchmarks import score_openstructure as canonical  # noqa: E402
from benchmarks import screen_scratch_architectures as scratch  # noqa: E402
from benchmarks.screen_architectures import ARCHITECTURE_KEYS, validate_anchor  # noqa: E402

ARMS = {
    "hierarchy": dict(shared_trunk_layers=6, latent_hierarchy=True),
    "flat_fine6": dict(shared_trunk_layers=6, n_latent_layers=6),
}
PARAMETERS = dict(hierarchy=271223962, flat_fine6=269123226)
ROLE = "fixed-final-checkpoint architecture comparison;no checkpoint or sampler tuning"
SAMPLER = dict(method="sde", n_steps=500, tau=0.01, seed=0, K=2)
TRAIN_GPU = "NVIDIA RTX 6000 Ada Generation"
INFER_GPUS = {TRAIN_GPU, "NVIDIA RTX A5000"}
QUALITY = ("lddt", "bb_lddt", "tm_score")
require, exact, read = proof.require, proof.exact, proof.read


@contextmanager
def registry():
    """Do not change the historical registry outside this adapter's calls."""
    old_scratch, old_proof = scratch.VARIANTS, proof.VARIANTS
    scratch.VARIANTS = {name: dict(value) for name, value in ARMS.items()}
    proof.VARIANTS = {name: dict(value) for name, value in ARMS.items()}
    try:
        yield
    finally:
        scratch.VARIANTS, proof.VARIANTS = old_scratch, old_proof


def sha(path):
    return screen.sha256(path)


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with path.open("x") as stream:
        stream.write(payload)


def local_path(run, name, *, regular=False):
    candidate = Path(name)
    require(not candidate.is_absolute() and ".." not in candidate.parts, "nonlocal evidence path")
    path = run / candidate
    require(path.resolve().is_relative_to(run.resolve()), "evidence path escapes run")
    if regular:
        require(path.is_file() and not path.is_symlink(), f"missing regular sealed file: {name}")
    return path


def relative(run, path):
    return str(Path(path).resolve().relative_to(run.resolve()))


def seals(run, values):
    require(isinstance(values, dict) and values, "missing evidence seals")
    for name, digest in values.items():
        require(proof.valid_sha(digest), "invalid evidence SHA256")
        require(sha(local_path(run, name, regular=True)) == digest, f"changed evidence: {name}")


def expected(arm):
    require(arm in ARMS, "undeclared comparison arm")
    with registry():
        return scratch.expected_config(arm)


def common_config(config):
    return {key: value for key, value in config.items()
            if key not in ARCHITECTURE_KEYS | proof.IGNORED_CONFIG and not key.startswith("wandb_")}


def validate_protocol(run, protocol):
    exact(protocol, dict(
        formal_applied_update_cap_per_arm=10000, microbatches_per_formal_run=40000,
        preflight_optimizer_updates=0, teacher_models=0, all_parameters_trainable=True,
        common_device_final_confirmation_required=True, benchmark_role=ROLE,
    ), "comparison budget, initialization or role differs")
    require(protocol.get("sampler") == SAMPLER, "fixed K2 sampler differs")
    exact(protocol["sampler"], SAMPLER, "typed sampler differs")
    require(set(protocol.get("arms", {})) == set(ARMS), "comparison requires exactly two arms")
    require(Path(protocol["panel"]).resolve() == run / "panel", "comparison panel is outside run/panel")
    require(isinstance(protocol.get("ost"), str) and Path(protocol["ost"]).is_absolute(),
            "canonical OST path must be absolute")
    configs = {}
    with registry():
        for arm, entry in protocol["arms"].items():
            exact(entry, dict(nominal_K=2, expected_parameters=PARAMETERS[arm]),
                  "nominal K or expected parameter count differs")
            require(set(entry.get("model_fields", {})) == set(proof.MODEL_FIELDS),
                    "all 34 typed model fields are required")
            exact(entry["model_fields"], {key: expected(arm)[key] for key in proof.MODEL_FIELDS},
                  "declared comparison architecture differs")
            for key, path in dict(config=run / f"configs/{arm}.yaml",
                                  preflight=run / f"preflight/{arm}",
                                  training=run / f"training/{arm}", u0=run / "u0").items():
                require(Path(entry[key]).resolve() == path, f"unexpected {arm} {key} path")
            config = yaml.safe_load(Path(entry["config"]).read_text())
            scratch.validate_config(config, arm)
            exact(config, expected(arm), "typed scratch configuration differs")
            configs[arm] = config
    require(common_config(configs["hierarchy"]) == common_config(configs["flat_fine6"]),
            "training configurations differ outside the declared architecture")
    return configs


def validate_panel(run, protocol):
    admission = read(run / "panel_admitted.json")
    panel_path = local_path(run, "panel/panel.json", regular=True)
    exact(admission, dict(admitted=True, protocol_sha256=sha(run / "protocol.json"),
                          panel_sha256=sha(panel_path)), "panel admission binding differs")
    seals(run, admission.get("evidence_sha256"))
    rows = read(panel_path).get("targets", [])
    names = [row.get("target") for row in rows]
    require(rows and len(names) == len(set(names)), "missing or duplicate panel targets")
    coverage = admission.get("coverage", {})
    exact(coverage, dict(eligible_after_homology=len(rows)), "panel admission coverage differs")
    require(type(coverage.get("candidates")) is int and coverage["candidates"] >= len(rows)
            and isinstance(coverage.get("by_dataset"), dict) and coverage["by_dataset"]
            and "excluded" in coverage, "missing candidate/exclusion/dataset coverage")
    excluded = coverage["excluded"]
    count = excluded if type(excluded) is int else len(excluded) if isinstance(excluded, (list, dict)) else -1
    require(count >= 0 and coverage["candidates"] == len(rows) + count,
            "candidate and exclusion coverage does not reconcile")
    targets = {}
    for row in rows:
        name, sequence = row.get("target"), row.get("sequence")
        require(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_-]+", name), "invalid target name")
        require(isinstance(row.get("entry_id"), str) and row["entry_id"], "missing panel entry_id")
        require(isinstance(sequence, str) and sequence and re.fullmatch(r"[A-Z]+", sequence),
                "invalid panel sequence")
        exact(row, dict(length=len(sequence), sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest()),
              "panel sequence identity differs")
        for kind, suffix, field in (("inputs", "npz", "packed_sha256"),
                                    ("references", "pdb", "reference_sha256")):
            path = local_path(run, f"panel/{kind}/{name}.{suffix}", regular=True)
            require(proof.valid_sha(row.get(field)) and sha(path) == row[field],
                    "panel packed input or reference changed")
        import numpy as np

        with np.load(run / f"panel/inputs/{name}.npz", allow_pickle=False) as packed:
            require("esm" in packed.files, "missing panel ESM conditioning")
            features = packed["esm"]
            require(features.shape == (len(sequence), 2560)
                    and np.issubdtype(features.dtype, np.floating) and np.isfinite(features).all(),
                    "invalid panel ESM conditioning")
        targets[name] = row
    return targets


def validate_run(run):
    run = Path(run).resolve()
    manifest = read(run / "source_manifest.json")
    require("source_manifest.json" not in manifest and "cpu_admitted.json" not in manifest,
            "source manifest cannot seal itself or its admission")
    require(all(local_path(run, name).resolve() not in {
        run / "source_manifest.json", run / "cpu_admitted.json"} for name in manifest),
        "source manifest contains a self-referential path alias")
    seals(run, manifest)
    modules = (Path(__file__), Path(scratch.__file__), Path(proof.__file__), Path(screen.__file__),
               Path(audit.__file__), Path(inputs.__file__), Path(canonical.__file__))
    for path in (*modules, run / "protocol.json"):
        require(manifest.get(relative(run, path)) == sha(path), "unsealed comparison dependency")
    exact(read(run / "cpu_admitted.json"), dict(admitted=True,
          source_manifest_sha256=sha(run / "source_manifest.json")), "CPU admission binding differs")
    protocol = read(run / "protocol.json")
    configs = validate_protocol(run, protocol)
    for entry in protocol["arms"].values():
        path = Path(entry["config"])
        require(manifest.get(relative(run, path)) == sha(path), "unsealed comparison configuration")
    targets = validate_panel(run, protocol)
    return dict(run=run, protocol=protocol, manifest=manifest, targets=targets, configs=configs,
                source_manifest_sha256=sha(run / "source_manifest.json"),
                protocol_sha256=sha(run / "protocol.json"),
                panel_admission_sha256=sha(run / "panel_admitted.json"))


def context_binding(context):
    return {key: context[key] for key in (
        "source_manifest_sha256", "protocol_sha256", "panel_admission_sha256")}


def unchanged(context):
    run = context["run"]
    require(sha(run / "source_manifest.json") == context["source_manifest_sha256"]
            and sha(run / "protocol.json") == context["protocol_sha256"]
            and sha(run / "panel_admitted.json") == context["panel_admission_sha256"],
            "admitted run changed during execution")
    seals(run, context["manifest"])
    validate_panel(run, context["protocol"])


def hardware(*, training=False):
    require(bool(os.environ.get("SLURM_JOB_ID")), "GPU execution requires a Slurm allocation")
    import torch

    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "GPU stage requires exactly one allocated CUDA device")
    require(torch.cuda.current_device() == 0, "unexpected current CUDA device")
    name = torch.cuda.get_device_name(0)
    require(name == TRAIN_GPU if training else name in INFER_GPUS, "GPU class differs")
    uuid = str(getattr(torch.cuda.get_device_properties(0), "uuid", ""))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    require(uuid not in ("", "None", "unavailable") and isinstance(visible, str) and visible,
            "physical GPU UUID and CUDA_VISIBLE_DEVICES are required")
    value = dict(name=name, uuid=uuid, visible_devices=visible, slurm_job=os.environ["SLURM_JOB_ID"],
                 pid=os.getpid(), torch=str(torch.__version__), cuda=torch.version.cuda)
    validate_hardware(value, training=training)
    return value


def validate_hardware(value, *, training=False):
    require(isinstance(value, dict), "missing physical GPU receipt")
    require(value.get("name") == TRAIN_GPU if training else value.get("name") in INFER_GPUS,
            "GPU receipt class differs")
    require(all(isinstance(value.get(key), str) and value[key]
                for key in ("uuid", "visible_devices", "slurm_job", "torch", "cuda"))
            and value["uuid"] not in ("None", "unavailable")
            and type(value.get("pid")) is int and value["pid"] > 0,
            "incomplete physical GPU receipt")


def no_outputs(paths):
    require(all(not Path(path).exists() and not Path(path).is_symlink() for path in paths),
            "refusing to overwrite existing comparison artifacts")


def training_receipt_path(run, arm):
    return run / f"train_admitted/{arm}.json"


def verify_actual_training(context, arm):
    with registry():
        config, complete, metrics, evidence = proof.verify_arm(
            context["run"], arm, context["protocol"], context["manifest"], context["targets"])
    initial = read(Path(context["protocol"]["arms"][arm]["training"]) / "audit/initial.json")
    exact(initial, dict(parameters=PARAMETERS[arm], trainable_parameters=PARAMETERS[arm]),
          "actual parameter count differs")
    return config, complete, metrics, {relative(context["run"], path): value
                                       for path, value in evidence.items()}


def train(context, arm):
    require(arm in ARMS, "train requires a declared arm")
    run, entry = context["run"], context["protocol"]["arms"][arm]
    preflight, home = Path(entry["preflight"]), Path(entry["training"])
    probe = Path(entry["u0"]) / f"{arm}-10000"
    destination = training_receipt_path(run, arm)
    no_outputs((preflight, home, probe, destination))
    device = hardware(training=True)
    import torch

    torch.set_num_threads(1)
    with registry():
        scratch.preflight(SimpleNamespace(arm=arm, config=Path(entry["config"]), out=preflight))
        exact(read(preflight / "initialization.json"), dict(
            parameters=PARAMETERS[arm], trainable_parameters=PARAMETERS[arm],
            prelude_layers=3, shared_trunk_layers=6, hierarchy=arm == "hierarchy",
        ), "native model layout differs before formal training")
        gc.collect()
        torch.cuda.empty_cache()
        require(hardware(training=True) == device, "preflight changed physical GPU")
        scratch.train(SimpleNamespace(arm=arm, config=Path(entry["config"]), out=home,
                                      native_receipt=preflight / "native_verified.json"))
        screen.probe(SimpleNamespace(checkpoint=home / "ckpt_0010000.pt", out=probe,
                                     audit=home / "audit", batches=16, expected_step=10000,
                                     validate_output=validate_anchor))
    config, complete, metrics, evidence = verify_actual_training(context, arm)
    require(hardware(training=True) == device, "training/probe changed physical GPU")
    gt = read(home / "ground_truth_verified.json")
    for path in (preflight / "initial_ema.pt", home / "initial_ema.pt", home / "ckpt_0010000.pt"):
        evidence[relative(run, path)] = sha(path)
    previous, weighted = 0, 0.0
    for row in metrics:
        require(proof.finite(row.get("perf_train_step_ms_max"), positive=True), "invalid step timer")
        weighted += row["perf_train_step_ms_max"] * (row["step"] - previous)
        previous = row["step"]
    require(previous == 10000, "incomplete timing windows")
    unchanged(context)
    write_new(destination, dict(admitted=True, arm=arm, **context_binding(context), hardware=device,
        parameters=PARAMETERS[arm], nominal_K=2, config=config, complete=complete,
        all_update_weighted_mean_step_ms=weighted / previous,
        final_checkpoint_sha256=gt["final_checkpoint_sha256"],
        final_ema_sha256=gt["final_ema_sha256"], evidence_sha256=evidence))


def admitted_training(context, arm):
    run = context["run"]
    value = read(training_receipt_path(run, arm))
    exact(value, dict(admitted=True, arm=arm, parameters=PARAMETERS[arm], nominal_K=2,
                      **context_binding(context)), "training admission identity differs")
    seals(run, value.get("evidence_sha256"))
    home = Path(context["protocol"]["arms"][arm]["training"])
    config, complete = read(home / "config.json"), read(home / "audit/complete.json")
    require(value.get("config") == config and all(value["complete"].get(k) == v
                                                 for k, v in complete.items()),
            "training admission differs from actual training")
    exact(complete, inputs.BUDGET, "incomplete admitted training budget")
    with registry():
        scratch.validate_config(config, arm)
        exact(config, expected(arm), "actual scratch configuration differs")
    gt = read(home / "ground_truth_verified.json")
    require(value.get("final_checkpoint_sha256") == gt["final_checkpoint_sha256"]
            and value.get("final_ema_sha256") == gt["final_ema_sha256"], "own EMA binding differs")
    validate_hardware(value.get("hardware"), training=True)
    for key in ("steady_mean_step_ms", "peak_allocated_gib", "peak_reserved_gib"):
        require(proof.finite(complete.get(key), positive=True), "invalid measured training cost")
    require(proof.finite(value.get("all_update_weighted_mean_step_ms"), positive=True),
            "invalid full-update training timer")
    return value


def evaluation(run, arm):
    return run / f"evaluation/final/{arm}-10000"


def verify_external_runtime(expected_runtime, current_runtime):
    require(set(expected_runtime) == set(current_runtime), "native runtime fields differ")
    # This aggregate exists only after the architecture has been constructed.
    external = {key: value for key, value in expected_runtime.items() if key != "outproj_norm"}
    exact(current_runtime, external, "external native runtime differs from training")


@contextmanager
def inference_model_runtime(expected_runtime):
    """Admit the actual constructed model before the sampler loads its own EMA."""
    from mambafold.model.bimamba3 import mamba3_runtime_info
    from mambafold.train import trainer

    verify_external_runtime(expected_runtime, mamba3_runtime_info())
    original = trainer.build_model
    constructions = 0

    def build(*args, **kwargs):
        nonlocal constructions
        model = original(*args, **kwargs)
        require(mamba3_runtime_info(model) == expected_runtime,
                "constructed model native runtime differs from its training model")
        constructions += 1
        return model

    trainer.build_model = build
    try:
        yield
        require(constructions == 1, "common inference must verify exactly one constructed model")
    finally:
        trainer.build_model = original


def infer(context):
    run = context["run"]
    no_outputs((run / "inference_admitted.json", *(evaluation(run, arm) for arm in ARMS)))
    device = hardware()
    import torch

    torch.set_num_threads(1)
    admitted = {arm: admitted_training(context, arm) for arm in ARMS}
    require(common_config(admitted["hierarchy"]["config"])
            == common_config(admitted["flat_fine6"]["config"]), "actual common configurations differ")
    streams, runtimes = {}, {}
    from mambafold.model.bimamba3 import mamba3_runtime_info

    runtime = mamba3_runtime_info()
    for arm in ARMS:
        verify_actual_training(context, arm)
        home = Path(context["protocol"]["arms"][arm]["training"])
        runtimes[arm] = read(home / "initialization.json")["mamba3_runtime"]
        verify_external_runtime(runtimes[arm], runtime)
        path = home / "audit/inputs.jsonl"
        streams[arm], digest = inputs.read_inputs(path, 40000)
        require(digest == admitted[arm]["complete"]["input_stream_sha256"], "input stream differs")
    require(streams["hierarchy"] == streams["flat_fine6"], "actual ordered 40000 inputs differ")
    bindings, evidence = {}, {}
    for arm in ARMS:
        home = Path(context["protocol"]["arms"][arm]["training"])
        directory = evaluation(run, arm)
        require(hardware() == device, "common inference changed physical GPU")
        with inference_model_runtime(runtimes[arm]):
            screen.infer(SimpleNamespace(checkpoint=home / "ckpt_0010000.pt", panel=run / "panel",
                                        out=directory, depths=[2]))
        require(hardware() == device, "common inference changed physical GPU")
        proof.verify_inference(directory, 10000, [2], admitted[arm]["final_checkpoint_sha256"],
                               context["targets"], arm == "hierarchy")
        binding = dict(hardware=device, checkpoint_sha256=admitted[arm]["final_checkpoint_sha256"],
                       ema_sha256=admitted[arm]["final_ema_sha256"], nominal_K=2)
        write_new(directory / "gpu_binding.json", binding)
        bindings[arm] = binding
        for path in (directory / "sampling.json", directory / "hardware.json",
                     directory / "gpu_binding.json", directory / "K2/pairs/manifest.json"):
            evidence[relative(run, path)] = sha(path)
        for target in context["targets"]:
            path = directory / f"K2/pairs/{target}_seed0_pred.pdb"
            evidence[relative(run, path)] = sha(path)
        gc.collect()
        torch.cuda.empty_cache()
    unchanged(context)
    write_new(run / "inference_admitted.json", dict(admitted=True, **context_binding(context),
        hardware=device, sampler=SAMPLER, coverage=len(context["targets"]), arms=bindings,
        training_receipts={arm: sha(training_receipt_path(run, arm)) for arm in ARMS},
        input_comparison=dict(method="byte_exact", microbatches_per_arm=40000), evidence_sha256=evidence))


def admitted_inference(context):
    run = context["run"]
    value = read(run / "inference_admitted.json")
    exact(value, dict(admitted=True, coverage=len(context["targets"]), sampler=SAMPLER,
                      **context_binding(context)), "inference admission binding differs")
    require(set(value.get("arms", {})) == set(value.get("training_receipts", {})) == set(ARMS),
            "incomplete common inference arm coverage")
    validate_hardware(value.get("hardware"))
    seals(run, value.get("evidence_sha256"))
    for arm in ARMS:
        require(sha(training_receipt_path(run, arm)) == value["training_receipts"][arm],
                "training receipt changed after common inference")
        training = admitted_training(context, arm)
        binding = value["arms"][arm]
        exact(binding, dict(hardware=value["hardware"], nominal_K=2,
                            checkpoint_sha256=training["final_checkpoint_sha256"],
                            ema_sha256=training["final_ema_sha256"]), "inference own EMA/GPU differs")
        require(read(evaluation(run, arm) / "gpu_binding.json") == binding, "per-arm GPU binding differs")
        proof.verify_inference(evaluation(run, arm), 10000, [2], binding["checkpoint_sha256"],
                               context["targets"], arm == "hierarchy")
    return value


def verify_canonical(summary, directory, context):
    proof.verify_summary(summary, context["targets"])
    exact(summary, dict(expected_targets=len(context["targets"]),
                        input_manifest_sha256=sha(directory / "K2/pairs/manifest.json")),
          "canonical scoring input manifest differs")
    scorer = summary.get("openstructure", {})
    require(scorer.get("compare_args") == list(canonical.COMPARE_ARGS)
            and scorer.get("executable") == context["protocol"]["ost"]
            and isinstance(scorer.get("version"), str) and scorer["version"], "canonical scorer differs")
    require(summary.get("failures") == [], "canonical scoring contains failures")
    return {row["target_id"].removesuffix("_seed0"): {key: row[key] for key in QUALITY}
            for row in summary["rows"]}


def score(context, arm=None):
    run = context["run"]
    admitted_inference(context)
    names = (arm,) if arm else tuple(ARMS)
    require(set(names) <= set(ARMS), "undeclared scoring arm")
    no_outputs([path for name in names for path in (run / f"scores/{name}.json",
        run / f"scores/{name}.failed.json", evaluation(run, name) / "K2/scores")])
    failed = []
    for name in names:
        directory = evaluation(run, name)
        completed = subprocess.run([
            sys.executable, str(Path(canonical.__file__)), "--in-dir", str(directory / "K2/pairs"),
            "--out-dir", str(directory / "K2/scores"), "--ost", context["protocol"]["ost"],
            "--expected", str(len(context["targets"]))], check=False)
        path = directory / "K2/scores/summary.json"
        summary = read(path) if path.is_file() else None
        if completed.returncode or summary is None or summary.get("evaluation_complete") is not True:
            write_new(run / f"scores/{name}.failed.json", dict(arm=name, admitted=False,
                returncode=completed.returncode, expected=len(context["targets"]), summary=summary))
            failed.append(name)
            continue
        vectors = verify_canonical(summary, directory, context)
        unchanged(context)
        write_new(run / f"scores/{name}.json", dict(admitted=True, arm=name,
            **context_binding(context), inference_admission_sha256=sha(run / "inference_admitted.json"),
            coverage=len(vectors), metrics=vectors, summary_sha256=sha(path),
            scorer=summary["openstructure"], ost_executable_sha256=sha(context["protocol"]["ost"])))
    require(not failed, "canonical scoring failed or was incomplete: " + ", ".join(failed))


def paired_metrics(targets, vectors):
    require(set(vectors) == set(ARMS), "paired metrics require both arms")
    names = set(targets)
    require(names and all(set(values) == names for values in vectors.values()),
            "paired quality coverage differs")
    for values in vectors.values():
        require(all(set(row) == set(QUALITY) and all(proof.finite(v) for v in row.values())
                    for row in values.values()), "nonfinite or incomplete quality metrics")
    entries = {}
    for name, target in targets.items():
        entries.setdefault(target["entry_id"], []).append(name)
    overall = {arm: dict(coverage=len(names), **{
        metric: statistics.fmean(values[name][metric] for name in sorted(names)) for metric in QUALITY
    }) for arm, values in vectors.items()}
    per_entry = {entry: dict(coverage=len(members), arms={arm: {
        metric: statistics.fmean(values[name][metric] for name in members) for metric in QUALITY
    } for arm, values in vectors.items()}) for entry, members in sorted(entries.items())}
    paired = {}
    for metric in QUALITY:
        deltas = {name: vectors["flat_fine6"][name][metric] - vectors["hierarchy"][name][metric]
                  for name in sorted(names)}
        entry_deltas = {entry: statistics.fmean(deltas[name] for name in members)
                        for entry, members in sorted(entries.items())}
        paired[metric] = dict(chain_mean_delta=statistics.fmean(deltas.values()),
                             equal_entry_mean_delta=statistics.fmean(entry_deltas.values()),
                             entry_deltas=entry_deltas, target_deltas=deltas)
    return dict(coverage=dict(targets=len(names), entries=len(entries)), overall=overall,
                per_entry=per_entry, flat_fine6_minus_hierarchy=paired,
                all_zero_all_atom=all(row["lddt"] == 0 for values in vectors.values()
                                     for row in values.values()))


def report(context):
    run = context["run"]
    no_outputs((run / "comparison.json", run / "comparison.csv"))
    inference = admitted_inference(context)
    vectors, costs, scorers = {}, {}, []
    for arm in ARMS:
        value = read(run / f"scores/{arm}.json")
        exact(value, dict(admitted=True, arm=arm, coverage=len(context["targets"]),
                          inference_admission_sha256=sha(run / "inference_admitted.json"),
                          **context_binding(context)), "score admission identity differs")
        directory = evaluation(run, arm)
        summary_path = directory / "K2/scores/summary.json"
        require(sha(summary_path) == value["summary_sha256"], "canonical score summary changed")
        vectors[arm] = verify_canonical(read(summary_path), directory, context)
        require(vectors[arm] == value["metrics"], "reported score vectors differ")
        require(sha(context["protocol"]["ost"]) == value["ost_executable_sha256"], "OST executable changed")
        require(value["scorer"] == read(summary_path)["openstructure"], "score receipt scorer differs")
        scorers.append(value["scorer"])
        receipt = admitted_training(context, arm)
        training = receipt["complete"]
        measured = read(directory / "sampling.json")
        costs[arm] = dict(parameters=PARAMETERS[arm],
            training=dict(steady_mean_step_ms=training["steady_mean_step_ms"],
                all_update_weighted_mean_step_ms=receipt["all_update_weighted_mean_step_ms"],
                peak_allocated_gib=training["peak_allocated_gib"],
                peak_reserved_gib=training["peak_reserved_gib"], timing_scope=training["timing_scope"]),
            inference=dict(coverage=len(measured), mean_seconds=statistics.fmean(row["seconds"] for row in measured),
                peak_allocated_gib=max(row["peak_allocated_gib"] for row in measured),
                seconds_by_target={row["target"]: row["seconds"] for row in measured}))
    require(scorers[0] == scorers[1], "arms used different canonical scorer identities")
    result = dict(benchmark_role=ROLE, nominal_K=2, sampler=SAMPLER, **context_binding(context),
        panel_coverage=read(run / "panel_admitted.json")["coverage"],
        **paired_metrics(context["targets"], vectors), costs=costs, inference_hardware=inference["hardware"],
        architecture_selected=False, parameter_or_FLOP_matched=False,
        limits="Single training seed, fixed 10000 applied updates; descriptive declared "
               "sequence-homology-excluded whole-target comparison. This does not exclude remote "
               "homology, coordinate overlap or ESMC pretraining exposure. CA-lDDT is canonical "
               "OST bb_lddt for peptides. Warm sampler times include instrumentation/final CPU "
               "copy and exclude loading/PDB/scoring; one observation per target. Parameter "
               "counts are close; FLOPs are not matched. No checkpoint or sampler tuning.")
    unchanged(context)
    write_new(run / "comparison.json", result)
    with (run / "comparison.csv").open("x", newline="") as stream:
        columns = ("arm", "coverage", "lddt", "bb_lddt", "tm_score", "parameters",
                   "steady_train_ms", "all_update_train_ms", "train_allocated_gib",
                   "train_reserved_gib", "inference_seconds", "inference_allocated_gib")
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for arm in ARMS:
            cost = costs[arm]
            writer.writerow(dict(arm=arm, **result["overall"][arm], parameters=PARAMETERS[arm],
                steady_train_ms=cost["training"]["steady_mean_step_ms"],
                all_update_train_ms=cost["training"]["all_update_weighted_mean_step_ms"],
                train_allocated_gib=cost["training"]["peak_allocated_gib"],
                train_reserved_gib=cost["training"]["peak_reserved_gib"],
                inference_seconds=cost["inference"]["mean_seconds"],
                inference_allocated_gib=cost["inference"]["peak_allocated_gib"]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "infer", "score", "report"))
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--arm", choices=tuple(ARMS))
    args = parser.parse_args()
    if (args.stage == "train" and args.arm is None
            or args.stage in ("infer", "report") and args.arm is not None):
        parser.error("train requires --arm; infer/report operate on both arms")
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("comparison execution requires a Slurm allocation, including CPU scoring/reporting")
    context = validate_run(args.run)
    if args.stage == "train":
        train(context, args.arm)
    elif args.stage == "infer":
        infer(context)
    elif args.stage == "score":
        score(context, args.arm)
    else:
        report(context)


if __name__ == "__main__":
    main()
