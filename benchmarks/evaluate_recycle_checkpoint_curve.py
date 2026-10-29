"""Evaluate own5000 EMA checkpoints of the three completed recycling models against reused10000 rows."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import shlex
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from benchmarks import input_equivalence as inputs  # noqa: E402  (torch-free; prepare runs on login)

MODELS = ("fixed2", "sampled16_tail2", "fixed4")
DEPTHS = dict(fixed2=2, sampled16_tail2=2, fixed4=4)
STEP, REFERENCE_STEP = 5000, 10000
SAMPLER = dict(method="sde", n_steps=500, tau=.01, seed=0)
REFERENCE_DEPTHS = [1, 2, 4, 8, 16]
WSD = dict(warmup_steps=500, stable_steps=7500, cosine_decay_steps=2000, scheduler_horizon=10000)
GAPS = {"fixed2_K2_minus_sampled16_tail2_K2": ("sampled16_tail2", "fixed2"),
        "fixed4_K4_minus_fixed2_K2": ("fixed2", "fixed4")}
GEOMETRY = ("model_clashes", "model_bad_bonds", "model_bad_angles")
ROLE = "recycling checkpoint curve;own5000 EMA versus reused10000 evaluation, no training or sampler tuning"
REFERENCE_FILES = dict(source_manifest_sha256="source_manifest.json", protocol_sha256="protocol.json",
    panel_admission_sha256="panel_admitted.json", comparison_sha256="comparison.json",
    inference_admission_sha256="inference_admitted.json",
    external_training_admission_sha256="external_training_admitted.json")
# Non-model drift from the 10k frozen code: an n_steps parameter with default500 and project prose.
ALLOWED_CODE_DRIFT = ("code/PROJECT.md", "code/benchmarks/compare_shared_trunk.py",
                      "code/benchmarks/report_scratch_architecture_screen.py")
FROZEN = dict(benchmarks=("*.py", "BENCHMARK_POLICY.md"), configs=("*.yaml",), pipeline=("**/*.py",),
              scripts=("*.py",), src=("**/*.py",), tests=("*.py",))
UNSEALED = ("source_manifest.json", "cpu_admitted.json", "reference10k_admitted.json")
TESTS = ("tests/test_recycle_checkpoint_curve.py", "tests/test_recycle_sampler_steps.py",
         "tests/test_completed_recycle_pair.py", "tests/test_scratch_recycle_full4.py",
         "tests/test_shared_trunk_audit.py", "tests/test_scratch_architecture_report.py",
         "tests/test_scratch_recycle_tail.py")
LIMITS = ("Single training and inference seed, one SDE500 sample per target on the eligible15-target/13-entry panel. "
    "The5000 checkpoint lies in the WSD stable phase (warmup500, stable7500, cosine2000) at peak learning rate, "
    "while10000 is post-cooldown, so5000->10000 mixes additional training length with learning-rate annealing. "
    "Two checkpoints cannot prove long-run recovery, a trend or its cause. The10000 rows are reused from the "
    "admitted three-way run, not re-inferred; the5000 rollouts may use another physical GPU of the same class. "
    "5000 checkpoint bytes were not hashed at training time; identity rests on its stored step/config/runtime/"
    "schedule index inside the admitted training home and sealed audits showing5000 applied updates. "
    "CA is canonical OST bb_lddt; geometry entries are unnormalised counts. No automatic model selection.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


sha = inputs.sha256


def bind():
    """Import the torch-dependent evaluator stack only for Slurm stages, never for prepare."""
    if "core" in globals():
        return
    from benchmarks import compare_scratch_recycle_full4 as module

    globals().update(core=module, base=module.base, proof=module.proof, screen=module.screen,
                     read=module.read, exact=module.exact, write_new=module.write_new, QUALITY=module.QUALITY)


def training_homes(reference_run, original):
    old, pair = Path(reference_run).resolve(), Path(original["external_pair"]["run"]).resolve()
    return {arm: (old if arm == "fixed4" else pair) / f"training/{arm}" for arm in MODELS}


def frozen_files(root):
    names = {Path(name) for name in ("PROJECT.md", "pyproject.toml", "uv.lock")}
    for directory, patterns in FROZEN.items():
        for pattern in patterns:
            names |= {path.relative_to(root) for path in Path(root).glob(f"{directory}/{pattern}")
                      if path.is_file() and "__pycache__" not in path.parts}
    return sorted(names)


def seal(run, names=None):
    names = names or sorted(str(path.relative_to(run)) for path in run.rglob("*")
                            if path.is_file() and "__pycache__" not in path.parts)
    return {name: sha(run / name) for name in names if name not in UNSEALED}


def evaluation(run, arm):
    require(arm in MODELS, "undeclared checkpoint-curve model")
    return run / f"evaluation/step{STEP}/{arm}"


def reference_path(run):
    return run / "reference10k_admitted.json"


def score_path(run, arm):
    return run / f"scores/{arm}-step{STEP}-K{DEPTHS[arm]}.json"


VERIFY_FROZEN = r'''"""Check static checkpoint-curve seals, the bound10k reference and identical model source."""
import hashlib,json
from pathlib import Path
RUN=Path(__file__).resolve().parent
FILES=__FILES__
ALLOWED=set(__ALLOWED__)

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
 return h.hexdigest()

def check_manifest(run):
 manifest=json.loads((run/'source_manifest.json').read_text())
 assert manifest and not set(manifest)&{'source_manifest.json','cpu_admitted.json','reference10k_admitted.json'}
 for name,digest in manifest.items():
  p=run/name
  assert not Path(name).is_absolute() and '..' not in Path(name).parts
  assert p.resolve().is_relative_to(run.resolve()) and p.is_file() and not p.is_symlink()
  assert p.suffix not in ('.pt','.pth','.safetensors'),name
  assert sha(p)==digest,name
 return manifest

def main():
 current=check_manifest(RUN)
 p=json.loads((RUN/'protocol.json').read_text());ref=p['reference10k'];old=Path(ref['run']).resolve()
 baseline=check_manifest(old)
 assert Path(ref['code']).resolve()==old/'code'
 for key,name in FILES.items():assert sha(old/name)==ref[key],key
 model={name for name in baseline if name.startswith('code/src/')}
 assert model and model=={name for name in current if name.startswith('code/src/')}
 assert all(current[name]==baseline[name] for name in model),'model source differs from10k'
 assert all(name in current and (current[name]==digest or name in ALLOWED) for name,digest in baseline.items() if name.startswith('code/')),'frozen code differs from10k'
 assert p['sampler']=={'method':'sde','n_steps':500,'tau':.01,'seed':0} and type(p['sampler']['n_steps']) is int and type(p['sampler']['seed']) is int
 assert p['inference_depths']=={'fixed2':2,'sampled16_tail2':2,'fixed4':4} and all(type(v) is int for v in p['inference_depths'].values())
 assert p['checkpoint_steps']==[5000,10000] and all(type(v) is int for v in p['checkpoint_steps'])
 assert Path(p['panel']).resolve()==old/'panel' and sha(old/'panel/panel.json')==p['panel_sha256']
 assert set(p['models'])=={'fixed2','sampled16_tail2','fixed4'}
 for arm,entry in p['models'].items():
  checkpoint=Path(entry['checkpoint'])
  assert checkpoint.name=='ckpt_0005000.pt' and checkpoint.parent==Path(entry['training']) and checkpoint.is_file() and not checkpoint.is_symlink(),arm
 print('Checkpoint-curve static seals, bound10k reference and identical model source verified',flush=True)

if __name__=='__main__':main()
'''

WRITE_CPU_ADMISSION = r'''"""Bind focused CPU checks and the reused admitted panel to the checkpoint-curve evaluator."""
import hashlib,json,os,sys
from pathlib import Path
import xml.etree.ElementTree as ET
assert os.environ.get('SLURM_JOB_ID'), 'Slurm allocation required'
run=Path(__file__).resolve().parent
code=run/'code';sys.path[:0]=[str(code),str(code/'src')]
from benchmarks import evaluate_recycle_checkpoint_curve as trial
assert Path(trial.__file__).resolve()==code/'benchmarks/evaluate_recycle_checkpoint_curve.py'
trial.bind()
protocol=json.loads((run/'protocol.json').read_text())
original=trial.validate_protocol(run,protocol)
targets=trial.validate_panel(protocol,original)
root=ET.parse(run/'checks.xml').getroot();suites=list(root) if root.tag=='testsuites' else [root]
counts={key:sum(int(suite.attrib.get(key,0)) for suite in suites) for key in ('tests','failures','errors','skipped')}
assert counts['tests']>0 and counts['failures']==counts['errors']==counts['skipped']==0,counts
value=dict(admitted=True,source_manifest_sha256=hashlib.sha256((run/'source_manifest.json').read_bytes()).hexdigest(),tests=counts,panel_targets=len(targets),panel_entries=len({row['entry_id'] for row in targets.values()}),panel_reuse='unchanged10k three-way sealed admission',checks_sha256=hashlib.sha256((run/'checks.xml').read_bytes()).hexdigest())
trial.write_new(run/'cpu_admitted.json',value)
print(json.dumps(value),flush=True)
'''


def scripts(run):
    common = f'''#!/usr/bin/env bash
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
RUN={shlex.quote(str(run))}
CODE="$RUN/code"
PY="$SLURM_SUBMIT_DIR/.venv/bin/python"
export PYTHONPATH="$CODE:$CODE/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TMPDIR="/tmp/mf-curve-${{SLURM_JOB_ID}}-${{SLURM_ARRAY_TASK_ID:-stage}}"
mkdir -p "$TMPDIR"
"$PY" "$RUN/verify_frozen.py"
'''
    stage = '"$PY" "$CODE/benchmarks/evaluate_recycle_checkpoint_curve.py" {} --run "$RUN"\n'
    return {"cpu_admit.sh": common + 'export CUDA_VISIBLE_DEVICES=""\ncd "$CODE"\n"$PY" -m pytest -q '
                + " ".join(TESTS) + ' --junitxml="$RUN/checks.xml"\n"$PY" "$RUN/write_cpu_admission.py"\n'
                + stage.format("reference"),
            "infer.sh": common + stage.format("infer"),
            "score.sh": common + 'export CUDA_VISIBLE_DEVICES=""\n' + stage.format("score") + stage.format("report")}


def prepare(out, reference, *, stamp=None):
    """Freeze a new run with file operations only; every model/torch check happens later under Slurm."""
    old = Path(reference).resolve()
    original = json.loads((old / "protocol.json").read_text())
    homes = training_homes(old, original)
    for home in homes.values():
        require(all((home / name).is_file() and not (home / name).is_symlink()
                    for name in ("ckpt_0005000.pt", "ckpt_0010000.pt")), "missing regular5000/10000 checkpoint")
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = Path(out).resolve() / f"recycle-checkpoint-curve-{stamp}"
    run.mkdir(parents=True, exist_ok=False)
    for name in frozen_files(ROOT):
        (run / "code" / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, run / "code" / name)
    protocol = dict(created_at=datetime.now(timezone.utc).isoformat(),
        authority="User authorized re-evaluating the existing own5000 EMA checkpoints of the three completed "
                  "recycling models with the unchanged10k SDE500 sampler; no training, sampler or checkpoint tuning.",
        benchmark_role=ROLE, sampler=dict(SAMPLER), checkpoint_steps=[STEP, REFERENCE_STEP],
        inference_depths=dict(DEPTHS), models={arm: dict(training=str(homes[arm]),
            checkpoint=str(homes[arm] / "ckpt_0005000.pt")) for arm in MODELS},
        reference10k=dict(run=str(old), code=str(old / "code"),
                          **{key: sha(old / name) for key, name in REFERENCE_FILES.items()}),
        panel=str(old / "panel"), panel_sha256=json.loads((old / "panel_admitted.json").read_text())["panel_sha256"],
        ost=original["ost"], WSD=original["WSD"], expected_targets=15, expected_entries=13,
        expected_new_predictions=15 * len(MODELS), maximum_concurrent_new_GPUs=1, limits=LIMITS)
    (run / "protocol.json").write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
    (run / "verify_frozen.py").write_text(VERIFY_FROZEN.replace("__FILES__", repr(REFERENCE_FILES))
                                          .replace("__ALLOWED__", repr(ALLOWED_CODE_DRIFT)))
    (run / "write_cpu_admission.py").write_text(WRITE_CPU_ADMISSION)
    for name, text in scripts(run).items():
        (run / name).write_text(text)
        (run / name).chmod(0o755)
    manifest = seal(run)
    with (run / "source_manifest.json").open("x") as stream:
        stream.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return dict(run=str(run), frozen_files=len(manifest), expected_new_predictions=protocol["expected_new_predictions"],
                source_manifest_sha256=sha(run / "source_manifest.json"), reference10k=str(old))


def validate_reference(protocol):
    reference = protocol.get("reference10k", {})
    require(isinstance(reference, dict) and set(reference) == {"run", "code", *REFERENCE_FILES},
            "10k reference binding fields differ")
    old, code = Path(reference["run"]), Path(reference["code"])
    require(old.is_absolute() and code.resolve() == old.resolve() / "code", "10k frozen code path differs")
    for key, name in REFERENCE_FILES.items():
        require(proof.valid_sha(reference[key]) and sha(old / name) == reference[key], f"10k reference {name} changed")
    original, manifest = read(old / "protocol.json"), read(old / "source_manifest.json")
    verifier = code / "benchmarks/compare_scratch_recycle_full4.py"
    require(manifest.get(str(verifier.resolve().relative_to(old.resolve()))) == sha(verifier),
            "10k three-way evaluator is not sealed")
    exact(original, dict(sampler=dict(SAMPLER, K=2), inference_depths=REFERENCE_DEPTHS), "10k sampler/depths differ")
    exact(original["sampler"], dict(SAMPLER, K=2), "typed10k sampler differs")
    exact(original.get("WSD", {}), WSD, "10k WSD schedule differs")
    require(set(original.get("arms", {})) == {"fixed4"}
            and Path(original["arms"]["fixed4"]["training"]).resolve() == old.resolve() / "training/fixed4",
            "10k fixed4 training home differs")
    exact(read(old / "comparison.json"), dict(benchmark_role=original.get("benchmark_role"), architecture_selected=False,
        **{key: reference[key] for key in ("source_manifest_sha256", "protocol_sha256", "panel_admission_sha256")}),
        "10k report context differs")
    return original


def validate_protocol(run, protocol):
    exact(protocol, dict(benchmark_role=ROLE, sampler=SAMPLER, checkpoint_steps=[STEP, REFERENCE_STEP],
        inference_depths=DEPTHS, expected_targets=15, expected_entries=13, expected_new_predictions=15 * len(MODELS)),
        "checkpoint-curve contract differs")
    exact(protocol["sampler"], SAMPLER, "typed sampler settings differ")
    exact(protocol["inference_depths"], DEPTHS, "typed per-model depths differ")
    require(all(type(value) is int for value in protocol["checkpoint_steps"]), "checkpoint steps must be integers")
    require(not any(key in protocol for key in ("arms", "training_policies", "training", "configs")),
            "checkpoint-curve evaluation cannot declare a training stage")
    original = validate_reference(protocol)
    old = Path(protocol["reference10k"]["run"]).resolve()
    require(Path(protocol.get("panel", "")).resolve() == old / "panel" and protocol.get("ost") == original.get("ost"),
            "10k panel/scorer must be reused read-only")
    require(protocol.get("panel_sha256") == read(old / "panel_admitted.json").get("panel_sha256")
            == sha(old / "panel/panel.json"), "10k panel hash binding differs")
    require(protocol.get("WSD") == original["WSD"], "declared WSD schedule differs from training")
    homes = training_homes(old, original)
    require(protocol.get("models") == {arm: dict(training=str(homes[arm]), checkpoint=str(homes[arm] / "ckpt_0005000.pt"))
                                       for arm in MODELS}, "own5000 checkpoint paths differ")
    return original


def validate_panel(protocol, original):
    targets = base.validate_panel(Path(protocol["reference10k"]["run"]).resolve(), original)
    require(len(targets) == protocol["expected_targets"] and len({row["entry_id"] for row in targets.values()})
            == protocol["expected_entries"], "checkpoint-curve panel must contain fifteen targets and thirteen entries")
    return targets


def validate_run(run):
    bind()
    run = Path(run).resolve()
    manifest = read(run / "source_manifest.json")
    require(not set(manifest) & set(UNSEALED), "manifest cannot seal itself or its admissions")
    base.seals(run, manifest)
    for path in (Path(__file__), Path(core.__file__), Path(core.atom.__file__), Path(base.__file__),
                 Path(base.scratch.__file__), Path(proof.__file__), Path(screen.__file__), Path(base.audit.__file__),
                 Path(inputs.__file__), Path(base.canonical.__file__), run / "protocol.json"):
        require(manifest.get(base.relative(run, path)) == sha(path), "unsealed checkpoint-curve dependency")
    exact(read(run / "cpu_admitted.json"), dict(admitted=True, source_manifest_sha256=sha(run / "source_manifest.json")),
          "CPU admission binding differs")
    protocol = read(run / "protocol.json")
    original = validate_protocol(run, protocol)
    baseline = read(Path(protocol["reference10k"]["run"]) / "source_manifest.json")
    model = {name for name in baseline if name.startswith("code/src/")}
    require(model and model == {name for name in manifest if name.startswith("code/src/")}
            and all(manifest[name] == baseline[name] for name in model), "model source differs from the10k evaluation")
    require(all(manifest.get(name) == digest or name in ALLOWED_CODE_DRIFT for name, digest in baseline.items()
                if name.startswith("code/")), "frozen code differs from the10k evaluation")
    return dict(run=run, protocol=protocol, reference_protocol=original, targets=validate_panel(protocol, original),
        manifest=manifest, source_manifest_sha256=sha(run / "source_manifest.json"),
        protocol_sha256=sha(run / "protocol.json"),
        panel_admission_sha256=protocol["reference10k"]["panel_admission_sha256"])


def unchanged(context):
    fresh = validate_run(context["run"])
    require(base.context_binding(fresh) == base.context_binding(context) and fresh["targets"] == context["targets"],
            "checkpoint-curve sources/protocol/panel changed")


REFERENCE10K_SCRIPT = r'''
import statistics, sys
from pathlib import Path
code, run, destination = (Path(value).resolve() for value in sys.argv[1:4])
sys.path[:0] = [str(code), str(code / "src")]
from benchmarks import compare_scratch_recycle_full4 as trial
assert Path(trial.__file__).resolve() == code / "benchmarks/compare_scratch_recycle_full4.py"
models = dict(fixed2=2, sampled16_tail2=2, fixed4=4)
context = trial.validate_run(run)
inference, training = trial.admitted_inference(context), trial.all_training(context)
binding = trial.base.context_binding(context)
comparison = trial.read(run / "comparison.json")
trial.exact(comparison, dict(benchmark_role=trial.ROLE, sampler=trial.SAMPLER, inference_depths=trial.DEPTHS,
    architecture_selected=False, **binding), "10k report context differs")
trial.require(comparison["inference_hardware"] == inference["hardware"], "10k report hardware differs")
selected, scorers = {}, []
for arm, depth in models.items():
    directory = trial.base.evaluation(run, arm)
    admission = trial.read(run / f"scores/{arm}-K{depth}.json")
    trial.exact(admission, dict(admitted=True, arm=arm, K=depth, coverage=len(context["targets"]),
        inference_admission_sha256=trial.sha(run / "inference_admitted.json"), **binding), "10k canonical admission differs")
    path = directory / f"K{depth}/scores/summary.json"
    trial.require(trial.sha(path) == admission["summary_sha256"]
        and trial.sha(context["protocol"]["ost"]) == admission["ost_executable_sha256"], "10k score evidence changed")
    summary = trial.read(path)
    values = trial.verify_canonical(summary, directory, context, depth)
    trial.require(values == admission["metrics"] and summary["openstructure"] == admission["scorer"],
        "10k canonical vector/scorer differs")
    geometry = trial.geometry_diagnostics(directory, context["targets"], values, admission["scorer"], depth)
    measured = trial.verify_inference(context, arm, inference["arms"][arm]["checkpoint_sha256"])
    timings = [row for row in measured if row["K"] == depth]
    rows = [row for row in comparison["rows"] if row["arm"] == arm and row["K"] == depth]
    trial.require(len(rows) == 1 and all(rows[0][metric] == statistics.fmean(value[metric] for value in values.values())
        for metric in trial.QUALITY) and rows[0]["inference_seconds"] == statistics.fmean(row["seconds"] for row in timings)
        and comparison["per_target"][arm][str(depth)] == values and comparison["geometry"][arm][str(depth)] == geometry,
        "10k report differs from admitted canonical measurements")
    home, receipt = trial.training_home(context, arm), training[arm]
    scorers.append(admission["scorer"])
    selected[arm] = dict(K=depth, training=str(home), row=rows[0], per_target=values, geometry=geometry,
        training_receipt=receipt, training_receipt_sha256=trial.receipt_sha(context, arm),
        checkpoint_sha256=inference["arms"][arm]["checkpoint_sha256"], ema_sha256=inference["arms"][arm]["ema_sha256"],
        config=receipt["config"], runtime=trial.read(home / "initialization.json")["mamba3_runtime"])
trial.require(all(scorer == scorers[0] for scorer in scorers), "10k scorers differ across models")
trial.base.unchanged(context)
trial.write_new(destination, dict(admitted=True, **binding, comparison_sha256=trial.sha(run / "comparison.json"),
    inference_admission_sha256=trial.sha(run / "inference_admitted.json"),
    external_training_admission_sha256=trial.sha(run / "external_training_admitted.json"),
    verifier_sha256=trial.sha(Path(trial.__file__)), ost_executable_sha256=trial.sha(context["protocol"]["ost"]),
    scorer=scorers[0], hardware=inference["hardware"], models=selected))
'''


def reference(context):
    """Admit the reused10k rows with the original frozen three-way evaluator in an isolated interpreter."""
    destination = reference_path(context["run"])
    base.no_outputs([destination])
    value = context["protocol"]["reference10k"]
    validate_reference(context["protocol"])
    subprocess.run([sys.executable, "-c", REFERENCE10K_SCRIPT, value["code"], value["run"], str(destination)],
                   cwd=value["code"], check=True)
    return admitted_reference(context["run"], context["protocol"], context["targets"])


def admitted_reference(run, protocol, targets):
    bound = protocol["reference10k"]
    original = validate_reference(protocol)
    old, code = Path(bound["run"]).resolve(), Path(bound["code"]).resolve()
    value = read(reference_path(run))
    exact(value, dict(admitted=True, **{key: bound[key] for key in REFERENCE_FILES},
        verifier_sha256=sha(code / "benchmarks/compare_scratch_recycle_full4.py"),
        ost_executable_sha256=sha(protocol["ost"])), "10k reference admission binding differs")
    require(set(value.get("models", {})) == set(MODELS), "10k reference model coverage differs")
    base.validate_hardware(value.get("hardware"), training=True)
    comparison, homes = read(old / "comparison.json"), training_homes(old, original)
    require(value["hardware"] == comparison["inference_hardware"], "10k reference hardware differs")
    for arm in MODELS:
        row, depth = value["models"][arm], DEPTHS[arm]
        exact(row, dict(K=depth, training=str(homes[arm])), "10k reference model/depth/home differs")
        path = homes[arm].parents[1] / f"train_admitted/{arm}.json"
        receipt = row.get("training_receipt", {})
        require(sha(path) == row.get("training_receipt_sha256") and read(path) == receipt, "10k training admission changed")
        require(row.get("checkpoint_sha256") == receipt["final_checkpoint_sha256"]
                and row.get("ema_sha256") == receipt["final_ema_sha256"] and row.get("config") == receipt["config"]
                and row.get("runtime") == read(homes[arm] / "initialization.json")["mamba3_runtime"],
                "10k checkpoint/config/runtime identity differs")
        require(set(row.get("per_target", {})) == set(targets) and comparison["per_target"][arm][str(depth)]
                == row["per_target"] and comparison["geometry"][arm][str(depth)] == row["geometry"]
                and row["row"] in comparison["rows"] and (row["row"]["arm"], row["row"]["K"]) == (arm, depth),
                "10k rows differ from the sealed three-way report")
    return value


def verify_progress(home, receipt, wsd):
    """Prove from sealed training audits that the5000 checkpoint follows exactly5000 applied stable-phase updates."""
    root, complete = Path(home).parents[1], receipt["complete"]
    paths = {name: Path(home) / f"audit/{name}" for name in ("metrics.jsonl", "used_lrs.jsonl", "inputs.jsonl")}
    seals = {base.relative(root, path): sha(path) for path in paths.values()}
    require(all(receipt["evidence_sha256"].get(name) == digest for name, digest in seals.items()),
            "training audit is not sealed by its admitted receipt")
    delivered, stream = inputs.read_inputs(paths["inputs.jsonl"], complete["microbatches"])
    require(stream == complete["input_stream_sha256"], "admitted training input stream differs")
    micro = complete["microbatches"] // complete["updates"] * STEP
    prefix = hashlib.sha256(b"".join(bytes.fromhex(value) for value in delivered[:micro])).hexdigest()
    rows = [row for row in proof.lines(paths["metrics.jsonl"]) if row.get("step") == STEP]
    require(len(rows) == 1, "missing step5000 training metrics")
    exact(rows[0], dict(applied_updates=STEP, update_applied=1.0, input_stream_sha256=prefix),
          "step5000 applied updates or delivered input prefix differ")
    lrs = proof.lines(paths["used_lrs.jsonl"])
    require([row.get("update") for row in lrs] == list(range(1, complete["updates"] + 1)),
            "used learning-rate audit coverage differs")
    exact(lrs[STEP - 1], dict(schedule_index=STEP - 1), "step5000 schedule index differs")
    rates = {group.get("kind"): group.get("lr") for group in lrs[STEP - 1].get("groups", [])}
    require(set(rates) == {"muon", "adamw"} and all(proof.finite(rate, positive=True) for rate in rates.values())
            and math.isclose(rates["muon"], wsd["muon_peak"], rel_tol=1e-9)
            and math.isclose(rates["adamw"], wsd["adamw_peak"], rel_tol=1e-9)
            and WSD["warmup_steps"] < STEP <= WSD["warmup_steps"] + WSD["stable_steps"],
            "update5000 is not at the WSD stable peak")
    return dict(applied_updates=STEP, microbatches=micro, input_prefix_sha256=prefix, schedule_index=STEP - 1,
                used_lrs=rates, sealed_audit_sha256=seals)


def verify_inference(context, arm, checkpoint_sha):
    measured = proof.verify_inference(evaluation(context["run"], arm), STEP, [DEPTHS[arm]], checkpoint_sha,
        context["targets"], False, atom_recycling="carry", n_steps=SAMPLER["n_steps"])
    for row in measured:
        exact(row, dict(recycle_policy=arm), "prediction policy differs")
        require(proof.finite(row.get("peak_reserved_gib"), positive=True), "reserved inference memory missing")
    return measured


def infer(context):
    import torch
    from mambafold.model.bimamba3 import mamba3_runtime_info

    run, protocol = context["run"], context["protocol"]
    base.no_outputs([run / "inference_admitted.json", *(evaluation(run, arm) for arm in MODELS)])
    device = base.hardware(training=True)
    torch.set_num_threads(1)
    old = admitted_reference(run, protocol, context["targets"])
    require(device["name"] == old["hardware"]["name"], "5000 GPU class differs from the10k evaluation")
    runtime = mamba3_runtime_info()
    evidence, models = {"reference10k_admitted.json": sha(reference_path(run))}, {}
    for arm in MODELS:
        row = old["models"][arm]
        checkpoint = Path(protocol["models"][arm]["checkpoint"])
        require(checkpoint.parent == Path(row["training"]) and checkpoint.is_file() and not checkpoint.is_symlink(),
                "5000 checkpoint is outside its admitted training home")
        base.verify_external_runtime(row["runtime"], runtime)
        progress = verify_progress(checkpoint.parent, row["training_receipt"], protocol["WSD"])
        checkpoint_sha = sha(checkpoint)
        ema = proof.checkpoint_identity(checkpoint, row["config"], STEP, runtime=row["runtime"])
        require(checkpoint_sha != row["checkpoint_sha256"] and ema != row["ema_sha256"],
                "5000 checkpoint duplicates the final10000 model")
        directory = evaluation(run, arm)
        require(base.hardware(training=True) == device, "common physical inference GPU changed")
        with base.inference_model_runtime(row["runtime"]):
            screen.infer(SimpleNamespace(checkpoint=checkpoint, panel=Path(protocol["panel"]), out=directory,
                                        depths=[DEPTHS[arm]], n_steps=SAMPLER["n_steps"]))
        require(base.hardware(training=True) == device, "common physical inference GPU changed")
        verify_inference(context, arm, checkpoint_sha)
        models[arm] = dict(hardware=device, step=STEP, K=DEPTHS[arm], checkpoint=str(checkpoint),
            checkpoint_sha256=checkpoint_sha, ema_sha256=ema, training_receipt_sha256=row["training_receipt_sha256"],
            progress=progress)
        write_new(directory / "gpu_binding.json", models[arm])
        paths = [*(directory / name for name in ("sampling.json", "hardware.json", "gpu_binding.json")),
                 directory / f"K{DEPTHS[arm]}/pairs/manifest.json"]
        paths.extend(directory / f"K{DEPTHS[arm]}/pairs/{target}_seed0_pred.pdb" for target in context["targets"])
        evidence.update({base.relative(run, path): sha(path) for path in paths})
        gc.collect()
        torch.cuda.empty_cache()
    require(len({row["progress"]["input_prefix_sha256"] for row in models.values()}) == 1,
            "first5000-update ordered training inputs differ between models")
    unchanged(context)
    admitted_reference(run, protocol, context["targets"])
    write_new(run / "inference_admitted.json", dict(admitted=True, **base.context_binding(context), hardware=device,
        sampler=SAMPLER, inference_depths=DEPTHS, checkpoint_step=STEP, coverage=len(context["targets"]), arms=models,
        reference10k_admission_sha256=sha(reference_path(run)), evidence_sha256=evidence))


def admitted_inference(context):
    run = context["run"]
    old = admitted_reference(run, context["protocol"], context["targets"])
    value = read(run / "inference_admitted.json")
    exact(value, dict(admitted=True, checkpoint_step=STEP, sampler=SAMPLER, inference_depths=DEPTHS,
        coverage=len(context["targets"]), reference10k_admission_sha256=sha(reference_path(run)),
        **base.context_binding(context)), "checkpoint-curve inference admission differs")
    require(set(value.get("arms", {})) == set(MODELS), "inference model coverage differs")
    base.validate_hardware(value.get("hardware"), training=True)
    require(value["hardware"]["name"] == old["hardware"]["name"], "5000 GPU class differs from the10k evaluation")
    base.seals(run, value.get("evidence_sha256"))
    for arm in MODELS:
        row, binding = old["models"][arm], value["arms"][arm]
        exact(binding, dict(hardware=value["hardware"], step=STEP, K=DEPTHS[arm],
            checkpoint=context["protocol"]["models"][arm]["checkpoint"],
            training_receipt_sha256=row["training_receipt_sha256"]), "own5000 EMA/GPU binding differs")
        require(proof.valid_sha(binding.get("checkpoint_sha256")) and proof.valid_sha(binding.get("ema_sha256"))
                and binding["checkpoint_sha256"] != row["checkpoint_sha256"] and binding["ema_sha256"] != row["ema_sha256"],
                "5000 checkpoint identity missing or equal to the final model")
        require(binding.get("progress") == verify_progress(Path(row["training"]), row["training_receipt"],
                context["protocol"]["WSD"]), "5000 training-progress evidence changed")
        require(read(evaluation(run, arm) / "gpu_binding.json") == binding, "per-model GPU binding changed")
        verify_inference(context, arm, binding["checkpoint_sha256"])
    return value, old


def score(context):
    run = context["run"]
    admitted_inference(context)
    base.no_outputs([path for arm in MODELS for path in (score_path(run, arm),
        score_path(run, arm).with_suffix(".failed.json"), evaluation(run, arm) / f"K{DEPTHS[arm]}/scores")])
    failed = []
    for arm in MODELS:
        directory, depth = evaluation(run, arm), DEPTHS[arm]
        home = directory / f"K{depth}"
        completed = subprocess.run([sys.executable, str(Path(base.canonical.__file__)), "--in-dir", str(home / "pairs"),
            "--out-dir", str(home / "scores"), "--ost", context["protocol"]["ost"],
            "--expected", str(len(context["targets"]))], check=False)
        path = home / "scores/summary.json"
        summary = read(path) if path.is_file() else None
        if completed.returncode or summary is None or summary.get("evaluation_complete") is not True:
            write_new(score_path(run, arm).with_suffix(".failed.json"), dict(admitted=False, arm=arm, K=depth,
                step=STEP, returncode=completed.returncode, expected=len(context["targets"]), summary=summary))
            failed.append(f"{arm}/K{depth}")
            continue
        vectors = core.verify_canonical(summary, directory, context, depth)
        unchanged(context)
        write_new(score_path(run, arm), dict(admitted=True, arm=arm, K=depth, step=STEP, **base.context_binding(context),
            inference_admission_sha256=sha(run / "inference_admitted.json"), coverage=len(vectors), metrics=vectors,
            summary_sha256=sha(path), scorer=summary["openstructure"], ost_executable_sha256=sha(context["protocol"]["ost"])))
    require(not failed, "canonical scoring failed or incomplete: " + ", ".join(failed))


def tally(effects):
    for value in effects.values():
        deltas = list(value["target_deltas"].values())
        value.update(wins=sum(delta > 0 for delta in deltas), ties=sum(delta == 0 for delta in deltas),
                     losses=sum(delta < 0 for delta in deltas))
    return effects


def curve_effects(targets, vectors, totals):
    """Per-model 10000-5000 effects and declared between-model gaps at each checkpoint; no selection."""
    early, late = str(STEP), str(REFERENCE_STEP)
    gaps = {step: {name: tally(core.paired_effects(targets, vectors[step][first], vectors[step][second]))
                   for name, (first, second) in GAPS.items()} for step in (early, late)}
    return dict(
        step10000_minus_5000={arm: tally(core.paired_effects(targets, vectors[early][arm], vectors[late][arm]))
                              for arm in MODELS},
        geometry_totals_10000_minus_5000={arm: {field: totals[late][arm][field] - totals[early][arm][field]
                                                for field in GEOMETRY} for arm in MODELS},
        between_model_gaps=gaps,
        gap_change_10000_minus_5000={name: {metric: {key: gaps[late][name][metric][key] - gaps[early][name][metric][key]
            for key in ("chain_mean_delta", "equal_entry_mean_delta")} for metric in QUALITY} for name in GAPS})


def report(context):
    run = context["run"]
    base.no_outputs([run / "comparison.json", run / "comparison.csv"])
    (inference, old), targets = admitted_inference(context), context["targets"]
    early, late = str(STEP), str(REFERENCE_STEP)
    rows, scorers = [], []
    vectors, geometry, totals = {early: {}, late: {}}, {early: {}, late: {}}, {early: {}, late: {}}
    for arm in MODELS:
        directory, depth = evaluation(run, arm), DEPTHS[arm]
        admission = read(score_path(run, arm))
        exact(admission, dict(admitted=True, arm=arm, K=depth, step=STEP, coverage=len(targets),
            inference_admission_sha256=sha(run / "inference_admitted.json"), **base.context_binding(context)),
            "checkpoint-curve score admission differs")
        path = directory / f"K{depth}/scores/summary.json"
        require(sha(path) == admission["summary_sha256"] and sha(context["protocol"]["ost"])
                == admission["ost_executable_sha256"], "canonical score evidence changed")
        summary = read(path)
        values = core.verify_canonical(summary, directory, context, depth)
        require(values == admission["metrics"] and summary["openstructure"] == admission["scorer"],
                "canonical vectors/scorer changed")
        scorers.append(admission["scorer"])
        reused = old["models"][arm]
        vectors[early][arm], vectors[late][arm] = values, reused["per_target"]
        geometry[early][arm] = core.geometry_diagnostics(directory, targets, values, admission["scorer"], depth)
        geometry[late][arm] = reused["geometry"]
        timings = [row for row in read(directory / "sampling.json") if row["K"] == depth]
        costs = {early: dict(inference_seconds=statistics.fmean(row["seconds"] for row in timings),
                             inference_allocated_gib=max(row["peak_allocated_gib"] for row in timings),
                             inference_reserved_gib=max(row["peak_reserved_gib"] for row in timings)),
                 late: {key: reused["row"][key] for key in ("inference_seconds", "inference_allocated_gib",
                                                            "inference_reserved_gib")}}
        for step in (early, late):
            totals[step][arm] = {field: geometry[step][arm]["totals"][field] for field in GEOMETRY}
            rows.append(dict(step=int(step), arm=arm, K=depth, coverage=len(vectors[step][arm]),
                **{metric: statistics.fmean(value[metric] for value in vectors[step][arm].values()) for metric in QUALITY},
                **totals[step][arm], **costs[step]))
        require(all(math.isclose(rows[-1][metric], reused["row"][metric], rel_tol=0, abs_tol=1e-12) for metric in QUALITY),
                "reused10k means differ from their per-target vectors")
    require(scorers and all(scorer == old["scorer"] for scorer in scorers),
            "canonical scorer differs between the5000 rollouts and the10k reference")
    hardware = {early: inference["hardware"], late: old["hardware"]}
    result = dict(benchmark_role=ROLE, **base.context_binding(context), sampler=SAMPLER,
        checkpoint_steps=[STEP, REFERENCE_STEP], inference_depths=DEPTHS,
        coverage=dict(targets=len(targets), entries=len({row["entry_id"] for row in targets.values()})),
        metric_names=dict(AA="lddt", CA="bb_lddt", TM="tm_score"), rows=rows, per_target=vectors, geometry=geometry,
        **curve_effects(targets, vectors, totals),
        checkpoints={arm: {early: {key: inference["arms"][arm][key] for key in ("checkpoint_sha256", "ema_sha256",
            "progress")}, late: {key: old["models"][arm][key] for key in ("checkpoint_sha256", "ema_sha256")}}
            for arm in MODELS},
        WSD=context["protocol"]["WSD"], per_step_hardware=hardware,
        same_physical_gpu=hardware[early]["uuid"] == hardware[late]["uuid"],
        reference10k_comparison_sha256=old["comparison_sha256"],
        reference10k_admission_sha256=sha(reference_path(run)), architecture_selected=False, limits=LIMITS)
    unchanged(context)
    write_new(run / "comparison.json", result)
    with (run / "comparison.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "reference", "infer", "score", "report"))
    parser.add_argument("--run", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--reference", type=Path, default=ROOT / "out/recycle-full4-scratch-wsd-20261026T031310Z")
    args = parser.parse_args()
    if args.stage == "prepare":
        if args.out is None or args.run is not None:
            parser.error("prepare requires --out and no --run")
        print(json.dumps(prepare(args.out, args.reference)))
        return
    if args.run is None or args.out is not None:
        parser.error(f"{args.stage} requires --run and no --out")
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("checkpoint-curve evaluation requires a Slurm allocation, including CPU stages")
    context = validate_run(args.run)
    {"reference": reference, "infer": infer, "score": score, "report": report}[args.stage](context)


if __name__ == "__main__":
    main()
