"""Seed replicates of scratch flat_fine6 versus fixed2, joined to hash-bound reused seed-20261023 points."""

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
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

# Both imports are torch-free; the ML stack is bound lazily so prepare can run on login.
from benchmarks import evaluate_recycle_checkpoint_curve as curve  # noqa: E402
from benchmarks import input_equivalence as inputs  # noqa: E402

SEEDS = (20261023, 20261024, 20261025)
REPRODUCTION_SEED = SEEDS[0]
ARCHITECTURES = dict(flat_fine6=dict(shared_trunk_layers=6, n_latent_layers=6),
                     fixed2=dict(shared_trunk_layers=6, n_latent_layers=6, atom_recycling="carry",
                                 recycle_policy="fixed2"))
ARCH_KEYS = ("atom_recycling", "recycle_policy")
# The trainer writes these non-recycling defaults into every config.json; absence and default are equivalent.
FLAT_DEFAULTS = dict(atom_recycling="none", recycle_policy="legacy")
NEW_SEEDS = dict(flat_fine6=SEEDS, fixed2=SEEDS[1:])


def arm_name(architecture, seed):
    return f"{architecture}_s{seed}"


ARMS = {arm_name(arch, seed): dict(ARCHITECTURES[arch], seed=seed) for arch, seeds in NEW_SEEDS.items()
        for seed in seeds}
ARCHITECTURE_OF = {arm_name(arch, seed): arch for arch, seeds in NEW_SEEDS.items() for seed in seeds}
SEED_OF = {arm: value["seed"] for arm, value in ARMS.items()}
REPRODUCTION_ARM = arm_name("flat_fine6", REPRODUCTION_SEED)
OLD_FLAT, REUSED_FIXED2 = "flat_fine6_s20261023_oldcode", "fixed2_s20261023_reused"
POINTS = (*ARMS, OLD_FLAT, REUSED_FIXED2)
PARAMETER_COUNT = 269123226
PARAMETERS = dict.fromkeys(ARMS, PARAMETER_COUNT)
DEPTH, STEP = 2, 10000
SAMPLER = dict(method="sde", n_steps=500, tau=0.01, seed=0, K=2)
QUALITY = ("lddt", "bb_lddt", "tm_score")
GEOMETRY = ("model_clashes", "model_bad_bonds", "model_bad_angles")
GROUPS = ("atom_encoder.", "residue_trunk.", "latent_residual.", "atom_decoder.")
ROLE = "seed-replicate fixed-final-checkpoint architecture comparison;no checkpoint or sampler tuning"
TEMPLATE_KEYS = frozenset({"config", "out_dir", "wandb_name"})
FLAT_FILES = dict(source_manifest_sha256="source_manifest.json", protocol_sha256="protocol.json",
    panel_admission_sha256="panel_admitted.json", comparison_sha256="comparison.json",
    inference_admission_sha256="inference_admitted.json", score_admission_sha256="scores/flat_fine6.json",
    training_admission_sha256="train_admitted/flat_fine6.json")
FLAT_VERIFIER = "benchmarks/compare_scratch_hierarchy_fine6.py"
UNSEALED = ("source_manifest.json", "cpu_admitted.json", "reference10k_admitted.json",
            "reference_flat_fine6_admitted.json", "preflight_verified.json")
TESTS = ("tests/test_seed_replicates.py", "tests/test_recycle_checkpoint_curve.py",
         "tests/test_scratch_recycle_full4.py", "tests/test_scratch_recycle_tail.py",
         "tests/test_scratch_hierarchy_fine6.py", "tests/test_shared_trunk_audit.py",
         "tests/test_scratch_architecture_report.py", "tests/test_scratch_architecture_screen.py")
DEFAULT_FLAT = ROOT / "out/hierarchy-fine6-scratch-wsd-20261025T050440Z"
DEFAULT_FULL4 = ROOT / "out/recycle-full4-scratch-wsd-20261026T031310Z"
# A follow-up run admits and evaluates the finished trainings of one frozen run without editing it.
FOLLOWUP_FILES = dict(protocol_sha256="protocol.json", source_manifest_sha256="source_manifest.json",
    cpu_admission_sha256="cpu_admitted.json", preflight_admission_sha256="preflight_verified.json",
    reference10k_admission_sha256="reference10k_admitted.json",
    reference_flat_admission_sha256="reference_flat_fine6_admitted.json", jobs_sha256="jobs.json",
    verify_frozen_sha256="verify_frozen.py")
FOLLOWUP_KEYS = ("created_at", "authority", "followup_role", "training_run")
FOLLOWUP_ROLE = ("retrospective admission and common evaluation of the finished seed-replicate trainings of the bound "
                 "training run; no retraining, the training run is read only")
UNRECORDED_GPU = ("training GPU UUID unrecorded: the training job failed at its own admission before writing a receipt; "
                  "the Slurm array task and ground_truth_verified.json identify the job and GPU class")
COMPLETED_TRAINING = "completed with its own training admission"
FAILED_AT_FLAT_CHECK = "failed only at the presence-based flat_fine6 admission check"
FLAT_FAILURE = "ValueError: flat_fine6 must keep the original non-recycling construction"
LIMITS = ("Three training seeds per architecture (flat_fine6: three new-code seeds; fixed2: two new seeds plus the "
    "reused seed-20261023 model), one SDE500 tau0.01 sample per target, scratch WSD10000 applied updates only. "
    "The eligible15-target/13-entry panel has now been reused across several architecture decisions, so it is no "
    "longer a clean held-out test; descriptive SDs from two or three seeds are not confidence intervals. The "
    "reused fixed2 seed-20261023 rows come from the three-way run's inference job (same model source, other job "
    "and possibly another physical GPU of the same class); the old flat_fine6 seed-20261023 run used older model "
    "source and harness code, so its paired delta is cross-code. Declared sequence-homology exclusion does not "
    "exclude remote homology, coordinate overlap or ESMC pretraining exposure. CA is canonical OST bb_lddt; "
    "geometry entries are unnormalised OST counts. Sampler times include instrumentation and final CPU copy. "
    "No automatic architecture selection.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


sha = inputs.sha256


def load(path):
    return json.loads(Path(path).read_text())


def bind():
    """Import the torch-dependent evaluator stack only for Slurm stages and tests, never for prepare."""
    if "tail" in globals():
        return
    curve.bind()
    from benchmarks import compare_scratch_recycle_tail as module

    base = module.base
    globals().update(tail=module, base=base, scratch=base.scratch, proof=base.proof, screen=base.screen,
                     read=base.read, exact=base.exact, write_new=base.write_new,
                     BASE_FIELDS=tuple(base.proof.MODEL_FIELDS), RECYCLE_FIELDS=tuple(module.MODEL_FIELDS),
                     _BASE_PREFLIGHT=base.scratch.preflight, _BASE_VERIFY=base.verify_actual_training,
                     _BASE_ADMITTED=base.admitted_training)


# ── torch-free preparation ──────────────────────────────────────────────


def recipe_templates(flat_reference, full4_reference):
    """The two frozen recipes; they may differ only by the declared atom-carry/fixed2 keys."""
    original = load(Path(full4_reference) / "protocol.json")
    pair = Path(original["external_pair"]["run"]).resolve()
    paths = dict(flat_fine6=Path(flat_reference).resolve() / "configs/flat_fine6.yaml",
                 fixed2=pair / "configs/fixed2.yaml")
    templates = {arch: yaml.safe_load(path.read_text()) for arch, path in paths.items()}
    verify_templates(templates)
    return paths, templates


def verify_templates(templates):
    flat, fixed = templates["flat_fine6"], templates["fixed2"]
    require(isinstance(flat, dict) and isinstance(fixed, dict), "recipe templates must be mappings")
    require(flat.get("seed") == fixed.get("seed") == REPRODUCTION_SEED, "recipe templates are not seed20261023")
    require(not set(flat) & set(ARCH_KEYS) and {key: fixed.get(key) for key in ARCH_KEYS}
            == {key: ARCHITECTURES["fixed2"][key] for key in ARCH_KEYS}, "architecture keys differ from declaration")
    common = lambda value: {key: item for key, item in value.items() if key not in TEMPLATE_KEYS | set(ARCH_KEYS)}  # noqa: E731
    require(common(flat) == common(fixed), "flat_fine6 and fixed2 recipes differ outside atom_recycling/recycle_policy")


def arm_config(template, run, arm):
    return dict(template, config=str(Path(run) / f"configs/{arm}.yaml"), out_dir=str(Path(run) / f"training/{arm}"),
                wandb_name=f"seed-replicate-{arm}", seed=SEED_OF[arm])


def model_source(manifest):
    return {name: digest for name, digest in manifest.items() if name.startswith("code/src/")}


def source_drift(old, new):
    old, new = model_source(old), model_source(new)
    return sorted(name for name in set(old) | set(new) if old.get(name) != new.get(name))


def flat_binding(old):
    old = Path(old).resolve()
    home = old / "training/flat_fine6"
    return dict(run=str(old), code=str(old / "code"), seed=REPRODUCTION_SEED,
                **{key: sha(old / name) for key, name in FLAT_FILES.items()},
                initial_model_sha256=load(home / "initialization.json")["target_sha256"],
                input_stream_sha256=load(home / "audit/complete.json")["input_stream_sha256"])


def regular(path):
    return Path(path).is_file() and not Path(path).is_symlink()


VERIFY_FROZEN = r'''"""Check static seed-replicate seals, both reused seed-20261023 bindings and the fixed2 model source."""
import hashlib,json,subprocess,sys
from pathlib import Path
import yaml
RUN=Path(__file__).resolve().parent
REF10K=__REF10K__
FLAT=__FLAT__
TEMPLATE_KEYS=set(__TEMPLATE__)
FOLLOWUP=__FOLLOWUP__
FOLLOWUP_KEYS=set(__FOLLOWUP_KEYS__)

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
 return h.hexdigest()

def check_manifest(run,unsealed=('source_manifest.json','cpu_admitted.json')):
 manifest=json.loads((run/'source_manifest.json').read_text())
 assert manifest and not set(manifest)&set(unsealed)
 for name,digest in manifest.items():
  p=run/name
  assert not Path(name).is_absolute() and '..' not in Path(name).parts
  assert p.resolve().is_relative_to(run.resolve()) and p.is_file() and not p.is_symlink()
  assert p.suffix not in ('.pt','.pth','.safetensors'),name
  assert sha(p)==digest,name
 return manifest

def main():
 current=check_manifest(RUN,__UNSEALED__)
 p=json.loads((RUN/'protocol.json').read_text())
 assert p['seeds']==[20261023,20261024,20261025] and all(type(s) is int for s in p['seeds'])
 assert set(p['arms'])=={'flat_fine6_s20261023','flat_fine6_s20261024','flat_fine6_s20261025','fixed2_s20261024','fixed2_s20261025'}
 assert p['sampler']=={'method':'sde','n_steps':500,'tau':.01,'seed':0,'K':2} and p['inference_depth']==2
 assert p['formal_applied_update_cap_per_arm']==10000 and p['microbatches_per_formal_run']==40000
 ref=p['reference10k'];full4=Path(ref['run']).resolve();baseline=check_manifest(full4)
 assert Path(ref['code']).resolve()==full4/'code'
 for key,name in REF10K.items():assert sha(full4/name)==ref[key],key
 model={n for n in baseline if n.startswith('code/src/')}
 assert model and model=={n for n in current if n.startswith('code/src/')} and all(current[n]==baseline[n] for n in model),'model source differs from the reused fixed2'
 assert Path(p['panel']).resolve()==full4/'panel' and sha(full4/'panel/panel.json')==p['panel_sha256']
 flat=p['reference_flat_fine6'];old=Path(flat['run']).resolve();check_manifest(old)
 assert Path(flat['code']).resolve()==old/'code'
 for key,name in FLAT.items():assert sha(old/name)==flat[key],key
 original=json.loads((full4/'protocol.json').read_text());pair=Path(original['external_pair']['run']).resolve()
 assert sha(pair/'source_manifest.json')==original['external_pair']['source_manifest_sha256']
 pairs=json.loads((pair/'source_manifest.json').read_text()); olds=json.loads((old/'source_manifest.json').read_text())
 templates={}
 for arch,entry in p['templates'].items():
  path=Path(entry['path']);owner,seal=(old,olds) if arch=='flat_fine6' else (pair,pairs)
  assert sha(path)==entry['sha256']==seal[str(path.resolve().relative_to(owner))],arch
  templates[arch]=yaml.safe_load(path.read_text())
 for arm,entry in p['arms'].items():
  c=yaml.safe_load(Path(entry['config']).read_text())
  expected=dict(templates[entry['architecture']],config=entry['config'],out_dir=entry['training'],wandb_name='seed-replicate-'+arm,seed=entry['seed'])
  assert c==expected,arm
  assert c['resume'] is None and c['initialize_refiner_from'] is None and c['initialize_model_from_ema'] is False
  assert c['keep_checkpoint_steps']==[5000,10000] and c['total_steps']==10000
 for path in (old/'training/flat_fine6/ckpt_0010000.pt',pair/'training/fixed2/ckpt_0010000.pt'):
  assert path.is_file() and not path.is_symlink(),path
 t=p.get('training_run')
 if t is not None:
  orig=Path(t['run']).resolve()
  for key,name in FOLLOWUP.items():assert sha(orig/name)==t[key],key
  assert json.loads((orig/'jobs.json').read_text())['train_array']==t['training_job']
  assert set(t['configs'])==set(p['arms'])
  for arm,digest in t['configs'].items():assert sha(orig/'configs'/(arm+'.yaml'))==digest,arm
  found={str(q.relative_to(orig)) for q in (orig/'preflight').rglob('*') if q.is_file() and q.suffix!='.pt'}
  assert found==set(t['preflight'])
  for name,digest in t['preflight'].items():
   q=orig/name;assert q.is_file() and not q.is_symlink() and sha(q)==digest,name
  for arm,entry in p['arms'].items():
   for key,name in (('config','configs/'+arm+'.yaml'),('preflight','preflight/'+arm),('training','training/'+arm),('u0','u0')):
    assert Path(entry[key]).resolve()==orig/name,(arm,key)
  source=json.loads((orig/'protocol.json').read_text())
  assert 'training_run' not in source
  assert {k:v for k,v in p.items() if k not in FOLLOWUP_KEYS}=={k:v for k,v in source.items() if k not in FOLLOWUP_KEYS}
  subprocess.run([sys.executable,str(orig/'verify_frozen.py')],check=True)
 print('Seed-replicate static seals, reused seed-20261023 bindings and fixed2 model source verified',flush=True)

if __name__=='__main__':main()
'''

WRITE_CPU_ADMISSION = r'''"""Bind focused CPU checks and the reused admitted panel to the seed-replicate trial."""
import hashlib,json,os,sys
from pathlib import Path
import xml.etree.ElementTree as ET
assert os.environ.get('SLURM_JOB_ID'), 'Slurm allocation required'
run=Path(__file__).resolve().parent
code=run/'code';sys.path[:0]=[str(code),str(code/'src')]
from benchmarks import compare_seed_replicates as trial
assert Path(trial.__file__).resolve()==code/'benchmarks/compare_seed_replicates.py'
trial.bind()
protocol=json.loads((run/'protocol.json').read_text())
trial.validate_protocol(run,protocol)
targets=trial.validate_panel(protocol)
root=ET.parse(run/'checks.xml').getroot();suites=list(root) if root.tag=='testsuites' else [root]
counts={key:sum(int(suite.attrib.get(key,0)) for suite in suites) for key in ('tests','failures','errors','skipped')}
assert counts['tests']>0 and counts['failures']==counts['errors']==counts['skipped']==0,counts
classes={case.attrib.get('classname','') for case in root.iter('testcase')}
for name in trial.TESTS:
 module=name.removesuffix('.py').replace('/','.')
 assert any(value==module or value.startswith(module+'.') for value in classes),name
value=dict(admitted=True,source_manifest_sha256=hashlib.sha256((run/'source_manifest.json').read_bytes()).hexdigest(),tests=counts,panel_targets=len(targets),panel_entries=len({row['entry_id'] for row in targets.values()}),panel_reuse='unchanged three-way sealed admission',checks_sha256=hashlib.sha256((run/'checks.xml').read_bytes()).hexdigest())
trial.write_new(run/'cpu_admitted.json',value)
print(json.dumps(value),flush=True)
'''


def scripts(run, *, followup=False):
    common = f'''#!/usr/bin/env bash
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
RUN={shlex.quote(str(run))}
CODE="$RUN/code"
PY="$SLURM_SUBMIT_DIR/.venv/bin/python"
export PYTHONPATH="$CODE:$CODE/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TMPDIR="/tmp/mf-seedrep-${{SLURM_JOB_ID}}-${{SLURM_ARRAY_TASK_ID:-stage}}"
mkdir -p "$TMPDIR"
"$PY" "$RUN/verify_frozen.py"
'''
    stage = '"$PY" "$CODE/benchmarks/compare_seed_replicates.py" {} --run "$RUN"{}\n'
    arms = " ".join(ARMS)
    cpu = 'export CUDA_VISIBLE_DEVICES=""\n'
    values = {"cpu_admit.sh": common + cpu + 'cd "$CODE"\n"$PY" -m pytest -q '
                  + " ".join(TESTS) + ' --junitxml="$RUN/checks.xml"\n"$PY" "$RUN/write_cpu_admission.py"\n'
                  + stage.format("reference", ""),
              "infer.sh": common + stage.format("infer", ""),
              "score.sh": common + cpu + stage.format("score", "") + stage.format("report", "")}
    if followup:
        return dict(values, **{"admit_training.sh": common + cpu + stage.format("admit-training", "")})
    return dict(values, **{"preflight.sh": common + stage.format("preflight", ""),
                           "train.sh": common + f'ARMS=({arms})\nARM="${{ARMS[$SLURM_ARRAY_TASK_ID]}}"\n'
                               + stage.format("train", ' --arm "$ARM"')})


def prepare(out, *, flat_reference=DEFAULT_FLAT, full4_reference=DEFAULT_FULL4, stamp=None):
    """Freeze a new run with file operations only; every model/torch check happens later under Slurm."""
    old, full4 = Path(flat_reference).resolve(), Path(full4_reference).resolve()
    original = load(full4 / "protocol.json")
    pair = Path(original["external_pair"]["run"]).resolve()
    paths, templates = recipe_templates(old, full4)
    for path in (old / "training/flat_fine6/ckpt_0010000.pt", pair / "training/fixed2/ckpt_0010000.pt"):
        require(regular(path), "missing regular reused10000 checkpoint")
    flat = flat_binding(old)
    old_protocol = load(old / "protocol.json")
    require(old_protocol.get("sampler") == SAMPLER and original.get("sampler") == SAMPLER, "reused sampler differs")
    require(old_protocol.get("WSD") == original.get("WSD") and old_protocol.get("ost") == original.get("ost"),
            "reused WSD/scorer differs")
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = Path(out).resolve() / f"seed-replicate-scratch-wsd-{stamp}"
    run.mkdir(parents=True, exist_ok=False)
    frozen = freeze_code(run)
    full4_manifest = load(full4 / "source_manifest.json")
    require(frozen == model_source(full4_manifest), "current model source differs from the reused fixed2 model source")
    arms = {}
    for arm in ARMS:
        config = arm_config(templates[ARCHITECTURE_OF[arm]], run, arm)
        (run / "configs").mkdir(exist_ok=True)
        (run / f"configs/{arm}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        arms[arm] = dict(architecture=ARCHITECTURE_OF[arm], seed=SEED_OF[arm], config=config["config"],
            preflight=str(run / f"preflight/{arm}"), training=config["out_dir"], u0=str(run / "u0"),
            nominal_K=2, expected_parameters=PARAMETER_COUNT)
    pair_home = pair / "training/fixed2"
    declared = dict(flat_fine6=old_protocol["arms"]["flat_fine6"]["model_fields"],
                    fixed2=load(pair / "protocol.json")["arms"]["fixed2"]["model_fields"])
    for arm, entry in arms.items():
        entry["model_fields"] = {key: templates[entry["architecture"]][key] for key in declared[entry["architecture"]]}
    protocol = dict(created_at=datetime.now(timezone.utc).isoformat(),
        authority="User authorized five scratch WSD10k single-GPU runs (flat_fine6 seeds20261023-25, fixed2 seeds "
                  "20261024-25), one common-GPU K2 SDE500 inference and canonical scoring; fixed2 seed20261023 and the "
                  "old flat_fine6 seed20261023 run are reused by hash binding, never retrained.",
        benchmark_role=ROLE, seeds=list(SEEDS), architectures=ARCHITECTURES, arms=arms,
        formal_applied_update_cap_per_arm=STEP, microbatches_per_formal_run=4 * STEP, preflight_optimizer_updates=0,
        teacher_models=0, all_parameters_trainable=True, sampler=dict(SAMPLER), inference_depth=DEPTH,
        maximum_concurrent_training_GPUs=len(ARMS), panel=str(full4 / "panel"),
        panel_sha256=load(full4 / "panel_admitted.json")["panel_sha256"], ost=original["ost"], WSD=original["WSD"],
        expected_targets=15, expected_entries=13, reproduction_seed=REPRODUCTION_SEED,
        templates={arch: dict(path=str(path), sha256=sha(path)) for arch, path in paths.items()},
        reference10k=dict(run=str(full4), code=str(full4 / "code"),
                          **{key: sha(full4 / name) for key, name in curve.REFERENCE_FILES.items()}),
        reused_fixed2=dict(seed=REPRODUCTION_SEED, training=str(pair_home),
            initial_model_sha256=load(pair_home / "initialization.json")["target_sha256"],
            input_stream_sha256=load(pair_home / "audit/complete.json")["input_stream_sha256"]),
        reference_flat_fine6=flat,
        reproduction=dict(arm=REPRODUCTION_ARM, expected_initial_model_sha256=flat["initial_model_sha256"],
                          expected_input_stream_sha256=flat["input_stream_sha256"]),
        model_source_changes_since_old_flat_fine6=source_drift(load(old / "source_manifest.json"), frozen),
        flat_fine6_reproduction_gate="GPU preflight requires the old seed20261023 initial tensors, an old-vs-current "
            "forward/LiFT-backward equivalence probe on perturbed weights, and the old first eight native inputs; "
            "training admission requires the old 40000-microbatch input stream.",
        limits=LIMITS)
    manifest = freeze_run(run, protocol)
    return dict(run=str(run), frozen_files=len(manifest), arms=list(ARMS),
                source_manifest_sha256=sha(run / "source_manifest.json"),
                model_source_changes_since_old_flat_fine6=protocol["model_source_changes_since_old_flat_fine6"])


def freeze_code(run):
    for name in curve.frozen_files(ROOT):
        (run / "code" / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, run / "code" / name)
    return {str(path.relative_to(run)): sha(path) for path in (run / "code/src").rglob("*.py")
            if "__pycache__" not in path.parts}


def freeze_run(run, protocol):
    """Write the protocol, verifiers and stage scripts, then seal everything except later admissions."""
    (run / "protocol.json").write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
    (run / "verify_frozen.py").write_text(VERIFY_FROZEN.replace("__REF10K__", repr(curve.REFERENCE_FILES))
        .replace("__FLAT__", repr(FLAT_FILES)).replace("__TEMPLATE__", repr(sorted(TEMPLATE_KEYS)))
        .replace("__UNSEALED__", repr(UNSEALED)).replace("__FOLLOWUP__", repr(FOLLOWUP_FILES))
        .replace("__FOLLOWUP_KEYS__", repr(sorted(FOLLOWUP_KEYS))))
    (run / "write_cpu_admission.py").write_text(WRITE_CPU_ADMISSION)
    for name, text in scripts(run, followup="training_run" in protocol).items():
        (run / name).write_text(text)
        (run / name).chmod(0o755)
    manifest = curve.seal(run)
    require(not set(manifest) & set(UNSEALED), "prepare sealed an admission artifact")
    with (run / "source_manifest.json").open("x") as stream:
        stream.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def preflight_files(original):
    """Every non-checkpoint preflight file; initial_ema.pt bytes are bound through initialization.json."""
    return {str(path.relative_to(original)): sha(path) for path in sorted((Path(original) / "preflight").rglob("*"))
            if path.is_file() and path.suffix != ".pt"}


def training_binding(original):
    original = Path(original).resolve()
    for name in FOLLOWUP_FILES.values():
        require(regular(original / name), f"training run lacks a regular {name}")
    require("training_run" not in load(original / "protocol.json"), "a follow-up cannot bind another follow-up run")
    job = load(original / "jobs.json").get("train_array")
    require(isinstance(job, str) and job.isdigit(), "training run lacks its training array job id")
    for arm in ARMS:
        require(regular(original / f"configs/{arm}.yaml") and (original / f"preflight/{arm}").is_dir(),
                "training run lacks an arm configuration or preflight")
    return dict(run=str(original), training_job=job,
                **{key: sha(original / name) for key, name in FOLLOWUP_FILES.items()},
                configs={arm: sha(original / f"configs/{arm}.yaml") for arm in ARMS},
                preflight=preflight_files(original))


def prepare_followup(out, training_run, *, stamp=None):
    """Freeze a follow-up run that admits the bound run's trainings; the bound run is only read."""
    original = Path(training_run).resolve()
    binding = training_binding(original)
    source = load(original / "protocol.json")
    full4 = Path(source["reference10k"]["run"]).resolve()
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = Path(out).resolve() / f"seed-replicate-eval-{stamp}"
    run.mkdir(parents=True, exist_ok=False)
    frozen = freeze_code(run)
    require(frozen == model_source(load(full4 / "source_manifest.json"))
            == model_source(load(original / "source_manifest.json")),
            "current model source differs from the training run or the reused fixed2 model source")
    protocol = dict(source, created_at=datetime.now(timezone.utc).isoformat(),
        authority="User authorized admitting the five finished WSD10k trainings of the bound run without retraining "
                  "(their own admission failed on a presence-based flat_fine6 check), one common-GPU K2 SDE500 "
                  "inference and canonical scoring; the bound run is never edited.",
        followup_role=FOLLOWUP_ROLE, training_run=binding)
    manifest = freeze_run(run, protocol)
    return dict(run=str(run), frozen_files=len(manifest), arms=list(ARMS), training_run=str(original),
                training_job=binding["training_job"], source_manifest_sha256=sha(run / "source_manifest.json"))


# ── admitted contract ───────────────────────────────────────────────────


@contextmanager
def registry():
    old = scratch.VARIANTS, proof.VARIANTS
    scratch.VARIANTS = {name: dict(value) for name, value in ARMS.items()}
    proof.VARIANTS = {name: dict(value) for name, value in ARMS.items()}
    try:
        yield
    finally:
        scratch.VARIANTS, proof.VARIANTS = old


def expected(arm):
    require(arm in ARMS, "undeclared seed-replicate arm")
    with registry():
        return scratch.expected_config(arm)


def fields(arm):
    return RECYCLE_FIELDS if ARCHITECTURE_OF[arm] == "fixed2" else BASE_FIELDS


def model_fields(arm):
    config = expected(arm)
    return {key: config[key] for key in fields(arm)}


def validate_flat_reference(protocol):
    bound = protocol.get("reference_flat_fine6", {})
    require(isinstance(bound, dict) and set(bound) == {"run", "code", "seed", "initial_model_sha256",
            "input_stream_sha256", *FLAT_FILES}, "old flat_fine6 binding fields differ")
    old, code = Path(bound["run"]), Path(bound["code"])
    require(old.is_absolute() and code.resolve() == old.resolve() / "code" and bound["seed"] == REPRODUCTION_SEED
            and type(bound["seed"]) is int, "old flat_fine6 run/code/seed differs")
    for key, name in FLAT_FILES.items():
        require(proof.valid_sha(bound[key]) and sha(old / name) == bound[key], f"old flat_fine6 {name} changed")
    original, manifest = read(old / "protocol.json"), read(old / "source_manifest.json")
    verifier = code / FLAT_VERIFIER
    require(manifest.get(str(verifier.resolve().relative_to(old.resolve()))) == sha(verifier),
            "old flat_fine6 evaluator is not sealed")
    exact(original, dict(sampler=SAMPLER, ost=protocol.get("ost")), "old flat_fine6 sampler/scorer differs")
    exact(original["sampler"], SAMPLER, "typed old flat_fine6 sampler differs")
    require(original.get("WSD") == protocol.get("WSD"), "old flat_fine6 WSD schedule differs")
    home = old.resolve() / "training/flat_fine6"
    require(Path(original.get("arms", {}).get("flat_fine6", {}).get("training", "")).resolve() == home,
            "old flat_fine6 training home differs")
    require(read(home / "initialization.json").get("target_sha256") == bound["initial_model_sha256"]
            and read(home / "audit/complete.json").get("input_stream_sha256") == bound["input_stream_sha256"],
            "old flat_fine6 initial tensors or input stream changed")
    return original


def shared_protocol(protocol):
    return {key: value for key, value in protocol.items() if key not in FOLLOWUP_KEYS}


def validate_training_binding(protocol):
    """A follow-up keeps every training-run protocol field and binds that run's admissions by hash."""
    bound = protocol.get("training_run", {})
    require(isinstance(bound, dict) and set(bound) == {"run", "training_job", "configs", "preflight", *FOLLOWUP_FILES},
            "training-run binding fields differ")
    require(Path(bound["run"]).is_absolute() and type(bound["training_job"]) is str and bound["training_job"].isdigit(),
            "training-run path or job id differs")
    original = Path(bound["run"]).resolve()
    for key, name in FOLLOWUP_FILES.items():
        require(proof.valid_sha(bound[key]) and regular(original / name) and sha(original / name) == bound[key],
                f"training run {name} changed")
    require(read(original / "jobs.json").get("train_array") == bound["training_job"], "training array job differs")
    require(set(bound["configs"]) == set(ARMS) and all(regular(original / f"configs/{arm}.yaml")
            and sha(original / f"configs/{arm}.yaml") == digest for arm, digest in bound["configs"].items()),
            "training-run arm configuration changed")
    require(bound["preflight"] and bound["preflight"] == preflight_files(original)
            and all(regular(original / name) for name in bound["preflight"]), "training-run preflight evidence changed")
    source = read(original / "protocol.json")
    require("training_run" not in source and shared_protocol(protocol) == shared_protocol(source),
            "follow-up protocol differs from its training run outside the follow-up binding")
    require(protocol.get("followup_role") == FOLLOWUP_ROLE, "follow-up role differs")
    return original


def owner(context):
    """The context whose run holds preflight, training homes and their admissions."""
    return context.get("training_context", context)


def validate_protocol(run, protocol):
    run = Path(run).resolve()
    home = validate_training_binding(protocol) if "training_run" in protocol else run
    exact(protocol, dict(benchmark_role=ROLE, formal_applied_update_cap_per_arm=STEP,
        microbatches_per_formal_run=4 * STEP, preflight_optimizer_updates=0, teacher_models=0,
        all_parameters_trainable=True, inference_depth=DEPTH, expected_targets=15, expected_entries=13,
        reproduction_seed=REPRODUCTION_SEED), "seed-replicate budget/role/coverage differs")
    require(protocol.get("sampler") == SAMPLER, "sampler differs")
    exact(protocol["sampler"], SAMPLER, "typed sampler differs")
    require(protocol.get("seeds") == list(SEEDS) and all(type(seed) is int for seed in protocol["seeds"]),
            "declared seeds differ")
    require(protocol.get("architectures") == ARCHITECTURES, "declared architectures differ")
    require(set(protocol.get("arms", {})) == set(ARMS), "exactly the five new seed-replicate arms are required")
    original = curve.validate_reference(protocol)
    full4 = Path(protocol["reference10k"]["run"]).resolve()
    require(protocol.get("ost") == original["ost"] and protocol.get("WSD") == original["WSD"],
            "scorer/WSD differs from the reused fixed2 evaluation")
    require(Path(protocol.get("panel", "")).resolve() == full4 / "panel"
            and protocol.get("panel_sha256") == read(full4 / "panel_admitted.json").get("panel_sha256")
            == sha(full4 / "panel/panel.json"), "panel must be the reused three-way panel")
    validate_flat_reference(protocol)
    flat = protocol["reference_flat_fine6"]
    exact(protocol.get("reproduction", {}), dict(arm=REPRODUCTION_ARM,
        expected_initial_model_sha256=flat["initial_model_sha256"],
        expected_input_stream_sha256=flat["input_stream_sha256"]), "flat_fine6 reproduction target differs")
    pair = Path(original["external_pair"]["run"]).resolve()
    pair_home = pair / "training/fixed2"
    exact(protocol.get("reused_fixed2", {}), dict(seed=REPRODUCTION_SEED, training=str(pair_home),
        initial_model_sha256=original["external_pair"]["expected_initial_model_sha256"],
        input_stream_sha256=read(pair / "train_admitted/fixed2.json")["complete"]["input_stream_sha256"]),
        "reused fixed2 identity differs")
    require(read(pair_home / "initialization.json").get("target_sha256")
            == protocol["reused_fixed2"]["initial_model_sha256"], "reused fixed2 initial tensors changed")
    old = Path(flat["run"]).resolve()
    templates, seals = {}, dict(flat_fine6=(old, read(old / "source_manifest.json")),
                                fixed2=(pair, read(pair / "source_manifest.json")))
    require(sha(pair / "source_manifest.json") == original["external_pair"]["source_manifest_sha256"],
            "reused fixed2 recipe manifest changed")
    require(set(protocol.get("templates", {})) == set(ARCHITECTURES), "recipe template coverage differs")
    for arch, entry in protocol["templates"].items():
        owner, manifest = seals[arch]
        path = Path(entry.get("path", ""))
        require(path.resolve() == owner / f"configs/{arch}.yaml"
                and sha(path) == entry.get("sha256") == manifest.get(str(path.resolve().relative_to(owner))),
                "recipe template is not the sealed reused configuration")
        templates[arch] = yaml.safe_load(path.read_text())
    verify_templates(templates)
    configs = {}
    for arm, entry in protocol["arms"].items():
        exact(entry, dict(architecture=ARCHITECTURE_OF[arm], seed=SEED_OF[arm], nominal_K=2,
                          expected_parameters=PARAMETER_COUNT), "arm architecture/seed/construction differs")
        require(set(entry.get("model_fields", {})) == set(fields(arm)), "architecture model fields are required")
        exact(entry["model_fields"], model_fields(arm), "declared model fields differ")
        for key, path in dict(config=home / f"configs/{arm}.yaml", preflight=home / f"preflight/{arm}",
                              training=home / f"training/{arm}", u0=home / "u0").items():
            require(Path(entry[key]).resolve() == path, f"unexpected {arm} {key} path")
        config = yaml.safe_load(Path(entry["config"]).read_text())
        require(config == arm_config(templates[entry["architecture"]], home, arm),
                "arm configuration differs from its reused recipe outside seed and paths")
        with registry():
            scratch.validate_config(config, arm)
        exact(config, expected(arm), "typed seed-replicate configuration differs")
        configs[arm] = config
    return configs


def validate_panel(protocol):
    original = read(Path(protocol["reference10k"]["run"]) / "protocol.json")
    targets = base.validate_panel(Path(protocol["reference10k"]["run"]).resolve(), original)
    require(len(targets) == protocol["expected_targets"] and len({row["entry_id"] for row in targets.values()})
            == protocol["expected_entries"], "seed-replicate panel must contain fifteen targets and thirteen entries")
    return targets


def validate_run(run, *, executing=True):
    """Admit a frozen run; executing=False admits a bound training run whose own evaluator is not running."""
    bind()
    run = Path(run).resolve()
    manifest = read(run / "source_manifest.json")
    require(not set(manifest) & set(UNSEALED), "manifest cannot seal itself or its admissions")
    base.seals(run, manifest)
    for path in (Path(__file__), Path(curve.__file__), Path(tail.__file__), Path(tail.atom.__file__),
                 Path(base.__file__), Path(scratch.__file__), Path(proof.__file__), Path(screen.__file__),
                 Path(base.audit.__file__), Path(inputs.__file__), Path(base.canonical.__file__)):
        if executing:
            require(manifest.get(base.relative(run, path)) == sha(path), "unsealed seed-replicate dependency")
        else:
            require("code/" + base.relative(ROOT, path) in manifest, "training run did not seal its evaluator")
    require(manifest.get("protocol.json") == sha(run / "protocol.json"), "unsealed seed-replicate protocol")
    exact(read(run / "cpu_admitted.json"), dict(admitted=True, source_manifest_sha256=sha(run / "source_manifest.json")),
          "CPU admission binding differs")
    protocol = read(run / "protocol.json")
    require(executing or "training_run" not in protocol, "a training run cannot itself be a follow-up")
    configs = validate_protocol(run, protocol)
    training = validate_run(protocol["training_run"]["run"], executing=False) if "training_run" in protocol else None
    sealed = training or dict(run=run, manifest=manifest)
    for entry in protocol["arms"].values():
        require(sealed["manifest"].get(base.relative(sealed["run"], entry["config"])) == sha(entry["config"]),
                "unsealed arm configuration")
    baseline = read(Path(protocol["reference10k"]["run"]) / "source_manifest.json")
    require(model_source(manifest) == model_source(baseline), "model source differs from the reused fixed2 model")
    require(training is None or model_source(manifest) == model_source(training["manifest"]),
            "follow-up model source differs from the model source that trained the bound checkpoints")
    require(protocol.get("model_source_changes_since_old_flat_fine6")
            == source_drift(read(Path(protocol["reference_flat_fine6"]["run"]) / "source_manifest.json"), manifest),
            "recorded model source drift differs")
    context = dict(run=run, protocol=protocol, manifest=manifest, configs=configs, targets=validate_panel(protocol),
        source_manifest_sha256=sha(run / "source_manifest.json"), protocol_sha256=sha(run / "protocol.json"),
        panel_admission_sha256=protocol["reference10k"]["panel_admission_sha256"])
    if training is not None:
        require(training["configs"] == configs and training["targets"] == context["targets"],
                "follow-up and training-run configurations or panel differ")
        context["training_context"] = training
    return context


def unchanged(context):
    fresh = validate_run(context["run"])
    require(base.context_binding(fresh) == base.context_binding(context) and fresh["targets"] == context["targets"]
            and base.context_binding(owner(fresh)) == base.context_binding(owner(context)),
            "seed-replicate sources/protocol/panel changed")


@contextmanager
def bindings():
    values = dict(ARMS=ARMS, PARAMETERS=PARAMETERS, registry=registry, verify_actual_training=verify_actual_training,
                  admitted_training=admitted_training, admitted_inference=admitted_inference, unchanged=unchanged)
    old = {key: getattr(base, key) for key in values}
    try:
        for key, value in values.items():
            setattr(base, key, value)
        with registry():
            yield
    finally:
        for key, value in old.items():
            setattr(base, key, value)


# ── reused seed-20261023 points ─────────────────────────────────────────


OLD_FLAT_SCRIPT = r'''
import math, statistics, sys
from pathlib import Path
code, run, destination = (Path(value).resolve() for value in sys.argv[1:4])
sys.path[:0] = [str(code), str(code / "src")]
from benchmarks import compare_scratch_hierarchy_fine6 as trial
assert Path(trial.__file__).resolve() == code / "benchmarks/compare_scratch_hierarchy_fine6.py"
arm = "flat_fine6"
context = trial.validate_run(run)
inference, receipt = trial.admitted_inference(context), trial.admitted_training(context, arm)
binding = trial.context_binding(context)
comparison = trial.read(run / "comparison.json")
trial.exact(comparison, dict(benchmark_role=trial.ROLE, nominal_K=2, sampler=trial.SAMPLER, architecture_selected=False,
    **binding), "old flat_fine6 report context differs")
trial.require(comparison["inference_hardware"] == inference["hardware"], "old report hardware differs")
admission = trial.read(run / f"scores/{arm}.json")
trial.exact(admission, dict(admitted=True, arm=arm, coverage=len(context["targets"]),
    inference_admission_sha256=trial.sha(run / "inference_admitted.json"), **binding), "old canonical admission differs")
directory = trial.evaluation(run, arm)
path = directory / "K2/scores/summary.json"
trial.require(trial.sha(path) == admission["summary_sha256"]
    and trial.sha(context["protocol"]["ost"]) == admission["ost_executable_sha256"], "old score evidence changed")
summary = trial.read(path)
values = trial.verify_canonical(summary, directory, context)
trial.require(values == admission["metrics"] and summary["openstructure"] == admission["scorer"],
    "old canonical vector/scorer differs")
trial.require(all(math.isclose(comparison["overall"][arm][metric], statistics.fmean(row[metric] for row in values.values()),
    rel_tol=0, abs_tol=1e-12) for metric in trial.QUALITY), "old report differs from admitted canonical vectors")
home = Path(context["protocol"]["arms"][arm]["training"])
initial = trial.read(home / "initialization.json")
trial.unchanged(context)
trial.write_new(destination, dict(admitted=True, arm=arm, seed=receipt["config"]["seed"], K=2, **binding,
    comparison_sha256=trial.sha(run / "comparison.json"), inference_admission_sha256=trial.sha(run / "inference_admitted.json"),
    score_admission_sha256=trial.sha(run / f"scores/{arm}.json"),
    training_admission_sha256=trial.sha(trial.training_receipt_path(run, arm)),
    verifier_sha256=trial.sha(Path(trial.__file__)), ost_executable_sha256=trial.sha(context["protocol"]["ost"]),
    scorer=admission["scorer"], hardware=inference["hardware"], per_target=values, overall=comparison["overall"][arm],
    costs=comparison["costs"][arm], training_receipt=receipt, training=str(home),
    checkpoint_sha256=inference["arms"][arm]["checkpoint_sha256"], ema_sha256=inference["arms"][arm]["ema_sha256"],
    config=receipt["config"], runtime=initial["mamba3_runtime"], initial_model_sha256=initial["target_sha256"],
    input_stream_sha256=receipt["complete"]["input_stream_sha256"]))
'''


def flat_reference_path(run):
    return Path(run) / "reference_flat_fine6_admitted.json"


def admitted_old_flat(run, protocol, targets):
    """Old flat_fine6 seed20261023 rows, admitted by its own frozen evaluator, plus canonical geometry."""
    bound = protocol["reference_flat_fine6"]
    validate_flat_reference(protocol)
    old, code = Path(bound["run"]).resolve(), Path(bound["code"]).resolve()
    value = read(flat_reference_path(run))
    exact(value, dict(admitted=True, arm="flat_fine6", seed=REPRODUCTION_SEED, K=2,
        **{key: bound[key] for key in FLAT_FILES}, verifier_sha256=sha(code / FLAT_VERIFIER),
        ost_executable_sha256=sha(protocol["ost"]), training=str(old / "training/flat_fine6"),
        initial_model_sha256=bound["initial_model_sha256"], input_stream_sha256=bound["input_stream_sha256"]),
        "old flat_fine6 admission binding differs")
    base.validate_hardware(value.get("hardware"), training=True)
    comparison, receipt = read(old / "comparison.json"), read(old / "train_admitted/flat_fine6.json")
    score = read(old / "scores/flat_fine6.json")
    require(value["hardware"] == comparison["inference_hardware"] and value.get("overall") == comparison["overall"]
            ["flat_fine6"] and value.get("costs") == comparison["costs"]["flat_fine6"], "old flat_fine6 report differs")
    require(value.get("training_receipt") == receipt and value.get("checkpoint_sha256")
            == receipt["final_checkpoint_sha256"] and value.get("ema_sha256") == receipt["final_ema_sha256"]
            and value.get("config") == receipt["config"], "old flat_fine6 training admission differs")
    require(set(value.get("per_target", {})) == set(targets) and value["per_target"] == score["metrics"]
            and value.get("scorer") == score["scorer"], "old flat_fine6 canonical vectors differ")
    require(sha(old / "panel/panel.json") == read(old / "panel_admitted.json").get("panel_sha256")
            == protocol["panel_sha256"], "old flat_fine6 panel differs from the reused panel")
    directory = base.evaluation(old, "flat_fine6")
    return dict(value, geometry=tail.geometry_diagnostics(directory, targets, value["per_target"], value["scorer"], DEPTH))


def reused_fixed2(context, reference=None):
    """The reused10000 fixed2 K2 row of the three-way evaluation, bound exactly as the checkpoint curve binds it."""
    reference = reference or curve.admitted_reference(context["run"], context["protocol"], context["targets"])
    row = reference["models"]["fixed2"]
    bound = context["protocol"]["reused_fixed2"]
    exact(row, dict(K=DEPTH, training=bound["training"]), "reused fixed2 depth/home differs")
    receipt = row["training_receipt"]
    require(row["config"].get("seed") == REPRODUCTION_SEED and row["config"].get("recycle_policy") == "fixed2"
            and row["config"].get("atom_recycling") == "carry", "reused fixed2 seed/policy differs")
    require(receipt["complete"]["input_stream_sha256"] == bound["input_stream_sha256"]
            and read(Path(row["training"]) / "initialization.json")["target_sha256"] == bound["initial_model_sha256"],
            "reused fixed2 initial tensors or inputs differ")
    drop = TEMPLATE_KEYS | {"seed", "requested_num_workers"}
    recipe = {key: value for key, value in context["configs"][arm_name("fixed2", SEEDS[1])].items() if key not in drop}
    require({key: value for key, value in row["config"].items() if key not in drop} == recipe,
            "reused fixed2 recipe differs from the new fixed2 seeds")
    return dict(row, scorer=reference["scorer"], hardware=reference["hardware"])


def admitted_references(context):
    run, protocol, targets = context["run"], context["protocol"], context["targets"]
    fixed = reused_fixed2(context)
    flat = admitted_old_flat(run, protocol, targets)
    require(fixed["scorer"] == flat["scorer"], "reused points used different canonical scorer identities")
    return dict(fixed2=fixed, flat_fine6=flat)


def reference(context):
    """Admit both reused seed20261023 points with their own frozen evaluators in isolated interpreters."""
    base.no_outputs([flat_reference_path(context["run"])])
    curve.reference(context)
    bound = context["protocol"]["reference_flat_fine6"]
    validate_flat_reference(context["protocol"])
    subprocess.run([sys.executable, "-c", OLD_FLAT_SCRIPT, bound["code"], bound["run"],
                    str(flat_reference_path(context["run"]))], cwd=bound["code"], check=True)
    return admitted_references(context)


# ── GPU preflight: construction pairing and old-vs-current flat_fine6 equivalence ──


EQUIVALENCE_SCRIPT = r'''
import json, sys
from dataclasses import replace
from pathlib import Path
code, config_path, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), Path(sys.argv[3])
sys.path[:0] = [str(code), str(code / "src")]
import numpy as np, torch, yaml
import mambafold
from benchmarks.compare_full_refinement import batch_digest, tensor_digest
from mambafold.train import engine, trainer
from mambafold.train.distributed import enable_cuda_perf_flags
assert Path(mambafold.__file__).resolve().is_relative_to(code / "src")
config = yaml.safe_load(config_path.read_text())
torch.set_num_threads(1)
enable_cuda_perf_flags()
trainer.seed_all(config["seed"])
model = trainer.build_model(config, "cuda")
initial = tensor_digest(model.state_dict().items())
generator = torch.Generator().manual_seed(20261027)
with torch.no_grad():
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        parameter.add_((torch.randn(parameter.shape, generator=generator) * .02).to(parameter.device, parameter.dtype))
perturbed = tensor_digest(model.state_dict().items())
trainer.seed_all(config["seed"] + 17)
batch = trainer._synthetic_batch(1, 32, 14, config["d_plm"], torch.device("cuda"))
residue_mask = batch.res_mask.clone()
residue_mask[:, -3:] = False
atom_mask = batch.atom_mask & residue_mask.unsqueeze(-1)
batch = replace(batch, x_clean=torch.randn_like(batch.x_clean) * .1, x_t=torch.randn_like(batch.x_t) * .1,
    eps=torch.randn_like(batch.eps), esm=torch.randn_like(batch.esm), res_mask=residue_mask,
    ca_mask=batch.ca_mask & residue_mask, atom_mask=atom_mask, valid_mask=batch.valid_mask & atom_mask)
model.eval()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    evaluated = model(batch)["v_atom"].float()
model.train()
trainer.seed_all(5)
with torch.autocast("cuda", dtype=torch.bfloat16):
    output = model(batch)
loss, metrics = engine.allatom_loss_surface(output, batch, alpha_mode=config["alpha_mode"],
    use_rigid_align=config["use_rigid_align"], w_fm=config["w_fm"], w_lddt_atom=config["w_lddt_atom"],
    lddt_cutoff_A=config["lddt_cutoff_A"], lddt_pair_chunk_size=config["lddt_pair_chunk_size"])
loss.backward()
norms = {}
for prefix in ("atom_encoder.", "residue_trunk.", "latent_residual.", "atom_decoder."):
    grads = [p.grad.float() for n, p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
    norms[prefix] = float(sum(g.square().sum() for g in grads).sqrt()) if grads else 0.0
out.mkdir(parents=True, exist_ok=False)
np.savez(out / "outputs.npz", eval_velocity=evaluated.cpu().numpy(),
    train_velocities=np.stack([v.detach().float().cpu().numpy() for v in output["v_atom_steps"]]),
    depths=output["latent_depths"].detach().float().cpu().numpy())
(out / "probe.json").write_text(json.dumps(dict(code=str(code), initial_model_sha256=initial,
    perturbed_model_sha256=perturbed, batch_sha256=batch_digest(batch), loss=float(loss.detach()),
    gradient_norms=norms, parameters=sum(p.numel() for p in model.parameters()),
    device=torch.cuda.get_device_name(), torch=str(torch.__version__)), indent=2, allow_nan=False) + "\n")
'''


def verify_equivalence(old, new, arrays, expected_initial):
    """Old frozen and current code must compute the same flat_fine6 function from the same tensors."""
    import numpy as np

    for key in ("initial_model_sha256", "perturbed_model_sha256", "batch_sha256", "parameters"):
        require(old.get(key) == new.get(key), f"old/current flat_fine6 {key} differs")
    require(old["initial_model_sha256"] == expected_initial and old["parameters"] == PARAMETER_COUNT,
            "flat_fine6 construction differs from the old seed20261023 run")
    require(old.get("code") != new.get("code"), "equivalence probe must compare two code trees")
    rows = {}
    for name in ("eval_velocity", "train_velocities", "depths"):
        first, second = arrays["old"][name], arrays["new"][name]
        require(first.shape == second.shape and np.isfinite(first).all() and np.isfinite(second).all(),
                f"flat_fine6 {name} shape or finiteness differs")
        scale, difference = float(np.abs(first).max()), float(np.abs(first - second).max())
        require(scale > 0, f"flat_fine6 {name} probe is trivially zero")
        require(difference <= 1e-3 * scale, f"old/current flat_fine6 {name} differs")
        rows[name] = dict(shape=list(first.shape), bitwise_equal=bool(np.array_equal(first, second)),
                          max_abs_difference=difference, max_abs_value=scale)
    require(proof.finite(old.get("loss"), positive=True) and proof.finite(new.get("loss"), positive=True)
            and math.isclose(old["loss"], new["loss"], rel_tol=1e-3), "old/current flat_fine6 LiFT loss differs")
    require(set(old.get("gradient_norms", {})) == set(new.get("gradient_norms", {})) == set(GROUPS)
            and all(proof.finite(old["gradient_norms"][key], positive=True)
                    and math.isclose(old["gradient_norms"][key], new["gradient_norms"][key], rel_tol=1e-2)
                    for key in GROUPS), "old/current flat_fine6 gradient groups differ")
    return dict(admitted=True, arrays=rows, loss=dict(old=old["loss"], new=new["loss"]),
                gradient_norms=dict(old=old["gradient_norms"], new=new["gradient_norms"]),
                tolerance="max|old-new| <= 1e-3 max|old| per output; loss rel1e-3; gradient-group norms rel1e-2")


def verify_preflight(value, protocol):
    """Construction pairing per seed; the seed20261023 points must equal both reused initial tensors."""
    rows = value.get("rows", {})
    require(set(rows) == {str(seed) for seed in SEEDS} and all(set(row) == set(ARCHITECTURES) for row in rows.values()),
            "preflight seed/architecture coverage differs")
    signatures = set()
    for seed, row in rows.items():
        for arch, entry in row.items():
            require(entry.get("parameters") == entry.get("trainable_parameters") == PARAMETER_COUNT
                    and type(entry["parameters"]) is int, "constructed parameter count differs")
            require(proof.valid_sha(entry.get("initial_model_sha256")) and proof.valid_sha(entry.get(
                "parameter_signature_sha256")), "constructed tensor identity missing")
            signatures.add(entry["parameter_signature_sha256"])
    require(len(signatures) == 1, "flat_fine6 and fixed2 parameter sets differ")
    first = rows[str(REPRODUCTION_SEED)]
    require(first["flat_fine6"]["initial_model_sha256"] == protocol["reproduction"]["expected_initial_model_sha256"],
            "current code does not rebuild the old flat_fine6 seed20261023 initial tensors")
    require(first["fixed2"]["initial_model_sha256"] == protocol["reused_fixed2"]["initial_model_sha256"],
            "current code does not rebuild the reused fixed2 seed20261023 initial tensors")
    for arch in ARCHITECTURES:
        require(len({rows[str(seed)][arch]["initial_model_sha256"] for seed in SEEDS}) == len(SEEDS),
                "seed does not change the constructed initial tensors")
    pairs = {str(seed): dict(identical_parameter_sets=True, identical_initial_tensors=
        rows[str(seed)]["flat_fine6"]["initial_model_sha256"] == rows[str(seed)]["fixed2"]["initial_model_sha256"],
        initial_model_sha256={arch: rows[str(seed)][arch]["initial_model_sha256"] for arch in ARCHITECTURES})
        for seed in SEEDS}
    require(value.get("pairs") == pairs, "recorded seed pairing differs from constructed tensors")
    require(isinstance(value.get("equivalence"), dict) and value["equivalence"].get("admitted") is True,
            "old/current flat_fine6 equivalence was not admitted")
    return pairs


def construction_config(context, arch, seed):
    arm = arm_name(arch, seed)
    if arm in context["configs"]:
        return dict(context["configs"][arm])
    return dict(context["configs"][arm_name(arch, SEEDS[1])], seed=seed)


def preflight_path(run):
    return Path(run) / "preflight_verified.json"


def preflight(context):
    import numpy as np
    import torch
    from mambafold.model.bimamba3 import mamba3_runtime_info
    from mambafold.train import trainer

    run, protocol = context["run"], context["protocol"]
    probes = run / "preflight/equivalence"
    base.no_outputs([preflight_path(run), probes])
    device = base.hardware(training=True)
    torch.set_num_threads(1)
    references = admitted_references(context)
    rows = {}
    with scratch.forbid_checkpoint_loads():
        for seed in SEEDS:
            for arch in ARCHITECTURES:
                trainer.seed_all(seed)
                model = trainer.build_model(construction_config(context, arch, seed), "cuda")
                signature = [(name, list(value.shape), str(value.dtype)) for name, value in model.state_dict().items()]
                rows.setdefault(str(seed), {})[arch] = dict(
                    initial_model_sha256=base.audit.tensor_digest(model.state_dict().items()),
                    parameters=sum(p.numel() for p in model.parameters()),
                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    parameter_signature_sha256=hashlib.sha256(json.dumps(signature).encode()).hexdigest(),
                    runtime=mamba3_runtime_info(model))
                del model
                gc.collect()
                torch.cuda.empty_cache()
    require(base.hardware(training=True) == device, "preflight changed physical GPU")
    flat = protocol["reference_flat_fine6"]
    config = Path(protocol["templates"]["flat_fine6"]["path"])
    probe, arrays = {}, {}
    for label, code in (("old", Path(flat["code"])), ("new", run / "code")):
        subprocess.run([sys.executable, "-c", EQUIVALENCE_SCRIPT, str(code), str(config), str(probes / label)],
                       cwd=code, env=dict(os.environ, PYTHONPATH=f"{code}:{code / 'src'}"), check=True)
        probe[label] = read(probes / label / "probe.json")
        with np.load(probes / label / "outputs.npz", allow_pickle=False) as packed:
            arrays[label] = {name: packed[name] for name in packed.files}
    require(base.hardware(training=True) == device, "equivalence probe changed physical GPU")
    equivalence = verify_equivalence(probe["old"], probe["new"], arrays, flat["initial_model_sha256"])
    value = dict(admitted=True, **base.context_binding(context), hardware=device, rows=rows, equivalence=equivalence,
        pairs={str(seed): dict(identical_parameter_sets=True, identical_initial_tensors=rows[str(seed)]["flat_fine6"]
            ["initial_model_sha256"] == rows[str(seed)]["fixed2"]["initial_model_sha256"],
            initial_model_sha256={arch: rows[str(seed)][arch]["initial_model_sha256"] for arch in ARCHITECTURES})
            for seed in SEEDS},
        old_runtime_equal={arch: rows[str(REPRODUCTION_SEED)][arch]["runtime"] == references[arch]["runtime"]
                           for arch in ARCHITECTURES},
        reference10k_admission_sha256=sha(curve.reference_path(run)),
        reference_flat_admission_sha256=sha(flat_reference_path(run)),
        evidence_sha256={base.relative(run, path): sha(path) for label in probe
                         for path in (probes / label / "probe.json", probes / label / "outputs.npz")})
    verify_preflight(value, protocol)
    unchanged(context)
    write_new(preflight_path(run), value)


def admitted_preflight(context):
    source = owner(context)
    run = source["run"]
    value = read(preflight_path(run))
    require(source is context or sha(preflight_path(run)) == context["protocol"]["training_run"]
            ["preflight_admission_sha256"], "training-run preflight admission changed")
    exact(value, dict(admitted=True, **base.context_binding(source),
        reference10k_admission_sha256=sha(curve.reference_path(run)),
        reference_flat_admission_sha256=sha(flat_reference_path(run))), "preflight admission binding differs")
    base.validate_hardware(value.get("hardware"), training=True)
    base.seals(run, value.get("evidence_sha256"))
    verify_preflight(value, context["protocol"])
    return value


# ── training ────────────────────────────────────────────────────────────


def reproduction_inputs(protocol):
    home = Path(protocol["reference_flat_fine6"]["run"]) / "training/flat_fine6"
    delivered, digest = inputs.read_inputs(home / "audit/inputs.jsonl", 4 * STEP)
    require(digest == protocol["reproduction"]["expected_input_stream_sha256"], "old flat_fine6 input audit changed")
    return delivered[:8]


def check_native(native, gate, arm, old_inputs=None):
    """Fail before formal training when fresh weights or the reproduced data stream differ."""
    seed, arch = SEED_OF[arm], ARCHITECTURE_OF[arm]
    require(native.get("arm") == arm and native.get("seed") == seed, "native receipt arm/seed differs")
    require(native.get("model_sha256") == gate["rows"][str(seed)][arch]["initial_model_sha256"],
            "native fresh weights differ from the admitted seed/architecture construction")
    if old_inputs is not None:
        require([row.get("input_sha256") for row in native.get("rows", [])] == list(old_inputs),
                "first eight native inputs differ from the old flat_fine6 data stream")


def train(context, arm):
    require(arm in ARMS, "train requires a declared seed-replicate arm")
    gate = admitted_preflight(context)
    admitted_references(context)
    old_inputs = reproduction_inputs(context["protocol"]) if arm == REPRODUCTION_ARM else None
    native = tail.native_preflight if ARCHITECTURE_OF[arm] == "fixed2" else _BASE_PREFLIGHT

    def gated(args):
        native(args)
        check_native(read(args.out / "native_verified.json"), gate, arm, old_inputs)

    original = scratch.preflight
    scratch.preflight = gated
    try:
        with bindings(), tail.policy_recording() if ARCHITECTURE_OF[arm] == "fixed2" else nullcontext():
            base.train(context, arm)
    finally:
        scratch.preflight = original


def training_home(context, arm):
    return Path(context["protocol"]["arms"][arm]["training"])


def verify_flat_construction(config, native):
    """flat_fine6 is non-recycling by value: absent keys and the trainer's written defaults are equivalent."""
    require(all(value.get(key, default) == default for value in (config, native)
                for key, default in FLAT_DEFAULTS.items()),
            "flat_fine6 must keep the original non-recycling construction")


def verify_actual_training(context, arm):
    source = owner(context)
    with bindings():
        saved, proof.MODEL_FIELDS = proof.MODEL_FIELDS, fields(arm)
        try:
            config, complete, metrics, evidence = _BASE_VERIFY(source, arm)
        finally:
            proof.MODEL_FIELDS = saved
    run, home = source["run"], training_home(context, arm)
    initial = read(home / "initialization.json")
    gate = admitted_preflight(context)
    require(initial["target_sha256"] == gate["rows"][str(SEED_OF[arm])][ARCHITECTURE_OF[arm]]["initial_model_sha256"],
            "formal initial tensors differ from the admitted construction")
    if arm == REPRODUCTION_ARM:
        target = context["protocol"]["reproduction"]
        require(initial["target_sha256"] == target["expected_initial_model_sha256"]
                and complete["input_stream_sha256"] == target["expected_input_stream_sha256"],
                "flat_fine6 seed20261023 does not reproduce the old initial tensors and input stream")
    native = read(Path(context["protocol"]["arms"][arm]["preflight"]) / "native_verified.json")
    if ARCHITECTURE_OF[arm] == "fixed2":
        exact(native, dict(recycle_policy="fixed2", atom_recycling="carry",
            model_fields={key: config[key] for key in RECYCLE_FIELDS}), "native fixed2 model identity differs")
        for index, row in enumerate(native["rows"]):
            loops = scratch.actual_training_loops(config, index // 4)
            exact(row, dict(observed_calls=screen.expected_call_counts(loops, atom_recycling="carry", training=True),
                supervised_steps=list(tail.supervised_steps("fixed2", loops))), "native recycle execution differs")
        path, policy_path = tail.semantic_path(context, arm), home / "audit/recycle_policy.jsonl"
        tail.verify_semantics(read(path), initial["target_sha256"])
        policy = tail.verify_policy_audit(policy_path, "fixed2", config["seed"])
        require(policy["depth_counts"] == complete["depth_counts"], "policy sequence and completed K counts differ")
        for item in (path, policy_path):
            evidence[base.relative(run, item)] = sha(item)
        complete = dict(complete, recycle_policy_audit=policy)
    else:
        verify_flat_construction(config, native)
    evidence["preflight_verified.json"] = sha(preflight_path(run))
    return config, complete, metrics, evidence


def admitted_training(context, arm):
    source = owner(context)
    if source is context:
        with bindings():
            value = _BASE_ADMITTED(context, arm)
    else:
        value = followup_receipt(context, arm)
    run, home = source["run"], training_home(context, arm)
    require(value["evidence_sha256"].get("preflight_verified.json") == sha(preflight_path(run)),
            "preflight admission not sealed into training")
    initial = read(home / "initialization.json")
    gate = admitted_preflight(context)
    require(initial["target_sha256"] == gate["rows"][str(SEED_OF[arm])][ARCHITECTURE_OF[arm]]["initial_model_sha256"],
            "admitted initial tensors differ from the admitted construction")
    if ARCHITECTURE_OF[arm] == "fixed2":
        for path in (tail.semantic_path(context, arm), home / "audit/recycle_policy.jsonl"):
            require(value["evidence_sha256"].get(base.relative(run, path)) == sha(path), "fixed2 audit not sealed")
        tail.verify_semantics(read(tail.semantic_path(context, arm)), initial["target_sha256"])
        require(value["complete"].get("recycle_policy_audit") == tail.verify_policy_audit(
            home / "audit/recycle_policy.jsonl", "fixed2", value["config"]["seed"]), "admitted policy sequence differs")
    return value


# ── follow-up: admit the bound run's finished trainings without retraining ──


def parse_queue(text, job):
    """Unfinished tasks of array `job` in `squeue -h -o %F|%i|%T` output, pending ranges included; a COMPLETING
    task has already exited its batch script and only runs the epilog, so it no longer writes training files."""
    rows = {}
    for line in text.splitlines():
        if line.strip():
            array, task, state = line.split("|")
            if array == job and state != "COMPLETING":
                rows[task] = state
    return rows


def queued_training_tasks(job):
    """Compute nodes cannot reach Slurm accounting here, so an ended task is one the controller no longer queues."""
    import getpass

    completed = subprocess.run(["squeue", "-h", "-u", getpass.getuser(), "-o", "%F|%i|%T"],
                               check=True, capture_output=True, text=True)
    return parse_queue(completed.stdout, job)


def known_flat_failure(log, original):
    """True only when training finished and its own admission stopped at the presence-based flat_fine6 check."""
    lines = [line.rstrip() for line in Path(log).read_text(errors="replace").splitlines() if line.strip()]
    marker = "Traceback (most recent call last):"
    if lines.count(marker) != 1 or "Done. Total steps: 10000" not in lines:
        return False
    tail_lines = lines[lines.index(marker):]
    code = Path(original) / "code/benchmarks"
    frames = [line.strip() for line in tail_lines if line.strip().startswith("File ")]
    return tail_lines[-1] == FLAT_FAILURE and frames[-3:] == [
        f'File "{code / "compare_scratch_hierarchy_fine6.py"}", line 296, in train',
        f'File "{code / "compare_seed_replicates.py"}", line 924, in verify_actual_training',
        f'File "{code / "compare_seed_replicates.py"}", line 87, in require']


def training_ended(context, arm, queued):
    """The whole training array left the queue and this arm ended with its own receipt or at the flat_fine6 bug."""
    original, job = owner(context)["run"], context["protocol"]["training_run"]["training_job"]
    require(not queued, f"training array {job} still has queued or running tasks: {sorted(queued)}")
    task = f"{job}_{list(ARMS).index(arm)}"
    log, path = original / f"logs/train-{task}.out", base.training_receipt_path(original, arm)
    require(regular(log), f"training task {task} ({arm}) never started or lacks its log")
    if path.exists() or path.is_symlink():
        require(regular(path), f"training {arm} admission is not a regular file")
        termination, receipt = COMPLETED_TRAINING, read(path)
    else:
        require(ARCHITECTURE_OF[arm] == "flat_fine6" and known_flat_failure(log, original),
                f"training {arm} ended without its own admission for a reason other than the flat_fine6 check")
        termination, receipt = FAILED_AT_FLAT_CHECK, None
    return dict(task=task, termination=termination, log=base.relative(original, log), log_sha256=sha(log)), receipt


def validate_training_hardware(value):
    require(isinstance(value, dict) and value.get("name") == base.TRAIN_GPU, "training GPU class differs")
    if value.get("uuid") is None:
        exact(value, dict(source=UNRECORDED_GPU), "unrecorded training GPU provenance differs")
        require(isinstance(value.get("slurm_job"), str) and value["slurm_job"],
                "unrecorded training GPU lacks its Slurm task")
    else:
        base.validate_hardware(value, training=True)


def training_admission(context, arm, queued):
    """Re-run the complete fixed training admission on the bound run's finished arm; the bound run is only read."""
    source = owner(context)
    require(source is not context, "training admission of another run requires a follow-up run")
    job, receipt = training_ended(context, arm, queued)
    run, entry = source["run"], source["protocol"]["arms"][arm]
    home, preflight = Path(entry["training"]), Path(entry["preflight"])
    require(regular(home / "ckpt_0010000.pt"), f"{arm} lacks a regular ckpt_0010000.pt")
    config, complete, metrics, evidence = verify_actual_training(context, arm)
    gt = read(home / "ground_truth_verified.json")
    exact(complete, inputs.BUDGET, "incomplete applied training budget")
    exact(gt, dict(arm=arm, optimizer_updates=STEP, microbatches=4 * STEP, final_schedule_index=STEP,
                   checkpoint_loads_during_training=0, device=base.TRAIN_GPU), "training ground truth differs")
    for path in (preflight / "initial_ema.pt", home / "initial_ema.pt", home / "ckpt_0010000.pt"):
        evidence[base.relative(run, path)] = sha(path)
    require(evidence[base.relative(run, home / "ckpt_0010000.pt")] == gt["final_checkpoint_sha256"],
            "final checkpoint bytes differ")
    previous, weighted = 0, 0.0
    for row in metrics:
        require(proof.finite(row.get("perf_train_step_ms_max"), positive=True), "invalid step timer")
        weighted += row["perf_train_step_ms_max"] * (row["step"] - previous)
        previous = row["step"]
    require(previous == STEP, "incomplete timing windows")
    value = dict(admitted=True, arm=arm, **base.context_binding(context), training_binding=base.context_binding(source),
                 parameters=PARAMETER_COUNT, nominal_K=2, config=config, complete=complete,
                 all_update_weighted_mean_step_ms=weighted / previous,
                 final_checkpoint_sha256=gt["final_checkpoint_sha256"], final_ema_sha256=gt["final_ema_sha256"],
                 evidence_sha256=evidence, training_job=job)
    if receipt is None:
        value.update(hardware=dict(name=gt["device"], uuid=None, slurm_job=job["task"], source=UNRECORDED_GPU),
                     original_training_admission_sha256=None)
    else:
        exact(receipt, dict(admitted=True, arm=arm, **base.context_binding(source)),
              "original training admission identity differs")
        keys = ("parameters", "nominal_K", "config", "complete", "all_update_weighted_mean_step_ms",
                "final_checkpoint_sha256", "final_ema_sha256", "evidence_sha256")
        require({key: receipt.get(key) for key in keys} == {key: value[key] for key in keys},
                "original training admission differs from the re-verified training")
        value.update(hardware=receipt["hardware"],
                     original_training_admission_sha256=sha(base.training_receipt_path(run, arm)))
    validate_training_hardware(value["hardware"])
    base.seals(run, evidence)
    return value


def followup_receipt(context, arm):
    source = owner(context)
    run, home = source["run"], training_home(context, arm)
    value = read(base.training_receipt_path(context["run"], arm))
    exact(value, dict(admitted=True, arm=arm, parameters=PARAMETER_COUNT, nominal_K=2, **base.context_binding(context)),
          "follow-up training admission identity differs")
    require(value.get("training_binding") == base.context_binding(source), "training-run binding differs")
    base.seals(run, value.get("evidence_sha256"))
    config, complete = read(home / "config.json"), read(home / "audit/complete.json")
    require(value.get("config") == config and all(value["complete"].get(k) == v for k, v in complete.items()),
            "training admission differs from actual training")
    exact(complete, inputs.BUDGET, "incomplete admitted training budget")
    with registry():
        scratch.validate_config(config, arm)
        exact(config, expected(arm), "actual scratch configuration differs")
    gt = read(home / "ground_truth_verified.json")
    require(value.get("final_checkpoint_sha256") == gt["final_checkpoint_sha256"]
            == value["evidence_sha256"].get(base.relative(run, home / "ckpt_0010000.pt"))
            and value.get("final_ema_sha256") == gt["final_ema_sha256"], "own checkpoint/EMA binding differs")
    job = value.get("training_job", {})
    require(job.get("task") == f'{context["protocol"]["training_run"]["training_job"]}_{list(ARMS).index(arm)}'
            and job.get("termination") in (COMPLETED_TRAINING, FAILED_AT_FLAT_CHECK)
            and sha(base.local_path(run, job.get("log", ""), regular=True)) == job.get("log_sha256"),
            "training job provenance differs")
    validate_training_hardware(value.get("hardware"))
    require((job["termination"] == COMPLETED_TRAINING) == (value["hardware"].get("uuid") is not None)
            == (value.get("original_training_admission_sha256") is not None), "training hardware provenance differs")
    if value["original_training_admission_sha256"] is not None:
        require(sha(base.training_receipt_path(run, arm)) == value["original_training_admission_sha256"],
                "original training admission changed")
    for key in ("steady_mean_step_ms", "peak_allocated_gib", "peak_reserved_gib"):
        require(proof.finite(complete.get(key), positive=True), "invalid measured training cost")
    require(proof.finite(value.get("all_update_weighted_mean_step_ms"), positive=True),
            "invalid full-update training timer")
    return value


def admit_training(context):
    """All five arms or nothing: no receipt is written unless every bound training is finished and admitted."""
    run = context["run"]
    require(owner(context) is not context, "admit-training requires a follow-up run")
    base.no_outputs([base.training_receipt_path(run, arm) for arm in ARMS])
    admitted_preflight(context)
    admitted_references(context)
    queued = queued_training_tasks(context["protocol"]["training_run"]["training_job"])
    values = {arm: training_admission(context, arm, queued) for arm in ARMS}
    unchanged(context)
    for arm, value in values.items():
        write_new(base.training_receipt_path(run, arm), value)
    for arm in ARMS:
        admitted_training(context, arm)
    return values


# ── common inference, canonical scoring ────────────────────────────────


def verify_inference(context, arm, checkpoint_sha):
    directory = base.evaluation(context["run"], arm)
    recycling = ARCHITECTURES[ARCHITECTURE_OF[arm]].get("atom_recycling", "none")
    measured = proof.verify_inference(directory, STEP, [DEPTH], checkpoint_sha, context["targets"], False,
                                      atom_recycling=recycling, n_steps=SAMPLER["n_steps"])
    for row in measured:
        if recycling == "none":
            verify_flat_construction(row, {})
        else:
            exact(row, dict(recycle_policy="fixed2"), "fixed2 inference policy differs")
        require(proof.finite(row.get("peak_reserved_gib"), positive=True), "missing reserved inference memory")
    return measured


def seed_pairing(context, receipts):
    """Per-seed initial-tensor and input-stream identities, including both reused seed20261023 points."""
    protocol, result = context["protocol"], {}
    for seed in SEEDS:
        initial = {arch: read(training_home(context, arm_name(arch, seed)) / "initialization.json")["target_sha256"]
                   for arch in ARCHITECTURES if arm_name(arch, seed) in ARMS}
        streams = {arch: receipts[arm_name(arch, seed)]["complete"]["input_stream_sha256"]
                   for arch in ARCHITECTURES if arm_name(arch, seed) in ARMS}
        if seed == REPRODUCTION_SEED:
            initial["fixed2_reused"] = protocol["reused_fixed2"]["initial_model_sha256"]
            streams["fixed2_reused"] = protocol["reused_fixed2"]["input_stream_sha256"]
            initial["flat_fine6_oldcode"] = protocol["reproduction"]["expected_initial_model_sha256"]
            streams["flat_fine6_oldcode"] = protocol["reproduction"]["expected_input_stream_sha256"]
        result[str(seed)] = dict(initial_model_sha256=initial, input_stream_sha256=streams,
            identical_initial_tensors=len(set(initial.values())) == 1,
            identical_input_streams=len(set(streams.values())) == 1)
    return result


def infer(context):
    import torch
    from mambafold.model.bimamba3 import mamba3_runtime_info

    run = context["run"]
    base.no_outputs([run / "inference_admitted.json", *(base.evaluation(run, arm) for arm in ARMS)])
    device = base.hardware(training=True)
    torch.set_num_threads(1)
    admitted_preflight(context)
    admitted_references(context)
    admitted = {arm: admitted_training(context, arm) for arm in ARMS}
    streams, runtimes = {}, {}
    runtime = mamba3_runtime_info()
    for arm in ARMS:
        verify_actual_training(context, arm)
        home = training_home(context, arm)
        runtimes[arm] = read(home / "initialization.json")["mamba3_runtime"]
        base.verify_external_runtime(runtimes[arm], runtime)
        streams[arm], digest = inputs.read_inputs(home / "audit/inputs.jsonl", 4 * STEP)
        require(digest == admitted[arm]["complete"]["input_stream_sha256"], "actual input stream digest differs")
    for seed in SEEDS[1:]:
        require(streams[arm_name("flat_fine6", seed)] == streams[arm_name("fixed2", seed)],
                "paired seed input streams are not byte-identical")
    pairing = seed_pairing(context, admitted)
    evidence, arms = {}, {}
    for arm in ARMS:
        home, directory = training_home(context, arm), base.evaluation(run, arm)
        require(base.hardware(training=True) == device, "common physical inference GPU changed")
        with base.inference_model_runtime(runtimes[arm]):
            screen.infer(SimpleNamespace(checkpoint=home / "ckpt_0010000.pt", panel=Path(context["protocol"]["panel"]),
                                        out=directory, depths=[DEPTH], n_steps=SAMPLER["n_steps"]))
        require(base.hardware(training=True) == device, "common physical inference GPU changed")
        verify_inference(context, arm, admitted[arm]["final_checkpoint_sha256"])
        arms[arm] = dict(hardware=device, nominal_K=2, checkpoint_sha256=admitted[arm]["final_checkpoint_sha256"],
                         ema_sha256=admitted[arm]["final_ema_sha256"])
        write_new(directory / "gpu_binding.json", arms[arm])
        paths = [directory / name for name in ("sampling.json", "hardware.json", "gpu_binding.json")]
        paths.append(directory / f"K{DEPTH}/pairs/manifest.json")
        paths.extend(directory / f"K{DEPTH}/pairs/{target}_seed0_pred.pdb" for target in context["targets"])
        evidence.update({base.relative(run, path): sha(path) for path in paths})
        gc.collect()
        torch.cuda.empty_cache()
    unchanged(context)
    write_new(run / "inference_admitted.json", dict(admitted=True, **base.context_binding(context), hardware=device,
        sampler=SAMPLER, inference_depth=DEPTH, coverage=len(context["targets"]), arms=arms, seeds=pairing,
        preflight_admission_sha256=sha(preflight_path(owner(context)["run"])),
        training_receipts={arm: sha(base.training_receipt_path(run, arm)) for arm in ARMS},
        input_comparison=dict(method="byte_exact_within_seed", microbatches_per_arm=4 * STEP),
        evidence_sha256=evidence))


def admitted_inference(context):
    run = context["run"]
    value = read(run / "inference_admitted.json")
    exact(value, dict(admitted=True, coverage=len(context["targets"]), sampler=SAMPLER, inference_depth=DEPTH,
        preflight_admission_sha256=sha(preflight_path(owner(context)["run"])), **base.context_binding(context)),
        "seed-replicate inference admission differs")
    require(set(value.get("arms", {})) == set(value.get("training_receipts", {})) == set(ARMS),
            "incomplete common inference arms")
    base.validate_hardware(value.get("hardware"), training=True)
    base.seals(run, value.get("evidence_sha256"))
    training = {}
    for arm in ARMS:
        require(sha(base.training_receipt_path(run, arm)) == value["training_receipts"][arm],
                "training receipt changed after inference")
        training[arm] = admitted_training(context, arm)
        binding = value["arms"][arm]
        exact(binding, dict(hardware=value["hardware"], nominal_K=2,
            checkpoint_sha256=training[arm]["final_checkpoint_sha256"], ema_sha256=training[arm]["final_ema_sha256"]),
            "inference own EMA/GPU binding differs")
        require(read(base.evaluation(run, arm) / "gpu_binding.json") == binding, "per-arm GPU binding differs")
        verify_inference(context, arm, binding["checkpoint_sha256"])
    require(value.get("seeds") == seed_pairing(context, training), "recorded seed pairing differs")
    return value


def score(context):
    with bindings():
        base.score(context)


# ── report ──────────────────────────────────────────────────────────────


def entries(targets):
    groups = {}
    for name, row in targets.items():
        groups.setdefault(row["entry_id"], []).append(name)
    return groups


def run_quality(targets, values):
    groups = entries(targets)
    return {metric: dict(chain_mean=statistics.fmean(values[name][metric] for name in sorted(targets)),
                         equal_entry_mean=statistics.fmean(statistics.fmean(values[name][metric] for name in names)
                                                           for _, names in sorted(groups.items())))
            for metric in QUALITY}


def spread(values):
    values = list(values)
    require(values and all(proof.finite(value) for value in values), "spread requires finite values")
    return dict(n=len(values), values=values, mean=statistics.fmean(values),
                sd=statistics.stdev(values) if len(values) > 1 else None, min=min(values), max=max(values))


PAIRS = {
    "20261024": (arm_name("fixed2", 20261024), arm_name("flat_fine6", 20261024),
                 dict(cross_code=False, same_inference_job=True)),
    "20261025": (arm_name("fixed2", 20261025), arm_name("flat_fine6", 20261025),
                 dict(cross_code=False, same_inference_job=True)),
    "20261023_oldcode_flat": (REUSED_FIXED2, OLD_FLAT, dict(cross_code=True, same_inference_job=False)),
    "20261023_newcode_flat": (REUSED_FIXED2, REPRODUCTION_ARM, dict(cross_code=False, same_inference_job=False)),
}
PAIR_SETS = {"seeds_20261024_20261025": ("20261024", "20261025"),
             "seeds_20261023oldcode_20261024_20261025": ("20261023_oldcode_flat", "20261024", "20261025"),
             "seeds_20261023newcode_20261024_20261025": ("20261023_newcode_flat", "20261024", "20261025")}
GROUPINGS = dict(flat_fine6=dict(new_code_seeds=tuple(arm_name("flat_fine6", seed) for seed in SEEDS)),
                 fixed2=dict(new_and_reused_seeds=(REUSED_FIXED2, *(arm_name("fixed2", seed) for seed in SEEDS[1:])),
                             new_code_seeds=tuple(arm_name("fixed2", seed) for seed in SEEDS[1:])))


def summarize(targets, vectors):
    """Descriptive seed statistics and paired flat_fine6-fixed2 effects; no selection or significance claim."""
    require(set(vectors) == set(POINTS), "seed-replicate report requires all new and reused points")
    for values in vectors.values():
        require(set(values) == set(targets) and all(set(row) == set(QUALITY) and all(proof.finite(v) for v in
                row.values()) for row in values.values()), "invalid or incomplete quality vectors")
    runs = {name: run_quality(targets, values) for name, values in vectors.items()}
    kinds = ("chain_mean", "equal_entry_mean")
    architectures = {arch: {label: dict(runs=list(names), **{metric: {kind: spread(runs[name][metric][kind]
                     for name in names) for kind in kinds} for metric in QUALITY}) for label, names in groups.items()}
                     for arch, groups in GROUPINGS.items()}
    architectures["flat_fine6"]["old_code_seed20261023"] = dict(run=OLD_FLAT, **runs[OLD_FLAT])
    paired = {label: dict(fixed2=first, flat_fine6=second, **flags, effects=curve.tally(
                  tail.paired_effects(targets, vectors[first], vectors[second])))
              for label, (first, second, flags) in PAIRS.items()}
    summaries = {name: dict(pairs=list(labels), **{metric: {kind: spread(paired[label]["effects"][metric]
                 [f"{kind}_delta"] for label in labels) for kind in kinds} for metric in QUALITY})
                 for name, labels in PAIR_SETS.items()}
    flat_sd = architectures["flat_fine6"]["new_code_seeds"]["lddt"]["chain_mean"]
    fixed_sd = architectures["fixed2"]["new_and_reused_seeds"]["lddt"]["chain_mean"]
    pooled = math.sqrt(((flat_sd["n"] - 1) * flat_sd["sd"] ** 2 + (fixed_sd["n"] - 1) * fixed_sd["sd"] ** 2)
                       / (flat_sd["n"] + fixed_sd["n"] - 2))
    gap = summaries["seeds_20261024_20261025"]["lddt"]["chain_mean"]["mean"]
    noise = dict(metric="lddt chain mean", flat_fine6_sd=flat_sd["sd"], fixed2_sd=fixed_sd["sd"],
                 pooled_within_architecture_sd=pooled, mean_paired_gap_seeds_20261024_20261025=gap,
                 gap_over_pooled_sd=gap / pooled if pooled > 0 else None,
                 between_architecture_mean_difference=flat_sd["mean"] - fixed_sd["mean"],
                 interpretation="descriptive only; two/three seeds give no calibrated test")
    reproduction = dict(new=REPRODUCTION_ARM, old=OLD_FLAT, effects=curve.tally(
        tail.paired_effects(targets, vectors[OLD_FLAT], vectors[REPRODUCTION_ARM])))
    return dict(runs=runs, architectures=architectures, paired_flat_fine6_minus_fixed2=paired,
                paired_delta_summaries=summaries, seed_noise=noise, reproduction_new_minus_old=reproduction)


def same_gpu(first, second):
    """None when either training GPU UUID was never recorded."""
    return None if first is None or second is None else first == second


def report(context):
    run, targets = context["run"], context["targets"]
    base.no_outputs([run / "comparison.json", run / "comparison.csv"])
    inference = admitted_inference(context)
    references = admitted_references(context)
    vectors, geometry, costs, identities, scorers = {}, {}, {}, {}, []
    for arm in ARMS:
        value = read(run / f"scores/{arm}.json")
        exact(value, dict(admitted=True, arm=arm, coverage=len(targets),
            inference_admission_sha256=sha(run / "inference_admitted.json"), **base.context_binding(context)),
            "score admission identity differs")
        directory = base.evaluation(run, arm)
        path = directory / f"K{DEPTH}/scores/summary.json"
        require(sha(path) == value["summary_sha256"] and sha(context["protocol"]["ost"]) == value["ost_executable_sha256"],
                "canonical score evidence changed")
        summary = read(path)
        vectors[arm] = base.verify_canonical(summary, directory, context)
        require(vectors[arm] == value["metrics"] and summary["openstructure"] == value["scorer"],
                "canonical vectors/scorer changed")
        scorers.append(value["scorer"])
        geometry[arm] = tail.geometry_diagnostics(directory, targets, vectors[arm], value["scorer"], DEPTH)
        receipt = admitted_training(context, arm)
        complete, measured = receipt["complete"], read(directory / "sampling.json")
        costs[arm] = dict(steady_train_ms=complete["steady_mean_step_ms"],
            all_update_train_ms=receipt["all_update_weighted_mean_step_ms"],
            train_allocated_gib=complete["peak_allocated_gib"], train_reserved_gib=complete["peak_reserved_gib"],
            inference_seconds=statistics.fmean(row["seconds"] for row in measured),
            inference_allocated_gib=max(row["peak_allocated_gib"] for row in measured),
            inference_reserved_gib=max(row["peak_reserved_gib"] for row in measured),
            training_gpu_uuid=receipt["hardware"].get("uuid"), inference_gpu_uuid=inference["hardware"]["uuid"])
        identities[arm] = dict(architecture=ARCHITECTURE_OF[arm], seed=SEED_OF[arm], code="new",
            initial_model_sha256=read(training_home(context, arm) / "initialization.json")["target_sha256"],
            input_stream_sha256=complete["input_stream_sha256"],
            checkpoint_sha256=receipt["final_checkpoint_sha256"], ema_sha256=receipt["final_ema_sha256"])
    flat, fixed = references["flat_fine6"], references["fixed2"]
    require(all(scorer == flat["scorer"] for scorer in scorers), "new and reused canonical scorers differ")
    vectors[OLD_FLAT], geometry[OLD_FLAT] = flat["per_target"], flat["geometry"]
    vectors[REUSED_FIXED2], geometry[REUSED_FIXED2] = fixed["per_target"], fixed["geometry"]
    old_costs = flat["costs"]
    costs[OLD_FLAT] = dict(steady_train_ms=old_costs["training"]["steady_mean_step_ms"],
        all_update_train_ms=old_costs["training"]["all_update_weighted_mean_step_ms"],
        train_allocated_gib=old_costs["training"]["peak_allocated_gib"],
        train_reserved_gib=old_costs["training"]["peak_reserved_gib"],
        inference_seconds=old_costs["inference"]["mean_seconds"],
        inference_allocated_gib=old_costs["inference"]["peak_allocated_gib"], inference_reserved_gib=None,
        training_gpu_uuid=flat["training_receipt"]["hardware"]["uuid"], inference_gpu_uuid=flat["hardware"]["uuid"])
    costs[REUSED_FIXED2] = dict({key: fixed["row"][key] for key in ("steady_train_ms", "all_update_train_ms",
        "train_allocated_gib", "train_reserved_gib", "inference_seconds", "inference_allocated_gib",
        "inference_reserved_gib")}, training_gpu_uuid=fixed["training_receipt"]["hardware"]["uuid"],
        inference_gpu_uuid=fixed["hardware"]["uuid"])
    identities[OLD_FLAT] = dict(architecture="flat_fine6", seed=REPRODUCTION_SEED, code="old",
        initial_model_sha256=flat["initial_model_sha256"], input_stream_sha256=flat["input_stream_sha256"],
        checkpoint_sha256=flat["checkpoint_sha256"], ema_sha256=flat["ema_sha256"])
    identities[REUSED_FIXED2] = dict(architecture="fixed2", seed=REPRODUCTION_SEED, code="reused_same_model_source",
        initial_model_sha256=context["protocol"]["reused_fixed2"]["initial_model_sha256"],
        input_stream_sha256=fixed["training_receipt"]["complete"]["input_stream_sha256"],
        checkpoint_sha256=fixed["checkpoint_sha256"], ema_sha256=fixed["ema_sha256"])
    totals = {name: {field: geometry[name]["totals"][field] for field in GEOMETRY} for name in POINTS}
    stats = summarize(targets, vectors)
    new, old = identities[REPRODUCTION_ARM], identities[OLD_FLAT]
    stats["reproduction_new_minus_old"].update(
        geometry_totals_delta={field: totals[REPRODUCTION_ARM][field] - totals[OLD_FLAT][field] for field in GEOMETRY},
        identical={key: new[key] == old[key] for key in ("initial_model_sha256", "input_stream_sha256", "ema_sha256")},
        same_training_gpu=same_gpu(costs[REPRODUCTION_ARM]["training_gpu_uuid"], costs[OLD_FLAT]["training_gpu_uuid"]),
        same_inference_gpu=costs[REPRODUCTION_ARM]["inference_gpu_uuid"] == costs[OLD_FLAT]["inference_gpu_uuid"],
        model_source_changes=context["protocol"]["model_source_changes_since_old_flat_fine6"])
    result = dict(benchmark_role=ROLE, sampler=SAMPLER, nominal_K=2, inference_depth=DEPTH, seeds=list(SEEDS),
        **base.context_binding(context), coverage=dict(targets=len(targets), entries=len(entries(targets))),
        panel_coverage=read(Path(context["protocol"]["reference10k"]["run"]) / "panel_admitted.json")["coverage"],
        metric_names=dict(AA="lddt", CA="bb_lddt", TM="tm_score"), points=list(POINTS), identities=identities,
        per_target=vectors, geometry_totals=totals, geometry_unit=geometry[REPRODUCTION_ARM]["unit"], costs=costs,
        seed_pairing=inference["seeds"], **stats, inference_hardware=inference["hardware"],
        reused_inference_hardware={OLD_FLAT: flat["hardware"], REUSED_FIXED2: fixed["hardware"]},
        reference10k_admission_sha256=sha(curve.reference_path(run)),
        reference_flat_admission_sha256=sha(flat_reference_path(run)),
        preflight_admission_sha256=sha(preflight_path(owner(context)["run"])), architecture_selected=False,
        limits=LIMITS)
    unchanged(context)
    write_new(run / "comparison.json", result)
    rows = [dict(run=name, architecture=identities[name]["architecture"], seed=identities[name]["seed"],
                 code=identities[name]["code"], coverage=len(vectors[name]),
                 **{metric: stats["runs"][name][metric]["chain_mean"] for metric in QUALITY},
                 **{f"{metric}_equal_entry": stats["runs"][name][metric]["equal_entry_mean"] for metric in QUALITY},
                 **totals[name], **{key: value for key, value in costs[name].items() if not key.endswith("uuid")})
            for name in POINTS]
    with (run / "comparison.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "reference", "preflight", "train", "admit-training", "infer",
                                          "score", "report"))
    parser.add_argument("--run", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--flat-reference", type=Path)
    parser.add_argument("--full4-reference", type=Path)
    parser.add_argument("--training-run", type=Path,
                        help="prepare a follow-up run that admits and evaluates this run's finished trainings")
    args = parser.parse_args()
    if args.stage == "prepare":
        if args.out is None or args.run is not None or args.arm is not None:
            parser.error("prepare requires --out and no --run/--arm")
        if args.training_run is not None:
            if args.flat_reference is not None or args.full4_reference is not None:
                parser.error("a follow-up reuses the training run's references; omit --flat/--full4-reference")
            print(json.dumps(prepare_followup(args.out, args.training_run)))
        else:
            print(json.dumps(prepare(args.out, flat_reference=args.flat_reference or DEFAULT_FLAT,
                                     full4_reference=args.full4_reference or DEFAULT_FULL4)))
        return
    if args.run is None or args.out is not None or args.training_run is not None:
        parser.error(f"{args.stage} requires --run and no --out/--training-run")
    if (args.stage == "train") != (args.arm is not None):
        parser.error("train requires exactly one --arm; other stages operate on all arms")
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("seed-replicate execution requires a Slurm allocation, including CPU stages")
    context = validate_run(args.run)
    followup = owner(context) is not context
    if args.stage in (("preflight", "train") if followup else ("admit-training",)):
        parser.error(f"{args.stage} is not a stage of this {'follow-up' if followup else 'training'} run")
    if args.stage == "train":
        train(context, args.arm)
    else:
        dict(reference=reference, preflight=preflight, infer=infer, score=score, report=report,
             **{"admit-training": admit_training})[args.stage](context)


if __name__ == "__main__":
    main()
