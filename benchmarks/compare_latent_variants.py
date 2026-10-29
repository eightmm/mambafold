"""Single-seed latent-usage screen around flat_fine6, joined to the hash-bound reused flat_fine6 seed-20261023 point.

Six scratch WSD10k arms differ from flat_fine6 only in the latent branch (compression, width, depth, coarse
hierarchy width). The baseline is never retrained: its training, inference and canonical scores are admitted by the
frozen evaluator of the seed-replicate follow-up run. Training arms are independent; inference and reporting
evaluate every arm that wrote its own training admission and list the others as excluded.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

# Both imports are torch-free; the ML stack is bound lazily so prepare can run on login.
from benchmarks import compare_seed_replicates as seedrep  # noqa: E402
from benchmarks import input_equivalence as inputs  # noqa: E402

SEED = 20261023
STEP, DEPTH = 10000, 2
SAMPLER = dict(method="sde", n_steps=500, tau=0.01, seed=0, K=2)
QUALITY = ("lddt", "bb_lddt", "tm_score")
GEOMETRY = ("model_clashes", "model_bad_bonds", "model_bad_angles")
BASELINE = "flat_fine6_s20261023"
BASELINE_PARAMETERS = 269123226
BASELINE_LATENT = dict(latent_compression=4, d_latent=512, n_latent_layers=6, latent_hierarchy=False,
                       latent_coarse_width=0)
LATENT_KEYS = tuple(BASELINE_LATENT)
ARMS = {
    "r4_w1024": dict(BASELINE_LATENT, d_latent=1024),
    "r8_w1024": dict(BASELINE_LATENT, latent_compression=8, d_latent=1024),
    "r16_w2048": dict(BASELINE_LATENT, latent_compression=16, d_latent=2048),
    # One width-1024 block is the only r8 layer count within +-5% of the baseline (two blocks are +6.0%;
    # width 704-736 at six blocks is +13-15% because compression/readout projections scale with width).
    "r8_w1024_pm": dict(BASELINE_LATENT, latent_compression=8, d_latent=1024, n_latent_layers=1),
    "hier_wide": dict(BASELINE_LATENT, latent_hierarchy=True, latent_coarse_width=1024),
    "fine_deep": dict(BASELINE_LATENT, n_latent_layers=12),
}
# Constructed by the frozen code on CPU before preparation; GPU preflight re-counts every arm.
PARAMETERS = dict(r4_w1024=376902682, r8_w1024=381101082, r16_w2048=839474970, r8_w1024_pm=261424282,
                  hier_wide=321192858, fine_deep=309153306)
PARAMETER_MATCHED = ("r8_w1024_pm",)
PARAMETER_TOLERANCE = 0.05
NOISE_BAND = 0.025
SHARED = dict(shared_trunk_layers=6)
TEMPLATE_KEYS = frozenset({"config", "out_dir", "wandb_name"})
ROLE = "single-seed latent-usage architecture screen;fixed-final-checkpoint;no checkpoint or sampler tuning"
BASELINE_FILES = dict(protocol_sha256="protocol.json", source_manifest_sha256="source_manifest.json",
    cpu_admission_sha256="cpu_admitted.json", comparison_sha256="comparison.json",
    inference_admission_sha256="inference_admitted.json", score_admission_sha256=f"scores/{BASELINE}.json",
    training_admission_sha256=f"train_admitted/{BASELINE}.json")
BASELINE_VERIFIER = "benchmarks/compare_seed_replicates.py"
EQUIVALENCE_ARRAYS = ("eval_velocity", "train_velocities", "depths")
EQUIVALENCE_FILES = ("probe.json", "outputs.npz")
# The reused training run's own current-code probe, which it admitted bitwise equal to its old-code probe.
EQUIVALENCE_REFERENCE = tuple(f"preflight/equivalence/new/{name}" for name in EQUIVALENCE_FILES)
EQUIVALENCE_LABELS = ("warmup", "new")
UNSEALED = ("source_manifest.json", "cpu_admitted.json", "reference_admitted.json", "preflight_verified.json")
TESTS = ("tests/test_latent_variants.py", "tests/test_latent_mamba.py", "tests/test_seed_replicates.py",
         "tests/test_scratch_hierarchy_fine6.py", "tests/test_scratch_architecture_screen.py",
         "tests/test_scratch_architecture_report.py", "tests/test_shared_trunk_audit.py")
DEFAULT_EVAL = ROOT / "out/seed-replicate-eval-20261027T192020Z"
MEMORY_UPDATES = 3
LIMITS = ("Single training seed per arm; the flat_fine6 seed SD from the three-seed replicate study is about .0115 "
    "all-atom lDDT, so |delta AA| < .025 is within seed noise. Scratch WSD10000 applied updates only; ranking may not "
    "transfer to longer schedules or the eight-protein DDP batch. The eligible15-target/13-entry panel has been "
    "reused across several architecture decisions and is not a clean held-out test. Only r8_w1024_pm is parameter "
    "matched (+-5%); the latent FLOP column is tokens x width^2 x layers relative to flat_fine6 and ignores "
    "projections and the unchanged full-width trunk. The reused baseline was trained in another job with "
    "unrecorded training GPU UUID and inferred in another job, possibly on another physical RTX 6000 Ada; its paired "
    "deltas are cross-job. Declared sequence-homology exclusion does not exclude remote homology, coordinate "
    "overlap or ESMC pretraining exposure. CA is canonical OST bb_lddt; geometry entries are unnormalised OST counts. "
    "Sampler times include instrumentation and final CPU copy. No automatic architecture selection.")

require, sha, regular = seedrep.require, seedrep.sha, seedrep.regular


def load(path):
    return json.loads(Path(path).read_text())


def bind():
    """Import the torch-dependent evaluator stack only for Slurm stages and tests, never for prepare."""
    if "base" in globals():
        return
    seedrep.bind()
    globals().update(base=seedrep.base, tail=seedrep.tail, scratch=seedrep.scratch, proof=seedrep.proof,
                     screen=seedrep.screen, read=seedrep.read, exact=seedrep.exact, write_new=seedrep.write_new,
                     LATENT_FIELDS=(*seedrep.proof.MODEL_FIELDS, "latent_coarse_width"))


# ── torch-free arithmetic and preparation ───────────────────────────────


def latent_cost(arch):
    """Latent tokens per residue x width^2 x blocks; the coarse stack has two blocks at L/(4r)."""
    fine = arch["n_latent_layers"] * arch["d_latent"] ** 2 / arch["latent_compression"]
    width = arch["latent_coarse_width"] or arch["d_latent"]
    return fine + (2 * width ** 2 / (4 * arch["latent_compression"]) if arch["latent_hierarchy"] else 0)


def flop_ratio(arch):
    return latent_cost(arch) / latent_cost(BASELINE_LATENT)


def parameter_ratio(arm):
    return PARAMETERS[arm] / BASELINE_PARAMETERS


def declared_arm(arm):
    return dict(architecture=dict(ARMS[arm]), expected_parameters=PARAMETERS[arm], nominal_K=DEPTH,
                latent_flop_ratio=flop_ratio(ARMS[arm]), parameter_ratio=parameter_ratio(arm),
                parameter_matched=arm in PARAMETER_MATCHED)


def verify_template(template):
    require(isinstance(template, dict) and template.get("seed") == SEED, "baseline recipe is not seed20261023")
    require("latent_coarse_width" not in template and all(template.get(key) == value for key, value in
            dict(BASELINE_LATENT, **SHARED).items() if key != "latent_coarse_width"),
            "baseline recipe is not the flat_fine6 latent construction")
    require(all(template.get(key, default) == default for key, default in seedrep.FLAT_DEFAULTS.items()),
            "baseline recipe is not non-recycling")
    require(template.get("resume") is None and template.get("initialize_refiner_from") is None
            and template.get("initialize_model_from_ema") is False and template.get("total_steps") == STEP,
            "baseline recipe is not a fresh scratch WSD10k recipe")


def arm_config(template, run, arm):
    return dict(template, **ARMS[arm], config=str(Path(run) / f"configs/{arm}.yaml"),
                out_dir=str(Path(run) / f"training/{arm}"), wandb_name=f"latent-variants-{arm}")


def baseline_binding(eval_run):
    """Every reused-baseline identity that later stages re-verify; reads files only."""
    eval_run = Path(eval_run).resolve()
    protocol = load(eval_run / "protocol.json")
    training = protocol.get("training_run", {})
    require(isinstance(training, dict) and training.get("run"), "reused evaluation is not a follow-up run")
    training_run = Path(training["run"]).resolve()
    entry = protocol["arms"][BASELINE]
    home = Path(entry["training"]).resolve()
    require(home == training_run / f"training/{BASELINE}", "reused baseline home is outside its training run")
    template = training_run / f"configs/{BASELINE}.yaml"
    for path in (*(eval_run / name for name in BASELINE_FILES.values()), template, home / "ckpt_0010000.pt",
                 home / "initialization.json", home / "audit/complete.json", home / "audit/inputs.jsonl"):
        require(regular(path), f"reused baseline lacks a regular {path}")
    manifest = load(training_run / "source_manifest.json")
    require(manifest.get(str(template.relative_to(training_run))) == sha(template),
            "reused baseline recipe is not sealed by its training run")
    comparison = load(eval_run / "comparison.json")
    noise = comparison["architectures"]["flat_fine6"]["new_code_seeds"]["lddt"]["chain_mean"]
    require(comparison.get("architecture_selected") is False and noise.get("n") == 3, "reused seed noise differs")
    initial = load(home / "initialization.json")
    require(initial.get("parameters") == BASELINE_PARAMETERS, "reused baseline parameter count differs")
    return dict(eval_run=str(eval_run), eval_code=str(eval_run / "code"), training_run=str(training_run),
                training_code=str(training_run / "code"), arm=BASELINE, training=str(home), seed=SEED,
                **{key: sha(eval_run / name) for key, name in BASELINE_FILES.items()},
                template=str(template), template_sha256=sha(template),
                initial_model_sha256=initial["target_sha256"],
                input_stream_sha256=load(home / "audit/complete.json")["input_stream_sha256"],
                parameters=BASELINE_PARAMETERS, seed_noise_lddt_chain_mean_sd=noise["sd"],
                seed_noise_runs=noise["n"])


def equivalence_binding(baseline):
    """Seal the historical current-code flat_fine6 probe; reads files only.

    Re-running the old code tree in a fresh process is not bitwise reproducible on the GPU (first-launch kernel
    selection), so the admitted historical probe, not a re-execution of old code, is the equivalence reference.
    """
    training_run = Path(baseline["training_run"]).resolve()
    sealed = load(Path(baseline["eval_run"]) / "protocol.json").get("training_run", {})
    admission = training_run / "preflight_verified.json"
    require(regular(admission) and sha(admission) == sealed.get("preflight_admission_sha256"),
            "historical preflight admission is not the one sealed by the reused evaluation")
    value = load(admission)
    equivalence = value.get("equivalence")
    require(value.get("admitted") is True and isinstance(equivalence, dict) and equivalence.get("admitted") is True
            and set(equivalence.get("arrays", {})) == set(EQUIVALENCE_ARRAYS)
            and all(row.get("bitwise_equal") is True for row in equivalence["arrays"].values()),
            "historical old/current flat_fine6 equivalence was not admitted bitwise equal")
    digests = {}
    for name in EQUIVALENCE_REFERENCE:
        path = training_run / name
        require(regular(path), f"historical equivalence lacks a regular {path}")
        digests[name] = sha(path)
        require(value.get("evidence_sha256", {}).get(name) == digests[name] == sealed.get("preflight", {}).get(name),
                f"historical equivalence {name} is not sealed by its admission and the reused evaluation")
    probe_path, outputs_path = (training_run / name for name in EQUIVALENCE_REFERENCE)
    probe = load(probe_path)
    require(Path(probe.get("code", "")).resolve() == training_run / "code"
            and probe.get("initial_model_sha256") == baseline["initial_model_sha256"]
            and probe.get("parameters") == BASELINE_PARAMETERS,
            "historical equivalence probe is not the reused flat_fine6 current-code probe")
    return dict(run=str(training_run), preflight_admission=str(admission), preflight_admission_sha256=sha(admission),
                probe=str(probe_path), probe_sha256=digests[EQUIVALENCE_REFERENCE[0]], outputs=str(outputs_path),
                outputs_sha256=digests[EQUIVALENCE_REFERENCE[1]], code=probe["code"], loss=probe["loss"])


VERIFY_FROZEN = r'''"""Check static latent-variant seals, the reused flat_fine6 baseline and equivalence-reference bindings and every arm recipe."""
import hashlib,json,sys
from pathlib import Path
import yaml
RUN=Path(__file__).resolve().parent
FILES=__FILES__
UNSEALED=__UNSEALED__

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
 return h.hexdigest()

def main():
 manifest=json.loads((RUN/'source_manifest.json').read_text())
 assert manifest and not set(manifest)&set(UNSEALED)
 for name,digest in manifest.items():
  p=RUN/name
  assert not Path(name).is_absolute() and '..' not in Path(name).parts
  assert p.resolve().is_relative_to(RUN.resolve()) and p.is_file() and not p.is_symlink(),name
  assert p.suffix not in ('.pt','.pth','.safetensors'),name
  assert sha(p)==digest,name
 p=json.loads((RUN/'protocol.json').read_text())
 assert p['seed']==20261023 and p['sampler']=={'method':'sde','n_steps':500,'tau':.01,'seed':0,'K':2}
 assert p['formal_applied_update_cap_per_arm']==10000 and p['microbatches_per_formal_run']==40000
 b=p['baseline'];old=Path(b['eval_run'])
 for key,name in FILES.items():assert sha(old/name)==b[key],key
 assert sha(b['template'])==b['template_sha256']
 e=p['equivalence_reference']
 for key in ('preflight_admission','probe','outputs'):assert sha(e[key])==e[key+'_sha256'],key
 template=yaml.safe_load(Path(b['template']).read_text())
 for arm,entry in p['arms'].items():
  c=yaml.safe_load(Path(entry['config']).read_text())
  assert c==dict(template,**entry['architecture'],config=entry['config'],out_dir=entry['training'],wandb_name='latent-variants-'+arm),arm
  assert c['resume'] is None and c['initialize_refiner_from'] is None and c['initialize_model_from_ema'] is False
  assert c['seed']==20261023 and c['total_steps']==10000
 assert Path(b['training'],'ckpt_0010000.pt').is_file()
 print('Latent-variant static seals, reused baseline binding and arm recipes verified',flush=True)

if __name__=='__main__':main()
'''

WRITE_CPU_ADMISSION = r'''"""Bind focused CPU checks to the latent-variant trial."""
import hashlib,json,os,sys
from pathlib import Path
import xml.etree.ElementTree as ET
assert os.environ.get('SLURM_JOB_ID'), 'Slurm allocation required'
run=Path(__file__).resolve().parent
code=run/'code';sys.path[:0]=[str(code),str(code/'src')]
from benchmarks import compare_latent_variants as trial
assert Path(trial.__file__).resolve()==code/'benchmarks/compare_latent_variants.py'
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
value=dict(admitted=True,source_manifest_sha256=hashlib.sha256((run/'source_manifest.json').read_bytes()).hexdigest(),tests=counts,panel_targets=len(targets),panel_entries=len({row['entry_id'] for row in targets.values()}),checks_sha256=hashlib.sha256((run/'checks.xml').read_bytes()).hexdigest())
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
export TMPDIR="/tmp/mf-latentvar-${{SLURM_JOB_ID}}-${{SLURM_ARRAY_TASK_ID:-stage}}"
mkdir -p "$TMPDIR"
"$PY" "$RUN/verify_frozen.py"
'''
    stage = '"$PY" "$CODE/benchmarks/compare_latent_variants.py" {} --run "$RUN"{}\n'
    cpu = 'export CUDA_VISIBLE_DEVICES=""\n'
    return {"cpu_admit.sh": common + cpu + 'cd "$CODE"\n"$PY" -m pytest -q ' + " ".join(TESTS)
                + ' --junitxml="$RUN/checks.xml"\n"$PY" "$RUN/write_cpu_admission.py"\n' + stage.format("reference", ""),
            "preflight.sh": common + stage.format("preflight", ""),
            "train.sh": common + f'ARMS=({" ".join(ARMS)})\nARM="${{ARMS[$SLURM_ARRAY_TASK_ID]}}"\n'
                + stage.format("train", ' --arm "$ARM"'),
            "infer.sh": common + stage.format("infer", ""),
            "score.sh": common + cpu + stage.format("score", "") + stage.format("report", "")}


def prepare(out, *, eval_run=DEFAULT_EVAL, stamp=None):
    """Freeze a new run with file operations only; every model/torch check happens later under Slurm."""
    baseline = baseline_binding(eval_run)
    template = yaml.safe_load(Path(baseline["template"]).read_text())
    verify_template(template)
    source = load(Path(baseline["eval_run"]) / "protocol.json")
    require(source.get("sampler") == SAMPLER and source.get("inference_depth") == DEPTH, "reused sampler differs")
    fields = source["arms"][BASELINE]["model_fields"]
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = Path(out).resolve() / f"latent-variants-scratch-wsd-{stamp}"
    run.mkdir(parents=True, exist_ok=False)
    frozen = seedrep.freeze_code(run)
    (run / "configs").mkdir()
    arms = {}
    for arm in ARMS:
        config = arm_config(template, run, arm)
        (run / f"configs/{arm}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        arms[arm] = dict(declared_arm(arm), config=config["config"], preflight=str(run / f"preflight/{arm}"),
                         training=config["out_dir"], u0=str(run / "u0"),
                         model_fields={key: config[key] for key in (*fields, "latent_coarse_width")})
    protocol = dict(created_at=datetime.now(timezone.utc).isoformat(),
        authority="User authorized a single-seed scratch WSD10k latent-usage screen of six arms around flat_fine6 "
                  "(one RTX 6000 Ada per arm), one common-GPU K2 SDE500 inference and canonical scoring; the "
                  "flat_fine6 seed20261023 baseline is reused by hash binding and never retrained.",
        benchmark_role=ROLE, seed=SEED, arms=arms, arm_order=list(ARMS), baseline=baseline,
        equivalence_reference=equivalence_binding(baseline),
        formal_applied_update_cap_per_arm=STEP, microbatches_per_formal_run=4 * STEP, preflight_optimizer_updates=0,
        teacher_models=0, all_parameters_trainable=True, sampler=dict(SAMPLER), inference_depth=DEPTH,
        maximum_concurrent_training_GPUs=len(ARMS), panel=source["panel"], panel_sha256=source["panel_sha256"],
        reference10k=source["reference10k"], ost=source["ost"], WSD=source["WSD"], expected_targets=15,
        expected_entries=13, noise_band_lddt=NOISE_BAND, parameter_tolerance=PARAMETER_TOLERANCE,
        latent_flop_definition="latent tokens per residue x width^2 x blocks relative to flat_fine6",
        model_source_changes_since_baseline=seedrep.source_drift(
            load(Path(baseline["training_run"]) / "source_manifest.json"), frozen),
        baseline_reproduction_gate="GPU preflight requires current code to rebuild the reused seed20261023 initial "
            "tensors and a forward/LiFT-backward equivalence probe on perturbed weights. The probe runs twice on "
            "current code in fresh processes; the first is a discarded warm-up recorded only as a diagnostic, and the "
            "second must match the sealed historical current-code probe of the reused training run "
            "(equivalence_reference, admitted there bitwise equal to its old-code probe) within the seed-replicate "
            "tolerance. Old code is not re-executed. Every arm's first eight native inputs and its admitted "
            "40000-microbatch stream must equal the baseline stream.",
        limits=LIMITS)
    manifest = freeze_run(run, protocol)
    return dict(run=str(run), frozen_files=len(manifest), arms=list(ARMS),
                source_manifest_sha256=sha(run / "source_manifest.json"),
                model_source_changes_since_baseline=protocol["model_source_changes_since_baseline"])


def freeze_run(run, protocol):
    (run / "protocol.json").write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
    (run / "verify_frozen.py").write_text(VERIFY_FROZEN.replace("__FILES__", repr(BASELINE_FILES))
                                          .replace("__UNSEALED__", repr(UNSEALED)))
    (run / "write_cpu_admission.py").write_text(WRITE_CPU_ADMISSION)
    for name, text in scripts(run).items():
        (run / name).write_text(text)
        (run / name).chmod(0o755)
    manifest = {str(path.relative_to(run)): sha(path) for path in sorted(run.rglob("*"))
                if path.is_file() and "__pycache__" not in path.parts}
    require(not set(manifest) & set(UNSEALED), "prepare sealed an admission artifact")
    with (run / "source_manifest.json").open("x") as stream:
        stream.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


# ── admitted contract ───────────────────────────────────────────────────


def variants():
    return {arm: dict(SHARED, **arch) for arm, arch in ARMS.items()}


@contextmanager
def registry():
    """Expose the six arms to the shared scratch verifiers with the coarse-width field; restore on exit."""
    old = scratch.VARIANTS, proof.VARIANTS, proof.MODEL_FIELDS
    scratch.VARIANTS, proof.VARIANTS, proof.MODEL_FIELDS = variants(), variants(), LATENT_FIELDS
    try:
        yield
    finally:
        scratch.VARIANTS, proof.VARIANTS, proof.MODEL_FIELDS = old


def expected(arm):
    require(arm in ARMS, "undeclared latent-variant arm")
    with registry():
        return scratch.expected_config(arm)


def validate_baseline(protocol):
    bound = protocol.get("baseline", {})
    eval_run = Path(bound.get("eval_run", ""))
    require(eval_run.is_absolute(), "reused baseline evaluation path must be absolute")
    exact(bound, baseline_binding(eval_run), "reused baseline binding changed")
    eval_manifest = read(eval_run / "source_manifest.json")
    verifier = eval_run / "code" / BASELINE_VERIFIER
    require(eval_manifest.get(str(verifier.relative_to(eval_run))) == sha(verifier),
            "reused baseline evaluator is not sealed")
    source = read(eval_run / "protocol.json")
    require(source.get("sampler") == SAMPLER and protocol.get("ost") == source.get("ost")
            and protocol.get("WSD") == source.get("WSD") and protocol.get("panel") == source.get("panel")
            and protocol.get("panel_sha256") == source.get("panel_sha256")
            and protocol.get("reference10k") == source.get("reference10k"),
            "sampler/scorer/WSD/panel differ from the reused baseline evaluation")
    return source


def validate_equivalence_reference(protocol):
    require(protocol.get("equivalence_reference") == equivalence_binding(protocol["baseline"]),
            "sealed historical equivalence reference changed")
    return protocol["equivalence_reference"]


def validate_protocol(run, protocol):
    run = Path(run).resolve()
    exact(protocol, dict(benchmark_role=ROLE, seed=SEED, formal_applied_update_cap_per_arm=STEP,
        microbatches_per_formal_run=4 * STEP, preflight_optimizer_updates=0, teacher_models=0,
        all_parameters_trainable=True, inference_depth=DEPTH, expected_targets=15, expected_entries=13,
        noise_band_lddt=NOISE_BAND, parameter_tolerance=PARAMETER_TOLERANCE, arm_order=list(ARMS),
        limits=LIMITS), "latent-variant budget/role/coverage differs")
    require(protocol.get("sampler") == SAMPLER, "sampler differs")
    exact(protocol["sampler"], SAMPLER, "typed sampler differs")
    require(set(protocol.get("arms", {})) == set(ARMS), "exactly the six latent-variant arms are required")
    validate_baseline(protocol)
    validate_equivalence_reference(protocol)
    template = yaml.safe_load(Path(protocol["baseline"]["template"]).read_text())
    verify_template(template)
    configs = {}
    for arm, entry in protocol["arms"].items():
        exact(entry, declared_arm(arm), "declared arm architecture/parameters/FLOP ratio differs")
        exact(entry["architecture"], ARMS[arm], "typed arm architecture differs")
        for key, path in dict(config=run / f"configs/{arm}.yaml", preflight=run / f"preflight/{arm}",
                              training=run / f"training/{arm}", u0=run / "u0").items():
            require(Path(entry[key]).resolve() == path, f"unexpected {arm} {key} path")
        config = yaml.safe_load(Path(entry["config"]).read_text())
        require(config == arm_config(template, run, arm), "arm recipe differs from flat_fine6 outside its latent keys")
        with registry():
            scratch.validate_config(config, arm)
        exact(config, expected(arm), "typed latent-variant configuration differs")
        require(set(entry.get("model_fields", {})) == set(LATENT_FIELDS), "every typed model field is required")
        exact(entry["model_fields"], {key: expected(arm)[key] for key in LATENT_FIELDS}, "declared model fields differ")
        configs[arm] = config
    return configs


def validate_panel(protocol):
    full4 = Path(protocol["reference10k"]["run"]).resolve()
    require(Path(protocol["panel"]).resolve() == full4 / "panel"
            and protocol["panel_sha256"] == read(full4 / "panel_admitted.json").get("panel_sha256")
            == sha(full4 / "panel/panel.json")
            and protocol["reference10k"].get("panel_admission_sha256") == sha(full4 / "panel_admitted.json"),
            "panel must be the reused three-way panel")
    targets = base.validate_panel(full4, read(full4 / "protocol.json"))
    require(len(targets) == protocol["expected_targets"] and len({row["entry_id"] for row in targets.values()})
            == protocol["expected_entries"], "latent-variant panel must contain fifteen targets and thirteen entries")
    return targets


def validate_run(run):
    bind()
    run = Path(run).resolve()
    manifest = read(run / "source_manifest.json")
    require(not set(manifest) & set(UNSEALED), "manifest cannot seal itself or its admissions")
    base.seals(run, manifest)
    for path in (Path(__file__), Path(seedrep.__file__), Path(seedrep.curve.__file__), Path(tail.__file__),
                 Path(tail.atom.__file__), Path(base.__file__), Path(scratch.__file__), Path(proof.__file__),
                 Path(screen.__file__), Path(base.audit.__file__), Path(inputs.__file__),
                 Path(base.canonical.__file__), run / "protocol.json"):
        require(manifest.get(base.relative(run, path)) == sha(path), "unsealed latent-variant dependency")
    exact(read(run / "cpu_admitted.json"), dict(admitted=True, source_manifest_sha256=sha(run / "source_manifest.json")),
          "CPU admission binding differs")
    protocol = read(run / "protocol.json")
    configs = validate_protocol(run, protocol)
    for entry in protocol["arms"].values():
        require(manifest.get(base.relative(run, entry["config"])) == sha(entry["config"]), "unsealed arm recipe")
    old = read(Path(protocol["baseline"]["training_run"]) / "source_manifest.json")
    require(protocol.get("model_source_changes_since_baseline") == seedrep.source_drift(old, manifest),
            "recorded model source drift differs")
    return dict(run=run, protocol=protocol, manifest=manifest, configs=configs, targets=validate_panel(protocol),
                source_manifest_sha256=sha(run / "source_manifest.json"), protocol_sha256=sha(run / "protocol.json"),
                panel_admission_sha256=protocol["reference10k"]["panel_admission_sha256"])


def unchanged(context):
    fresh = validate_run(context["run"])
    require(base.context_binding(fresh) == base.context_binding(context) and fresh["targets"] == context["targets"],
            "latent-variant sources/protocol/panel changed")


# ── reused baseline ─────────────────────────────────────────────────────


REFERENCE_SCRIPT = r'''
import math, sys
from pathlib import Path
code, run, destination = (Path(value).resolve() for value in sys.argv[1:4])
sys.path[:0] = [str(code), str(code / "src")]
from benchmarks import compare_seed_replicates as trial
assert Path(trial.__file__).resolve() == code / "benchmarks/compare_seed_replicates.py"
arm = "flat_fine6_s20261023"
context = trial.validate_run(run)
inference, receipt = trial.admitted_inference(context), trial.admitted_training(context, arm)
binding = trial.base.context_binding(context)
comparison = trial.read(run / "comparison.json")
trial.exact(comparison, dict(architecture_selected=False, nominal_K=2, sampler=trial.SAMPLER, **binding),
    "reused report context differs")
trial.require(comparison["inference_hardware"] == inference["hardware"], "reused report hardware differs")
admission = trial.read(run / f"scores/{arm}.json")
trial.exact(admission, dict(admitted=True, arm=arm, coverage=len(context["targets"]),
    inference_admission_sha256=trial.sha(run / "inference_admitted.json"), **binding), "reused score admission differs")
directory = trial.base.evaluation(run, arm)
path = directory / "K2/scores/summary.json"
trial.require(trial.sha(path) == admission["summary_sha256"]
    and trial.sha(context["protocol"]["ost"]) == admission["ost_executable_sha256"], "reused score evidence changed")
values = trial.base.verify_canonical(trial.read(path), directory, context)
trial.require(values == admission["metrics"] == comparison["per_target"][arm], "reused canonical vectors differ")
runs = trial.run_quality(context["targets"], values)
trial.require(all(math.isclose(runs[m][k], comparison["runs"][arm][m][k], rel_tol=0, abs_tol=1e-12)
    for m in trial.QUALITY for k in ("chain_mean", "equal_entry_mean")), "reused report differs from its vectors")
noise = comparison["architectures"]["flat_fine6"]["new_code_seeds"]["lddt"]["chain_mean"]
home = Path(context["protocol"]["arms"][arm]["training"])
initial = trial.read(home / "initialization.json")
trial.unchanged(context)
trial.write_new(destination, dict(admitted=True, arm=arm, seed=receipt["config"]["seed"], K=2, **binding,
    verifier_sha256=trial.sha(Path(trial.__file__)), ost_executable_sha256=trial.sha(context["protocol"]["ost"]),
    scorer=admission["scorer"], hardware=inference["hardware"], per_target=values, overall=comparison["runs"][arm],
    costs=comparison["costs"][arm], geometry_totals=comparison["geometry_totals"][arm], training_receipt=receipt,
    training=str(home), checkpoint_sha256=inference["arms"][arm]["checkpoint_sha256"],
    ema_sha256=inference["arms"][arm]["ema_sha256"], config=receipt["config"], runtime=initial["mamba3_runtime"],
    initial_model_sha256=initial["target_sha256"], input_stream_sha256=receipt["complete"]["input_stream_sha256"],
    seed_noise=noise))
'''


def reference_path(run):
    return Path(run) / "reference_admitted.json"


def admitted_reference(context):
    """The reused baseline rows admitted by its own frozen evaluator, plus re-derived canonical geometry."""
    run, protocol, targets = context["run"], context["protocol"], context["targets"]
    bound = protocol["baseline"]
    validate_baseline(protocol)
    value = read(reference_path(run))
    exact(value, dict(admitted=True, arm=BASELINE, seed=SEED, K=DEPTH, training=bound["training"],
        verifier_sha256=sha(Path(bound["eval_code"]) / BASELINE_VERIFIER), ost_executable_sha256=sha(protocol["ost"]),
        initial_model_sha256=bound["initial_model_sha256"], input_stream_sha256=bound["input_stream_sha256"]),
        "reused baseline admission binding differs")
    eval_run = Path(bound["eval_run"])
    score, receipt = read(eval_run / f"scores/{BASELINE}.json"), read(eval_run / f"train_admitted/{BASELINE}.json")
    require(set(value.get("per_target", {})) == set(targets) and value["per_target"] == score["metrics"]
            and value.get("scorer") == score["scorer"], "reused baseline canonical vectors differ")
    require(value.get("training_receipt") == receipt and value.get("checkpoint_sha256")
            == receipt["final_checkpoint_sha256"] and value.get("ema_sha256") == receipt["final_ema_sha256"],
            "reused baseline training admission differs")
    require(value.get("seed_noise", {}).get("sd") == bound["seed_noise_lddt_chain_mean_sd"],
            "reused seed noise differs")
    base.validate_hardware(value.get("hardware"), training=True)
    geometry = tail.geometry_diagnostics(base.evaluation(eval_run, BASELINE), targets, value["per_target"],
                                         value["scorer"], DEPTH)
    require({field: geometry["totals"][field] for field in GEOMETRY} == value.get("geometry_totals"),
            "reused baseline geometry differs")
    return dict(value, geometry=geometry)


def reference(context):
    base.no_outputs([reference_path(context["run"])])
    bound = context["protocol"]["baseline"]
    validate_baseline(context["protocol"])
    subprocess.run([sys.executable, "-c", REFERENCE_SCRIPT, bound["eval_code"], bound["eval_run"],
                    str(reference_path(context["run"]))], cwd=bound["eval_code"],
                   env=dict(os.environ, PYTHONPATH=f"{bound['eval_code']}:{bound['eval_code']}/src"), check=True)
    return admitted_reference(context)


# ── GPU preflight: construction, memory and baseline equivalence ────────


def construction_config(context, name):
    if name == BASELINE:
        return yaml.safe_load(Path(context["protocol"]["baseline"]["template"]).read_text())
    return dict(context["configs"][name])


def memory_probe(model, cfg):
    """Updates of a throwaway model at the largest declared shape; formal models never pass through here."""
    import torch
    from dataclasses import replace

    from mambafold.train import engine, optim, trainer
    from mambafold.train.ema import EMA

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ema, optimizer = EMA(model, decay=cfg["ema_decay"]), optim.build_optimizer(model, cfg)
    generator = torch.Generator().manual_seed(SEED)
    batch = trainer._synthetic_batch(cfg["batch_size"] * cfg["copies_per_protein"], cfg["max_length"], 14,
                                     cfg["d_plm"], torch.device("cuda"))
    noise = lambda value, scale: (torch.randn(value.shape, generator=generator) * scale).to(value)  # noqa: E731
    batch = replace(batch, x_clean=noise(batch.x_clean, 10.), x_t=noise(batch.x_t, 10.), eps=noise(batch.eps, 1.),
                    esm=noise(batch.esm, 1.))
    model.train()
    seconds, losses = [], []
    for _ in range(MEMORY_UPDATES):
        torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(cfg["grad_accum_steps"]):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(batch)
            loss, _ = engine.allatom_loss_surface(output, batch, alpha_mode=cfg["alpha_mode"],
                use_rigid_align=cfg["use_rigid_align"], w_fm=cfg["w_fm"], w_lddt_atom=cfg["w_lddt_atom"],
                lddt_cutoff_A=cfg["lddt_cutoff_A"], lddt_pair_chunk_size=cfg["lddt_pair_chunk_size"])
            (loss / cfg["grad_accum_steps"]).backward()
            losses.append(float(loss.detach()))
            del output, loss
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()
        ema.update(model)
        torch.cuda.synchronize()
        seconds.append(time.perf_counter() - start)
    total = torch.cuda.get_device_properties(0).total_memory
    value = dict(feasible=True, shape=dict(batch=cfg["batch_size"] * cfg["copies_per_protein"],
        length=cfg["max_length"], microbatches_per_update=cfg["grad_accum_steps"]), updates=MEMORY_UPDATES,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2 ** 30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2 ** 30, device_total_gib=total / 2 ** 30,
        warm_update_ms=1000 * statistics.fmean(seconds[1:]), first_update_ms=1000 * seconds[0],
        losses_finite=all(math.isfinite(loss) for loss in losses),
        scope="throwaway seeded model, synthetic dense L1024 batch, full update incl. optimizer and EMA; "
              "not the formal model and not a training-cost measurement")
    del ema, optimizer, batch
    return value


def preflight_path(run):
    return Path(run) / "preflight_verified.json"


def verify_preflight(value, protocol):
    rows, memory = value.get("rows", {}), value.get("memory", {})
    require(set(rows) == {BASELINE, *ARMS} and set(memory) == {BASELINE, *ARMS}, "preflight arm coverage differs")
    expected_parameters = dict(PARAMETERS, **{BASELINE: BASELINE_PARAMETERS})
    for name, row in rows.items():
        require(type(row.get("parameters")) is int and row["parameters"] == row.get("trainable_parameters")
                == expected_parameters[name], f"{name} constructed parameter count differs")
        require(proof.valid_sha(row.get("initial_model_sha256")) and proof.valid_sha(
            row.get("parameter_signature_sha256")), "constructed tensor identity missing")
    require(rows[BASELINE]["initial_model_sha256"] == protocol["baseline"]["initial_model_sha256"],
            "current code does not rebuild the reused flat_fine6 seed20261023 initial tensors")
    require(len({row["initial_model_sha256"] for row in rows.values()}) == len(rows)
            and len({row["parameter_signature_sha256"] for row in rows.values()}) == len(rows),
            "latent variants do not change the constructed tensors")
    for arm in PARAMETER_MATCHED:
        require(abs(rows[arm]["parameters"] / BASELINE_PARAMETERS - 1) <= PARAMETER_TOLERANCE,
                "parameter-matched arm is outside the declared tolerance")
    require(memory[BASELINE].get("feasible") is True, "the baseline construction must fit the training GPU")
    for name, row in memory.items():
        if row.get("feasible") is True:
            require(all(proof.finite(row.get(key), positive=True) for key in (
                "peak_allocated_gib", "peak_reserved_gib", "device_total_gib", "warm_update_ms"))
                and row["peak_reserved_gib"] < row["device_total_gib"] and row.get("losses_finite") is True,
                f"{name} memory probe is incomplete or nonfinite")
        else:
            exact(row, dict(feasible=False), f"{name} infeasibility record differs")
            require(isinstance(row.get("error"), str) and row["error"], f"{name} infeasibility lacks its error")
    equivalence = value.get("equivalence")
    require(isinstance(equivalence, dict) and equivalence.get("admitted") is True,
            "historical/current flat_fine6 equivalence was not admitted")
    require(equivalence.get("reference") == reference_record(protocol["equivalence_reference"]),
            "equivalence was not admitted against the sealed historical reference")
    for record in (equivalence, equivalence.get("warmup")):
        require(isinstance(record, dict) and set(record.get("arrays", {})) == set(EQUIVALENCE_ARRAYS)
                and all(type(row.get("bitwise_equal")) is bool for row in record["arrays"].values()),
                "per-array bitwise equivalence record is missing")
    require(set(value.get("evidence_sha256", {})) == {f"preflight/equivalence/{label}/{name}"
            for label in EQUIVALENCE_LABELS for name in EQUIVALENCE_FILES},
            "warm-up and current-code equivalence probes are not both sealed")
    return {name: row["feasible"] for name, row in memory.items()}


def reference_record(binding):
    return {key: binding[key] for key in ("probe_sha256", "outputs_sha256", "code")}


def probe_arrays(path):
    import numpy as np

    with np.load(path, allow_pickle=False) as packed:
        return {key: packed[key] for key in packed.files}


def probe_difference(reference, other, reference_arrays, other_arrays):
    """How far a probe is from the sealed reference; recorded, never gated on."""
    import numpy as np

    rows = {}
    for name in EQUIVALENCE_ARRAYS:
        first, second = reference_arrays.get(name), other_arrays.get(name)
        shaped = first is not None and second is not None and first.shape == second.shape
        difference = float(np.abs(first.astype(np.float64) - second).max()) if shaped and first.size else None
        rows[name] = dict(shape_equal=shaped, bitwise_equal=bool(shaped and np.array_equal(first, second)),
                          max_abs_difference=difference if proof.finite(difference) else None)
    loss = other.get("loss")
    return dict(arrays=rows, loss=loss if proof.finite(loss) else None,
                identity_equal={key: reference.get(key) == other.get(key) for key in
                                ("initial_model_sha256", "perturbed_model_sha256", "batch_sha256", "parameters")})


def admit_equivalence(binding, initial, probes, arrays):
    """Gate the second current-code probe against the sealed historical one with the seed-replicate tolerance."""
    require(probes["reference"].get("code") == binding["code"] != probes["new"].get("code"),
            "equivalence reference is not the sealed historical probe")
    verified = seedrep.verify_equivalence(probes["reference"], probes["new"],
                                          dict(old=arrays["reference"], new=arrays["new"]), initial)
    return dict(verified, compared="old=sealed historical current-code probe; new=this run's second current-code probe",
                reference=reference_record(binding),
                warmup=probe_difference(probes["reference"], probes["warmup"], arrays["reference"], arrays["warmup"]))


def preflight(context):
    import torch
    from mambafold.model.bimamba3 import mamba3_runtime_info
    from mambafold.train import trainer
    from mambafold.train.distributed import enable_cuda_perf_flags

    run, protocol = context["run"], context["protocol"]
    probes = run / "preflight/equivalence"
    base.no_outputs([preflight_path(run), probes])
    device = base.hardware(training=True)
    torch.set_num_threads(1)
    enable_cuda_perf_flags()
    reference_value = admitted_reference(context)
    rows, memory = {}, {}
    with scratch.forbid_checkpoint_loads():
        for name in (BASELINE, *ARMS):
            cfg = construction_config(context, name)
            trainer.seed_all(SEED)
            model = trainer.build_model(cfg, "cuda")
            signature = [(key, list(value.shape), str(value.dtype)) for key, value in model.state_dict().items()]
            rows[name] = dict(initial_model_sha256=base.audit.tensor_digest(model.state_dict().items()),
                parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                parameter_signature_sha256=hashlib.sha256(json.dumps(signature).encode()).hexdigest(),
                latent_parameters=sum(p.numel() for p in model.latent_residual.stack.parameters()),
                runtime=mamba3_runtime_info(model))
            try:
                memory[name] = memory_probe(model, cfg)
            except torch.cuda.OutOfMemoryError as error:
                memory[name] = dict(feasible=False, error=f"{type(error).__name__}: {error}"[:2000])
            del model
            gc.collect()
            torch.cuda.empty_cache()
    require(base.hardware(training=True) == device, "preflight changed physical GPU")
    bound, code = protocol["baseline"], run / "code"
    sealed = validate_equivalence_reference(protocol)
    probe, arrays = dict(reference=read(sealed["probe"])), dict(reference=probe_arrays(sealed["outputs"]))
    for label in EQUIVALENCE_LABELS:
        subprocess.run([sys.executable, "-c", seedrep.EQUIVALENCE_SCRIPT, str(code), bound["template"],
                        str(probes / label)], cwd=code, env=dict(os.environ, PYTHONPATH=f"{code}:{code / 'src'}"),
                       check=True)
        probe[label] = read(probes / label / "probe.json")
        arrays[label] = probe_arrays(probes / label / "outputs.npz")
    require(base.hardware(training=True) == device, "equivalence probe changed physical GPU")
    require(validate_equivalence_reference(protocol) == sealed, "historical equivalence reference changed")
    equivalence = admit_equivalence(sealed, bound["initial_model_sha256"], probe, arrays)
    value = dict(admitted=True, **base.context_binding(context), hardware=device, rows=rows, memory=memory,
        equivalence=equivalence, baseline_runtime_equal=rows[BASELINE]["runtime"] == reference_value["runtime"],
        reference_admission_sha256=sha(reference_path(run)),
        evidence_sha256={base.relative(run, probes / label / name): sha(probes / label / name)
                         for label in EQUIVALENCE_LABELS for name in EQUIVALENCE_FILES})
    verify_preflight(value, protocol)
    unchanged(context)
    write_new(preflight_path(run), value)


def admitted_preflight(context):
    run = context["run"]
    value = read(preflight_path(run))
    exact(value, dict(admitted=True, **base.context_binding(context), reference_admission_sha256=sha(reference_path(run))),
          "preflight admission binding differs")
    base.validate_hardware(value.get("hardware"), training=True)
    base.seals(run, value.get("evidence_sha256"))
    validate_equivalence_reference(context["protocol"])
    verify_preflight(value, context["protocol"])
    return value


# ── training ────────────────────────────────────────────────────────────


def baseline_inputs(protocol):
    bound = protocol["baseline"]
    delivered, digest = inputs.read_inputs(Path(bound["training"]) / "audit/inputs.jsonl", 4 * STEP)
    require(digest == bound["input_stream_sha256"], "reused baseline input audit changed")
    return delivered


def check_native(native, gate, arm, first_inputs):
    """Fail before formal training when fresh weights or the shared data stream differ."""
    require(native.get("arm") == arm and native.get("seed") == SEED, "native receipt arm/seed differs")
    require(native.get("model_sha256") == gate["rows"][arm]["initial_model_sha256"],
            "native fresh weights differ from the admitted construction")
    require([row.get("input_sha256") for row in native.get("rows", [])] == list(first_inputs),
            "first eight native inputs differ from the baseline data stream")


def training_home(context, arm):
    return Path(context["protocol"]["arms"][arm]["training"])


def verify_actual_training(context, arm):
    run = context["run"]
    with registry():
        config, complete, metrics, evidence = proof.verify_arm(run, arm, context["protocol"], context["manifest"],
                                                               context["targets"])
    home = training_home(context, arm)
    exact(read(home / "audit/initial.json"), dict(parameters=PARAMETERS[arm], trainable_parameters=PARAMETERS[arm]),
          "actual parameter count differs")
    initial = read(home / "initialization.json")
    gate = admitted_preflight(context)
    require(initial["target_sha256"] == gate["rows"][arm]["initial_model_sha256"],
            "formal initial tensors differ from the admitted construction")
    native = read(Path(context["protocol"]["arms"][arm]["preflight"]) / "native_verified.json")
    seedrep.verify_flat_construction(config, native)
    require(complete["input_stream_sha256"] == context["protocol"]["baseline"]["input_stream_sha256"],
            "formal 40000-microbatch stream differs from the baseline stream")
    evidence = {base.relative(run, path): value for path, value in evidence.items()}
    evidence["preflight_verified.json"] = sha(preflight_path(run))
    return config, complete, metrics, evidence


def train(context, arm):
    import torch
    from benchmarks.screen_architectures import validate_anchor

    require(arm in ARMS, "train requires a declared latent-variant arm")
    gate = admitted_preflight(context)
    require(gate["memory"][arm]["feasible"] is True, f"{arm} did not fit the training GPU in preflight")
    admitted_reference(context)
    first = baseline_inputs(context["protocol"])[:scratch.PREFLIGHT_BATCHES]
    run, entry = context["run"], context["protocol"]["arms"][arm]
    preflight_dir, home = Path(entry["preflight"]), Path(entry["training"])
    probe, destination = Path(entry["u0"]) / f"{arm}-{STEP}", base.training_receipt_path(run, arm)
    base.no_outputs((preflight_dir, home, probe, destination))
    device = base.hardware(training=True)
    torch.set_num_threads(1)
    with registry():
        scratch.preflight(SimpleNamespace(arm=arm, config=Path(entry["config"]), out=preflight_dir))
        check_native(read(preflight_dir / "native_verified.json"), gate, arm, first)
        exact(read(preflight_dir / "initialization.json"), dict(parameters=PARAMETERS[arm],
            trainable_parameters=PARAMETERS[arm], prelude_layers=3, shared_trunk_layers=6,
            hierarchy=ARMS[arm]["latent_hierarchy"]), "native model layout differs before formal training")
        gc.collect()
        torch.cuda.empty_cache()
        require(base.hardware(training=True) == device, "preflight changed physical GPU")
        scratch.train(SimpleNamespace(arm=arm, config=Path(entry["config"]), out=home,
                                      native_receipt=preflight_dir / "native_verified.json"))
        screen.probe(SimpleNamespace(checkpoint=home / f"ckpt_{STEP:07d}.pt", out=probe, audit=home / "audit",
                                     batches=16, expected_step=STEP, validate_output=validate_anchor))
    config, complete, metrics, evidence = verify_actual_training(context, arm)
    require(base.hardware(training=True) == device, "training/probe changed physical GPU")
    gt = read(home / "ground_truth_verified.json")
    exact(gt, dict(arm=arm, optimizer_updates=STEP, microbatches=4 * STEP, final_schedule_index=STEP,
                   checkpoint_loads_during_training=0), "training ground truth differs")
    for path in (preflight_dir / "initial_ema.pt", home / "initial_ema.pt", home / f"ckpt_{STEP:07d}.pt"):
        evidence[base.relative(run, path)] = sha(path)
    unchanged(context)
    write_new(destination, dict(admitted=True, arm=arm, **base.context_binding(context), hardware=device,
        parameters=PARAMETERS[arm], nominal_K=DEPTH, config=config, complete=complete,
        all_update_weighted_mean_step_ms=weighted_step_ms(metrics),
        final_checkpoint_sha256=gt["final_checkpoint_sha256"], final_ema_sha256=gt["final_ema_sha256"],
        evidence_sha256=evidence))


def weighted_step_ms(metrics):
    previous, weighted = 0, 0.0
    for row in metrics:
        require(proof.finite(row.get("perf_train_step_ms_max"), positive=True), "invalid step timer")
        weighted += row["perf_train_step_ms_max"] * (row["step"] - previous)
        previous = row["step"]
    require(previous == STEP, "incomplete timing windows")
    return weighted / previous


def admitted_training(context, arm):
    run = context["run"]
    value = read(base.training_receipt_path(run, arm))
    exact(value, dict(admitted=True, arm=arm, parameters=PARAMETERS[arm], nominal_K=DEPTH,
                      **base.context_binding(context)), "training admission identity differs")
    base.seals(run, value.get("evidence_sha256"))
    require(value["evidence_sha256"].get("preflight_verified.json") == sha(preflight_path(run)),
            "preflight admission not sealed into training")
    home = training_home(context, arm)
    config, complete = read(home / "config.json"), read(home / "audit/complete.json")
    require(value.get("config") == config and all(value["complete"].get(k) == v for k, v in complete.items()),
            "training admission differs from actual training")
    exact(complete, inputs.BUDGET, "incomplete admitted training budget")
    require(complete["input_stream_sha256"] == context["protocol"]["baseline"]["input_stream_sha256"],
            "admitted stream differs from the baseline stream")
    with registry():
        scratch.validate_config(config, arm)
    exact(config, expected(arm), "actual scratch configuration differs")
    seedrep.verify_flat_construction(config, {})
    gt = read(home / "ground_truth_verified.json")
    require(value.get("final_checkpoint_sha256") == gt["final_checkpoint_sha256"]
            == value["evidence_sha256"].get(base.relative(run, home / f"ckpt_{STEP:07d}.pt"))
            and value.get("final_ema_sha256") == gt["final_ema_sha256"], "own checkpoint/EMA binding differs")
    base.validate_hardware(value.get("hardware"), training=True)
    for key in ("steady_mean_step_ms", "peak_allocated_gib", "peak_reserved_gib"):
        require(proof.finite(complete.get(key), positive=True), "invalid measured training cost")
    require(proof.finite(value.get("all_update_weighted_mean_step_ms"), positive=True), "invalid full-update timer")
    initial = read(home / "initialization.json")
    require(initial["target_sha256"] == admitted_preflight(context)["rows"][arm]["initial_model_sha256"],
            "admitted initial tensors differ from the admitted construction")
    return value


# ── inference over every admitted arm ──────────────────────────────────


def split_arms(run):
    """Arms that wrote their own training admission; every other arm is excluded, never silently dropped."""
    present, excluded = [], {}
    for arm in ARMS:
        path = base.training_receipt_path(run, arm)
        if path.exists() or path.is_symlink():
            require(regular(path), f"{arm} training admission is not a regular file")
            present.append(arm)
        else:
            home = Path(run) / f"training/{arm}"
            excluded[arm] = dict(reason="no training admission", training_home_exists=home.exists(),
                                 final_checkpoint_exists=(home / f"ckpt_{STEP:07d}.pt").exists())
    return present, excluded


def verify_inference(context, arm, checkpoint_sha):
    directory = base.evaluation(context["run"], arm)
    measured = proof.verify_inference(directory, STEP, [DEPTH], checkpoint_sha, context["targets"],
                                      ARMS[arm]["latent_hierarchy"], n_steps=SAMPLER["n_steps"])
    for row in measured:
        seedrep.verify_flat_construction(row, {})
        require(proof.finite(row.get("peak_reserved_gib"), positive=True), "missing reserved inference memory")
    return measured


def infer(context):
    import torch
    from mambafold.model.bimamba3 import mamba3_runtime_info

    run = context["run"]
    base.no_outputs([run / "inference_admitted.json", *(base.evaluation(run, arm) for arm in ARMS)])
    device = base.hardware(training=True)
    torch.set_num_threads(1)
    admitted_preflight(context)
    admitted_reference(context)
    present, excluded = split_arms(run)
    require(present, "no latent-variant arm wrote a training admission")
    admitted = {arm: admitted_training(context, arm) for arm in present}
    runtime, runtimes = mamba3_runtime_info(), {}
    for arm in present:
        verify_actual_training(context, arm)
        runtimes[arm] = read(training_home(context, arm) / "initialization.json")["mamba3_runtime"]
        base.verify_external_runtime(runtimes[arm], runtime)
        _, digest = inputs.read_inputs(training_home(context, arm) / "audit/inputs.jsonl", 4 * STEP)
        require(digest == admitted[arm]["complete"]["input_stream_sha256"], "actual input stream digest differs")
    evidence, arms = {}, {}
    for arm in present:
        home, directory = training_home(context, arm), base.evaluation(run, arm)
        require(base.hardware(training=True) == device, "common physical inference GPU changed")
        with base.inference_model_runtime(runtimes[arm]):
            screen.infer(SimpleNamespace(checkpoint=home / f"ckpt_{STEP:07d}.pt", panel=Path(context["protocol"]["panel"]),
                                        out=directory, depths=[DEPTH], n_steps=SAMPLER["n_steps"]))
        require(base.hardware(training=True) == device, "common physical inference GPU changed")
        verify_inference(context, arm, admitted[arm]["final_checkpoint_sha256"])
        arms[arm] = dict(hardware=device, nominal_K=DEPTH, checkpoint_sha256=admitted[arm]["final_checkpoint_sha256"],
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
        sampler=SAMPLER, inference_depth=DEPTH, coverage=len(context["targets"]), arms=arms, excluded=excluded,
        preflight_admission_sha256=sha(preflight_path(run)),
        training_receipts={arm: sha(base.training_receipt_path(run, arm)) for arm in present},
        input_comparison=dict(method="digest_equal_to_reused_baseline_stream", microbatches_per_arm=4 * STEP),
        evidence_sha256=evidence))


def admitted_inference(context):
    run = context["run"]
    value = read(run / "inference_admitted.json")
    exact(value, dict(admitted=True, coverage=len(context["targets"]), sampler=SAMPLER, inference_depth=DEPTH,
        preflight_admission_sha256=sha(preflight_path(run)), **base.context_binding(context)),
        "latent-variant inference admission differs")
    present, excluded = split_arms(run)
    require(list(value.get("arms", {})) == present and set(value.get("training_receipts", {})) == set(present)
            and value.get("excluded") == excluded and present, "inference arm coverage differs from admissions")
    base.validate_hardware(value.get("hardware"), training=True)
    base.seals(run, value.get("evidence_sha256"))
    for arm in present:
        require(sha(base.training_receipt_path(run, arm)) == value["training_receipts"][arm],
                "training receipt changed after inference")
        training = admitted_training(context, arm)
        binding = value["arms"][arm]
        exact(binding, dict(hardware=value["hardware"], nominal_K=DEPTH,
            checkpoint_sha256=training["final_checkpoint_sha256"], ema_sha256=training["final_ema_sha256"]),
            "inference own EMA/GPU binding differs")
        require(read(base.evaluation(run, arm) / "gpu_binding.json") == binding, "per-arm GPU binding differs")
        verify_inference(context, arm, binding["checkpoint_sha256"])
    return value


def score(context):
    run = context["run"]
    names = tuple(admitted_inference(context)["arms"])
    base.no_outputs([path for name in names for path in (run / f"scores/{name}.json", run / f"scores/{name}.failed.json",
                                                         base.evaluation(run, name) / f"K{DEPTH}/scores")])
    failed = []
    for name in names:
        directory = base.evaluation(run, name)
        completed = subprocess.run([sys.executable, str(Path(base.canonical.__file__)), "--in-dir",
            str(directory / f"K{DEPTH}/pairs"), "--out-dir", str(directory / f"K{DEPTH}/scores"), "--ost",
            context["protocol"]["ost"], "--expected", str(len(context["targets"]))], check=False)
        path = directory / f"K{DEPTH}/scores/summary.json"
        summary = read(path) if path.is_file() else None
        if completed.returncode or summary is None or summary.get("evaluation_complete") is not True:
            write_new(run / f"scores/{name}.failed.json", dict(arm=name, admitted=False,
                returncode=completed.returncode, expected=len(context["targets"]), summary=summary))
            failed.append(name)
            continue
        vectors = base.verify_canonical(summary, directory, context)
        unchanged(context)
        write_new(run / f"scores/{name}.json", dict(admitted=True, arm=name, **base.context_binding(context),
            inference_admission_sha256=sha(run / "inference_admitted.json"), coverage=len(vectors), metrics=vectors,
            summary_sha256=sha(path), scorer=summary["openstructure"],
            ost_executable_sha256=sha(context["protocol"]["ost"])))
    require(not failed, "canonical scoring failed or was incomplete: " + ", ".join(failed))


# ── report ──────────────────────────────────────────────────────────────


def summarize(targets, vectors, noise_sd):
    """Per-arm quality and paired arm-minus-baseline effects with win/tie/loss; descriptive only, no selection."""
    require(BASELINE in vectors and len(vectors) > 1 and set(vectors) - {BASELINE} <= set(ARMS),
            "latent-variant report requires the baseline and at least one admitted arm")
    for values in vectors.values():
        require(set(values) == set(targets) and all(set(row) == set(QUALITY) and all(proof.finite(v) for v in
                row.values()) for row in values.values()), "invalid or incomplete quality vectors")
    runs = {name: seedrep.run_quality(targets, values) for name, values in vectors.items()}
    paired = {}
    for arm in (name for name in ARMS if name in vectors):
        effects = seedrep.curve.tally(tail.paired_effects(targets, vectors[BASELINE], vectors[arm]))
        aa = effects["lddt"]["chain_mean_delta"]
        paired[arm] = dict(effects=effects, aa_chain_mean_delta=aa, aa_delta_over_seed_sd=aa / noise_sd,
                           within_seed_noise=abs(aa) < NOISE_BAND)
    return dict(runs=runs, paired_minus_baseline=paired,
                noise=dict(metric="lddt chain mean", flat_fine6_seed_sd=noise_sd, band=NOISE_BAND,
                           interpretation="single seed per arm; |delta AA| below the band is within seed noise"))


def report(context):
    run, targets, protocol = context["run"], context["targets"], context["protocol"]
    base.no_outputs([run / "comparison.json", run / "comparison.csv"])
    inference, reference_value = admitted_inference(context), admitted_reference(context)
    gate = admitted_preflight(context)
    present = list(inference["arms"])
    vectors, geometry, costs = {}, {}, {}
    for arm in present:
        value = read(run / f"scores/{arm}.json")
        exact(value, dict(admitted=True, arm=arm, coverage=len(targets),
            inference_admission_sha256=sha(run / "inference_admitted.json"), **base.context_binding(context)),
            "score admission identity differs")
        directory = base.evaluation(run, arm)
        path = directory / f"K{DEPTH}/scores/summary.json"
        require(sha(path) == value["summary_sha256"] and sha(protocol["ost"]) == value["ost_executable_sha256"],
                "canonical score evidence changed")
        summary = read(path)
        vectors[arm] = base.verify_canonical(summary, directory, context)
        require(vectors[arm] == value["metrics"] and summary["openstructure"] == value["scorer"]
                == reference_value["scorer"], "canonical vectors/scorer differ")
        geometry[arm] = {field: tail.geometry_diagnostics(directory, targets, vectors[arm], value["scorer"],
                                                          DEPTH)["totals"][field] for field in GEOMETRY}
        receipt, measured = admitted_training(context, arm), read(directory / "sampling.json")
        complete = receipt["complete"]
        costs[arm] = dict(steady_train_ms=complete["steady_mean_step_ms"],
            all_update_train_ms=receipt["all_update_weighted_mean_step_ms"],
            train_allocated_gib=complete["peak_allocated_gib"], train_reserved_gib=complete["peak_reserved_gib"],
            inference_seconds=statistics.fmean(row["seconds"] for row in measured),
            inference_allocated_gib=max(row["peak_allocated_gib"] for row in measured),
            inference_reserved_gib=max(row["peak_reserved_gib"] for row in measured),
            training_gpu_uuid=receipt["hardware"]["uuid"], inference_gpu_uuid=inference["hardware"]["uuid"])
    vectors[BASELINE], geometry[BASELINE] = reference_value["per_target"], reference_value["geometry_totals"]
    costs[BASELINE] = dict(reference_value["costs"])
    stats = summarize(targets, vectors, reference_value["seed_noise"]["sd"])
    names = [BASELINE, *present]
    architecture = dict(ARMS, **{BASELINE: dict(BASELINE_LATENT)})
    parameters = {name: gate["rows"][name]["parameters"] for name in names}
    sizes = {name: dict(parameters=parameters[name], parameter_ratio=parameters[name] / BASELINE_PARAMETERS,
                        latent_flop_ratio=flop_ratio(architecture[name]), parameter_matched=name in PARAMETER_MATCHED,
                        preflight_memory=gate["memory"][name]) for name in names}
    result = dict(benchmark_role=ROLE, sampler=SAMPLER, nominal_K=DEPTH, inference_depth=DEPTH, seed=SEED,
        **base.context_binding(context), coverage=dict(targets=len(targets), entries=len(seedrep.entries(targets))),
        metric_names=dict(AA="lddt", CA="bb_lddt", TM="tm_score"), baseline=BASELINE, arms=present,
        excluded=inference["excluded"], architectures={name: architecture[name] for name in names}, sizes=sizes,
        per_target=vectors, geometry_totals=geometry,
        geometry_unit="OST violation entries; unnormalised counts", costs=costs, **stats,
        inference_hardware=inference["hardware"], baseline_inference_hardware=reference_value["hardware"],
        same_inference_gpu_as_baseline=inference["hardware"]["uuid"] == reference_value["hardware"]["uuid"],
        reference_admission_sha256=sha(reference_path(run)), preflight_admission_sha256=sha(preflight_path(run)),
        latent_flop_definition=protocol["latent_flop_definition"], architecture_selected=False, limits=LIMITS)
    unchanged(context)
    write_new(run / "comparison.json", result)
    rows = []
    for name in names:
        delta = stats["paired_minus_baseline"].get(name)
        rows.append(dict(arm=name, **architecture[name], parameters=parameters[name],
            parameter_ratio=sizes[name]["parameter_ratio"], latent_flop_ratio=sizes[name]["latent_flop_ratio"],
            **{metric: stats["runs"][name][metric]["chain_mean"] for metric in QUALITY},
            **{f"{metric}_delta": delta["effects"][metric]["chain_mean_delta"] if delta else None for metric in QUALITY},
            **{f"lddt_{key}": delta["effects"]["lddt"][key] if delta else None for key in ("wins", "ties", "losses")},
            within_seed_noise=delta["within_seed_noise"] if delta else None, **geometry[name],
            **{key: value for key, value in costs[name].items() if not key.endswith("uuid")}))
    with (run / "comparison.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "reference", "preflight", "train", "infer", "score", "report"))
    parser.add_argument("--run", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--eval-run", type=Path, help="seed-replicate follow-up run holding the reused baseline")
    args = parser.parse_args()
    if args.stage == "prepare":
        if args.out is None or args.run is not None or args.arm is not None:
            parser.error("prepare requires --out and no --run/--arm")
        print(json.dumps(prepare(args.out, eval_run=args.eval_run or DEFAULT_EVAL)))
        return
    if args.run is None or args.out is not None or args.eval_run is not None:
        parser.error(f"{args.stage} requires --run and no --out/--eval-run")
    if (args.stage == "train") != (args.arm is not None):
        parser.error("train requires exactly one --arm; other stages operate on all arms")
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("latent-variant execution requires a Slurm allocation, including CPU stages")
    context = validate_run(args.run)
    if args.stage == "train":
        train(context, args.arm)
    else:
        dict(reference=reference, preflight=preflight, infer=infer, score=score, report=report)[args.stage](context)


if __name__ == "__main__":
    main()
