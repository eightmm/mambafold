"""One longer scratch flat_fine6 run (WSD 500/25500/4000 = 30000 updates) evaluated at 10k/20k/30k.

The recipe is the admitted seed-20261023 flat_fine6 seed-replicate recipe except for the declared horizon,
cooldown and retained checkpoints. The 10k-horizon screen modules are reused by import; their hard-coded
horizon constants are rebound only inside `horizon()`. Under the 10k horizon every rebinding is the
identity, which is how the training admission is dry-run against the real finished 10k trainer output.
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
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

# Both imports are torch-free; the ML stack is bound lazily so prepare never imports torch.
from benchmarks import compare_seed_replicates as seedrep  # noqa: E402
from benchmarks import evaluate_recycle_checkpoint_curve as curve  # noqa: E402
from benchmarks import input_equivalence as inputs  # noqa: E402

sha = inputs.sha256


@dataclass(frozen=True)
class Horizon:
    """A WSD schedule exactly as `trainer.linear_warmup_lr` builds it (floor = min_lr/max_lr = 0.01)."""

    total: int
    warmup: int
    cooldown: int
    milestones: tuple

    @property
    def stable(self):
        return self.total - self.warmup - self.cooldown

    @property
    def decay_start(self):
        return self.total - self.cooldown

    @property
    def boundaries(self):
        """The 10k screen's nine boundary indices, placed on this horizon."""
        start = self.decay_start
        return (0, 1, self.warmup - 1, self.warmup, start - 1, start, start + self.cooldown // 2,
                self.total - 1, self.total)

    def multiplier(self, step):
        # Same arithmetic as screen_scratch_architectures.schedule_multiplier, so the 10k horizon is bit-identical.
        if type(step) is not int or not 0 <= step <= self.total:
            raise ValueError(f"schedule index must be within the declared {self.total} horizon")
        floor = 0.01
        if step < self.warmup:
            return floor + (1 - floor) * step / self.warmup
        if step < self.decay_start:
            return 1.0
        return floor + (1 - floor) * (1 + math.cos(math.pi * (step - self.decay_start) / self.cooldown)) / 2


TEN_K = Horizon(total=10000, warmup=500, cooldown=2000, milestones=(5000, 10000))
LONG = Horizon(total=30000, warmup=500, cooldown=4000, milestones=(10000, 20000, 30000))
SEED = 20261023
ARM = f"flat_fine6_long30k_s{SEED}"
REF_ARM = f"flat_fine6_s{SEED}"
FLAT_ARMS = tuple(f"flat_fine6_s{seed}" for seed in seedrep.SEEDS)
ARCHITECTURE = dict(seedrep.ARCHITECTURES["flat_fine6"])
VARIANT = dict(ARCHITECTURE, seed=SEED)
PARAMETER_COUNT = seedrep.PARAMETER_COUNT
DEPTH = 2
SAMPLER = dict(seedrep.SAMPLER)
PEAKS = dict(muon=0.003, adamw=0.0001)
QUALITY, GEOMETRY = seedrep.QUALITY, seedrep.GEOMETRY
KINDS = ("chain_mean", "equal_entry_mean")
CONFIG_CHANGES = dict(total_steps=LONG.total, lr_cooldown_steps=LONG.cooldown,
                      keep_checkpoint_steps=list(LONG.milestones))
TEMPLATE_RECIPE = dict(total_steps=TEN_K.total, warmup_steps=500, lr_cooldown_steps=TEN_K.cooldown,
                       keep_checkpoint_steps=list(TEN_K.milestones), ckpt_interval=5000, keep_last_checkpoints=1,
                       seed=SEED, ema_decay=0.999, muon_lr=0.003, lr=0.0001, min_lr=1e-06, batch_size=1,
                       copies_per_protein=4, grad_accum_steps=4, max_length=1024, resume=None,
                       initialize_refiner_from=None, initialize_model_from_ema=False)
WANDB_NAME = "flat-fine6-long-30k"
PATH_KEYS = ("config", "out_dir", "wandb_name", "wandb_tags")
ROLE = "single long scratch flat_fine6 run;fixed 10k/20k/30k checkpoints, no checkpoint or sampler tuning"
PHASES = {str(LONG.milestones[0]): "pre-cooldown diagnostic: mid-stable phase at the peak learning rate",
          str(LONG.milestones[1]): "pre-cooldown diagnostic: mid-stable phase at the peak learning rate",
          str(LONG.total): "post-cooldown final checkpoint: comparable to the final checkpoints of the 10k-schedule runs"}
REFERENCE_FILES = ("source_manifest.json", "protocol.json", "comparison.json", "inference_admitted.json",
                   "cpu_admitted.json", *(f"train_admitted/{arm}.json" for arm in FLAT_ARMS),
                   *(f"scores/{arm}.json" for arm in FLAT_ARMS))
REFERENCE_VERIFIER = "benchmarks/compare_seed_replicates.py"
PREFIX_MICROBATCHES = 4 * TEN_K.total
NATIVE_GATE_MICROBATCHES = 8
UNSEALED = ("source_manifest.json", "cpu_admitted.json", "reference_admitted.json", "checks.xml", "jobs.json",
            "training_hardware.json", "train_admitted.json", "inference_admitted.json")
TESTS = ("tests/test_flat_fine6_long.py", "tests/test_seed_replicates.py", "tests/test_scratch_hierarchy_fine6.py",
         "tests/test_scratch_architecture_report.py", "tests/test_scratch_architecture_screen.py",
         "tests/test_shared_trunk_audit.py", "tests/test_recycle_checkpoint_curve.py")
DEFAULT_EVAL = ROOT / "out/seed-replicate-eval-20261027T192020Z"
CAVEAT = ("Only the 30000-update checkpoint is post-cooldown and comparable to the final checkpoints of the "
          "10k-schedule runs. The 10000- and 20000-update checkpoints are taken mid-stable-phase at the peak "
          "learning rate; they are pre-cooldown diagnostics, not what a 10k or 20k schedule would reach (the "
          "5k-versus-10k study found near-zero all-atom lDDT before cooldown).")
LIMITS = (CAVEAT + " Single training seed (20261023) and one SDE500 tau0.01 sample per target on the reused "
    "eligible15-target/13-entry panel, which has been reused across several architecture decisions and is no "
    "longer a clean held-out test. 30k versus the 10k-schedule runs mixes three times the updates with a twice "
    "longer cooldown (4000 versus 2000 updates). Deltas against the three-seed 10k flat_fine6 mean/SD are "
    "descriptive; an n=3 SD is not a confidence interval. The paired seed-20261023 delta shares initial tensors "
    "and the recorded data-stream prefix, but the learning rates differ from update 8001 on. Declared "
    "sequence-homology exclusion does not exclude remote homology, coordinate overlap or ESMC pretraining "
    "exposure. CA is canonical OST bb_lddt; geometry entries are unnormalised OST counts. Sampler times include "
    "instrumentation and final CPU copy. No automatic architecture or checkpoint selection.")
QOS = dict(cpu="normal", train="verylong", infer="short")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text())


def regular(path):
    return Path(path).is_file() and not Path(path).is_symlink()


def bind():
    """Import the torch-dependent stack only for Slurm stages and tests, never for prepare."""
    if "base" in globals():
        return
    seedrep.bind()
    from benchmarks.screen_architectures import validate_anchor
    from mambafold.train import trainer

    globals().update(base=seedrep.base, scratch=seedrep.scratch, proof=seedrep.proof, screen=seedrep.screen,
                     tail=seedrep.tail, read=seedrep.read, exact=seedrep.exact, write_new=seedrep.write_new,
                     trainer=trainer, validate_anchor=validate_anchor)
    # The unpatched 10k-screen callables; `horizon()` wraps these, never its own wrappers.
    globals().update(_ORIGINAL=dict(expected_config=scratch.expected_config))


# ── horizon rebinding of the reused 10k-screen modules ──────────────────


class _ScheduleLabels:
    """Proxy for the screen's `audit` module whose only change is the hard-coded 10k WSD labels it writes."""

    def __init__(self, module, h):
        self._module, self._h = module, h

    def __getattr__(self, name):
        return getattr(self._module, name)

    def write_json(self, path, value):
        if Path(path).name == "schedule_verified.json":
            require(value.get("total_updates") == self._h.total and value.get("warmup_updates") == self._h.warmup
                    and value.get("stable_updates") == TEN_K.stable
                    and value.get("cooldown_updates") == TEN_K.cooldown, "unexpected screen schedule labels")
            value = dict(value, stable_updates=self._h.stable, cooldown_updates=self._h.cooldown)
        return self._module.write_json(path, value)


def horizon_verify_schedule(h):
    def verify_schedule(scheduler):
        require(scheduler.last_epoch == 0 and len(scheduler.lr_lambdas) == 2, "scheduler did not start fresh at index zero")
        for step in h.boundaries:
            require(all(math.isclose(fn(step), h.multiplier(step), abs_tol=1e-15) for fn in scheduler.lr_lambdas),
                    "existing scheduler differs from declared WSD boundaries")
        scratch.verify_group_lrs(scheduler.optimizer, 0)
        return {str(step): dict(multiplier=h.multiplier(step), muon_lr=PEAKS["muon"] * h.multiplier(step),
                                adamw_lr=PEAKS["adamw"] * h.multiplier(step)) for step in h.boundaries}
    return verify_schedule


@contextmanager
def horizon(h):
    """Rebind the screen's 10k constants to `h`; every binding is restored, also after a failure."""
    bind()
    original = _ORIGINAL["expected_config"]
    values = [(scratch, "TOTAL_STEPS", h.total), (scratch, "schedule_multiplier", h.multiplier),
              (scratch, "verify_schedule", horizon_verify_schedule(h)),
              (scratch, "expected_config", lambda arm: dict(original(arm), total_steps=h.total,
                                                            warmup_steps=h.warmup, lr_cooldown_steps=h.cooldown)),
              (scratch, "audit", _ScheduleLabels(scratch.audit, h)),
              (proof, "STEPS", h.total), (proof, "MICROBATCHES", 4 * h.total), (inputs, "MICROBATCHES", 4 * h.total),
              (inputs, "BUDGET", dict(updates=h.total, attempts=h.total, skipped_updates=0,
                                      microbatches=4 * h.total, finite_losses=4 * h.total))]
    saved = [(module, name, getattr(module, name)) for module, name, _ in values]
    try:
        for module, name, value in values:
            setattr(module, name, value)
        yield
    finally:
        for module, name, value in saved:
            setattr(module, name, value)


@contextmanager
def preflight_schedule(h):
    """The screen preflight passes the 10k cooldown literally; substitute the declared one and nothing else."""
    original = trainer.linear_warmup_lr

    def schedule(optimizer, warmup, max_lr, min_lr, *, total_steps, cooldown_steps):
        require(warmup == h.warmup and total_steps == h.total and cooldown_steps == TEN_K.cooldown,
                "unexpected preflight schedule arguments")
        return original(optimizer, warmup, max_lr, min_lr, total_steps=total_steps, cooldown_steps=h.cooldown)

    trainer.linear_warmup_lr = schedule
    try:
        yield
    finally:
        trainer.linear_warmup_lr = original


@contextmanager
def registry(arm):
    bind()
    old = scratch.VARIANTS, proof.VARIANTS
    scratch.VARIANTS, proof.VARIANTS = {arm: dict(VARIANT)}, {arm: dict(VARIANT)}
    try:
        yield
    finally:
        scratch.VARIANTS, proof.VARIANTS = old


def expected(arm, h):
    with registry(arm), horizon(h):
        return scratch.expected_config(arm)


# ── torch-free preparation ──────────────────────────────────────────────


def long_config(template, run):
    tags = list(template.get("wandb_tags") or [])
    require(tags.count("max10k") == 1, "template wandb tags lack the 10k horizon tag")
    return dict(template, config=str(Path(run) / f"configs/{ARM}.yaml"), out_dir=str(Path(run) / f"training/{ARM}"),
                wandb_name=WANDB_NAME, wandb_tags=["max30k" if tag == "max10k" else tag for tag in tags],
                **CONFIG_CHANGES)


def verify_template(template):
    require(isinstance(template, dict), "recipe template must be a mapping")
    require(all(type(template.get(key)) is type(value) and template[key] == value
                for key, value in TEMPLATE_RECIPE.items()), "template is not the seed-20261023 WSD10k recipe")
    # By value: the trainer writes the non-recycling defaults into config.json, so presence proves nothing.
    seedrep.verify_flat_construction(template, {})
    require(all(template.get(key) == value for key, value in ARCHITECTURE.items()), "template is not flat_fine6")


def reference_binding(eval_run):
    """Hash-bind the admitted seed-replicate evaluation and its seed-20261023 10k training home."""
    eval_run = Path(eval_run).resolve()
    protocol = load(eval_run / "protocol.json")
    training_run = Path(protocol["training_run"]["run"]).resolve()
    entry = protocol["arms"][REF_ARM]
    home, template = Path(entry["training"]).resolve(), Path(entry["config"]).resolve()
    require(home == training_run / f"training/{REF_ARM}" and template == training_run / f"configs/{REF_ARM}.yaml",
            "reference arm is outside its training run")
    manifest = load(training_run / "source_manifest.json")
    require(manifest.get(f"configs/{REF_ARM}.yaml") == sha(template) == protocol["training_run"]["configs"][REF_ARM],
            "reference recipe is not sealed by its training run")
    receipt, comparison = load(eval_run / f"train_admitted/{REF_ARM}.json"), load(eval_run / "comparison.json")
    initial = load(home / "initialization.json")["target_sha256"]
    stream = load(home / "audit/complete.json")["input_stream_sha256"]
    require(receipt["complete"]["input_stream_sha256"] == stream == comparison["identities"][REF_ARM]
            ["input_stream_sha256"] and comparison["identities"][REF_ARM]["initial_model_sha256"] == initial,
            "reference training identities differ between home and admitted report")
    require(comparison.get("architecture_selected") is False and comparison.get("sampler") == SAMPLER,
            "reference report selection/sampler differs")
    return dict(run=str(eval_run), code=str(eval_run / "code"), arm=REF_ARM, arms=list(FLAT_ARMS),
                files={name: sha(eval_run / name) for name in REFERENCE_FILES},
                verifier_sha256=sha(eval_run / "code" / REFERENCE_VERIFIER),
                training_run=str(training_run), training_run_manifest_sha256=sha(training_run / "source_manifest.json"),
                training=str(home), preflight=str(Path(entry["preflight"]).resolve()), u0=str(Path(entry["u0"]).resolve()),
                config=str(template), config_sha256=sha(template), model_fields=entry["model_fields"],
                initial_model_sha256=initial, input_stream_sha256=stream, inputs_sha256=sha(home / "audit/inputs.jsonl"),
                final_checkpoint_sha256=receipt["final_checkpoint_sha256"], final_ema_sha256=receipt["final_ema_sha256"])


VERIFY_FROZEN = r'''"""Check the static long-run seals, the recipe template and the reused seed-replicate bindings."""
import hashlib,json
from pathlib import Path
import yaml
RUN=Path(__file__).resolve().parent
UNSEALED=__UNSEALED__
CHANGES=__CHANGES__

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
 assert p['horizon']['total_updates']==30000 and p['horizon']['warmup_updates']==500 and p['horizon']['cooldown_updates']==4000
 assert p['horizon']['milestones']==[10000,20000,30000] and p['seed']==20261023 and type(p['seed']) is int
 assert p['sampler']=={'method':'sde','n_steps':500,'tau':.01,'seed':0,'K':2} and p['inference_depth']==2
 ref=p['reference'];old=Path(ref['run'])
 for name,digest in ref['files'].items():assert sha(old/name)==digest,name
 assert sha(old/'code'/'benchmarks/compare_seed_replicates.py')==ref['verifier_sha256']
 training=Path(ref['training_run'])
 assert sha(training/'source_manifest.json')==ref['training_run_manifest_sha256']
 seal=json.loads((training/'source_manifest.json').read_text())
 template=Path(ref['config'])
 assert sha(template)==ref['config_sha256']==seal[str(template.relative_to(training))]
 home=Path(ref['training'])
 assert json.loads((home/'initialization.json').read_text())['target_sha256']==ref['initial_model_sha256']
 assert json.loads((home/'audit/complete.json').read_text())['input_stream_sha256']==ref['input_stream_sha256']
 assert sha(home/'audit/inputs.jsonl')==ref['inputs_sha256']
 config=yaml.safe_load(Path(p['config']).read_text())
 base=yaml.safe_load(template.read_text())
 tags=['max30k' if t=='max10k' else t for t in base['wandb_tags']]
 assert config==dict(base,config=p['config'],out_dir=p['training'],wandb_name=p['wandb_name'],wandb_tags=tags,**CHANGES)
 assert config['resume'] is None and config['initialize_refiner_from'] is None and config['initialize_model_from_ema'] is False
 assert config.get('recycle_policy','legacy')=='legacy' and config.get('atom_recycling','none')=='none'
 assert sha(Path(p['panel'])/'panel.json')==p['panel_sha256'] and sha(Path(p['panel_run'])/'panel_admitted.json')==p['panel_admission_sha256']
 print('Long-run static seals, recipe template and reused seed-replicate bindings verified',flush=True)

if __name__=='__main__':main()
'''

WRITE_CPU_ADMISSION = r'''"""Bind focused CPU checks to the long flat_fine6 run."""
import hashlib,json,os,sys
from pathlib import Path
import xml.etree.ElementTree as ET
assert os.environ.get('SLURM_JOB_ID'), 'Slurm allocation required'
run=Path(__file__).resolve().parent
code=run/'code';sys.path[:0]=[str(code),str(code/'src')]
from benchmarks import train_flat_fine6_long as trial
assert Path(trial.__file__).resolve()==code/'benchmarks/train_flat_fine6_long.py'
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
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TMPDIR="/tmp/mf-long30k-${{SLURM_JOB_ID}}"
mkdir -p "$TMPDIR"
"$PY" "$RUN/verify_frozen.py"
'''
    stage = '"$PY" "$CODE/benchmarks/train_flat_fine6_long.py" {} --run "$RUN"\n'
    cpu = 'export CUDA_VISIBLE_DEVICES=""\n'
    return {"cpu_admit.sh": common + cpu + 'cd "$CODE"\n"$PY" -m pytest -q ' + " ".join(TESTS)
                + ' --junitxml="$RUN/checks.xml"\n"$PY" "$RUN/write_cpu_admission.py"\n' + stage.format("reference"),
            "train.sh": common + stage.format("train"),
            "admit_training.sh": common + cpu + stage.format("admit-training"),
            "infer.sh": common + stage.format("infer"),
            "score.sh": common + cpu + stage.format("score") + stage.format("report")}


def submit_chain(run, repo):
    """The sbatch chain for this run; prepare writes it but never executes it."""
    run = shlex.quote(str(run))
    lines = ["#!/usr/bin/env bash",
             "# Submission chain of this frozen run; prepare never executes it. GPU stages pass no -c/--mem.",
             "set -euo pipefail", f"cd {shlex.quote(str(repo))}", f"RUN={run}",
             'test ! -e "$RUN/jobs.json"',
             'CPU=$(sbatch --parsable -p cpu_only -A lcdd -q normal -c 4 --mem=16G --time=01:30:00 -J long30k-cpu '
             '-o "$RUN/logs/cpu_admit-%j.out" "$RUN/cpu_admit.sh")',
             f'TRAIN=$(sbatch --parsable -p 6000ada -A lcdd -q {QOS["train"]} --gres=gpu:1 --time=1-00:00:00 '
             '-J long30k-train -o "$RUN/logs/train-%j.out" --dependency=afterok:$CPU --kill-on-invalid-dep=yes '
             '"$RUN/train.sh")',
             'ADMIT=$(sbatch --parsable -p cpu_only -A lcdd -q normal -c 4 --mem=32G --time=02:00:00 -J long30k-admit '
             '-o "$RUN/logs/admit_training-%j.out" --dependency=afterok:$TRAIN --kill-on-invalid-dep=yes '
             '"$RUN/admit_training.sh")',
             f'INFER=$(sbatch --parsable -p 6000ada -A lcdd -q {QOS["infer"]} --gres=gpu:1 --time=03:00:00 '
             '-J long30k-infer -o "$RUN/logs/infer-%j.out" --dependency=afterok:$ADMIT --kill-on-invalid-dep=yes '
             '"$RUN/infer.sh")',
             'SCORE=$(sbatch --parsable -p cpu_only -A lcdd -q normal -c 4 --mem=16G --time=02:00:00 -J long30k-score '
             '-o "$RUN/logs/score-%j.out" --dependency=afterok:$INFER --kill-on-invalid-dep=yes "$RUN/score.sh")',
             'printf \'{"cpu_admit":"%s","train":"%s","admit_training":"%s","infer":"%s","score_report":"%s"}\\n\' '
             '"$CPU" "$TRAIN" "$ADMIT" "$INFER" "$SCORE" > "$RUN/jobs.json"', 'cat "$RUN/jobs.json"']
    return "\n".join(lines) + "\n"


def freeze_code(run):
    import shutil

    for name in curve.frozen_files(ROOT):
        (run / "code" / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, run / "code" / name)
    return seedrep.model_source({str(path.relative_to(run)): sha(path) for path in (run / "code/src").rglob("*.py")
                                 if "__pycache__" not in path.parts})


def seal(run):
    names = sorted(str(path.relative_to(run)) for path in run.rglob("*")
                   if path.is_file() and "__pycache__" not in path.parts)
    return {name: sha(run / name) for name in names if name not in UNSEALED}


def prepare(out, *, eval_run=DEFAULT_EVAL, stamp=None):
    """Freeze a new run with file operations only; every model/torch check happens later under Slurm."""
    reference = reference_binding(eval_run)
    eval_protocol = load(Path(reference["run"]) / "protocol.json")
    template = yaml.safe_load(Path(reference["config"]).read_text())
    verify_template(template)
    panel_run = Path(eval_protocol["reference10k"]["run"]).resolve()
    source = seedrep.model_source(load(Path(reference["run"]) / "source_manifest.json"))
    current = {f"code/{name}": sha(ROOT / name) for name in map(str, curve.frozen_files(ROOT)) if name.startswith("src/")}
    require(current == source, "current model source differs from the model source of the seed-replicate runs")
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = Path(out).resolve() / f"flat-fine6-long-30k-{stamp}"
    run.mkdir(parents=True, exist_ok=False)
    (run / "logs").mkdir()
    require(freeze_code(run) == source, "frozen model source differs from the copied working tree")
    config = long_config(template, run)
    (run / "configs").mkdir()
    (run / f"configs/{ARM}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    protocol = dict(created_at=datetime.now(timezone.utc).isoformat(),
        authority="User authorized one longer single-GPU scratch flat_fine6 training (seed20261023, 30000 updates, "
                  "WSD 500/25500/4000) and K2 SDE500 evaluation of its 10000/20000/30000 checkpoints; no extra arms.",
        benchmark_role=ROLE, arm=ARM, architecture="flat_fine6", architecture_fields=ARCHITECTURE, seed=SEED,
        horizon=dict(total_updates=LONG.total, warmup_updates=LONG.warmup, stable_updates=LONG.stable,
                     cooldown_updates=LONG.cooldown, milestones=list(LONG.milestones), muon_peak=0.003,
                     muon_floor=3e-05, adamw_peak=0.0001, adamw_floor=1e-06, ema_decay=0.999,
                     ckpt_interval=template["ckpt_interval"]),
        microbatches=4 * LONG.total, checkpoint_phases=PHASES, caveat=CAVEAT,
        config=config["config"], preflight=str(run / f"preflight/{ARM}"), training=config["out_dir"],
        u0=str(run / "u0"), wandb_name=WANDB_NAME, nominal_K=2, expected_parameters=PARAMETER_COUNT,
        model_fields=reference["model_fields"], declared_config_changes=dict(CONFIG_CHANGES, wandb_tags="max10k->max30k"),
        reference=reference, sampler=SAMPLER, inference_depth=DEPTH, panel=eval_protocol["panel"],
        panel_sha256=eval_protocol["panel_sha256"], panel_run=str(panel_run),
        panel_admission_sha256=sha(panel_run / "panel_admitted.json"), ost=eval_protocol["ost"],
        expected_targets=15, expected_entries=13, preflight_optimizer_updates=0, teacher_models=0,
        all_parameters_trainable=True,
        reproduction=dict(initial_tensors="gate: native and formal initial tensors equal the seed-20261023 10k run",
                          native_inputs=f"gate: the first {NATIVE_GATE_MICROBATCHES} native inputs equal its first inputs",
                          data_prefix=f"recorded, not gated: first {PREFIX_MICROBATCHES} delivered microbatches "
                                      "compared byte-exactly with the 10k run"),
        resumability="PreemptMode=OFF on 6000ada; one uninterrupted job under a 24h limit (about 2x the expected "
                     "11h). The scratch harness forbids checkpoint loads, so a resumed run would not be admitted; "
                     "ckpt_latest.pt is retained for manual salvage only.",
        limits=LIMITS)
    require(Path(protocol["panel"]).resolve() == panel_run / "panel"
            and sha(panel_run / "panel/panel.json") == protocol["panel_sha256"], "reference panel binding differs")
    (run / "protocol.json").write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
    (run / "verify_frozen.py").write_text(VERIFY_FROZEN.replace("__UNSEALED__", repr(UNSEALED))
                                          .replace("__CHANGES__", repr(CONFIG_CHANGES)))
    (run / "write_cpu_admission.py").write_text(WRITE_CPU_ADMISSION)
    for name, text in dict(scripts(run), **{"submit_chain.sh": submit_chain(run, ROOT)}).items():
        (run / name).write_text(text)
        (run / name).chmod(0o755)
    manifest = seal(run)
    with (run / "source_manifest.json").open("x") as stream:
        stream.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return dict(run=str(run), frozen_files=len(manifest), arm=ARM, source_manifest_sha256=sha(run / "source_manifest.json"))


# ── admitted contract ───────────────────────────────────────────────────


def validate_protocol(run, protocol):
    run = Path(run).resolve()
    require(protocol.get("benchmark_role") == ROLE and protocol.get("arm") == ARM and protocol.get("seed") == SEED
            and type(protocol["seed"]) is int and protocol.get("architecture_fields") == ARCHITECTURE,
            "long-run role/arm/seed/architecture differs")
    require(protocol.get("horizon", {}).get("total_updates") == LONG.total
            and protocol["horizon"].get("warmup_updates") == LONG.warmup
            and protocol["horizon"].get("cooldown_updates") == LONG.cooldown
            and protocol["horizon"].get("stable_updates") == LONG.stable
            and protocol["horizon"].get("milestones") == list(LONG.milestones)
            and protocol.get("microbatches") == 4 * LONG.total, "declared horizon differs")
    require(protocol.get("sampler") == SAMPLER and protocol.get("inference_depth") == DEPTH, "sampler differs")
    exact(protocol["sampler"], SAMPLER, "typed sampler differs")
    exact(protocol, dict(nominal_K=2, expected_parameters=PARAMETER_COUNT, expected_targets=15, expected_entries=13,
                         preflight_optimizer_updates=0, teacher_models=0, all_parameters_trainable=True,
                         checkpoint_phases=PHASES, limits=LIMITS), "long-run budget/coverage/limits differ")
    for key, path in dict(config=run / f"configs/{ARM}.yaml", preflight=run / f"preflight/{ARM}",
                          training=run / f"training/{ARM}", u0=run / "u0").items():
        require(Path(protocol[key]).resolve() == path, f"unexpected {key} path")
    reference = validate_reference_binding(protocol)
    template = yaml.safe_load(Path(reference["config"]).read_text())
    verify_template(template)
    config = yaml.safe_load(Path(protocol["config"]).read_text())
    require(config == long_config(template, run), "configuration differs from its template outside the declared changes")
    seedrep.verify_flat_construction(config, {})
    with registry(ARM), horizon(LONG):
        scratch.validate_config(config, ARM)
        exact(config, scratch.expected_config(ARM), "typed long-run configuration differs")
    require(set(protocol.get("model_fields", {})) == set(proof.MODEL_FIELDS)
            and protocol["model_fields"] == reference["model_fields"], "declared model fields differ")
    exact(protocol["model_fields"], {key: config[key] for key in proof.MODEL_FIELDS}, "typed model fields differ")
    return config


def validate_reference_binding(protocol):
    bound = protocol.get("reference", {})
    old = Path(bound.get("run", ""))
    require(old.is_absolute() and Path(bound["code"]).resolve() == old.resolve() / "code" and bound["arm"] == REF_ARM
            and bound["arms"] == list(FLAT_ARMS) and set(bound["files"]) == set(REFERENCE_FILES),
            "reference binding fields differ")
    for name, digest in bound["files"].items():
        require(proof.valid_sha(digest) and regular(old / name) and sha(old / name) == digest, f"reference {name} changed")
    require(sha(old / "code" / REFERENCE_VERIFIER) == bound["verifier_sha256"], "reference evaluator changed")
    current = reference_binding(old)
    require(current == bound, "reference training home or recipe changed")
    return bound


def validate_panel(protocol):
    panel_run = Path(protocol["panel_run"]).resolve()
    require(Path(protocol["panel"]).resolve() == panel_run / "panel" and sha(panel_run / "panel/panel.json")
            == protocol["panel_sha256"] and sha(panel_run / "panel_admitted.json") == protocol["panel_admission_sha256"],
            "panel binding differs")
    targets = base.validate_panel(panel_run, read(panel_run / "protocol.json"))
    require(len(targets) == protocol["expected_targets"]
            and len({row["entry_id"] for row in targets.values()}) == protocol["expected_entries"],
            "panel must contain fifteen targets and thirteen entries")
    return targets


def modules():
    return (Path(__file__), Path(seedrep.__file__), Path(curve.__file__), Path(inputs.__file__), Path(base.__file__),
            Path(scratch.__file__), Path(proof.__file__), Path(screen.__file__), Path(tail.__file__),
            Path(base.audit.__file__), Path(base.canonical.__file__))


def validate_run(run):
    bind()
    run = Path(run).resolve()
    manifest = read(run / "source_manifest.json")
    require(not set(manifest) & set(UNSEALED), "manifest cannot seal itself or later admissions")
    base.seals(run, manifest)
    for path in modules():
        require(manifest.get(base.relative(run, path)) == sha(path), "unsealed long-run dependency")
    require(manifest.get("protocol.json") == sha(run / "protocol.json"), "unsealed long-run protocol")
    exact(read(run / "cpu_admitted.json"), dict(admitted=True, source_manifest_sha256=sha(run / "source_manifest.json")),
          "CPU admission binding differs")
    protocol = read(run / "protocol.json")
    config = validate_protocol(run, protocol)
    require(manifest.get(base.relative(run, protocol["config"])) == sha(protocol["config"]), "unsealed configuration")
    return dict(run=run, protocol=protocol, manifest=manifest, config=config, targets=validate_panel(protocol),
                source_manifest_sha256=sha(run / "source_manifest.json"), protocol_sha256=sha(run / "protocol.json"),
                panel_admission_sha256=protocol["panel_admission_sha256"])


def unchanged(context):
    fresh = validate_run(context["run"])
    require(base.context_binding(fresh) == base.context_binding(context) and fresh["targets"] == context["targets"],
            "long-run sources/protocol/panel changed")


# ── reused seed-replicate points, admitted by their own frozen evaluator ──


REFERENCE_SCRIPT = r'''
import sys
from pathlib import Path
code, run, destination = (Path(value).resolve() for value in sys.argv[1:4])
sys.path[:0] = [str(code), str(code / "src")]
from benchmarks import compare_seed_replicates as trial
assert Path(trial.__file__).resolve() == code / "benchmarks/compare_seed_replicates.py"
trial.bind()
context = trial.validate_run(run)
inference = trial.admitted_inference(context)
binding = trial.base.context_binding(context)
comparison = trial.read(run / "comparison.json")
trial.exact(comparison, dict(benchmark_role=trial.ROLE, sampler=trial.SAMPLER, nominal_K=2, inference_depth=2,
    architecture_selected=False, limits=trial.LIMITS, **binding), "seed-replicate report context differs")
trial.require(comparison["inference_hardware"] == inference["hardware"], "seed-replicate report hardware differs")
targets, rows = context["targets"], {}
arms = [trial.arm_name("flat_fine6", seed) for seed in trial.SEEDS]
for arm in arms:
    score = trial.read(run / f"scores/{arm}.json")
    trial.exact(score, dict(admitted=True, arm=arm, coverage=len(targets),
        inference_admission_sha256=trial.sha(run / "inference_admitted.json"), **binding), "score admission differs")
    directory = trial.base.evaluation(run, arm)
    path = directory / "K2/scores/summary.json"
    trial.require(trial.sha(path) == score["summary_sha256"]
        and trial.sha(context["protocol"]["ost"]) == score["ost_executable_sha256"], "score evidence changed")
    summary = trial.read(path)
    values = trial.base.verify_canonical(summary, directory, context)
    trial.require(values == score["metrics"] == comparison["per_target"][arm]
        and summary["openstructure"] == score["scorer"], "canonical vectors/scorer differ")
    geometry = trial.tail.geometry_diagnostics(directory, targets, values, score["scorer"], 2)
    totals = {field: geometry["totals"][field] for field in trial.GEOMETRY}
    trial.require(totals == comparison["geometry_totals"][arm], "geometry totals differ")
    receipt_path = trial.base.training_receipt_path(run, arm)
    trial.require(trial.sha(receipt_path) == inference["training_receipts"][arm], "training receipt changed")
    receipt = trial.read(receipt_path)
    rows[arm] = dict(per_target=values, runs=comparison["runs"][arm], geometry_totals=totals,
        costs=comparison["costs"][arm], identity=comparison["identities"][arm], scorer=score["scorer"],
        checkpoint_sha256=inference["arms"][arm]["checkpoint_sha256"], ema_sha256=inference["arms"][arm]["ema_sha256"],
        config=receipt["config"], complete=receipt["complete"])
spread = comparison["architectures"]["flat_fine6"]["new_code_seeds"]
trial.require(spread["runs"] == arms, "three-seed flat_fine6 grouping differs")
trial.unchanged(context)
trial.write_new(destination, dict(admitted=True, run=str(run), **binding,
    comparison_sha256=trial.sha(run / "comparison.json"), inference_admission_sha256=trial.sha(run / "inference_admitted.json"),
    verifier_sha256=trial.sha(Path(trial.__file__)), inference_hardware=inference["hardware"], arms=rows, spread=spread,
    seed_pairing=comparison["seed_pairing"], architecture_selected=comparison["architecture_selected"],
    limits=comparison["limits"], targets=sorted(targets)))
'''


def reference_path(run):
    return Path(run) / "reference_admitted.json"


def reference(context):
    """Admit the three seed-replicate flat_fine6 points with their own frozen evaluator in an isolated interpreter."""
    base.no_outputs([reference_path(context["run"])])
    bound = context["protocol"]["reference"]
    code = Path(bound["code"]).resolve()
    subprocess.run([sys.executable, "-c", REFERENCE_SCRIPT, str(code), bound["run"], str(reference_path(context["run"]))],
                   cwd=code, env=dict(os.environ, PYTHONPATH=f"{code}:{code / 'src'}"), check=True)
    return admitted_reference(context)


def admitted_reference(context):
    bound, targets = context["protocol"]["reference"], context["targets"]
    value = read(reference_path(context["run"]))
    old = Path(bound["run"])
    exact(value, dict(admitted=True, run=str(old.resolve()), comparison_sha256=bound["files"]["comparison.json"],
                      inference_admission_sha256=bound["files"]["inference_admitted.json"],
                      source_manifest_sha256=bound["files"]["source_manifest.json"],
                      protocol_sha256=bound["files"]["protocol.json"], verifier_sha256=bound["verifier_sha256"],
                      architecture_selected=False), "reference admission binding differs")
    require(value.get("targets") == sorted(targets) and set(value.get("arms", {})) == set(FLAT_ARMS),
            "reference coverage differs")
    base.validate_hardware(value.get("inference_hardware"), training=True)
    for arm, row in value["arms"].items():
        require(set(row["per_target"]) == set(targets) and row["runs"] == seedrep.run_quality(targets, row["per_target"]),
                "reference per-target vectors and run means differ")
    ref = value["arms"][REF_ARM]
    require(ref["identity"]["initial_model_sha256"] == bound["initial_model_sha256"]
            and ref["identity"]["input_stream_sha256"] == bound["input_stream_sha256"]
            and ref["identity"]["ema_sha256"] == ref["ema_sha256"] == bound["final_ema_sha256"]
            and ref["checkpoint_sha256"] == bound["final_checkpoint_sha256"], "reference seed-20261023 identity differs")
    for metric in QUALITY:
        for kind in KINDS:
            recorded = value["spread"][metric][kind]
            fresh = seedrep.spread(value["arms"][arm]["runs"][metric][kind] for arm in FLAT_ARMS)
            require(set(recorded) == set(fresh) and all(math.isclose(recorded[key], fresh[key], rel_tol=0, abs_tol=1e-12)
                    if isinstance(fresh[key], float) else recorded[key] == fresh[key] for key in fresh if key != "values")
                    and recorded["values"] == fresh["values"], "reference three-seed spread differs")
    require(len({json.dumps(row["scorer"], sort_keys=True) for row in value["arms"].values()}) == 1,
            "reference points used different canonical scorers")
    return value


# ── training ────────────────────────────────────────────────────────────


def reference_inputs(protocol):
    bound = protocol["reference"]
    values, digest = inputs.read_inputs(Path(bound["training"]) / "audit/inputs.jsonl", PREFIX_MICROBATCHES)
    require(digest == bound["input_stream_sha256"], "reference input audit changed")
    return values


def check_native(native, protocol, reference_values):
    """Fail before formal training when fresh weights or the first native inputs differ from the 10k run."""
    require(native.get("arm") == ARM and native.get("seed") == SEED, "native receipt arm/seed differs")
    require(native.get("model_sha256") == protocol["reference"]["initial_model_sha256"],
            "native fresh weights differ from the seed-20261023 10k initial tensors")
    require([row.get("input_sha256") for row in native.get("rows", [])]
            == list(reference_values[:NATIVE_GATE_MICROBATCHES]),
            "first native inputs differ from the seed-20261023 10k data stream")


def hardware_path(run):
    return Path(run) / "training_hardware.json"


def train(context):
    import torch

    run, protocol = context["run"], context["protocol"]
    preflight, home = Path(protocol["preflight"]), Path(protocol["training"])
    probe = Path(protocol["u0"]) / f"{ARM}-{LONG.total}"
    base.no_outputs((preflight, home, probe, hardware_path(run), training_receipt_path(run)))
    device = base.hardware(training=True)
    torch.set_num_threads(1)
    admitted_reference(context)
    values = reference_inputs(protocol)
    config = Path(protocol["config"])
    with registry(ARM), horizon(LONG):
        with preflight_schedule(LONG):
            scratch.preflight(SimpleNamespace(arm=ARM, config=config, out=preflight))
        exact(read(preflight / "initialization.json"), dict(parameters=PARAMETER_COUNT,
              trainable_parameters=PARAMETER_COUNT, prelude_layers=3, shared_trunk_layers=6, hierarchy=False),
              "native model layout differs before formal training")
        check_native(read(preflight / "native_verified.json"), protocol, values)
        gc.collect()
        torch.cuda.empty_cache()
        require(base.hardware(training=True) == device, "preflight changed physical GPU")
        scratch.train(SimpleNamespace(arm=ARM, config=config, out=home, native_receipt=preflight / "native_verified.json"))
        screen.probe(SimpleNamespace(checkpoint=home / f"ckpt_{LONG.total:07d}.pt", out=probe, audit=home / "audit",
                                     batches=16, expected_step=LONG.total, validate_output=validate_anchor))
    require(base.hardware(training=True) == device, "training/probe changed physical GPU")
    unchanged(context)
    write_new(hardware_path(run), dict(admitted=True, **base.context_binding(context), hardware=device))


def verify_lrs(h, used, metrics, schedule, cfg):
    """`report_scratch_architecture_screen.verify_lrs` on horizon `h`: labels, boundaries and every applied update."""
    exact(schedule, dict(total_updates=h.total, warmup_updates=h.warmup, stable_updates=h.stable,
                         cooldown_updates=h.cooldown), "wrong WSD horizon")
    require(set(schedule.get("boundaries", {})) == {str(step) for step in h.boundaries}, "missing WSD boundary evidence")
    for step in h.boundaries:
        row, factor = schedule["boundaries"][str(step)], h.multiplier(step)
        for key, value in dict(multiplier=factor, muon_lr=PEAKS["muon"] * factor, adamw_lr=PEAKS["adamw"] * factor).items():
            require(proof.finite(row.get(key)) and math.isclose(row[key], value, rel_tol=1e-12, abs_tol=1e-15),
                    "wrong WSD boundary")
    require(len(used) == h.total, "missing actual optimizer-step LR evidence")
    for index, row in enumerate(used):
        exact(row, dict(update=index + 1, schedule_index=index), "reordered optimizer LR audit")
        groups = row.get("groups", [])
        require(len(groups) == 2 and {group.get("kind") for group in groups} == set(PEAKS), "wrong optimizer LR groups")
        for group in groups:
            require(proof.finite(group.get("lr")) and math.isclose(group["lr"], PEAKS[group["kind"]] * h.multiplier(index),
                    rel_tol=1e-12, abs_tol=1e-15), "actual used LR differs from WSD")
    interval = cfg["log_interval"]
    require([row.get("step") for row in metrics] == list(range(interval, h.total + 1, interval)),
            "missing/reordered training metric intervals")
    previous = 0
    for row in metrics:
        require(row.get("applied_updates") == row["step"], "metric update count differs")
        for kind, peak in PEAKS.items():
            mean = peak * statistics.fmean(h.multiplier(step) for step in range(previous + 1, row["step"] + 1))
            require(proof.finite(row.get(f"lr_{kind}")) and math.isclose(row[f"lr_{kind}"], mean, rel_tol=1e-10,
                    abs_tol=1e-14), "logged next-step LR interval differs")
        previous = row["step"]


def verify_checkpoints(home, cfg, h, gt, runtime):
    """Exactly the declared milestones are retained, ckpt_latest names the final one, and each binds its step."""
    numbered = sorted(path.name for path in home.glob("ckpt_[0-9][0-9][0-9][0-9][0-9][0-9][0-9].pt"))
    require(numbered == [f"ckpt_{step:07d}.pt" for step in h.milestones], "retained checkpoints differ from milestones")
    latest = home / "ckpt_latest.pt"
    require(latest.is_symlink() and os.readlink(latest) == f"ckpt_{h.total:07d}.pt", "ckpt_latest is not the final checkpoint")
    rows = {}
    for step in h.milestones:
        path = home / f"ckpt_{step:07d}.pt"
        require(regular(path), f"missing regular checkpoint {path.name}")
        rows[str(step)] = dict(path=path.name, checkpoint_sha256=sha(path),
                               ema_sha256=proof.checkpoint_identity(path, cfg, step, runtime=runtime))
    final = rows[str(h.total)]
    require(final["checkpoint_sha256"] == gt["final_checkpoint_sha256"] and final["ema_sha256"] == gt["final_ema_sha256"],
            "final checkpoint bytes or EMA differ from training ground truth")
    return rows


def verify_training(run, manifest, arm, entry, h):
    """Complete scratch training admission of one home on horizon `h`, by value; nothing is written."""
    run = Path(run).resolve()
    home, preflight, config_path = (Path(entry[key]) for key in ("training", "preflight", "config"))
    config_sha = sha(config_path)
    require(manifest.get(base.relative(run, config_path)) == config_sha, "configuration is not sealed")
    cfg, declared = read(home / "config.json"), yaml.safe_load(config_path.read_text())
    require(all(cfg.get(key) == value for key, value in declared.items() if key not in proof.IGNORED_CONFIG),
            "actual training configuration differs from source YAML")
    fields = entry["model_fields"]
    require(set(fields) == set(proof.MODEL_FIELDS) and entry["nominal_K"] == cfg["latent_loops"]
            and all(type(cfg.get(key)) is type(value) and cfg[key] == value for key, value in fields.items()),
            "declared architecture differs")
    initial, complete = read(home / "audit/initial.json"), read(home / "audit/complete.json")
    receipt, gt = read(home / "initialization.json"), read(home / "ground_truth_verified.json")
    native_path = preflight / "native_verified.json"
    native = read(native_path)
    seedrep.verify_flat_construction(cfg, native)
    require(receipt.get("native_receipt_sha256") == gt.get("native_receipt_sha256") == sha(native_path)
            and Path(receipt["native_receipt"]).resolve() == native_path.resolve(), "native receipt binding differs")
    delivered, stream = inputs.read_inputs(home / "audit/inputs.jsonl", 4 * h.total)
    require(stream == complete.get("input_stream_sha256"), "delivered input stream differs")
    exact(initial, dict(parameters=PARAMETER_COUNT, trainable_parameters=PARAMETER_COUNT), "actual parameter count differs")
    with registry(arm), horizon(h):
        proof.verify_training(cfg, arm, initial, complete, receipt, native, gt, delivered, config_sha)
    exact(gt, dict(arm=arm, optimizer_updates=h.total, microbatches=4 * h.total, final_schedule_index=h.total,
                   checkpoint_loads_during_training=0, device=base.TRAIN_GPU), "training ground truth differs")
    metrics = proof.lines(home / "audit/metrics.jsonl")
    proof.verify_metrics(metrics, cfg)
    schedule = read(home / "schedule_verified.json")
    require(native.get("schedule_boundaries") == schedule.get("boundaries"), "native/formal WSD boundaries differ")
    verify_lrs(h, proof.lines(home / "audit/used_lrs.jsonl"), metrics, schedule, cfg)
    native_initial, native_cfg = read(preflight / "initialization.json"), proof.read_snapshot_config(preflight)
    require(all(native_cfg.get(key) == cfg.get(key) for key in set(native_cfg) | set(cfg)
                if key not in proof.IGNORED_CONFIG | {"requested_num_workers"}),
            "native checkpoint full configuration differs from formal training")
    for directory, saved, expected_cfg in ((home, receipt, cfg), (preflight, native_initial, native_cfg)):
        require(saved.get("initialization") == "scratch"
                and saved.get("target_sha256") == saved.get("initial_ema_sha256") == initial["model_sha256"],
                "fresh snapshot model binding differs")
        require(sha(directory / "initial_ema.pt") == saved.get("initial_snapshot_sha256"), "initial checkpoint bytes differ")
        proof.checkpoint_identity(directory / "initial_ema.pt", expected_cfg, 0, initial["model_sha256"],
                                  initial["model_sha256"], runtime=receipt["mamba3_runtime"])
    milestones = verify_checkpoints(home, cfg, h, gt, receipt["mamba3_runtime"])
    probe_path = Path(entry["u0"]) / f"{arm}-{h.total}/u0.json"
    probe = read(probe_path)
    require(probe.get("checkpoint_sha256") == gt["final_checkpoint_sha256"]
            and probe.get("initial_model_sha256") == gt["final_ema_sha256"] and probe.get("checkpoint_step") == h.total
            and probe.get("updates") == 0 and probe.get("weight_source") == "checkpoint EMA; no probe updates",
            "final u0 probe EMA/checkpoint binding differs")
    rows = probe.get("rows", [])
    prelude = cfg["n_trunk"] - cfg["shared_trunk_layers"] * cfg["latent_loops"]
    require(len(rows) == 16 and [row.get("microbatch") for row in rows] == list(range(1, 17))
            and tuple(row.get("input_sha256") for row in rows) == delivered[:16]
            and all(row.get("u0_matches_actual_K0") is True and row.get("detached_anchor_gradient_verified") is True
                    and row.get("core_passes") == cfg["latent_loops"] and row.get("readouts") == cfg["latent_loops"] + 1
                    and row.get("prelude_layers") == prelude
                    and all(math.isfinite(v) for v in row.values() if isinstance(v, float)) for row in rows),
            "final probe input/routing coverage differs")
    require(not (home / ".requeue_requested").exists(), "interrupted formal training")
    previous, weighted = 0, 0.0
    for row in metrics:
        require(proof.finite(row.get("perf_train_step_ms_max"), positive=True), "invalid step timer")
        weighted += row["perf_train_step_ms_max"] * (row["step"] - previous)
        previous = row["step"]
    for key in ("steady_mean_step_ms", "peak_allocated_gib", "peak_reserved_gib", "end_to_end_seconds"):
        require(proof.finite(complete.get(key), positive=True), "invalid measured training cost")
    paths = [config_path, home / "config.json", home / "audit/initial.json", home / "audit/complete.json",
             home / "initialization.json", home / "ground_truth_verified.json", native_path, probe_path,
             preflight / "initialization.json", home / "audit/inputs.jsonl", home / "audit/metrics.jsonl",
             home / "audit/used_lrs.jsonl", home / "schedule_verified.json", home / "initial_ema.pt",
             preflight / "initial_ema.pt", *(home / row["path"] for row in milestones.values())]
    evidence = {base.relative(run, path): sha(path) for path in paths}
    return dict(config=cfg, complete=dict(complete, probe_input_sha256=list(delivered[:16])), delivered=delivered,
                milestones=milestones, evidence=evidence, all_update_weighted_mean_step_ms=weighted / previous,
                initial_model_sha256=initial["model_sha256"], final_checkpoint_sha256=gt["final_checkpoint_sha256"],
                final_ema_sha256=gt["final_ema_sha256"])


def prefix_comparison(values, reference_values):
    """Byte-exact comparison of the first delivered microbatches with the 10k run; recorded, never gated."""
    count = len(reference_values)
    require(len(values) >= count, "long run delivered fewer microbatches than the compared prefix")
    divergent = next((index for index in range(count) if values[index] != reference_values[index]), None)
    stream = hashlib.sha256()
    for value in values[:count]:
        stream.update(bytes.fromhex(value))
    return dict(compared_microbatches=count, identical=divergent is None,
                matching_prefix_microbatches=count if divergent is None else divergent,
                first_divergent_microbatch=None if divergent is None else divergent + 1,
                prefix_stream_sha256=stream.hexdigest(), gated=False)


def training_receipt_path(run):
    return Path(run) / "train_admitted.json"


def admit_training(context):
    run, protocol = context["run"], context["protocol"]
    base.no_outputs([training_receipt_path(run)])
    admitted_reference(context)
    device = read(hardware_path(run))
    exact(device, dict(admitted=True, **base.context_binding(context)), "training hardware binding differs")
    base.validate_hardware(device.get("hardware"), training=True)
    result = verify_training(run, context["manifest"], ARM, protocol, LONG)
    bound = protocol["reference"]
    require(result["initial_model_sha256"] == bound["initial_model_sha256"],
            "formal initial tensors differ from the seed-20261023 10k run")
    native = read(Path(protocol["preflight"]) / "native_verified.json")
    check_native(native, protocol, reference_inputs(protocol))
    prefix = prefix_comparison(result["delivered"], reference_inputs(protocol))
    evidence = dict(result["evidence"], **{"training_hardware.json": sha(hardware_path(run))})
    value = dict(admitted=True, arm=ARM, **base.context_binding(context), parameters=PARAMETER_COUNT, nominal_K=2,
                 hardware=device["hardware"], config=result["config"], complete=result["complete"],
                 milestones=result["milestones"], all_update_weighted_mean_step_ms=result["all_update_weighted_mean_step_ms"],
                 final_checkpoint_sha256=result["final_checkpoint_sha256"], final_ema_sha256=result["final_ema_sha256"],
                 reproduction=dict(initial_model_sha256=result["initial_model_sha256"],
                                   identical_initial_tensors=True, native_inputs_identical=NATIVE_GATE_MICROBATCHES,
                                   reference_input_stream_sha256=bound["input_stream_sha256"], data_prefix=prefix),
                 evidence_sha256=evidence)
    base.seals(run, evidence)
    unchanged(context)
    write_new(training_receipt_path(run), value)
    return admitted_training(context)


def admitted_training(context):
    run, protocol = context["run"], context["protocol"]
    value = read(training_receipt_path(run))
    exact(value, dict(admitted=True, arm=ARM, parameters=PARAMETER_COUNT, nominal_K=2, **base.context_binding(context)),
          "training admission identity differs")
    base.seals(run, value.get("evidence_sha256"))
    home = Path(protocol["training"])
    complete = read(home / "audit/complete.json")
    require(value.get("config") == read(home / "config.json")
            and all(value["complete"].get(key) == item for key, item in complete.items()),
            "training admission differs from actual training")
    with horizon(LONG):
        exact(complete, inputs.BUDGET, "incomplete admitted training budget")
    require(set(value.get("milestones", {})) == {str(step) for step in LONG.milestones}
            and all(value["evidence_sha256"].get(base.relative(run, home / row["path"])) == row["checkpoint_sha256"]
                    for row in value["milestones"].values()), "milestone checkpoints are not sealed")
    final = value["milestones"][str(LONG.total)]
    require(final["checkpoint_sha256"] == value["final_checkpoint_sha256"] and final["ema_sha256"]
            == value["final_ema_sha256"], "final milestone binding differs")
    require(value["reproduction"]["initial_model_sha256"] == protocol["reference"]["initial_model_sha256"],
            "admitted initial tensors differ from the reference")
    base.validate_hardware(value.get("hardware"), training=True)
    require(proof.finite(value.get("all_update_weighted_mean_step_ms"), positive=True), "invalid full-update timer")
    return value


def dry_run_admission(eval_run):
    """Run the training-admission logic on the real finished 10k seed-20261023 home under the 10k horizon."""
    bind()
    bound = reference_binding(eval_run)
    training_run = Path(bound["training_run"])
    manifest = load(training_run / "source_manifest.json")
    entry = dict(config=bound["config"], preflight=bound["preflight"], training=bound["training"], u0=bound["u0"],
                 model_fields=bound["model_fields"], nominal_K=2)
    result = verify_training(training_run, manifest, REF_ARM, entry, TEN_K)
    native = read(Path(bound["preflight"]) / "native_verified.json")
    config_keys = {key: result["config"].get(key, "<absent>") for key in seedrep.ARCH_KEYS}
    prefix = prefix_comparison(result["delivered"], inputs.read_inputs(Path(bound["training"]) / "audit/inputs.jsonl",
                                                                       PREFIX_MICROBATCHES)[0])
    require(result["initial_model_sha256"] == bound["initial_model_sha256"]
            and result["final_checkpoint_sha256"] == bound["final_checkpoint_sha256"]
            and result["final_ema_sha256"] == bound["final_ema_sha256"], "dry run identities differ from the admission")
    return dict(admitted=True, horizon=dict(total=TEN_K.total, cooldown=TEN_K.cooldown, milestones=list(TEN_K.milestones)),
                home=bound["training"], trainer_written_recycle_keys=config_keys,
                native_recycle_keys={key: native.get(key, "<absent>") for key in seedrep.ARCH_KEYS},
                milestones=result["milestones"], data_prefix=prefix, evidence_files=len(result["evidence"]),
                all_update_weighted_mean_step_ms=result["all_update_weighted_mean_step_ms"],
                initial_model_sha256=result["initial_model_sha256"], final_ema_sha256=result["final_ema_sha256"])


# ── inference at the three checkpoints, canonical scoring ───────────────


def label(step):
    return f"{ARM}-{step}"


def evaluation(run, step):
    return Path(run) / f"evaluation/{label(step)}"


def verify_inference(context, step, checkpoint_sha):
    measured = proof.verify_inference(evaluation(context["run"], step), step, [DEPTH], checkpoint_sha,
                                      context["targets"], False, atom_recycling="none", n_steps=SAMPLER["n_steps"])
    for row in measured:
        seedrep.verify_flat_construction(row, {})
        require(proof.finite(row.get("peak_reserved_gib"), positive=True), "missing reserved inference memory")
    return measured


def infer(context):
    import torch
    from mambafold.model.bimamba3 import mamba3_runtime_info

    run, protocol = context["run"], context["protocol"]
    base.no_outputs([run / "inference_admitted.json", *(evaluation(run, step) for step in LONG.milestones)])
    device = base.hardware(training=True)
    torch.set_num_threads(1)
    admitted_reference(context)
    training = admitted_training(context)
    home = Path(protocol["training"])
    runtime = read(home / "initialization.json")["mamba3_runtime"]
    base.verify_external_runtime(runtime, mamba3_runtime_info())
    arms, evidence = {}, {}
    for step in LONG.milestones:
        row, directory = training["milestones"][str(step)], evaluation(run, step)
        require(sha(home / row["path"]) == row["checkpoint_sha256"], "milestone checkpoint changed")
        require(base.hardware(training=True) == device, "common physical inference GPU changed")
        with base.inference_model_runtime(runtime):
            screen.infer(SimpleNamespace(checkpoint=home / row["path"], panel=Path(protocol["panel"]), out=directory,
                                         depths=[DEPTH], n_steps=SAMPLER["n_steps"]))
        require(base.hardware(training=True) == device, "common physical inference GPU changed")
        verify_inference(context, step, row["checkpoint_sha256"])
        arms[str(step)] = dict(hardware=device, nominal_K=2, step=step, checkpoint_sha256=row["checkpoint_sha256"],
                               ema_sha256=row["ema_sha256"])
        write_new(directory / "gpu_binding.json", arms[str(step)])
        paths = [directory / name for name in ("sampling.json", "hardware.json", "gpu_binding.json")]
        paths += [directory / f"K{DEPTH}/pairs/manifest.json",
                  *(directory / f"K{DEPTH}/pairs/{target}_seed0_pred.pdb" for target in context["targets"])]
        evidence.update({base.relative(run, path): sha(path) for path in paths})
        gc.collect()
        torch.cuda.empty_cache()
    unchanged(context)
    write_new(run / "inference_admitted.json", dict(admitted=True, **base.context_binding(context), hardware=device,
        sampler=SAMPLER, inference_depth=DEPTH, coverage=len(context["targets"]), checkpoints=arms,
        training_receipt_sha256=sha(training_receipt_path(run)), evidence_sha256=evidence))


def admitted_inference(context):
    run = context["run"]
    value = read(run / "inference_admitted.json")
    exact(value, dict(admitted=True, coverage=len(context["targets"]), sampler=SAMPLER, inference_depth=DEPTH,
                      training_receipt_sha256=sha(training_receipt_path(run)), **base.context_binding(context)),
          "inference admission differs")
    require(set(value.get("checkpoints", {})) == {str(step) for step in LONG.milestones}, "incomplete checkpoint inference")
    base.validate_hardware(value.get("hardware"), training=True)
    base.seals(run, value.get("evidence_sha256"))
    training = admitted_training(context)
    for step in LONG.milestones:
        binding, row = value["checkpoints"][str(step)], training["milestones"][str(step)]
        exact(binding, dict(hardware=value["hardware"], nominal_K=2, step=step, checkpoint_sha256=row["checkpoint_sha256"],
                            ema_sha256=row["ema_sha256"]), "inference checkpoint/EMA/GPU binding differs")
        require(read(evaluation(run, step) / "gpu_binding.json") == binding, "per-checkpoint GPU binding differs")
        verify_inference(context, step, binding["checkpoint_sha256"])
    return value


def score_path(run, step):
    return Path(run) / f"scores/{label(step)}.json"


def score(context):
    run = context["run"]
    admitted_inference(context)
    base.no_outputs([path for step in LONG.milestones for path in (score_path(run, step),
                     run / f"scores/{label(step)}.failed.json", evaluation(run, step) / f"K{DEPTH}/scores")])
    failed = []
    for step in LONG.milestones:
        directory = evaluation(run, step)
        completed = subprocess.run([sys.executable, str(Path(base.canonical.__file__)), "--in-dir",
            str(directory / f"K{DEPTH}/pairs"), "--out-dir", str(directory / f"K{DEPTH}/scores"), "--ost",
            context["protocol"]["ost"], "--expected", str(len(context["targets"]))], check=False)
        path = directory / f"K{DEPTH}/scores/summary.json"
        summary = read(path) if path.is_file() else None
        if completed.returncode or summary is None or summary.get("evaluation_complete") is not True:
            write_new(run / f"scores/{label(step)}.failed.json", dict(step=step, admitted=False,
                      returncode=completed.returncode, expected=len(context["targets"]), summary=summary))
            failed.append(label(step))
            continue
        vectors = base.verify_canonical(summary, directory, context)
        unchanged(context)
        write_new(score_path(run, step), dict(admitted=True, step=step, **base.context_binding(context),
            inference_admission_sha256=sha(run / "inference_admitted.json"), coverage=len(vectors), metrics=vectors,
            summary_sha256=sha(path), scorer=summary["openstructure"], ost_executable_sha256=sha(context["protocol"]["ost"])))
    require(not failed, "canonical scoring failed or was incomplete: " + ", ".join(failed))


# ── report ──────────────────────────────────────────────────────────────


def versus_spread(value, spread):
    delta = value - spread["mean"]
    return dict(value=value, reference_mean=spread["mean"], reference_sd=spread["sd"], delta=delta,
                delta_over_sd=delta / spread["sd"] if spread["sd"] else None,
                within_reference_range=spread["min"] <= value <= spread["max"])


def summarize(targets, vectors, reference):
    """Descriptive per-checkpoint deltas; only the final checkpoint is post-cooldown. No selection."""
    require(set(vectors) == {str(step) for step in LONG.milestones}, "report requires every milestone")
    ref = reference["arms"][REF_ARM]["per_target"]
    runs = {step: seedrep.run_quality(targets, values) for step, values in vectors.items()}
    rows = {}
    for step, values in vectors.items():
        rows[step] = dict(phase=PHASES[step], post_cooldown=int(step) == LONG.total, **runs[step],
            versus_three_seed_10k={metric: {kind: versus_spread(runs[step][metric][kind], reference["spread"][metric][kind])
                                            for kind in KINDS} for metric in QUALITY},
            paired_minus_seed20261023_10k=curve.tally(tail.paired_effects(targets, ref, values)))
    steps = [str(step) for step in LONG.milestones]
    progression = {f"{late}_minus_{early}": curve.tally(tail.paired_effects(targets, vectors[early], vectors[late]))
                   for early, late in ((steps[0], steps[1]), (steps[1], steps[2]), (steps[0], steps[2]))}
    return dict(checkpoints=rows, progression=progression)


def report(context):
    run, targets = context["run"], context["targets"]
    base.no_outputs([run / "comparison.json", run / "comparison.csv"])
    inference, reference = admitted_inference(context), admitted_reference(context)
    training = admitted_training(context)
    vectors, geometry, costs, scorers = {}, {}, {}, []
    for step in LONG.milestones:
        key, value = str(step), read(score_path(run, step))
        exact(value, dict(admitted=True, step=step, coverage=len(targets),
              inference_admission_sha256=sha(run / "inference_admitted.json"), **base.context_binding(context)),
              "score admission identity differs")
        directory = evaluation(run, step)
        path = directory / f"K{DEPTH}/scores/summary.json"
        require(sha(path) == value["summary_sha256"] and sha(context["protocol"]["ost"]) == value["ost_executable_sha256"],
                "canonical score evidence changed")
        summary = read(path)
        vectors[key] = base.verify_canonical(summary, directory, context)
        require(vectors[key] == value["metrics"] and summary["openstructure"] == value["scorer"], "canonical vectors changed")
        scorers.append(value["scorer"])
        diagnostics = tail.geometry_diagnostics(directory, targets, vectors[key], value["scorer"], DEPTH)
        geometry[key] = {field: diagnostics["totals"][field] for field in GEOMETRY}
        measured = read(directory / "sampling.json")
        costs[key] = dict(inference_seconds=statistics.fmean(row["seconds"] for row in measured),
                          inference_allocated_gib=max(row["peak_allocated_gib"] for row in measured),
                          inference_reserved_gib=max(row["peak_reserved_gib"] for row in measured))
    require(all(scorer == reference["arms"][REF_ARM]["scorer"] for scorer in scorers),
            "long-run and reference canonical scorers differ")
    stats = summarize(targets, vectors, reference)
    ref_geometry = {arm: reference["arms"][arm]["geometry_totals"] for arm in FLAT_ARMS}
    ref_seconds = [reference["arms"][arm]["costs"]["inference_seconds"] for arm in FLAT_ARMS]
    for key, row in stats["checkpoints"].items():
        row.update(geometry_totals=geometry[key], costs=costs[key],
            geometry_minus_seed20261023_10k={field: geometry[key][field] - ref_geometry[REF_ARM][field] for field in GEOMETRY},
            geometry_minus_three_seed_10k_mean={field: geometry[key][field] - statistics.fmean(
                ref_geometry[arm][field] for arm in FLAT_ARMS) for field in GEOMETRY},
            inference_seconds_minus_three_seed_10k_mean=costs[key]["inference_seconds"] - statistics.fmean(ref_seconds))
    complete = training["complete"]
    result = dict(benchmark_role=ROLE, arm=ARM, sampler=SAMPLER, nominal_K=2, inference_depth=DEPTH,
        **base.context_binding(context), coverage=dict(targets=len(targets), entries=len(seedrep.entries(targets))),
        metric_names=dict(AA="lddt", CA="bb_lddt", TM="tm_score"), caveat=CAVEAT,
        horizon=context["protocol"]["horizon"], per_target=vectors, **stats,
        reference=dict(three_seed_10k_spread=reference["spread"], seed20261023_10k=dict(
            runs=reference["arms"][REF_ARM]["runs"], geometry_totals=ref_geometry[REF_ARM],
            identity=reference["arms"][REF_ARM]["identity"], costs=reference["arms"][REF_ARM]["costs"]),
            admission_sha256=sha(reference_path(run)), inference_hardware=reference["inference_hardware"]),
        seed_pairing=dict(training["reproduction"], same_inference_gpu_as_reference=
                          inference["hardware"]["uuid"] == reference["inference_hardware"]["uuid"]),
        training=dict(steady_train_ms=complete["steady_mean_step_ms"],
                      all_update_train_ms=training["all_update_weighted_mean_step_ms"],
                      end_to_end_seconds=complete["end_to_end_seconds"], train_allocated_gib=complete["peak_allocated_gib"],
                      train_reserved_gib=complete["peak_reserved_gib"], training_gpu_uuid=training["hardware"]["uuid"],
                      milestones=training["milestones"]),
        inference_hardware=inference["hardware"], architecture_selected=reference["architecture_selected"],
        checkpoint_selected=False, limits=LIMITS)
    unchanged(context)
    write_new(run / "comparison.json", result)
    rows = [dict(step=int(key), phase=row["phase"], post_cooldown=row["post_cooldown"], coverage=len(vectors[key]),
                 **{metric: row[metric]["chain_mean"] for metric in QUALITY},
                 **{f"{metric}_equal_entry": row[metric]["equal_entry_mean"] for metric in QUALITY},
                 **{f"{metric}_minus_3seed_mean": row["versus_three_seed_10k"][metric]["chain_mean"]["delta"]
                    for metric in QUALITY},
                 **{f"{metric}_minus_s20261023_10k": row["paired_minus_seed20261023_10k"][metric]["chain_mean_delta"]
                    for metric in QUALITY}, **row["geometry_totals"], **row["costs"])
            for key, row in result["checkpoints"].items()]
    with (run / "comparison.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


STAGES = ("reference", "train", "admit-training", "infer", "score", "report")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "dry-run-admission", *STAGES))
    parser.add_argument("--run", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--eval-run", type=Path)
    args = parser.parse_args()
    if args.stage == "prepare":
        if args.out is None or args.run is not None:
            parser.error("prepare requires --out and no --run")
        print(json.dumps(prepare(args.out, eval_run=args.eval_run or DEFAULT_EVAL)))
        return
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("long-run execution requires a Slurm allocation, including CPU stages")
    if args.stage == "dry-run-admission":
        if args.run is not None or args.out is not None:
            parser.error("dry-run-admission reads only --eval-run")
        print(json.dumps(dry_run_admission(args.eval_run or DEFAULT_EVAL), indent=2))
        return
    if args.run is None or args.out is not None or args.eval_run is not None:
        parser.error(f"{args.stage} requires --run and no --out/--eval-run")
    context = validate_run(args.run)
    dict(reference=reference, train=train, score=score, report=report, infer=infer,
         **{"admit-training": admit_training})[args.stage](context)


if __name__ == "__main__":
    main()
