"""Batch-confound probe: flat_fine6 with four distinct proteins per update at a fixed 16 samples per update.

Every screen so far trained on one GPU with batch 1 x copies 4 x grad accum 4: sixteen samples of one protein
per update. Arm `ddp4_c4a1` runs the unmodified `scripts/train.py` under torchrun on four ranks of one node with
batch 1 x copies 4 x accum 1, so each update averages four proteins x four copies. Everything else is the
admitted seed-20261023 flat_fine6 WSD10k recipe; the learning rate is not rescaled. The final EMA checkpoint is
evaluated exactly like the three-seed flat_fine6 reference (SDE500 tau0.01 seed0 K2, the eligible 15-target/
13-entry panel, canonical OpenStructure) and compared against that reference by description only.

The single-process audit stack (compare_full_refinement.train and the screens built on it) refuses DDP, so the
training stage bypasses it; `PROVENANCE_LOST` names what that costs. The model source is copied from the
reference evaluation's sealed code, not from the working tree, so model code is byte-identical to the reference.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import shlex
import shutil
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

# Torch-free imports; the ML stack is bound lazily so prepare never imports torch.
from benchmarks import compare_seed_replicates as seedrep  # noqa: E402
from benchmarks import evaluate_recycle_checkpoint_curve as curve  # noqa: E402
from benchmarks import input_equivalence as inputs  # noqa: E402
from benchmarks import train_flat_fine6_long as long  # noqa: E402

sha = inputs.sha256

SEED = long.SEED
H = long.TEN_K
ARM = f"ddp4_c4a1_s{SEED}"
REF_ARM = long.REF_ARM
FLAT_ARMS = long.FLAT_ARMS
WORLD = 4
STEPS = H.total
SAMPLER = dict(seedrep.SAMPLER)
DEPTH = 2
QUALITY, GEOMETRY, KINDS = seedrep.QUALITY, seedrep.GEOMETRY, long.KINDS
PEAKS = dict(long.PEAKS)
TRAIN_GPU = "NVIDIA RTX 6000 Ada Generation"
CONFIG_CHANGES = dict(grad_accum_steps=1)
WANDB_NAME = "batch-confound-ddp4-c4a1"
TAG_CHANGE = ("single-GPU-comparison", "ddp4-c4a1-batch-confound")
CHECKPOINTS = (5000, 10000)
ROLE = "batch-confound probe: four distinct proteins x four copies per update versus one protein x sixteen"
NCCL = dict(NCCL_P2P_DISABLE="1", probe="benchmarks/probe_nccl.py", timeout_seconds=900)
SKIP_PATTERNS = ("OOM/NaN", "gnorm spike skipped")
PROVENANCE_LOST = (
    "per-microbatch delivered-input hashes and the input-stream digest (audit/inputs.jsonl)",
    "per-update actual optimizer learning-rate audit (audit/used_lrs.jsonl); only 100-update interval means "
    "from the trainer's own metrics.jsonl are checked against WSD",
    "initial-tensor identity with the seed-20261023 reference (initialization.json); same seed and model source, "
    "and DDP broadcasts rank-0 parameters, but it is not hashed",
    "native preflight receipt, schedule_verified.json, ground_truth_verified.json and the u0 routing probe",
    "per-update attempt/finite-loss counters; replaced by metrics update_applied==1 for every interval, "
    "data_state micro_batches_consumed==10000 and no skip lines in the torchrun log",
)
LIMITS = ("One training seed (20261023) and one SDE500 tau0.01 sample per target on the reused eligible "
          "15-target/13-entry panel, which has informed several decisions and is no longer a clean held-out test. "
          "The comparison is descriptive against an n=3 seed spread; delta/SD is not a confidence interval. The arm "
          "changes three things at once relative to the reference: four distinct proteins per update, four ranks "
          "under DDP (gradient all-reduce, per-rank RNG streams) and accum 1. The four proteins of an update are "
          "consecutive in a length-sorted megabatch, so they are length-similar, unlike the uniform single-protein "
          "draw. Ten thousand updates see four times as many proteins. Paired seed-20261023 deltas share the seed "
          "but not the data stream. The single-process audit stack is bypassed; see provenance_lost. CA is canonical "
          "OST bb_lddt; geometry entries are unnormalised OST counts. No architecture or batch policy is selected.")
QOS = dict(train="short", infer="short")
UNSEALED = ("source_manifest.json", "cpu_admitted.json", "reference_admitted.json", "checks.xml", "jobs.json",
            "training_run.json", "nccl_probe.json", "train_admitted.json", "inference_admitted.json")
TESTS = ("tests/test_batch_confound.py", "tests/test_core_runtime.py", "tests/test_flat_fine6_long.py",
         "tests/test_seed_replicates.py")


def repo_root():
    """The repository, also when this module runs from a frozen run's code copy (<repo>/out/<run>/code)."""
    if ROOT.name == "code" and (ROOT.parent / "source_manifest.json").is_file():
        return ROOT.parents[2]
    return ROOT


# The only checks a CPU node may skip: they need a CUDA allocation or driver.
CUDA_ONLY_TESTS = ("test_batched_ema_preserves_scalar_updates_and_buffers[cuda]",
                   "test_batch_padding_preserves_fields_and_loader_pinning[True]")
DEFAULT_EVAL = repo_root() / long.DEFAULT_EVAL.relative_to(long.ROOT)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text())


def bind():
    """Import the torch-dependent stack only for Slurm stages and tests, never for prepare."""
    if "base" in globals():
        return
    long.bind()
    globals().update(base=long.base, proof=long.proof, screen=long.screen, tail=long.tail, read=long.read,
                     exact=long.exact, write_new=long.write_new)


# ── torch-free preparation ──────────────────────────────────────────────


def derive_config(template, run):
    tags = list(template.get("wandb_tags") or [])
    require(tags.count(TAG_CHANGE[0]) == 1, "template wandb tags lack the single-GPU tag")
    return dict(template, config=str(Path(run) / f"configs/{ARM}.yaml"), out_dir=str(Path(run) / f"training/{ARM}"),
                wandb_name=WANDB_NAME, wandb_tags=[TAG_CHANGE[1] if tag == TAG_CHANGE[0] else tag for tag in tags],
                **CONFIG_CHANGES)


def reference_src(reference):
    """Model source of the reference evaluation, by its sealed manifest; prepare copies exactly these bytes."""
    manifest = load(Path(reference["run"]) / "source_manifest.json")
    source = seedrep.model_source(manifest)
    code = Path(reference["code"])
    for name, digest in source.items():
        require(sha(code.parent / name) == digest, f"reference model source changed: {name}")
    return source


def freeze_code(run, reference):
    source = reference_src(reference)
    for name in curve.frozen_files(ROOT):
        if str(name).startswith("src/"):
            continue
        (run / "code" / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, run / "code" / name)
    code = Path(reference["code"])
    for name in source:
        relative = Path(name).relative_to("code")
        (run / "code" / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(code / relative, run / "code" / relative)
    frozen = {str(path.relative_to(run)): sha(path) for path in (run / "code/src").rglob("*.py")
              if "__pycache__" not in path.parts}
    require(frozen == source, "frozen model source differs from the reference evaluation")
    return source


def seal(run):
    names = sorted(str(path.relative_to(run)) for path in run.rglob("*")
                   if path.is_file() and "__pycache__" not in path.parts)
    return {name: sha(run / name) for name in names if name not in UNSEALED}


VERIFY_FROZEN = r'''"""Check the static batch-confound seals, its recipe and the reference model source."""
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
 assert p['world_size']==4 and p['seed']==20261023 and p['samples_per_update']==16
 assert p['per_rank']=={'batch_size':1,'copies_per_protein':4,'grad_accum_steps':1}
 assert p['sampler']=={'method':'sde','n_steps':500,'tau':.01,'seed':0,'K':2} and p['inference_depth']==2
 ref=p['reference'];old=Path(ref['run'])
 for name,digest in ref['files'].items():assert sha(old/name)==digest,name
 src={k:v for k,v in json.loads((old/'source_manifest.json').read_text()).items() if k.startswith('code/src/')}
 mine={str(q.relative_to(RUN)):sha(q) for q in (RUN/'code/src').rglob('*.py') if '__pycache__' not in q.parts}
 assert src==mine,'model source differs from the reference evaluation'
 template=Path(ref['config']);assert sha(template)==ref['config_sha256']
 base=yaml.safe_load(template.read_text())
 config=yaml.safe_load(Path(p['config']).read_text())
 tags=[__TAG__[1] if t==__TAG__[0] else t for t in base['wandb_tags']]
 assert config==dict(base,config=p['config'],out_dir=p['training'],wandb_name=p['wandb_name'],wandb_tags=tags,**CHANGES)
 assert config['resume'] is None and config['initialize_refiner_from'] is None and config['total_steps']==10000
 assert sha(Path(p['panel'])/'panel.json')==p['panel_sha256'] and sha(Path(p['panel_run'])/'panel_admitted.json')==p['panel_admission_sha256']
 print('Batch-confound static seals, recipe and reference model source verified',flush=True)

if __name__=='__main__':main()
'''


def scripts(run, repo):
    common = f'''#!/usr/bin/env bash
set -euo pipefail
cd {shlex.quote(str(repo))}
RUN={shlex.quote(str(run))}
CODE="$RUN/code"
PY={shlex.quote(str(repo / ".venv/bin/python"))}
export PYTHONPATH="$CODE:$CODE/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TMPDIR="/tmp/mf-batchconfound-${{SLURM_JOB_ID}}"
mkdir -p "$TMPDIR"
"$PY" "$RUN/verify_frozen.py"
'''
    stage = '"$PY" "$CODE/benchmarks/train_batch_confound.py" {} --run "$RUN"\n'
    cpu = 'export CUDA_VISIBLE_DEVICES=""\n'
    return {"cpu_admit.sh": common + cpu + '(cd "$CODE" && "$PY" -m pytest -q -p no:cacheprovider ' + " ".join(TESTS)
                + ' --junitxml="$RUN/checks.xml")\n' + stage.format("admit-cpu") + stage.format("reference"),
            "train.sh": common + f'export NCCL_P2P_DISABLE={NCCL["NCCL_P2P_DISABLE"]}\n' + stage.format("train"),
            "infer.sh": common + stage.format("infer"),
            "score.sh": common + cpu + stage.format("score") + stage.format("report")}


def submit_chain(run, repo, cpu_job=None):
    """The GPU sbatch chain; prepare writes it and never executes it. GPU stages pass no -c/--mem."""
    run_q = shlex.quote(str(run))
    dep = f"--dependency=afterok:{cpu_job} --kill-on-invalid-dep=yes " if cpu_job else ""
    lines = ["#!/usr/bin/env bash",
             "# GPU chain of this frozen run (after its CPU admission). GPU stages pass no -c/--mem.",
             "set -euo pipefail", f"cd {shlex.quote(str(repo))}", f"RUN={run_q}",
             'test -f "$RUN/cpu_admitted.json" -a -f "$RUN/reference_admitted.json"', 'test ! -e "$RUN/jobs.json"',
             f'TRAIN=$(sbatch --parsable -p 6000ada -A lcdd -q {QOS["train"]} --gres=gpu:4 --nodes=1 --ntasks=1 '
             f'--time=10:00:00 -J bc-ddp4-train -o "$RUN/logs/train-%j.out" {dep}"$RUN/train.sh")',
             f'INFER=$(sbatch --parsable -p 6000ada -A lcdd -q {QOS["infer"]} --gres=gpu:1 --time=02:00:00 '
             '-J bc-ddp4-infer -o "$RUN/logs/infer-%j.out" --dependency=afterok:$TRAIN --kill-on-invalid-dep=yes '
             '"$RUN/infer.sh")',
             'SCORE=$(sbatch --parsable -p cpu_only -A lcdd -q normal -c 4 --mem=16G --time=02:00:00 -J bc-ddp4-score '
             '-o "$RUN/logs/score-%j.out" --dependency=afterok:$INFER --kill-on-invalid-dep=yes "$RUN/score.sh")',
             'printf \'{"train":"%s","infer":"%s","score_report":"%s"}\\n\' "$TRAIN" "$INFER" "$SCORE" '
             '> "$RUN/jobs.json"', 'cat "$RUN/jobs.json"']
    return "\n".join(lines) + "\n"


def prepare(out, *, eval_run=DEFAULT_EVAL, stamp=None):
    """Freeze a new run with file operations only; every model/torch check happens later under Slurm."""
    reference = long.reference_binding(eval_run)
    eval_protocol = load(Path(reference["run"]) / "protocol.json")
    template = yaml.safe_load(Path(reference["config"]).read_text())
    long.verify_template(template)
    panel_run = Path(eval_protocol["reference10k"]["run"]).resolve()
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = Path(out).resolve() / f"batch-confound-ddp4-{stamp}"
    run.mkdir(parents=True, exist_ok=False)
    (run / "logs").mkdir()
    source = freeze_code(run, reference)
    current = {f"code/{name}": sha(ROOT / name) for name in map(str, curve.frozen_files(ROOT)) if name.startswith("src/")}
    drift = sorted(name for name in set(source) | set(current) if source.get(name) != current.get(name))
    config = derive_config(template, run)
    (run / "configs").mkdir()
    (run / f"configs/{ARM}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    protocol = dict(created_at=datetime.now(timezone.utc).isoformat(),
        authority="Batch-confound probe requested by the user: one 4-rank DDP flat_fine6 arm (batch1 copies4 "
                  "accum1, 16 samples/update) on the seed-20261023 WSD10k recipe, K2 SDE500 evaluation of its final "
                  "EMA; GPU jobs are submitted only on explicit user approval.",
        benchmark_role=ROLE, arm=ARM, architecture="flat_fine6", seed=SEED, world_size=WORLD,
        per_rank=dict(batch_size=1, copies_per_protein=4, grad_accum_steps=1), samples_per_update=16,
        distinct_proteins_per_update=WORLD, reference_per_update=dict(world_size=1, batch_size=1, copies_per_protein=4,
                                                                     grad_accum_steps=4, distinct_proteins=1),
        lr_scaling=None, horizon=dict(total_updates=H.total, warmup_updates=H.warmup, stable_updates=H.stable,
                                      cooldown_updates=H.cooldown, muon_peak=PEAKS["muon"], adamw_peak=PEAKS["adamw"],
                                      floor_fraction=0.01, ema_decay=0.999),
        checkpoints=list(CHECKPOINTS), evaluated_checkpoint=STEPS, config=config["config"], training=config["out_dir"],
        repo=str(repo_root()), wandb_name=WANDB_NAME, declared_config_changes=dict(CONFIG_CHANGES, wandb_tags=
            f"{TAG_CHANGE[0]}->{TAG_CHANGE[1]}"), per_rank_rng="scripts/train.py seed_rank_streams: rank0 keeps the "
            "run seed, ranks 1-3 reseed host/CUDA RNG from sha256(seed,rank) before the loader starts; the batch "
            "sampler's global draw stays shared",
        nccl=NCCL, launcher=f"python -m torch.distributed.run --standalone --nproc_per_node={WORLD} scripts/train.py",
        model_source="copied from the reference evaluation code", working_tree_model_source_drift=drift,
        reference=reference, sampler=SAMPLER, inference_depth=DEPTH, panel=eval_protocol["panel"],
        panel_sha256=eval_protocol["panel_sha256"], panel_run=str(panel_run),
        panel_admission_sha256=sha(panel_run / "panel_admitted.json"), ost=eval_protocol["ost"],
        expected_targets=15, expected_entries=13, expected_parameters=seedrep.PARAMETER_COUNT,
        provenance_lost=list(PROVENANCE_LOST), limits=LIMITS)
    require(Path(protocol["panel"]).resolve() == panel_run / "panel"
            and sha(panel_run / "panel/panel.json") == protocol["panel_sha256"], "reference panel binding differs")
    (run / "protocol.json").write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
    (run / "verify_frozen.py").write_text(VERIFY_FROZEN.replace("__UNSEALED__", repr(UNSEALED))
                                          .replace("__CHANGES__", repr(CONFIG_CHANGES)).replace("__TAG__", repr(TAG_CHANGE)))
    for name, text in dict(scripts(run, repo_root()), **{"submit_chain.sh": submit_chain(run, repo_root())}).items():
        (run / name).write_text(text)
        (run / name).chmod(0o755)
    manifest = seal(run)
    with (run / "source_manifest.json").open("x") as stream:
        stream.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return dict(run=str(run), frozen_files=len(manifest), arm=ARM, model_source_drift_in_working_tree=drift,
                source_manifest_sha256=sha(run / "source_manifest.json"))


# ── admitted contract ───────────────────────────────────────────────────


def validate_protocol(run, protocol):
    run = Path(run).resolve()
    require(protocol.get("benchmark_role") == ROLE and protocol.get("arm") == ARM and protocol.get("seed") == SEED
            and type(protocol["seed"]) is int and protocol.get("world_size") == WORLD
            and protocol.get("samples_per_update") == 16, "batch-confound role/arm/seed/world differs")
    exact(protocol, dict(per_rank=dict(batch_size=1, copies_per_protein=4, grad_accum_steps=1), sampler=SAMPLER,
                         inference_depth=DEPTH, expected_targets=15, expected_entries=13, evaluated_checkpoint=STEPS,
                         checkpoints=list(CHECKPOINTS), nccl=NCCL, provenance_lost=list(PROVENANCE_LOST),
                         limits=LIMITS, expected_parameters=seedrep.PARAMETER_COUNT), "batch-confound protocol differs")
    for key, path in dict(config=run / f"configs/{ARM}.yaml", training=run / f"training/{ARM}").items():
        require(Path(protocol[key]).resolve() == path, f"unexpected {key} path")
    reference = long.validate_reference_binding(protocol)
    template = yaml.safe_load(Path(reference["config"]).read_text())
    long.verify_template(template)
    config = yaml.safe_load(Path(protocol["config"]).read_text())
    require(config == derive_config(template, run), "configuration differs from its template outside the declared changes")
    seedrep.verify_flat_construction(config, {})
    require(config["batch_size"] * config["copies_per_protein"] * config["grad_accum_steps"] * WORLD == 16,
            "samples per update differ from 16")
    exact({key: config[key] for key in proof.MODEL_FIELDS}, reference["model_fields"], "model fields differ")
    return config


def validate_run(run, *, admitted=True):
    bind()
    run = Path(run).resolve()
    manifest = read(run / "source_manifest.json")
    require(not set(manifest) & set(UNSEALED), "manifest cannot seal itself or later admissions")
    base.seals(run, manifest)
    for path in (Path(__file__), *long.modules(), Path(long.__file__)):
        require(manifest.get(base.relative(run, path)) == sha(path), "unsealed batch-confound dependency")
    for name in ("protocol.json", "code/scripts/train.py", "code/benchmarks/probe_nccl.py"):
        require(name in manifest, f"unsealed {name}")
    if admitted:
        exact(read(run / "cpu_admitted.json"), dict(admitted=True, source_manifest_sha256=sha(run / "source_manifest.json")),
              "CPU admission binding differs")
    protocol = read(run / "protocol.json")
    config = validate_protocol(run, protocol)
    require(manifest.get(base.relative(run, protocol["config"])) == sha(protocol["config"]), "unsealed configuration")
    return dict(run=run, protocol=protocol, manifest=manifest, config=config, targets=long.validate_panel(protocol),
                source_manifest_sha256=sha(run / "source_manifest.json"), protocol_sha256=sha(run / "protocol.json"),
                panel_admission_sha256=protocol["panel_admission_sha256"])


def unchanged(context):
    fresh = validate_run(context["run"])
    require(base.context_binding(fresh) == base.context_binding(context) and fresh["targets"] == context["targets"],
            "batch-confound sources/protocol/panel changed")


def admit_cpu(run):
    """Bind the focused CPU checks; runs before cpu_admitted.json exists."""
    import xml.etree.ElementTree as ET

    require(bool(os.environ.get("SLURM_JOB_ID")), "CPU admission requires a Slurm allocation")
    context = validate_run(run, admitted=False)
    root = ET.parse(context["run"] / "checks.xml").getroot()
    suites = list(root) if root.tag == "testsuites" else [root]
    counts = {key: sum(int(suite.attrib.get(key, 0)) for suite in suites) for key in ("tests", "failures", "errors", "skipped")}
    require(counts["tests"] > 0 and counts["failures"] == counts["errors"] == 0, f"CPU checks: {counts}")
    skipped = [(case.attrib.get("name", ""), child.attrib.get("message", "")) for case in root.iter("testcase")
               for child in case if child.tag == "skipped"]
    require(len(skipped) == counts["skipped"] and all(name.startswith(CUDA_ONLY_TESTS) and "CUDA" in message
                                                      for name, message in skipped), f"unexpected skips: {skipped}")
    classes = {case.attrib.get("classname", "") for case in root.iter("testcase")}
    for name in TESTS:
        module = name.removesuffix(".py").replace("/", ".")
        require(any(value == module or value.startswith(module + ".") for value in classes), f"missing checks: {name}")
    value = dict(admitted=True, source_manifest_sha256=context["source_manifest_sha256"], tests=counts, skipped=skipped,
                 panel_targets=len(context["targets"]), panel_entries=len(seedrep.entries(context["targets"])),
                 checks_sha256=sha(context["run"] / "checks.xml"))
    write_new(context["run"] / "cpu_admitted.json", value)
    return value


# ── training: four ranks under torchrun, after an NCCL preflight ────────


INVENTORY = r'''
import json, os, torch
props = [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]
print(json.dumps(dict(count=len(props), names=[p.name for p in props], uuids=[str(p.uuid) for p in props],
                      visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), torch=torch.__version__,
                      cuda=torch.version.cuda)))
'''


def gpu_inventory():
    """Device identity from a short-lived child so the stage process never holds a CUDA context."""
    completed = subprocess.run([sys.executable, "-c", INVENTORY], check=True, capture_output=True, text=True)
    value = json.loads(completed.stdout.strip().splitlines()[-1])
    require(value["count"] == WORLD and set(value["names"]) == {TRAIN_GPU} and len(set(value["uuids"])) == WORLD
            and all(uuid not in ("", "None") for uuid in value["uuids"]), f"training needs {WORLD} distinct {TRAIN_GPU}")
    return dict(value, host=socket.gethostname(), slurm_job=os.environ["SLURM_JOB_ID"])


def nccl_preflight(run, code):
    job = os.environ["SLURM_JOB_ID"]
    log = run / f"logs/nccl-{job}.out"
    env = dict(os.environ, NCCL_P2P_DISABLE=NCCL["NCCL_P2P_DISABLE"], PYTHONFAULTHANDLER="1")
    started = time.time()
    with log.open("x") as stream:
        try:
            completed = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone",
                                        f"--nproc_per_node={WORLD}", "--tee", "3", str(code / NCCL["probe"])],
                                       stdout=stream, stderr=subprocess.STDOUT, env=env, timeout=NCCL["timeout_seconds"])
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            returncode = "timeout"
    text = log.read_text()
    value = dict(returncode=returncode, seconds=time.time() - started, log=base.relative(run, log), log_sha256=sha(log),
                 all_reduce_ok=text.count("all_reduce ok"), done=text.count("DONE"), host=socket.gethostname(),
                 NCCL_P2P_DISABLE=env["NCCL_P2P_DISABLE"])
    write_new(run / "nccl_probe.json", value)
    require(returncode == 0 and value["all_reduce_ok"] == WORLD and value["done"] == WORLD,
            f"NCCL preflight failed on {value['host']}: {value}; refusing to train (no gloo fallback)")
    return value


def train(context):
    run, protocol = context["run"], context["protocol"]
    home = Path(protocol["training"])
    base.no_outputs((home, run / "training_run.json", run / "nccl_probe.json"))
    long.admitted_reference(context)
    hardware = gpu_inventory()
    code = run / "code"
    probe = nccl_preflight(run, code)
    job = os.environ["SLURM_JOB_ID"]
    log = run / f"logs/torchrun-{job}.out"
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={WORLD}",
               str(code / "scripts/train.py"), "--config", protocol["config"]]
    env = dict(os.environ, NCCL_P2P_DISABLE=NCCL["NCCL_P2P_DISABLE"], PYTHONPATH=f"{code}:{code / 'src'}")
    started, clock = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    with log.open("x") as stream:
        # Data paths in the recipe are repository-relative, exactly as for the reference runs.
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env, cwd=protocol["repo"])
    seconds = time.perf_counter() - clock
    require(gpu_inventory()["uuids"] == hardware["uuids"], "training GPUs changed")
    value = dict(admitted=completed.returncode == 0, **base.context_binding(context), hardware=hardware,
                 nccl_probe_sha256=sha(run / "nccl_probe.json"), nccl_probe_seconds=probe["seconds"],
                 command=command[1:], cwd=protocol["repo"], returncode=completed.returncode, started_at=started,
                 ended_at=datetime.now(timezone.utc).isoformat(), torchrun_seconds=seconds,
                 log=base.relative(run, log), log_sha256=sha(log))
    write_new(run / "training_run.json", value)
    require(completed.returncode == 0, f"torchrun training failed with {completed.returncode}; see {log}")
    unchanged(context)


# ── training admission (by value; no audit stack) ───────────────────────


def verify_metrics(rows, interval):
    """Every interval applied all its updates and logged the WSD next-step learning rates; no update was skipped."""
    require([row.get("step") for row in rows] == list(range(interval, STEPS + 1, interval)),
            "missing/reordered training metric intervals")
    previous = 0
    for row in rows:
        require(row.get("update_applied") == 1.0, f"skipped update in interval ending {row['step']}")
        for kind, peak in PEAKS.items():
            mean = peak * statistics.fmean(H.multiplier(step) for step in range(previous + 1, row["step"] + 1))
            # Under DDP the trainer reduces interval means as float32, hence the tolerance.
            require(proof.finite(row.get(f"lr_{kind}")) and math.isclose(row[f"lr_{kind}"], mean, rel_tol=1e-6,
                    abs_tol=1e-14), "logged next-step LR interval differs from WSD")
        require(proof.finite(row.get("perf_train_step_ms_max"), positive=True), "invalid step timer")
        previous = row["step"]


def step_costs(rows):
    """Slowest-rank ms/update, weighted by updates; `steady` drops the first interval (prewarm/first-touch)."""
    previous, weighted = 0, 0.0
    for row in rows:
        weighted += row["perf_train_step_ms_max"] * (row["step"] - previous)
        previous = row["step"]
    return dict(all_update_ms=weighted / previous,
                steady_ms=statistics.fmean(row["perf_train_step_ms_max"] for row in rows[1:]),
                data_wait_ms=statistics.fmean(row["perf_data_wait_ms_max"] for row in rows))


def reference_runtime(context):
    return read(Path(context["protocol"]["reference"]["training"]) / "initialization.json")["mamba3_runtime"]


def admit_training(context):
    import torch

    run, protocol = context["run"], context["protocol"]
    receipt = read(run / "training_run.json")
    exact(receipt, dict(admitted=True, returncode=0, **base.context_binding(context)), "training run receipt differs")
    require(sha(run / "nccl_probe.json") == receipt["nccl_probe_sha256"] and sha(run / receipt["log"])
            == receipt["log_sha256"], "training/NCCL evidence changed")
    log_text = (run / receipt["log"]).read_text()
    require(not any(pattern in log_text for pattern in SKIP_PATTERNS), "torchrun log reports a skipped update")
    require(f"world_size: {WORLD}" in log_text and "total_batch: 4" in log_text, "trainer did not run on four ranks")
    home = Path(protocol["training"])
    cfg, declared = read(home / "config.json"), yaml.safe_load(Path(protocol["config"]).read_text())
    require(all(cfg.get(key) == value for key, value in declared.items() if key not in proof.IGNORED_CONFIG),
            "actual training configuration differs from source YAML")
    require(cfg["requested_num_workers"] == declared["num_workers"], "worker request differs")
    numbered = sorted(path.name for path in home.glob("ckpt_[0-9][0-9][0-9][0-9][0-9][0-9][0-9].pt"))
    require(numbered == [f"ckpt_{step:07d}.pt" for step in CHECKPOINTS], "retained checkpoints differ")
    latest = home / "ckpt_latest.pt"
    require(latest.is_symlink() and os.readlink(latest) == f"ckpt_{STEPS:07d}.pt", "ckpt_latest is not the final one")
    require(not (home / ".requeue_requested").exists(), "interrupted training")
    runtime = reference_runtime(context)
    final = home / f"ckpt_{STEPS:07d}.pt"
    ema = proof.checkpoint_identity(final, cfg, STEPS, runtime=runtime)
    state = torch.load(final, map_location="cpu", mmap=True, weights_only=False)
    data = state.get("data_state", {})
    exact(data, dict(micro_batches_consumed=STEPS, world_size=WORLD, batch_size=1, grad_accum_steps=1,
                     sampler_type="LengthBucketedDistributedBatchSampler", seed=SEED), "checkpoint data state differs")
    states = state.get("rng_states") or []
    require(len(states) == WORLD and len({bytes(s["torch"].numpy().tobytes()) for s in states}) == WORLD
            and len({bytes(s["cuda"].numpy().tobytes()) for s in states if "cuda" in s}) == WORLD,
            "per-rank RNG streams are not distinct at the final checkpoint")
    parameters = sum(value.numel() for value in state["ema"].values())
    del state
    gc.collect()
    rows = proof.lines(home / "metrics.jsonl")
    verify_metrics(rows, cfg["log_interval"])
    costs = step_costs(rows)
    paths = [Path(protocol["config"]), home / "config.json", home / "metrics.jsonl", run / "training_run.json",
             run / "nccl_probe.json", run / receipt["log"], *(home / name for name in numbered)]
    evidence = {base.relative(run, path): sha(path) for path in paths}
    base.seals(run, evidence)
    unchanged(context)
    value = dict(admitted=True, arm=ARM, **base.context_binding(context), hardware=receipt["hardware"],
                 config=cfg, mamba3_runtime=runtime, final_checkpoint_sha256=sha(final), final_ema_sha256=ema,
                 ema_tensor_elements=parameters, data_state=data, costs=costs, torchrun_seconds=receipt["torchrun_seconds"],
                 evidence_sha256=evidence)
    write_new(run / "train_admitted.json", value)
    return admitted_training(context)


def admitted_training(context):
    run = context["run"]
    value = read(run / "train_admitted.json")
    exact(value, dict(admitted=True, arm=ARM, **base.context_binding(context)), "training admission identity differs")
    base.seals(run, value["evidence_sha256"])
    require(value["final_checkpoint_sha256"] == value["evidence_sha256"][base.relative(
        run, Path(context["protocol"]["training"]) / f"ckpt_{STEPS:07d}.pt")], "final checkpoint is not sealed")
    return value


# ── inference and canonical scoring (the reference evaluation path) ─────


def evaluation(run):
    return Path(run) / f"evaluation/{ARM}-{STEPS}"


def infer(context):
    import torch
    from mambafold.model.bimamba3 import mamba3_runtime_info

    run, protocol = context["run"], context["protocol"]
    base.no_outputs([run / "train_admitted.json", run / "inference_admitted.json", evaluation(run)])
    device = base.hardware(training=True)
    torch.set_num_threads(1)
    long.admitted_reference(context)
    training = admit_training(context)
    runtime = training["mamba3_runtime"]
    base.verify_external_runtime(runtime, mamba3_runtime_info())
    checkpoint, directory = Path(protocol["training"]) / f"ckpt_{STEPS:07d}.pt", evaluation(run)
    require(sha(checkpoint) == training["final_checkpoint_sha256"], "final checkpoint changed")
    started = time.perf_counter()
    with base.inference_model_runtime(runtime):
        long.screen.infer(SimpleNamespace(checkpoint=checkpoint, panel=Path(protocol["panel"]), out=directory,
                                          depths=[DEPTH], n_steps=SAMPLER["n_steps"]))
    seconds = time.perf_counter() - started
    require(base.hardware(training=True) == device, "inference changed physical GPU")
    verify_inference(context, training["final_checkpoint_sha256"])
    binding = dict(hardware=device, nominal_K=2, step=STEPS, checkpoint_sha256=training["final_checkpoint_sha256"],
                   ema_sha256=training["final_ema_sha256"])
    write_new(directory / "gpu_binding.json", binding)
    paths = [directory / name for name in ("sampling.json", "hardware.json", "gpu_binding.json")]
    paths += [directory / f"K{DEPTH}/pairs/manifest.json",
              *(directory / f"K{DEPTH}/pairs/{target}_seed0_pred.pdb" for target in context["targets"])]
    unchanged(context)
    write_new(run / "inference_admitted.json", dict(admitted=True, **base.context_binding(context), hardware=device,
        sampler=SAMPLER, inference_depth=DEPTH, coverage=len(context["targets"]), checkpoint=binding,
        stage_seconds=seconds, training_receipt_sha256=sha(run / "train_admitted.json"),
        evidence_sha256={base.relative(run, path): sha(path) for path in paths}))


def verify_inference(context, checkpoint_sha):
    measured = proof.verify_inference(evaluation(context["run"]), STEPS, [DEPTH], checkpoint_sha, context["targets"],
                                      False, atom_recycling="none", n_steps=SAMPLER["n_steps"])
    for row in measured:
        seedrep.verify_flat_construction(row, {})
        require(proof.finite(row.get("peak_reserved_gib"), positive=True), "missing reserved inference memory")
    return measured


def admitted_inference(context):
    run = context["run"]
    value = read(run / "inference_admitted.json")
    exact(value, dict(admitted=True, coverage=len(context["targets"]), sampler=SAMPLER, inference_depth=DEPTH,
                      training_receipt_sha256=sha(run / "train_admitted.json"), **base.context_binding(context)),
          "inference admission differs")
    base.validate_hardware(value["hardware"], training=True)
    base.seals(run, value["evidence_sha256"])
    training = admitted_training(context)
    exact(value["checkpoint"], dict(hardware=value["hardware"], nominal_K=2, step=STEPS,
                                    checkpoint_sha256=training["final_checkpoint_sha256"],
                                    ema_sha256=training["final_ema_sha256"]), "inference checkpoint binding differs")
    require(read(evaluation(run) / "gpu_binding.json") == value["checkpoint"], "GPU binding differs")
    verify_inference(context, value["checkpoint"]["checkpoint_sha256"])
    return value


def score(context):
    run = context["run"]
    admitted_inference(context)
    directory = evaluation(run)
    base.no_outputs([run / "scores" / f"{ARM}.json", run / "scores" / f"{ARM}.failed.json", directory / f"K{DEPTH}/scores"])
    completed = subprocess.run([sys.executable, str(Path(base.canonical.__file__)), "--in-dir",
        str(directory / f"K{DEPTH}/pairs"), "--out-dir", str(directory / f"K{DEPTH}/scores"), "--ost",
        context["protocol"]["ost"], "--expected", str(len(context["targets"]))], check=False)
    path = directory / f"K{DEPTH}/scores/summary.json"
    summary = read(path) if path.is_file() else None
    if completed.returncode or summary is None or summary.get("evaluation_complete") is not True:
        write_new(run / f"scores/{ARM}.failed.json", dict(arm=ARM, admitted=False, returncode=completed.returncode,
                  expected=len(context["targets"]), summary=summary))
        raise ValueError("canonical scoring failed or was incomplete")
    vectors = base.verify_canonical(summary, directory, context)
    unchanged(context)
    write_new(run / f"scores/{ARM}.json", dict(admitted=True, arm=ARM, **base.context_binding(context),
        inference_admission_sha256=sha(run / "inference_admitted.json"), coverage=len(vectors), metrics=vectors,
        summary_sha256=sha(path), scorer=summary["openstructure"], ost_executable_sha256=sha(context["protocol"]["ost"])))


# ── report ──────────────────────────────────────────────────────────────


def reference_costs(reference):
    """Single-GPU reference costs from each seed's own trainer metrics, computed exactly like this arm's."""
    rows = {}
    for arm in FLAT_ARMS:
        complete = reference["arms"][arm]["complete"]
        home = Path(reference["arms"][arm]["config"]["out_dir"])
        costs = step_costs(proof.lines(home / "metrics.jsonl"))
        rows[arm] = dict(costs, end_to_end_seconds=complete["end_to_end_seconds"], gpus=1,
                         gpu_hours=complete["end_to_end_seconds"] / 3600, metrics_sha256=sha(home / "metrics.jsonl"),
                         inference_seconds=reference["arms"][arm]["costs"]["inference_seconds"])
    return rows


def summarize(targets, values, reference):
    """Descriptive comparison with the three-seed reference; nothing is selected."""
    runs = seedrep.run_quality(targets, values)
    spread = reference["spread"]
    versus = {metric: {kind: long.versus_spread(runs[metric][kind], spread[metric][kind]) for kind in KINDS}
              for metric in QUALITY}
    paired = {arm: curve.tally(tail.paired_effects(targets, reference["arms"][arm]["per_target"], values))
              for arm in FLAT_ARMS}
    seed_mean = {target: {metric: statistics.fmean(reference["arms"][arm]["per_target"][target][metric]
                                                   for arm in FLAT_ARMS) for metric in QUALITY} for target in targets}
    per_target = {target: {metric: dict(value=values[target][metric], reference_seed_mean=seed_mean[target][metric],
                                        delta=values[target][metric] - seed_mean[target][metric],
                                        reference_seeds=[reference["arms"][arm]["per_target"][target][metric]
                                                         for arm in FLAT_ARMS]) for metric in QUALITY}
                  for target in sorted(targets)}
    return dict(runs=runs, versus_three_seed_10k=versus, paired_minus_reference_seed=paired,
                minus_reference_seed_mean=curve.tally(tail.paired_effects(targets, seed_mean, values)),
                per_target=per_target)


def report(context):
    run, targets = context["run"], context["targets"]
    base.no_outputs([run / "comparison.json", run / "comparison.csv"])
    inference, reference, training = admitted_inference(context), long.admitted_reference(context), admitted_training(context)
    value = read(run / f"scores/{ARM}.json")
    exact(value, dict(admitted=True, arm=ARM, coverage=len(targets), inference_admission_sha256=sha(run / "inference_admitted.json"),
                      **base.context_binding(context)), "score admission identity differs")
    directory = evaluation(run)
    path = directory / f"K{DEPTH}/scores/summary.json"
    require(sha(path) == value["summary_sha256"], "canonical score evidence changed")
    summary = read(path)
    values = base.verify_canonical(summary, directory, context)
    require(values == value["metrics"] and summary["openstructure"] == value["scorer"]
            == reference["arms"][REF_ARM]["scorer"], "canonical vectors or scorer differ from the reference")
    geometry = tail.geometry_diagnostics(directory, targets, values, value["scorer"], DEPTH)
    totals = {field: geometry["totals"][field] for field in GEOMETRY}
    stats = summarize(targets, values, reference)
    measured = read(directory / "sampling.json")
    train_hours = training["torchrun_seconds"] / 3600
    infer_hours = inference["stage_seconds"] / 3600
    ref_costs = reference_costs(reference)
    costs = dict(training["costs"], torchrun_seconds=training["torchrun_seconds"], gpus=WORLD,
                 train_gpu_hours=WORLD * train_hours, infer_gpu_hours=infer_hours,
                 inference_seconds=statistics.fmean(row["seconds"] for row in measured),
                 inference_reserved_gib=max(row["peak_reserved_gib"] for row in measured),
                 reference=ref_costs,
                 reference_mean_all_update_ms=statistics.fmean(row["all_update_ms"] for row in ref_costs.values()),
                 reference_mean_gpu_hours=statistics.fmean(row["gpu_hours"] for row in ref_costs.values()))
    costs["all_update_ms_ratio_to_reference"] = costs["all_update_ms"] / costs["reference_mean_all_update_ms"]
    result = dict(benchmark_role=ROLE, arm=ARM, sampler=SAMPLER, nominal_K=2, inference_depth=DEPTH,
        **base.context_binding(context), coverage=dict(targets=len(targets), entries=len(seedrep.entries(targets))),
        metric_names=dict(AA="lddt", CA="bb_lddt", TM="tm_score"), per_update=dict(
            arm=context["protocol"]["per_rank"] | dict(world_size=WORLD, distinct_proteins=WORLD, samples=16),
            reference=context["protocol"]["reference_per_update"] | dict(samples=16)),
        per_target_values=values, **stats, geometry_totals=totals,
        reference=dict(three_seed_10k_spread=reference["spread"], admission_sha256=sha(run / "reference_admitted.json"),
                       geometry_totals={arm: reference["arms"][arm]["geometry_totals"] for arm in FLAT_ARMS},
                       runs={arm: reference["arms"][arm]["runs"] for arm in FLAT_ARMS},
                       inference_hardware=reference["inference_hardware"]),
        costs=costs, training_hardware=training["hardware"], inference_hardware=inference["hardware"],
        same_inference_gpu_as_reference=inference["hardware"]["uuid"] == reference["inference_hardware"]["uuid"],
        provenance_lost=list(PROVENANCE_LOST), batch_policy_selected=False, limits=LIMITS)
    unchanged(context)
    write_new(run / "comparison.json", result)
    rows = [dict(arm=ARM, **{metric: stats["runs"][metric]["chain_mean"] for metric in QUALITY},
                 **{f"{metric}_minus_3seed_mean": stats["versus_three_seed_10k"][metric]["chain_mean"]["delta"]
                    for metric in QUALITY},
                 **{f"{metric}_delta_over_sd": stats["versus_three_seed_10k"][metric]["chain_mean"]["delta_over_sd"]
                    for metric in QUALITY}, all_update_ms=costs["all_update_ms"], gpu_hours=costs["train_gpu_hours"])]
    rows += [dict(arm=arm, **{metric: reference["arms"][arm]["runs"][metric]["chain_mean"] for metric in QUALITY},
                  **{f"{metric}_minus_3seed_mean": None for metric in QUALITY},
                  **{f"{metric}_delta_over_sd": None for metric in QUALITY},
                  all_update_ms=ref_costs[arm]["all_update_ms"], gpu_hours=ref_costs[arm]["gpu_hours"])
             for arm in FLAT_ARMS]
    with (run / "comparison.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


STAGES = ("admit-cpu", "reference", "train", "infer", "score", "report")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", *STAGES))
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
        parser.error("batch-confound execution requires a Slurm allocation, including CPU stages")
    if args.run is None or args.out is not None or args.eval_run is not None:
        parser.error(f"{args.stage} requires --run and no --out/--eval-run")
    if args.stage == "admit-cpu":
        print(json.dumps(admit_cpu(args.run)))
        return
    context = validate_run(args.run)
    if args.stage == "reference":
        long.reference(context)
        return
    dict(train=train, infer=infer, score=score, report=report)[args.stage](context)


if __name__ == "__main__":
    main()
