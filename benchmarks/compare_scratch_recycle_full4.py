"""Matched K4 full-BPTT trial and a common three-model evaluation."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from benchmarks import compare_scratch_atom_recycling as atom  # noqa: E402

base = atom.base
scratch, proof, screen, inputs = base.scratch, base.proof, base.screen, base.inputs
ARMS = {policy: dict(shared_trunk_layers=6, n_latent_layers=6, atom_recycling="carry",
                     recycle_policy=policy) for policy in ("fixed4",)}
MODEL_FIELDS = (*atom.MODEL_FIELDS, "recycle_policy")
DEPTHS = [1, 2, 4, 8, 16]
ROLE, SAMPLER, QUALITY = base.ROLE, base.SAMPLER, base.QUALITY
ALL_ARMS = ("fixed2", "sampled16_tail2", "fixed4")
PARAMETERS = dict.fromkeys(ALL_ARMS, 269123226)
POLICIES = dict(fixed4=dict(K="fixed4", BPTT="full", trajectory_loss_reduction="full_trajectory_mean"))

ARCHITECTURE = dict(residue_pre_layers=3, residue_post_layers=3, shared_residue_layers=6,
    fine_latent_layers=6, coarse_latent_layers=0, fine_compression=4, nominal_K=2,
    atom_recycling="carry", initial_residue_and_PLM_reinjection_every_pass=True,
    FM_coordinates_and_time_fixed_inside_call=True, depth_condition_entire_core=True,
    atom_state_carry_within_velocity_call_only=True)
read, sha, write_new, require, exact = base.read, base.sha, base.write_new, base.require, base.exact
validate_panel = base.validate_panel
_BASE_VALIDATE, _BASE_VERIFY = base.validate_run, base.verify_actual_training
_BASE_ADMITTED, _BASE_PREFLIGHT = base.admitted_training, scratch.preflight
_BASE_VALIDATE_ANCHOR = scratch.validate_anchor


@contextmanager
def registry():
    old = scratch.VARIANTS, proof.VARIANTS, proof.MODEL_FIELDS
    scratch.VARIANTS = {name: dict(value) for name, value in ARMS.items()}
    proof.VARIANTS = {name: dict(value) for name, value in ARMS.items()}
    proof.MODEL_FIELDS = MODEL_FIELDS
    try:
        yield
    finally:
        scratch.VARIANTS, proof.VARIANTS, proof.MODEL_FIELDS = old


def expected(arm):
    require(arm in ARMS, "undeclared recycle-depth arm")
    with registry():
        return scratch.expected_config(arm)


def common_config(config):
    return base.common_config({key: value for key, value in config.items() if key != "recycle_policy"})


_BASE_COMMON = base.common_config


@contextmanager
def bindings():
    values = dict(ARMS=ARMS, PARAMETERS=PARAMETERS, registry=registry,
        common_config=lambda config: _BASE_COMMON({key: value for key, value in config.items()
                                                 if key != "recycle_policy"}),
        validate_protocol=validate_protocol, verify_actual_training=verify_actual_training,
        admitted_training=admitted_training)
    old = {key: getattr(base, key) for key in values}
    try:
        for key, value in values.items():
            setattr(base, key, value)
        with registry():
            yield
    finally:
        for key, value in old.items():
            setattr(base, key, value)


def validate_protocol(run, protocol):
    exact(protocol, dict(formal_applied_update_cap_per_arm=10000, microbatches_per_formal_run=40000,
        preflight_optimizer_updates=0, teacher_models=0, all_parameters_trainable=True,
        common_device_final_confirmation_required=True, benchmark_role=ROLE), "recycle budget differs")
    require(set(protocol.get("arms", {})) == set(ARMS), "exactly the new fixed4 training arm is required")
    require(protocol.get("sampler") == SAMPLER, "sampler differs")
    exact(protocol["sampler"], SAMPLER, "typed sampler differs")
    require(protocol.get("inference_depths") == DEPTHS
            and all(type(value) is int for value in protocol["inference_depths"]), "inference depths differ")
    require(protocol.get("training_policies") == POLICIES, "training policy contract differs")
    require(protocol.get("architecture_contract") == ARCHITECTURE, "common recycle architecture differs")
    exact(protocol["architecture_contract"], ARCHITECTURE, "typed common recycle architecture differs")
    require(Path(protocol["panel"]).resolve() == run / "panel", "unexpected recycle panel path")
    require(Path(protocol["ost"]).is_absolute(), "canonical OST must be absolute")
    configs = {}
    for arm, entry in protocol["arms"].items():
        exact(entry, dict(nominal_K=2, expected_parameters=PARAMETERS[arm]), "model construction differs")
        require(set(entry.get("model_fields", {})) == set(MODEL_FIELDS), "all 36 model fields required")
        exact(entry["model_fields"], {key: expected(arm)[key] for key in MODEL_FIELDS}, "model fields differ")
        for key, path in dict(config=run / f"configs/{arm}.yaml", preflight=run / f"preflight/{arm}",
                              training=run / f"training/{arm}", u0=run / "u0").items():
            require(Path(entry[key]).resolve() == path, "recycle artifact path differs")
        config = yaml.safe_load(Path(entry["config"]).read_text())
        with registry():
            scratch.validate_config(config, arm)
        exact(config, expected(arm), "typed recycle configuration differs")
        configs[arm] = config
    validate_external_protocol(protocol.get("external_pair", {}))
    return configs


def validate_run(run):
    with bindings():
        context = _BASE_VALIDATE(run)
    for path in (Path(__file__), Path(atom.__file__)):
        require(context["manifest"].get(base.relative(context["run"], path)) == sha(path),
                "unsealed recycle adapter dependency")
    prior = context["protocol"].get("previous_panel_admission", {})
    exact(prior, dict(expected_targets=15, expected_entries=13, recheck_full_corpus_membership=True),
          "declared panel comparison differs")
    require(len(context["targets"]) == 15
            and len({row["entry_id"] for row in context["targets"].values()}) == 13, "panel coverage differs")
    previous = Path(prior["run"])
    receipt = read(previous / "panel_admitted.json")
    require(sha(previous / "panel_admitted.json") == prior["panel_admission_sha256"]
            and sha(previous / "panel/panel.json") == receipt["panel_sha256"], "prior panel seal differs")
    previous_targets = {row["target"]: row for row in read(previous / "panel/panel.json")["targets"]}
    require(atom.panel_signature(context["targets"]) == atom.panel_signature(previous_targets),
            "target/reference/sequence/ESM panel signature differs")
    old = Path(context["protocol"]["external_pair"]["run"])
    old_targets = {row["target"]: row for row in read(old / "panel/panel.json")["targets"]}
    require(atom.panel_signature(context["targets"]) == atom.panel_signature(old_targets),
            "new and running-pair panel signatures differ")
    return context


def supervised_steps(policy, loops):
    require(policy in ARMS and type(loops) is int and 1 <= loops <= 16, "invalid recycle policy/K")
    require(policy == "fixed4" and loops == 4, "fixed4 formal training must execute four passes")
    return (1, 2, 3, 4)


def verify_policy_row(row, policy, micro, seed):
    loops = scratch.actual_training_loops(dict(recycle_policy=policy, seed=seed), (micro - 1) // 4)
    selected = supervised_steps(policy, loops)
    exact(row, dict(micro=micro, optimizer_step=(micro - 1) // 4, policy=policy, K=loops,
        supervised_steps=list(selected), full_grid_columns=loops,
        readout_requires_grad=[True, *(index in selected for index in range(1, loops + 1))]),
        "actual recycle depth/supervision/gradient policy differs")


def verify_policy_audit(path, policy, seed, *, microbatches=40000):
    rows = proof.lines(path)
    require(len(rows) == microbatches, "incomplete per-microbatch recycle policy audit")
    for micro, row in enumerate(rows, 1):
        verify_policy_row(row, policy, micro, seed)
    counts = Counter(row["K"] for row in rows[::4])
    require(sum(counts.values()) == microbatches // 4, "recycle update coverage differs")
    return dict(microbatches=microbatches, updates=microbatches // 4,
                depth_counts={str(key): value for key, value in counts.items()},
                mean_K=statistics.fmean(row["K"] for row in rows[::4]))


def semantic_path(context, arm):
    return Path(context["protocol"]["arms"][arm]["preflight"]) / "recycle_semantics_verified.json"


def verify_semantics(value, initial_sha):
    exact(value, dict(admitted=True, optimizer_updates=0, checkpoint_loads=0, teacher_models=0,
        initial_model_sha256=initial_sha, restored_model_sha256=initial_sha,
        model_and_rng_restored=True, masked_padding_verified=True, valid_residues=29,
        policy="fixed4", K=4, supervised_steps=[1, 2, 3, 4], readout_requires_grad=[True] * 5,
        residue_requires_grad=[True] * 4, atom_requires_grad=[True] * 4,
        observed_calls=screen.expected_call_counts(4, atom_recycling="carry", training=True),
        original_grid_endpoint=True, fixed_FM_inputs=True), "fixed4 semantic proof differs")
    require(value.get("rng_before_sha256") == value.get("rng_after_sha256")
            and proof.valid_sha(value.get("rng_before_sha256")), "semantic RNG restoration differs")
    require(proof.valid_sha(value.get("input_sha256")), "semantic input binding missing")
    require(proof.finite(value.get("loss"), positive=True), "invalid fixed4 semantic loss")
    require(set(value.get("groups", {})) == set(proof.GROUPS)
            and all(proof.finite(group.get("norm"), positive=True) for group in value["groups"].values()),
            "fixed4 semantic full-loss gradient group missing")
    names = {"R0", "R1", "A1", "R2", "A2", "R3", "A3", "residue_trunk", "plm_proj", "plm_reinject"}
    require(set(value.get("final_only_gradient_norms", {})) == names
            and all(proof.finite(norm, positive=True) for norm in value["final_only_gradient_norms"].values()),
            "final K4 loss is disconnected from an early recurrent/initial/PLM state")


def verify_actual_training(context, arm):
    with bindings():
        config, complete, metrics, evidence = _BASE_VERIFY(context, arm)
    home = Path(context["protocol"]["arms"][arm]["training"])
    initial = read(home / "initialization.json")
    require(initial["target_sha256"] == context["protocol"]["external_pair"]["expected_initial_model_sha256"],
            "new scratch initialization differs from the running pair")
    native_path = Path(context["protocol"]["arms"][arm]["preflight"]) / "native_verified.json"
    native = read(native_path)
    exact(native, dict(recycle_policy=arm, atom_recycling="carry",
        model_fields={key: config[key] for key in MODEL_FIELDS}), "native 36-field model identity differs")
    for index, row in enumerate(native["rows"]):
        loops = scratch.actual_training_loops(config, index // 4)
        exact(row, dict(observed_calls=screen.expected_call_counts(loops, atom_recycling="carry", training=True),
            supervised_steps=list(supervised_steps(arm, loops))), "native recycle execution differs")
    path = semantic_path(context, arm)
    verify_semantics(read(path), initial["target_sha256"])
    policy_path = home / "audit/recycle_policy.jsonl"
    policy = verify_policy_audit(policy_path, arm, config["seed"])
    require(policy["depth_counts"] == complete["depth_counts"], "policy sequence and completed K counts differ")
    for item in (path, policy_path):
        evidence[base.relative(context["run"], item)] = sha(item)
    return config, dict(complete, recycle_policy_audit=policy), metrics, evidence


def admitted_training(context, arm):
    with bindings():
        value = _BASE_ADMITTED(context, arm)
    initial = read(Path(context["protocol"]["arms"][arm]["training"]) / "initialization.json")
    path = semantic_path(context, arm)
    require(value["evidence_sha256"].get(base.relative(context["run"], path)) == sha(path),
            "semantic admission not sealed into training")
    verify_semantics(read(path), initial["target_sha256"])
    home = Path(context["protocol"]["arms"][arm]["training"])
    policy_path = home / "audit/recycle_policy.jsonl"
    require(value["evidence_sha256"].get(base.relative(context["run"], policy_path)) == sha(policy_path),
            "actual policy sequence not sealed")
    require(value["complete"].get("recycle_policy_audit")
            == verify_policy_audit(policy_path, arm, value["config"]["seed"]), "admitted policy sequence differs")
    return value


def semantic_probe(config, native_sha, destination):
    """Verify full native K4 credit assignment with temporary readouts and no update."""
    import torch
    from mambafold.train import trainer

    base.no_outputs([destination])
    caller_rng = trainer.capture_rng_state()
    before_rng = atom.rng_digest(caller_rng)
    result = None
    try:
        with scratch.forbid_checkpoint_loads():
            trainer.seed_all(config["seed"])
            model = trainer.build_model(config, "cuda").train()
            initial = base.audit.tensor_digest(model.state_dict().items())
            old_loops = model.refinement_training_loops
            require(initial == native_sha and all(parameter.requires_grad for parameter in model.parameters()),
                    "fixed4 semantic model differs from its native fresh model")
            trainer.seed_all(config["seed"] + 17)
            batch = trainer._synthetic_batch(1, 32, 14, config["d_plm"], torch.device("cuda"))
            residue_mask = batch.res_mask.clone()
            residue_mask[:, -3:] = False
            atom_mask = batch.atom_mask & residue_mask.unsqueeze(-1)
            batch = replace(batch, x_clean=torch.randn_like(batch.x_clean) * .1,
                x_t=torch.randn_like(batch.x_t) * .1, eps=torch.randn_like(batch.eps),
                esm=torch.randn_like(batch.esm), res_mask=residue_mask,
                ca_mask=batch.ca_mask & residue_mask, atom_mask=atom_mask,
                valid_mask=batch.valid_mask & atom_mask)
            input_sha = base.audit.batch_digest(batch)
            counts, residues, atoms, R0 = Counter(), [], [], []
            handles = screen.execution_counter_hooks(model, counts, atom_recycling="carry")
            handles.append(model.residue_trunk.register_forward_hook(lambda module, args, out: R0.append(out)))
            handles.append(model.latent_residual.residue_stack.register_forward_hook(
                lambda module, args, out: residues.append(out)))
            def capture_atoms(module, args, output):
                if isinstance(output, tuple):
                    atoms.append(output[1])
            handles.append(model.atom_decoder.register_forward_hook(capture_atoms))
            try:
                with atom.temporary_readouts(model):
                    model.refinement_training_loops = 4
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        output = model(batch)
                    require(len(R0) == 1 and len(residues) == len(atoms) == 4, "missing full K4 hidden states")
                    require(torch.count_nonzero(output["v_atom"][~batch.atom_mask]).item() == 0,
                            "fixed4 semantic padded velocity is nonzero")
                    depths = output["latent_depths"]
                    require(torch.isfinite(depths).all().item() and depths.shape == (1, 4)
                            and torch.equal(depths[:, -1], torch.ones_like(depths[:, -1]))
                            and torch.all(depths[:, 1:] > depths[:, :-1]).item(), "full K4 depth grid differs")
                    final = output["v_atom"].float().square().mean()
                    state_names = ("R0", "R1", "A1", "R2", "A2", "R3", "A3")
                    leaves = [R0[0], residues[0], atoms[0], residues[1], atoms[1], residues[2], atoms[2]]
                    parameter_groups = {name: [parameter for key, parameter in model.named_parameters()
                        if key.startswith(name + ".")] for name in ("residue_trunk", "plm_proj", "plm_reinject")}
                    for parameters in parameter_groups.values():
                        leaves.extend(parameters)
                    gradients = torch.autograd.grad(final, leaves, allow_unused=True, retain_graph=True)
                    def norm(values):
                        return math.sqrt(sum(float(value.float().square().sum())
                                             for value in values if value is not None))
                    norms = {name: norm(gradients[index:index + 1]) for index, name in enumerate(state_names)}
                    offset = len(state_names)
                    for name, parameters in parameter_groups.items():
                        norms[name] = norm(gradients[offset:offset + len(parameters)])
                        offset += len(parameters)
                    loss, _ = scratch.engine.allatom_loss_surface(output, batch,
                        alpha_mode=config["alpha_mode"], use_rigid_align=config["use_rigid_align"],
                        w_fm=config["w_fm"], w_lddt_atom=config["w_lddt_atom"],
                        lddt_cutoff_A=config["lddt_cutoff_A"], lddt_pair_chunk_size=config["lddt_pair_chunk_size"])
                    require(torch.isfinite(loss).item(), "nonfinite K4 semantic loss")
                    loss.backward()
                    result = dict(admitted=True, optimizer_updates=0, checkpoint_loads=0, teacher_models=0,
                        initial_model_sha256=initial, policy="fixed4", K=4,
                        supervised_steps=list(output["latent_supervised_steps"]),
                        readout_requires_grad=[value.requires_grad for value in output["v_atom_steps"]],
                        residue_requires_grad=[value.requires_grad for value in residues],
                        atom_requires_grad=[value.requires_grad for value in atoms],
                        final_only_gradient_norms=norms, observed_calls=dict(counts),
                        groups=scratch._gradient_rows(model), loss=float(loss.detach()),
                        input_sha256=input_sha, fixed_FM_inputs=base.audit.batch_digest(batch) == input_sha,
                        original_grid_endpoint=True, masked_padding_verified=True, valid_residues=int(residue_mask.sum()))
            finally:
                for handle in handles:
                    handle.remove()
                model.refinement_training_loops = old_loops
                model.zero_grad(set_to_none=True)
            restored = base.audit.tensor_digest(model.state_dict().items())
            require(initial == restored, "K4 semantic probe changed model state")
            result["restored_model_sha256"] = restored
    finally:
        trainer.restore_rng_state(caller_rng)
        after_rng = atom.rng_digest(trainer.capture_rng_state())
        require(before_rng == after_rng, "K4 semantic probe did not restore caller RNG")
    result.update(rng_before_sha256=before_rng, rng_after_sha256=after_rng, model_and_rng_restored=True)
    verify_semantics(result, native_sha)
    write_new(destination, result)


def native_preflight(args):
    from mambafold.train import trainer

    original_build, original_anchor = trainer.build_model, scratch.validate_anchor
    counts, handles, observed = Counter(), [], []
    def build(*values, **kwargs):
        model = original_build(*values, **kwargs)
        handles.extend(screen.execution_counter_hooks(model, counts, atom_recycling="carry"))
        return model
    def anchor(model, batch, output, config):
        loops = output["latent_depths"].shape[1]
        require(dict(counts) == screen.expected_call_counts(loops, atom_recycling="carry", training=True),
                "native actual recycle calls differ")
        observed.append(dict(observed_calls=dict(counts), supervised_steps=list(output["latent_supervised_steps"])))
        try:
            return original_anchor(model, batch, output, config)
        finally:
            counts.clear()
    trainer.build_model, scratch.validate_anchor = build, anchor
    try:
        _BASE_PREFLIGHT(args)
    finally:
        trainer.build_model, scratch.validate_anchor = original_build, original_anchor
        for handle in handles:
            handle.remove()
    config, _ = scratch._read_config(args)
    native_path = args.out / "native_verified.json"
    native = read(native_path)
    exact(read(args.out / "initialization.json"), dict(parameters=269123226,
        trainable_parameters=269123226, prelude_layers=3, shared_trunk_layers=6, hierarchy=False),
        "native recycle layout differs")
    require(len(observed) == len(native["rows"]) == 8, "incomplete native recycle rows")
    for row, measured in zip(native["rows"], observed, strict=True):
        row.update(measured)
    native.update(recycle_policy=config["recycle_policy"], atom_recycling="carry",
                  model_fields={key: config[key] for key in MODEL_FIELDS})
    base.audit.write_json(native_path, native)
    gc.collect()
    semantic_probe(config, native["model_sha256"], args.out / "recycle_semantics_verified.json")


@contextmanager
def policy_recording():
    original = scratch.engine.allatom_forward_and_loss
    micro = 0
    def forward(model, *values, **kwargs):
        nonlocal micro
        raw, captured = getattr(model, "module", model), []
        def observe(module, args, output):
            loops = output["latent_depths"].shape[1]
            captured.append(dict(policy=raw.recycle_policy, K=loops,
                supervised_steps=list(output["latent_supervised_steps"]), full_grid_columns=loops,
                readout_requires_grad=[velocity.requires_grad for velocity in output["v_atom_steps"]]))
        handle = raw.register_forward_hook(observe)
        try:
            result = original(model, *values, **kwargs)
        finally:
            handle.remove()
        require(len(captured) == 1, "actual training must execute exactly one velocity rollout")
        micro += 1
        row = dict(micro=micro, optimizer_step=(micro - 1) // 4, **captured[0])
        config = raw._recycle_audit_config
        verify_policy_row(row, raw.recycle_policy, micro, config["seed"])
        path = Path(config["out_dir"]) / "audit/recycle_policy.jsonl"
        with path.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        return result
    original_build = scratch.trainer.build_model
    def build(config, device):
        model = original_build(config, device)
        model._recycle_audit_config = dict(config)
        return model
    scratch.engine.allatom_forward_and_loss, scratch.trainer.build_model = forward, build
    try:
        yield
    finally:
        scratch.engine.allatom_forward_and_loss, scratch.trainer.build_model = original, original_build


def train(context, arm):
    original = scratch.preflight
    def preflight(args):
        native_preflight(args)
        native = read(args.out / "native_verified.json")
        require(native["model_sha256"] == context["protocol"]["external_pair"]["expected_initial_model_sha256"],
                "fresh K4 native weights differ from the running pair")
    scratch.preflight = preflight
    try:
        with bindings(), policy_recording():
            base.train(context, arm)
    finally:
        scratch.preflight = original


def validate_external_protocol(value):
    require(isinstance(value, dict), "missing external running-pair contract")
    required = {"run", "code", "source_manifest_sha256", "protocol_sha256", "panel_admission_sha256",
                "expected_initial_model_sha256"}
    require(set(value) == required and all(isinstance(item, str) and item for item in value.values()),
            "external pair contract fields differ")
    old, code = Path(value["run"]), Path(value["code"])
    require(old.is_absolute() and code.resolve() == old.resolve() / "code", "external frozen code path differs")
    for key, filename in (("source_manifest_sha256", "source_manifest.json"),
                          ("protocol_sha256", "protocol.json"), ("panel_admission_sha256", "panel_admitted.json")):
        require(proof.valid_sha(value[key]) and sha(old / filename) == value[key], "external immutable binding differs")
    require(proof.valid_sha(value["expected_initial_model_sha256"]), "external initial tensor hash missing")
    manifest, protocol = read(old / "source_manifest.json"), read(old / "protocol.json")
    verifier = code / "benchmarks/compare_scratch_recycle_tail.py"
    require(manifest.get(str(verifier.resolve().relative_to(old.resolve()))) == sha(verifier),
            "external frozen verifier is not sealed")
    require(set(protocol.get("arms", {})) == set(ALL_ARMS) - set(ARMS)
            and protocol.get("architecture_contract") == ARCHITECTURE
            and protocol.get("sampler") == SAMPLER and protocol.get("inference_depths") == DEPTHS,
            "external architecture/sampler/coverage contract differs")
    for arm in ALL_ARMS[:2]:
        entry = protocol["arms"][arm]
        require(Path(entry["training"]).resolve() == old.resolve() / f"training/{arm}"
                and Path(entry["preflight"]).resolve() == old.resolve() / f"preflight/{arm}",
                "external training/preflight path differs")
        initial = read(Path(entry["training"]) / "initialization.json")
        require(initial.get("target_sha256") == value["expected_initial_model_sha256"],
                "external fresh initial tensors differ")
    return protocol


OLD_ADMISSION_SCRIPT = r'''
import hashlib, json, sys
from pathlib import Path
code, run, output = (Path(value).resolve() for value in sys.argv[1:])
sys.path[:0] = [str(code), str(code / "src")]
from benchmarks import compare_scratch_recycle_tail as trial
assert Path(trial.__file__).resolve() == code / "benchmarks/compare_scratch_recycle_tail.py"
context = trial.validate_run(run)
arms = {}
for arm in ("fixed2", "sampled16_tail2"):
    admitted = trial.admitted_training(context, arm)
    config, complete, metrics, evidence = trial.verify_actual_training(context, arm)
    home = Path(context["protocol"]["arms"][arm]["training"])
    initial = trial.read(home / "initialization.json")
    arms[arm] = dict(training_receipt=admitted,
        training_receipt_sha256=trial.sha(trial.base.training_receipt_path(run, arm)),
        initial_model_sha256=initial["target_sha256"], runtime=initial["mamba3_runtime"],
        checkpoint_sha256=admitted["final_checkpoint_sha256"], ema_sha256=admitted["final_ema_sha256"],
        input_stream_sha256=complete["input_stream_sha256"])
trial.base.unchanged(context)
trial.write_new(output, dict(admitted=True, run=str(run), code=str(code),
    verifier_sha256=trial.sha(Path(trial.__file__)), verifier_source="original frozen running-pair evaluator",
    **trial.base.context_binding(context), arms=arms))
'''


def run_external_admission(context):
    """Use the original sealed evaluator in a separate interpreter, never a patched registry."""
    destination = context["run"] / "external_training_admitted.json"
    base.no_outputs([destination])
    value = context["protocol"]["external_pair"]
    validate_external_protocol(value)
    subprocess.run([sys.executable, "-c", OLD_ADMISSION_SCRIPT, value["code"], value["run"], str(destination)],
                   cwd=value["code"], check=True)
    return admitted_external(context)


def admitted_external(context):
    external = context["protocol"]["external_pair"]
    protocol = validate_external_protocol(external)
    value = read(context["run"] / "external_training_admitted.json")
    exact(value, dict(admitted=True, run=str(Path(external["run"]).resolve()),
        code=str(Path(external["code"]).resolve()), source_manifest_sha256=external["source_manifest_sha256"],
        protocol_sha256=external["protocol_sha256"], panel_admission_sha256=external["panel_admission_sha256"],
        verifier_sha256=sha(Path(external["code"]) / "benchmarks/compare_scratch_recycle_tail.py"),
        verifier_source="original frozen running-pair evaluator"), "external validation receipt binding differs")
    require(set(value.get("arms", {})) == set(ALL_ARMS[:2]), "external training coverage differs")
    for arm, row in value["arms"].items():
        path = Path(external["run"]) / f"train_admitted/{arm}.json"
        receipt = read(path)
        require(sha(path) == row.get("training_receipt_sha256") and receipt == row.get("training_receipt"),
                "original external training admission changed")
        require(row.get("initial_model_sha256") == external["expected_initial_model_sha256"]
                and row.get("checkpoint_sha256") == receipt["final_checkpoint_sha256"]
                and row.get("ema_sha256") == receipt["final_ema_sha256"]
                and row.get("input_stream_sha256") == receipt["complete"]["input_stream_sha256"],
                "external weight/input identity differs")
        config = receipt["config"]
        require(config.get("recycle_policy") == arm, "external policy identity differs")
        new_config = read(training_home(context, "fixed4") / "config.json")
        require(common_config(config) == common_config(new_config),
                "external data/loss/optimizer settings differ from the new arm")
        require(receipt["parameters"] == PARAMETERS["fixed4"] and receipt["nominal_K"] == 2,
                "external construction/parameter topology differs")
        exact(receipt["complete"], inputs.BUDGET, "external applied-update budget differs")
        initial = read(Path(protocol["arms"][arm]["training"]) / "initialization.json")
        require(row.get("runtime") == initial["mamba3_runtime"], "external runtime metadata changed")
    return value


def training_home(context, arm):
    require(arm in ALL_ARMS, "undeclared three-way arm")
    if arm == "fixed4":
        return Path(context["protocol"]["arms"][arm]["training"])
    return Path(context["protocol"]["external_pair"]["run"]) / f"training/{arm}"


def all_training(context):
    external = admitted_external(context)
    return {**{arm: row["training_receipt"] for arm, row in external["arms"].items()},
            "fixed4": admitted_training(context, "fixed4")}


def receipt_sha(context, arm):
    run = context["run"] if arm == "fixed4" else Path(context["protocol"]["external_pair"]["run"])
    return sha(base.training_receipt_path(run, arm))


def admitted_inference(context):
    run, value = context["run"], read(context["run"] / "inference_admitted.json")
    exact(value, dict(admitted=True, identical_initial_weights=True, inference_depths=DEPTHS,
        coverage=len(context["targets"]), sampler=SAMPLER, **base.context_binding(context),
        external_training_admission_sha256=sha(run / "external_training_admitted.json")),
        "three-way inference admission binding differs")
    require(set(value.get("arms", {})) == set(value.get("training_receipts", {})) == set(ALL_ARMS),
            "incomplete three-way inference arms")
    base.validate_hardware(value.get("hardware"))
    base.seals(run, value.get("evidence_sha256"))
    training = all_training(context)
    initials = set()
    for arm in ALL_ARMS:
        require(receipt_sha(context, arm) == value["training_receipts"][arm],
                "training admission changed after common inference")
        initial = read(training_home(context, arm) / "initialization.json")
        initials.add(initial["target_sha256"])
        binding = value["arms"][arm]
        exact(binding, dict(hardware=value["hardware"], nominal_K=2, inference_depths=DEPTHS,
            checkpoint_sha256=training[arm]["final_checkpoint_sha256"],
            ema_sha256=training[arm]["final_ema_sha256"]), "three-way own EMA/GPU differs")
        require(read(base.evaluation(run, arm) / "gpu_binding.json") == binding, "per-model GPU binding differs")
        verify_inference(context, arm, binding["checkpoint_sha256"])
    require(initials == {value.get("initial_model_sha256")}, "admitted three-way initial tensors differ")
    return value


def verify_inference(context, arm, checkpoint_sha):
    directory = base.evaluation(context["run"], arm)
    measured = proof.verify_inference(directory, 10000, DEPTHS, checkpoint_sha,
        context["targets"], False, atom_recycling="carry")
    for row in measured:
        exact(row, dict(recycle_policy=arm), "inference recycle policy differs")
        require(proof.finite(row.get("peak_reserved_gib"), positive=True), "missing reserved inference memory")
    return measured


def infer(context):
    import torch
    from mambafold.model.bimamba3 import mamba3_runtime_info

    run = context["run"]
    base.no_outputs([run / "inference_admitted.json", *(base.evaluation(run, arm) for arm in ALL_ARMS)])
    device = base.hardware()
    torch.set_num_threads(1)
    external = run_external_admission(context)
    admitted = all_training(context)
    reference_config = common_config(admitted["fixed4"]["config"])
    require(all(common_config(value["config"]) == reference_config for value in admitted.values()),
            "actual common training configurations differ")
    streams, initials, runtimes = {}, {}, {}
    runtime = mamba3_runtime_info()
    for arm in ALL_ARMS:
        if arm == "fixed4":
            verify_actual_training(context, arm)
        home = training_home(context, arm)
        initial = read(home / "initialization.json")
        initials[arm], runtimes[arm] = initial["target_sha256"], initial["mamba3_runtime"]
        base.verify_external_runtime(runtimes[arm], runtime)
        streams[arm], digest = inputs.read_inputs(home / "audit/inputs.jsonl", 40000)
        require(digest == admitted[arm]["complete"]["input_stream_sha256"], "actual input stream digest differs")
    require(set(initials.values()) == {context["protocol"]["external_pair"]["expected_initial_model_sha256"]},
            "three-way formal initial tensors are not identical")
    require(all(stream == streams["fixed4"] for stream in streams.values()),
            "three-way actual ordered 40000 input streams differ")
    evidence, arms = {"external_training_admitted.json": sha(run / "external_training_admitted.json")}, {}
    for arm in ALL_ARMS:
        home = training_home(context, arm)
        directory = base.evaluation(run, arm)
        require(base.hardware() == device, "common physical inference GPU changed")
        with base.inference_model_runtime(runtimes[arm]):
            screen.infer(SimpleNamespace(checkpoint=home / "ckpt_0010000.pt", panel=run / "panel",
                                        out=directory, depths=DEPTHS))
        require(base.hardware() == device, "common physical inference GPU changed")
        verify_inference(context, arm, admitted[arm]["final_checkpoint_sha256"])
        arms[arm] = dict(hardware=device, nominal_K=2, inference_depths=DEPTHS,
            checkpoint_sha256=admitted[arm]["final_checkpoint_sha256"], ema_sha256=admitted[arm]["final_ema_sha256"])
        write_new(directory / "gpu_binding.json", arms[arm])
        paths = [directory / name for name in ("sampling.json", "hardware.json", "gpu_binding.json")]
        for depth in DEPTHS:
            paths.append(directory / f"K{depth}/pairs/manifest.json")
            paths.extend(directory / f"K{depth}/pairs/{target}_seed0_pred.pdb" for target in context["targets"])
        for path in paths:
            evidence[base.relative(run, path)] = sha(path)
        gc.collect()
        torch.cuda.empty_cache()
    base.unchanged(context)
    write_new(run / "inference_admitted.json", dict(admitted=True, **base.context_binding(context),
        hardware=device, sampler=SAMPLER, inference_depths=DEPTHS, coverage=len(context["targets"]), arms=arms,
        initial_model_sha256=initials["fixed4"], identical_initial_weights=True,
        external_training_admission_sha256=sha(run / "external_training_admitted.json"),
        training_receipts={arm: receipt_sha(context, arm) for arm in ALL_ARMS},
        input_comparison=dict(method="byte_exact", microbatches_per_arm=40000), evidence_sha256=evidence))


def verify_canonical(summary, directory, context, depth):
    proof.verify_summary(summary, context["targets"])
    exact(summary, dict(expected_targets=len(context["targets"]),
        input_manifest_sha256=sha(directory / f"K{depth}/pairs/manifest.json")), "canonical input manifest differs")
    scorer = summary.get("openstructure", {})
    require(scorer.get("compare_args") == list(base.canonical.COMPARE_ARGS)
            and scorer.get("executable") == context["protocol"]["ost"]
            and isinstance(scorer.get("version"), str) and scorer["version"]
            and summary.get("failures") == [], "canonical scorer/failure contract differs")
    return {row["target_id"].removesuffix("_seed0"): {key: row[key] for key in QUALITY}
            for row in summary["rows"]}


def score(context, arm=None):
    run = context["run"]
    admitted_inference(context)
    names = (arm,) if arm else ALL_ARMS
    require(set(names) <= set(ALL_ARMS), "undeclared scoring arm")
    base.no_outputs([path for name in names for depth in DEPTHS for path in (
        run / f"scores/{name}-K{depth}.json", run / f"scores/{name}-K{depth}.failed.json",
        base.evaluation(run, name) / f"K{depth}/scores")])
    failed = []
    for name in names:
        directory = base.evaluation(run, name)
        for depth in DEPTHS:
            home = directory / f"K{depth}"
            completed = subprocess.run([sys.executable, str(Path(base.canonical.__file__)),
                "--in-dir", str(home / "pairs"), "--out-dir", str(home / "scores"),
                "--ost", context["protocol"]["ost"], "--expected", str(len(context["targets"]))], check=False)
            path = home / "scores/summary.json"
            summary = read(path) if path.is_file() else None
            if completed.returncode or summary is None or summary.get("evaluation_complete") is not True:
                write_new(run / f"scores/{name}-K{depth}.failed.json", dict(arm=name, K=depth, admitted=False,
                    returncode=completed.returncode, expected=len(context["targets"]), summary=summary))
                failed.append(f"{name}/K{depth}")
                continue
            vectors = verify_canonical(summary, directory, context, depth)
            base.unchanged(context)
            write_new(run / f"scores/{name}-K{depth}.json", dict(admitted=True, arm=name, K=depth,
                **base.context_binding(context), inference_admission_sha256=sha(run / "inference_admitted.json"),
                coverage=len(vectors), metrics=vectors, summary_sha256=sha(path),
                scorer=summary["openstructure"], ost_executable_sha256=sha(context["protocol"]["ost"])))
    require(not failed, "canonical scoring failed or incomplete: " + ", ".join(failed))


def paired_effects(targets, first, second):
    require(targets and set(first) == set(second) == set(targets), "paired quality coverage differs")
    require(all(set(row) == set(QUALITY) and all(proof.finite(value) for value in row.values())
                for values in (first, second) for row in values.values()), "invalid paired quality metrics")
    entries = {}
    for target, row in targets.items():
        entries.setdefault(row["entry_id"], []).append(target)
    result = {}
    for metric in QUALITY:
        deltas = {target: second[target][metric] - first[target][metric] for target in sorted(targets)}
        entry_deltas = {entry: statistics.fmean(deltas[target] for target in names)
                        for entry, names in sorted(entries.items())}
        result[metric] = dict(chain_mean_delta=statistics.fmean(deltas.values()),
            equal_entry_mean_delta=statistics.fmean(entry_deltas.values()),
            target_deltas=deltas, entry_deltas=entry_deltas)
    return result


def geometry_diagnostics(directory, targets, vectors, scorer, depth):
    fields = tuple(f"{role}_{kind}" for role in ("model", "reference")
                   for kind in ("clashes", "bad_bonds", "bad_angles"))
    rows, evidence = {}, {}
    require(set(vectors) == set(targets), "geometry quality coverage differs")
    for target in sorted(targets):
        raw_path = directory / f"K{depth}/scores/raw/{target}_seed0.json"
        identity_path = raw_path.with_name(f"{target}_seed0.inputs.json")
        raw, identity = read(raw_path), read(identity_path)
        require(raw.get("status") == "SUCCESS", "geometry raw score failed")
        for metric in QUALITY:
            require(proof.finite(raw.get(metric)) and math.isclose(raw[metric], vectors[target][metric],
                rel_tol=1e-12, abs_tol=1e-15), "raw geometry quality differs")
        exact(identity, dict(schema_version=1,
            prediction_sha256=sha(directory / f"K{depth}/pairs/{target}_seed0_pred.pdb"),
            reference_sha256=targets[target]["reference_sha256"],
            openstructure_version=scorer["version"], compare_args=scorer["compare_args"]),
            "geometry scorer/input binding differs")
        require(all(isinstance(raw.get(field), list) for field in fields), "missing canonical geometry arrays")
        rows[target] = {field: len(raw[field]) for field in fields}
        for path in (raw_path, identity_path):
            evidence[str(path)] = sha(path)
    return dict(coverage_targets=len(rows), unit="OST violation entries; unnormalised counts",
        per_target=rows, totals={field: sum(row[field] for row in rows.values()) for field in fields},
        evidence_sha256=evidence)


def report(context):
    run = context["run"]
    base.no_outputs([run / "comparison.json", run / "comparison.csv"])
    inference = admitted_inference(context)
    vectors, rows, geometry, training_costs, scorers = {}, [], {}, {}, []
    admitted = all_training(context)
    for arm in ALL_ARMS:
        receipt = admitted[arm]
        training = receipt["complete"]
        training_costs[arm] = dict(steady_mean_step_ms=training["steady_mean_step_ms"],
            all_update_weighted_mean_step_ms=receipt["all_update_weighted_mean_step_ms"],
            peak_allocated_gib=training["peak_allocated_gib"], peak_reserved_gib=training["peak_reserved_gib"],
            timing_scope=training["timing_scope"], actual_policy=training["recycle_policy_audit"])
        directory = base.evaluation(run, arm)
        measured = read(directory / "sampling.json")
        vectors[arm], geometry[arm] = {}, {}
        for depth in DEPTHS:
            value = read(run / f"scores/{arm}-K{depth}.json")
            exact(value, dict(admitted=True, arm=arm, K=depth, coverage=len(context["targets"]),
                inference_admission_sha256=sha(run / "inference_admitted.json"), **base.context_binding(context)),
                "score admission differs")
            path = directory / f"K{depth}/scores/summary.json"
            require(sha(path) == value["summary_sha256"], "canonical summary changed")
            summary = read(path)
            quality = verify_canonical(summary, directory, context, depth)
            require(quality == value["metrics"] and summary["openstructure"] == value["scorer"]
                    and sha(context["protocol"]["ost"]) == value["ost_executable_sha256"],
                    "canonical vectors/scorer changed")
            scorers.append(value["scorer"])
            vectors[arm][str(depth)] = quality
            geometry[arm][str(depth)] = geometry_diagnostics(directory, context["targets"], quality,
                                                           value["scorer"], depth)
            timings = [row for row in measured if row["K"] == depth]
            rows.append(dict(arm=arm, K=depth, coverage=len(quality), parameters=PARAMETERS[arm],
                **{metric: statistics.fmean(row[metric] for row in quality.values()) for metric in QUALITY},
                steady_train_ms=training_costs[arm]["steady_mean_step_ms"],
                all_update_train_ms=training_costs[arm]["all_update_weighted_mean_step_ms"],
                train_allocated_gib=training["peak_allocated_gib"], train_reserved_gib=training["peak_reserved_gib"],
                inference_seconds=statistics.fmean(row["seconds"] for row in timings),
                inference_allocated_gib=max(row["peak_allocated_gib"] for row in timings),
                inference_reserved_gib=max(row["peak_reserved_gib"] for row in timings)))
    require(all(scorer == scorers[0] for scorer in scorers), "canonical scorer differs across arms/depths")
    harmonic = sum(1 / depth for depth in range(1, 17))
    result = dict(benchmark_role=ROLE, nominal_K=2, inference_depths=DEPTHS, sampler=SAMPLER,
        **base.context_binding(context), coverage=dict(targets=len(context["targets"]),
            entries=len({row["entry_id"] for row in context["targets"].values()})),
        panel_coverage=read(run / "panel_admitted.json")["coverage"], rows=rows, per_target=vectors,
        between_models_at_matched_K={f"{second}_minus_{first}": {
            str(depth): paired_effects(context["targets"], vectors[first][str(depth)], vectors[second][str(depth)])
            for depth in DEPTHS} for first, second in (("fixed2", "sampled16_tail2"),
                ("fixed2", "fixed4"), ("sampled16_tail2", "fixed4"))},
        depth_minus_K1={arm: {str(depth): paired_effects(context["targets"], vectors[arm]["1"],
            vectors[arm][str(depth)]) for depth in DEPTHS[1:]} for arm in ALL_ARMS},
        training_costs=training_costs, geometry=geometry, inference_hardware=inference["hardware"],
        external_training_admission_sha256=sha(run / "external_training_admitted.json"),
        identical_initial_weights=True, parameters_matched=True, FLOPs_matched=False,
        endpoint_FM_coefficients=dict(fixed2=.5, fixed4=.25, sampled16_tail2=.5 + .5 / harmonic),
        architecture_selected=False,
        limits="Single seed and scratch WSD10000 applied updates, same data budget and initial tensors. "
            "All three models use the same live initial-residue/PLM reinjection and atom carry. "
            "Sampled arm jointly changes K, gradient truncation and LiFT loss allocation; this is not a pure K effect. "
            "FixedK2/K4 supervise their complete trajectories; sampled16 trains only the suffix and records prefix diagnostics. "
            "The two running models were admitted with their original frozen verifier in an isolated interpreter; "
            "all three EMA checkpoints were sampled by this one frozen evaluator on one physical GPU. "
            "Declared sequence-homology-excluded whole targets do not exclude remote homology, coordinate overlap "
            "or ESMC pretraining exposure. CA is canonical OST bb_lddt. Sampler times include instrumentation "
            "and final CPU copy, exclude loading/PDB/scoring; one observation per target. No automatic selection.")
    base.unchanged(context)
    write_new(run / "comparison.json", result)
    with (run / "comparison.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "infer", "score", "report"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--arm", choices=ALL_ARMS)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("all comparison execution requires a Slurm allocation")
    if args.stage == "train" and args.arm != "fixed4" or args.stage in ("infer", "report") and args.arm:
        parser.error("train requires --arm fixed4; infer/report operate on all three models")
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

