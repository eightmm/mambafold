"""Matched scratch recycle-depth and tail-gradient trial, with sealed admission."""

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
                     recycle_policy=policy) for policy in ("fixed2", "sampled16_tail2")}
PARAMETERS = dict.fromkeys(ARMS, 269123226)
MODEL_FIELDS = (*atom.MODEL_FIELDS, "recycle_policy")
DEPTHS = [1, 2, 4, 8, 16]
ROLE, SAMPLER, QUALITY = base.ROLE, base.SAMPLER, base.QUALITY
POLICIES = dict(fixed2=dict(K="fixed2", BPTT="full"), sampled16_tail2=dict(
    K="inverse_integer_1_to_16", BPTT="last_min_K_2", prefix_velocity_losses="diagnostic_only",
    trajectory_loss_reduction="supervised_tail_mean"))
ARCHITECTURE = dict(residue_pre_layers=3, residue_post_layers=3, shared_residue_layers=6,
    fine_latent_layers=6, coarse_latent_layers=0, fine_compression=4, nominal_K=2,
    atom_recycling="carry", initial_residue_and_PLM_reinjection_every_pass=True,
    FM_coordinates_and_time_fixed_inside_call=True, depth_condition_entire_core=True,
    atom_state_carry_within_velocity_call_only=True)
read, sha, write_new, require, exact = base.read, base.sha, base.write_new, base.require, base.exact
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
    require(set(protocol.get("arms", {})) == set(ARMS), "exactly fixed2 and sampled16_tail2 arms required")
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
    require(common_config(configs["fixed2"]) == common_config(configs["sampled16_tail2"]),
            "training differs outside the declared recycle policy")
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
    return context


def supervised_steps(policy, loops):
    require(policy in ARMS and type(loops) is int and 1 <= loops <= 16, "invalid recycle policy/K")
    return tuple(range(max(1, loops - 1), loops + 1)) if policy == "sampled16_tail2" else tuple(range(1, loops + 1))


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
        constructor_pair_identical=True, model_and_rng_restored=True, K2_outputs_identical=True,
        K2_grids_identical=True, masked_padding_verified=True, valid_residues=29),
        "recycle semantic identity/restoration differs")
    require(value.get("rng_before_sha256") == value.get("rng_after_sha256")
            and proof.valid_sha(value.get("rng_before_sha256")), "semantic RNG restoration differs")
    require(proof.valid_sha(value.get("input_sha256")), "semantic input binding missing")
    rows = value.get("rows", [])
    require([(row.get("policy"), row.get("K")) for row in rows]
            == [("fixed2", 2), ("sampled16_tail2", 1), ("sampled16_tail2", 2), ("sampled16_tail2", 16)],
            "semantic policy/K coverage differs")
    for row in rows:
        policy, loops = row["policy"], row["K"]
        selected = supervised_steps(policy, loops)
        exact(row, dict(supervised_steps=list(selected),
            readout_requires_grad=[True, *(index in selected for index in range(1, loops + 1))],
            prefix_residue_requires_grad=[False] * (loops - len(selected)),
            prefix_atom_requires_grad=[False] * (loops - len(selected)),
            observed_calls=screen.expected_call_counts(loops, atom_recycling="carry", training=True),
            fixed_FM_inputs=True, original_grid_endpoint=True), "semantic unroll/gradient policy differs")
        require(proof.finite(row.get("loss"), positive=True), "nonfinite semantic LiFT loss")
        require(set(row.get("groups", {})) == set(proof.GROUPS)
                and all(proof.finite(group.get("norm"), positive=True) for group in row["groups"].values()),
                "missing semantic full-loss gradient group")
        require(set(row.get("final_only_gradient_norms", {}))
                == {"R0", "residue_trunk", "plm_proj", "plm_reinject"}
                and all(proof.finite(norm, positive=True) for norm in row["final_only_gradient_norms"].values()),
                "final loss is disconnected from initial/prelude/PLM paths")
        require(proof.finite(row.get("tail_atom_gradient_norm"), positive=True) if loops > 1
                else row.get("tail_atom_gradient_norm") is None,
                "final loss did not traverse the retained atom-state tail")


def verify_actual_training(context, arm):
    with bindings():
        config, complete, metrics, evidence = _BASE_VERIFY(context, arm)
    home = Path(context["protocol"]["arms"][arm]["training"])
    initial = read(home / "initialization.json")
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
    """Native kernels with opened readouts; the complete state and RNG are restored."""
    import torch
    from mambafold.train import trainer

    base.no_outputs([destination])
    caller_rng = trainer.capture_rng_state()
    before_rng = atom.rng_digest(caller_rng)
    model, result = None, None
    try:
        with scratch.forbid_checkpoint_loads():
            trainer.seed_all(config["seed"])
            model = trainer.build_model(config, "cuda").train()
            initial = base.audit.tensor_digest(model.state_dict().items())
            require(initial == native_sha and all(parameter.requires_grad for parameter in model.parameters()),
                    "semantic model differs from native random model")
            other_config = dict(config, recycle_policy="sampled16_tail2" if config["recycle_policy"] == "fixed2" else "fixed2")
            trainer.seed_all(config["seed"])
            other = trainer.build_model(other_config, "cuda")
            require(base.audit.tensor_digest(other.state_dict().items()) == initial,
                    "policy constructors do not have bitwise identical initial tensors")
            del other
            gc.collect()
            torch.cuda.empty_cache()
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
            input_sha, grid_rng = base.audit.batch_digest(batch), trainer.capture_rng_state()
            old_policy, old_loops = model.recycle_policy, model.refinement_training_loops
            rows, K2_outputs, K2_depths = [], {}, {}
            try:
                with atom.temporary_readouts(model):
                    for policy, loops in (("fixed2", 2), ("sampled16_tail2", 1),
                                          ("sampled16_tail2", 2), ("sampled16_tail2", 16)):
                        trainer.restore_rng_state(grid_rng)
                        model.recycle_policy, model.refinement_training_loops = policy, loops
                        model.zero_grad(set_to_none=True)
                        counts, residues, atoms, initial_residue = Counter(), [], [], []
                        handles = screen.execution_counter_hooks(model, counts, atom_recycling="carry")
                        handles.append(model.residue_trunk.register_forward_hook(
                            lambda module, args, output: initial_residue.append(output)))
                        handles.append(model.latent_residual.residue_stack.register_forward_hook(
                            lambda module, args, output: residues.append(output)))
                        def capture_atoms(module, args, output):
                            if isinstance(output, tuple):
                                atoms.append(output[1])
                        handles.append(model.atom_decoder.register_forward_hook(capture_atoms))
                        try:
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                output = model(batch)
                            selected = supervised_steps(policy, loops)
                            require(len(residues) == len(atoms) == loops and len(initial_residue) == 1,
                                    "semantic state hooks did not capture the actual rollout")
                            require(torch.count_nonzero(output["v_atom"][~batch.atom_mask]).item() == 0,
                                    "semantic padded velocity is nonzero")
                            require(torch.isfinite(output["latent_depths"]).all().item()
                                    and torch.equal(output["latent_depths"][:, -1],
                                                    torch.ones_like(output["latent_depths"][:, -1])),
                                    "semantic full-grid endpoint differs")
                            final = output["v_atom"].float().square().mean()
                            names = dict(model.named_parameters())
                            parameter_groups = {
                                "residue_trunk": [parameter for name, parameter in names.items()
                                                  if name.startswith("residue_trunk.")],
                                "plm_proj": [parameter for name, parameter in names.items()
                                             if name.startswith("plm_proj.")],
                                "plm_reinject": [parameter for name, parameter in names.items()
                                                 if name.startswith("plm_reinject.")],
                            }
                            flat_parameters = [parameter for values in parameter_groups.values() for parameter in values]
                            leaves = [initial_residue[0], *flat_parameters]
                            if loops > 1:
                                leaves.append(atoms[-2])
                            gradients = torch.autograd.grad(final, leaves, allow_unused=True, retain_graph=True)
                            def norm(values):
                                return math.sqrt(sum(float(value.float().square().sum())
                                                     for value in values if value is not None))
                            norms, offset = {"R0": norm(gradients[:1])}, 1
                            for name, parameters in parameter_groups.items():
                                norms[name] = norm(gradients[offset:offset + len(parameters)])
                                offset += len(parameters)
                            loss, _ = scratch.engine.allatom_loss_surface(output, batch,
                                alpha_mode=config["alpha_mode"], use_rigid_align=config["use_rigid_align"],
                                w_fm=config["w_fm"], w_lddt_atom=config["w_lddt_atom"],
                                lddt_cutoff_A=config["lddt_cutoff_A"],
                                lddt_pair_chunk_size=config["lddt_pair_chunk_size"])
                            require(torch.isfinite(loss).item(), "nonfinite semantic LiFT loss")
                            loss.backward()
                            prefix = loops - len(selected)
                            rows.append(dict(policy=policy, K=loops, supervised_steps=list(output["latent_supervised_steps"]),
                                readout_requires_grad=[velocity.requires_grad for velocity in output["v_atom_steps"]],
                                prefix_residue_requires_grad=[state.requires_grad for state in residues[:prefix]],
                                prefix_atom_requires_grad=[state.requires_grad for state in atoms[:prefix]],
                                final_only_gradient_norms=norms,
                                tail_atom_gradient_norm=norm(gradients[-1:]) if loops > 1 else None,
                                groups=scratch._gradient_rows(model), observed_calls=dict(counts),
                                loss=float(loss.detach()), original_grid_endpoint=True,
                                fixed_FM_inputs=base.audit.batch_digest(batch) == input_sha))
                            if loops == 2:
                                K2_outputs[policy] = tuple(velocity.detach().clone() for velocity in output["v_atom_steps"])
                                K2_depths[policy] = output["latent_depths"].detach().clone()
                        finally:
                            for handle in handles:
                                handle.remove()
                        del output, loss, final, gradients, leaves, flat_parameters, residues, atoms, initial_residue
                    require(all(torch.equal(a, b) for a, b in zip(K2_outputs["fixed2"],
                        K2_outputs["sampled16_tail2"], strict=True)), "matched K2 policy outputs differ")
                    require(torch.equal(K2_depths["fixed2"], K2_depths["sampled16_tail2"]),
                            "matched K2 policy depth grids differ")
            finally:
                model.recycle_policy, model.refinement_training_loops = old_policy, old_loops
                model.zero_grad(set_to_none=True)
            restored = base.audit.tensor_digest(model.state_dict().items())
            require(restored == initial, "semantic probe did not restore exact model state")
            result = dict(admitted=True, optimizer_updates=0, checkpoint_loads=0, teacher_models=0,
                initial_model_sha256=initial, restored_model_sha256=restored,
                constructor_pair_identical=True, K2_outputs_identical=True, K2_grids_identical=True,
                input_sha256=input_sha, masked_padding_verified=True, valid_residues=int(residue_mask.sum()), rows=rows)
    finally:
        trainer.restore_rng_state(caller_rng)
        after_rng = atom.rng_digest(trainer.capture_rng_state())
        require(before_rng == after_rng, "semantic probe did not restore caller RNG")
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
    scratch.preflight = native_preflight
    try:
        with bindings(), policy_recording():
            base.train(context, arm)
    finally:
        scratch.preflight = original


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
    base.no_outputs([run / "inference_admitted.json", *(base.evaluation(run, arm) for arm in ARMS)])
    device = base.hardware()
    torch.set_num_threads(1)
    admitted = {arm: admitted_training(context, arm) for arm in ARMS}
    require(common_config(admitted["fixed2"]["config"]) == common_config(admitted["sampled16_tail2"]["config"]),
            "actual common training configurations differ")
    streams, initials, runtimes, semantic_inputs = {}, {}, {}, {}
    runtime = mamba3_runtime_info()
    for arm in ARMS:
        verify_actual_training(context, arm)
        home = Path(context["protocol"]["arms"][arm]["training"])
        initial = read(home / "initialization.json")
        initials[arm], runtimes[arm] = initial["target_sha256"], initial["mamba3_runtime"]
        semantic_inputs[arm] = read(semantic_path(context, arm))["input_sha256"]
        base.verify_external_runtime(runtimes[arm], runtime)
        streams[arm], digest = inputs.read_inputs(home / "audit/inputs.jsonl", 40000)
        require(digest == admitted[arm]["complete"]["input_stream_sha256"], "actual input stream digest differs")
    require(initials["fixed2"] == initials["sampled16_tail2"], "formal initial tensors are not identical")
    require(streams["fixed2"] == streams["sampled16_tail2"], "actual ordered 40000 input streams differ")
    require(semantic_inputs["fixed2"] == semantic_inputs["sampled16_tail2"],
            "paired semantic inputs differ")
    evidence, arms = {}, {}
    for arm in ARMS:
        home = Path(context["protocol"]["arms"][arm]["training"])
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
        initial_model_sha256=initials["fixed2"], identical_initial_weights=True,
        training_receipts={arm: sha(base.training_receipt_path(run, arm)) for arm in ARMS},
        input_comparison=dict(method="byte_exact", microbatches_per_arm=40000), evidence_sha256=evidence))


def admitted_inference(context):
    run, value = context["run"], read(context["run"] / "inference_admitted.json")
    exact(value, dict(admitted=True, identical_initial_weights=True, inference_depths=DEPTHS,
        coverage=len(context["targets"]), sampler=SAMPLER, **base.context_binding(context)),
        "inference admission binding differs")
    require(set(value.get("arms", {})) == set(value.get("training_receipts", {})) == set(ARMS),
            "incomplete common inference arms")
    base.validate_hardware(value.get("hardware"))
    base.seals(run, value.get("evidence_sha256"))
    initials = set()
    for arm in ARMS:
        require(sha(base.training_receipt_path(run, arm)) == value["training_receipts"][arm],
                "training receipt changed after inference")
        training = admitted_training(context, arm)
        initial = read(Path(context["protocol"]["arms"][arm]["training"]) / "initialization.json")
        initials.add(initial["target_sha256"])
        binding = value["arms"][arm]
        exact(binding, dict(hardware=value["hardware"], nominal_K=2, inference_depths=DEPTHS,
            checkpoint_sha256=training["final_checkpoint_sha256"], ema_sha256=training["final_ema_sha256"]),
            "inference own EMA/GPU binding differs")
        require(read(base.evaluation(run, arm) / "gpu_binding.json") == binding, "per-arm GPU binding differs")
        verify_inference(context, arm, binding["checkpoint_sha256"])
    require(initials == {value.get("initial_model_sha256")}, "admitted initial tensors differ")
    return value


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
    names = (arm,) if arm else tuple(ARMS)
    require(set(names) <= set(ARMS), "undeclared scoring arm")
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
    for arm in ARMS:
        receipt = admitted_training(context, arm)
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
        sampled_minus_fixed_at_matched_K={str(depth): paired_effects(context["targets"],
            vectors["fixed2"][str(depth)], vectors["sampled16_tail2"][str(depth)]) for depth in DEPTHS},
        depth_minus_K1={arm: {str(depth): paired_effects(context["targets"], vectors[arm]["1"],
            vectors[arm][str(depth)]) for depth in DEPTHS[1:]} for arm in ARMS},
        training_costs=training_costs, geometry=geometry, inference_hardware=inference["hardware"],
        identical_initial_weights=True, parameters_matched=True, FLOPs_matched=False,
        endpoint_FM_coefficients=dict(fixed2=.5, sampled16_tail2=.5 + .5 / harmonic),
        architecture_selected=False,
        limits="Single seed and scratch WSD10000 applied updates, same data budget and initial tensors. "
            "Both arms use new live initial-residue/PLM reinjection; neither reproduces the old carry function. "
            "Sampled arm jointly changes K, gradient truncation and LiFT loss allocation; this is not a pure K effect. "
            "Only suffix predictions receive trajectory gradients; prefix velocities are diagnostic. "
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
    parser.add_argument("--arm", choices=tuple(ARMS))
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("all comparison execution requires a Slurm allocation")
    if args.stage == "train" and not args.arm or args.stage in ("infer", "report") and args.arm:
        parser.error("train requires --arm; infer/report operate on both arms")
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
