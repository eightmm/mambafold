"""Sealed scratch reset/carry experiment with native zero-update semantic admission."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import sys
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from benchmarks import compare_scratch_hierarchy_fine6 as base  # noqa: E402

scratch, proof, screen, inputs = base.scratch, base.proof, base.screen, base.inputs
ARMS = {f"atom_{mode}": dict(shared_trunk_layers=6, n_latent_layers=6, atom_recycling=mode)
        for mode in ("reset", "carry")}
PARAMETERS = dict.fromkeys(ARMS, 269123226)
MODEL_FIELDS = (*proof.MODEL_FIELDS, "atom_recycling")
ROLE, SAMPLER, QUALITY = base.ROLE, base.SAMPLER, base.QUALITY
read, sha, write_new, require, exact = base.read, base.sha, base.write_new, base.require, base.exact
validate_panel = base.validate_panel
_BASE_VALIDATE = base.validate_run
_BASE_VERIFY = base.verify_actual_training
_BASE_ADMITTED = base.admitted_training
_BASE_ADMITTED_INFERENCE = base.admitted_inference
_BASE_PREFLIGHT = scratch.preflight
_BASE_VERIFY_INFERENCE = proof.verify_inference


@contextmanager
def registry():
    old = scratch.VARIANTS, proof.VARIANTS, proof.MODEL_FIELDS
    scratch.VARIANTS = {key: dict(value) for key, value in ARMS.items()}
    proof.VARIANTS = {key: dict(value) for key, value in ARMS.items()}
    proof.MODEL_FIELDS = MODEL_FIELDS
    try:
        yield
    finally:
        scratch.VARIANTS, proof.VARIANTS, proof.MODEL_FIELDS = old


def expected(arm):
    require(arm in ARMS, "undeclared atom recycling arm")
    with registry():
        return scratch.expected_config(arm)


def common_config(config):
    return base_common({key: value for key, value in config.items() if key != "atom_recycling"})


base_common = base.common_config


@contextmanager
def bindings():
    values = dict(ARMS=ARMS, PARAMETERS=PARAMETERS, registry=registry, common_config=common_config,
                  validate_protocol=validate_protocol, verify_actual_training=verify_actual_training,
                  admitted_training=admitted_training, admitted_inference=admitted_inference)
    old = {key: getattr(base, key) for key in values}
    old_inference = proof.verify_inference
    proof.verify_inference = recycling_verifier
    for key, value in values.items():
        setattr(base, key, value)
    try:
        with registry():
            yield
    finally:
        for key, value in old.items():
            setattr(base, key, value)
        proof.verify_inference = old_inference


def validate_protocol(run, protocol):
    exact(protocol, dict(formal_applied_update_cap_per_arm=10000, microbatches_per_formal_run=40000,
        preflight_optimizer_updates=0, teacher_models=0, all_parameters_trainable=True,
        common_device_final_confirmation_required=True, benchmark_role=ROLE), "atom experiment budget differs")
    require(set(protocol.get("arms", {})) == set(ARMS), "exactly reset/carry arms are required")
    require(protocol.get("sampler") == SAMPLER, "fixed sampler differs")
    exact(protocol["sampler"], SAMPLER, "typed sampler differs")
    require(Path(protocol["panel"]).resolve() == run / "panel", "unexpected atom panel path")
    require(Path(protocol["ost"]).is_absolute(), "canonical OST must be absolute")
    exact(protocol.get("architecture_contract", {}), dict(
        residue_pre_layers=3, residue_post_layers=3, shared_residue_layers=6,
        fine_latent_layers=6, coarse_latent_layers=0, fine_compression=4, K=2,
        full_BPTT=True, FM_coordinates_and_time_fixed_inside_call=True,
        depth_condition_entire_core=True, atom_state_carry_within_velocity_call_only=True),
        "atom architecture contract differs")
    require(protocol["architecture_contract"].get("atom_recycling_modes") == ["reset", "carry"],
            "atom recycling modes differ")
    configs = {}
    for arm, entry in protocol["arms"].items():
        exact(entry, dict(nominal_K=2, expected_parameters=PARAMETERS[arm]), "atom construction differs")
        require(set(entry.get("model_fields", {})) == set(MODEL_FIELDS), "all 35 model fields required")
        exact(entry["model_fields"], {key: expected(arm)[key] for key in MODEL_FIELDS}, "atom model fields differ")
        for key, path in dict(config=run / f"configs/{arm}.yaml", preflight=run / f"preflight/{arm}",
                              training=run / f"training/{arm}", u0=run / "u0").items():
            require(Path(entry[key]).resolve() == path, "atom artifact path differs")
        config = yaml.safe_load(Path(entry["config"]).read_text())
        with registry():
            scratch.validate_config(config, arm)
        exact(config, expected(arm), "typed atom scratch configuration differs")
        configs[arm] = config
    require(common_config(configs["atom_reset"]) == common_config(configs["atom_carry"]),
            "training differs outside atom reset/carry")
    return configs


def validate_run(run):
    with bindings():
        context = _BASE_VALIDATE(run)
    require(context["manifest"].get(base.relative(context["run"], __file__)) == sha(__file__),
            "unsealed atom comparison adapter")
    prior = context["protocol"].get("previous_panel_admission", {})
    exact(prior, dict(expected_targets=15, expected_entries=13, recheck_full_corpus_membership=True),
          "declared panel comparison coverage differs")
    require(len(context["targets"]) == 15
            and len({row["entry_id"] for row in context["targets"].values()}) == 13, "atom panel coverage differs")
    previous = Path(prior["run"])
    receipt = read(previous / "panel_admitted.json")
    require(sha(previous / "panel_admitted.json") == prior["panel_admission_sha256"]
            and sha(previous / "panel/panel.json") == receipt["panel_sha256"], "previous panel seal differs")
    old_targets = {row["target"]: row for row in read(previous / "panel/panel.json")["targets"]}
    require(panel_signature(context["targets"]) == panel_signature(old_targets),
            "atom panel target IDs or sequence/reference/ESM semantics differ from the declared panel")
    return context


def panel_signature(targets):
    fields = ("entry_id", "sequence_sha256", "reference_sha256", "embedding_sha256")
    return {name: tuple(row[key] for key in fields) for name, row in targets.items()}


def semantic_path(context, arm):
    return Path(context["protocol"]["arms"][arm]["preflight"]) / "atom_semantics_verified.json"


def verify_semantics(value, initial_sha):
    exact(value, dict(admitted=True, optimizer_updates=0, checkpoint_loads=0, teacher_models=0,
        initial_model_sha256=initial_sha, restored_model_sha256=initial_sha,
        constructor_pair_identical=True, model_and_rng_restored=True,
        u0_equal=True, u1_equal=True, fixed_input=True, matched_depths=True,
        masked_padding_verified=True, valid_residues=29), "atom semantic proof differs")
    require(value.get("rng_before_sha256") == value.get("rng_after_sha256")
            and proof.valid_sha(value.get("rng_before_sha256")), "semantic RNG restoration differs")
    require(proof.valid_sha(value.get("input_sha256")), "semantic input identity is missing")
    require(proof.finite(value.get("u2_difference_norm"), positive=True), "reset/carry u2 did not diverge")
    require(proof.finite(value.get("reset_final_gradient_norm")) and value["reset_final_gradient_norm"] == 0
            and proof.finite(value.get("carry_final_gradient_norm"), positive=True), "final-only carry gradient differs")
    require(set(value.get("groups", {})) == {"reset", "carry"}, "missing semantic gradient modes")
    for mode, groups in value["groups"].items():
        require(set(groups) == set(proof.GROUPS), "missing semantic parameter gradient groups")
        require(all(proof.finite(row.get("norm"), positive=True) for row in groups.values()),
                "missing/nonfinite semantic gradient group")
        require(value["calls"][mode] == screen.expected_call_counts(2, atom_recycling=mode, training=True),
                "semantic execution counts differ")
        require(value["entire_core_depth_condition"][mode] is True, "entire-core depth condition differs")


def verify_actual_training(context, arm):
    with bindings():
        config, complete, metrics, evidence = _BASE_VERIFY(context, arm)
    home = Path(context["protocol"]["arms"][arm]["training"])
    initial = read(home / "initialization.json")
    native_path = Path(context["protocol"]["arms"][arm]["preflight"]) / "native_verified.json"
    native = read(native_path)
    exact(native, dict(atom_recycling=ARMS[arm]["atom_recycling"],
                       model_fields={key: config[key] for key in MODEL_FIELDS}), "native 35-field identity differs")
    require(all(row.get("observed_calls") == screen.expected_call_counts(2,
        atom_recycling=config["atom_recycling"], training=True) for row in native["rows"]),
        "native atom execution differs")
    path = semantic_path(context, arm)
    verify_semantics(read(path), initial["target_sha256"])
    evidence[base.relative(context["run"], path)] = sha(path)
    return config, complete, metrics, evidence


def admitted_training(context, arm):
    with bindings():
        value = _BASE_ADMITTED(context, arm)
    path = semantic_path(context, arm)
    require(value["evidence_sha256"].get(base.relative(context["run"], path)) == sha(path),
            "semantic admission is not sealed into training")
    initial = read(Path(context["protocol"]["arms"][arm]["training"]) / "initialization.json")
    verify_semantics(read(path), initial["target_sha256"])
    return value


def verify_inference(context, arm, checkpoint_sha):
    return _BASE_VERIFY_INFERENCE(base.evaluation(context["run"], arm), 10000, [2], checkpoint_sha,
        context["targets"], False, atom_recycling=ARMS[arm]["atom_recycling"])


def recycling_verifier(directory, *args, **kwargs):
    arm = Path(directory).name.removesuffix("-10000")
    require(arm in ARMS, "inference directory does not name a declared recycling arm")
    kwargs["atom_recycling"] = ARMS[arm]["atom_recycling"]
    return _BASE_VERIFY_INFERENCE(directory, *args, **kwargs)


def rng_digest(state):
    digest = hashlib.sha256(repr(state["python"]).encode())
    numpy = state["numpy"]
    digest.update(repr((numpy[0], *numpy[2:])).encode())
    digest.update(numpy[1].tobytes())
    digest.update(base.audit.tensor_digest((key, state[key]) for key in ("torch", "cuda")
                                          if key in state).encode())
    return digest.hexdigest()


@contextmanager
def temporary_readouts(model):
    import torch

    parameters = dict(model.named_parameters())
    names = ("atom_decoder.out.weight", "latent_residual.output_proj.weight")
    saved = {name: parameters[name].detach().clone() for name in names}
    try:
        with torch.no_grad():
            for name in names:
                parameter = parameters[name]
                require(torch.count_nonzero(parameter).item() == 0, "probe readout was not freshly zero")
                value = torch.arange(parameter.numel(), device=parameter.device, dtype=parameter.dtype)
                parameter.copy_((value.remainder(17) - 8).reshape_as(parameter) * 0.001)
        yield
    finally:
        with torch.no_grad():
            for name, value in saved.items():
                parameters[name].copy_(value)


def semantic_probe(config, native_sha, destination):
    """Small synthetic input; native production model and kernels, no optimizer."""
    import torch
    from mambafold.train import trainer

    base.no_outputs([destination])
    caller_rng = trainer.capture_rng_state()
    before_rng = rng_digest(caller_rng)
    model = None
    result = None
    try:
        with scratch.forbid_checkpoint_loads():
            trainer.seed_all(config["seed"])
            model = trainer.build_model(config, "cuda").train()
            initial = base.audit.tensor_digest(model.state_dict().items())
            require(initial == native_sha and all(p.requires_grad for p in model.parameters()),
                    "semantic model differs from the native random model")
            other_config = dict(config, atom_recycling="carry" if config["atom_recycling"] == "reset" else "reset")
            trainer.seed_all(config["seed"])
            other = trainer.build_model(other_config, "cuda")
            require(base.audit.tensor_digest(other.state_dict().items()) == initial
                    and all(parameter.requires_grad for parameter in other.parameters()),
                    "reset/carry constructors did not produce identical initial weights")
            del other
            gc.collect()
            torch.cuda.empty_cache()
            trainer.seed_all(config["seed"] + 17)
            batch = trainer._synthetic_batch(1, 32, 14, config["d_plm"], torch.device("cuda"))
            batch = replace(batch, x_clean=torch.randn_like(batch.x_clean) * .1,
                x_t=torch.randn_like(batch.x_t) * .1, eps=torch.randn_like(batch.eps),
                esm=torch.randn_like(batch.esm))
            residue_mask = batch.res_mask.clone()
            residue_mask[:, -3:] = False
            atom_mask = batch.atom_mask & residue_mask.unsqueeze(-1)
            batch = replace(batch, res_mask=residue_mask, ca_mask=batch.ca_mask & residue_mask,
                            atom_mask=atom_mask, valid_mask=batch.valid_mask & atom_mask)
            input_sha = base.audit.batch_digest(batch)
            grid_rng = trainer.capture_rng_state()
            old_mode, old_loops = model.atom_recycling, model.refinement_training_loops
            groups, calls, condition_checks, gradients, outputs, depths = {}, {}, {}, {}, {}, {}
            try:
                with temporary_readouts(model):
                    for mode in ("reset", "carry"):
                        trainer.restore_rng_state(grid_rng)
                        model.atom_recycling, model.refinement_training_loops = mode, 2
                        model.zero_grad(set_to_none=True)
                        counts, features = Counter(), []
                        conditions = {key: [] for key in ("encoder", "residue", "latent", "decoder")}
                        handles = screen.execution_counter_hooks(model, counts, atom_recycling=mode)
                        for key, module in (("encoder", model.atom_encoder.mamba),
                                            ("residue", model.latent_residual.residue_stack),
                                            ("latent", model.latent_residual.stack),
                                            ("decoder", model.atom_decoder)):
                            def capture(module, args, key=key):
                                index = 3 if key == "decoder" else 2
                                conditions[key].append(args[index].detach().clone())
                            handles.append(module.register_forward_pre_hook(capture))
                        def atoms(module, args, output):
                            if isinstance(output, tuple):
                                features.append(output[1])
                        handles.append(model.atom_decoder.register_forward_hook(atoms))
                        try:
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                output = model(batch)
                            require(torch.count_nonzero(output["v_atom"][~batch.atom_mask]).item() == 0,
                                    "semantic padding produced nonzero atom velocity")
                            require(dict(counts) == screen.expected_call_counts(2,
                                atom_recycling=mode, training=True), "native semantic execution differs")
                            require(len(features) == 2 and features[0].requires_grad, "missing differentiable A1")
                            final = output["v_atom"].float().square().mean()
                            (gradient,) = torch.autograd.grad(final, features[0], allow_unused=True, retain_graph=True)
                            gradients[mode] = 0.0 if gradient is None else float(gradient.float().norm())
                            loss, _ = scratch.engine.allatom_loss_surface(output, batch,
                                alpha_mode=config["alpha_mode"], use_rigid_align=config["use_rigid_align"],
                                w_fm=config["w_fm"], w_lddt_atom=config["w_lddt_atom"],
                                lddt_cutoff_A=config["lddt_cutoff_A"],
                                lddt_pair_chunk_size=config["lddt_pair_chunk_size"])
                            require(torch.isfinite(loss).item(), "nonfinite semantic LiFT loss")
                            loss.backward()
                            groups[mode] = scratch._gradient_rows(model)
                            outputs[mode] = tuple(value.detach().clone() for value in output["v_atom_steps"])
                            depths[mode] = output["latent_depths"].detach().clone()
                            calls[mode] = dict(counts)
                            condition_checks[mode] = (
                                all(len(conditions[key]) == n for key, n in
                                    (("encoder", 3), ("decoder", 3), ("residue", 2), ("latent", 2)))
                                and all(torch.equal(conditions["encoder"][i + 1], conditions[key][i + offset])
                                    for i in range(2) for key, offset in
                                    (("residue", 0), ("latent", 0), ("decoder", 1)))
                                and not torch.equal(conditions["residue"][0], conditions["residue"][1]))
                            require(base.audit.batch_digest(batch) == input_sha, "semantic FM inputs changed")
                        finally:
                            for handle in handles:
                                handle.remove()
                        del output, loss, final, gradient, features
                require(torch.equal(outputs["reset"][0], outputs["carry"][0])
                        and torch.equal(outputs["reset"][1], outputs["carry"][1]), "u0/u1 differ between reset/carry")
                require(torch.equal(depths["reset"], depths["carry"]), "semantic depth grids differ")
                difference = float((outputs["carry"][2].float() - outputs["reset"][2].float()).norm())
                result = dict(admitted=True, optimizer_updates=0, checkpoint_loads=0, teacher_models=0,
                    initial_model_sha256=initial, constructor_pair_identical=True,
                    u0_equal=True, u1_equal=True, u2_difference_norm=difference, fixed_input=True,
                    input_sha256=input_sha, matched_depths=True, groups=groups, calls=calls,
                    masked_padding_verified=True, valid_residues=int(residue_mask.sum()),
                    entire_core_depth_condition=condition_checks,
                    reset_final_gradient_norm=gradients["reset"], carry_final_gradient_norm=gradients["carry"],
                    input_scope="one native full-size 32-position batch, 29 valid residues and right padding")
            finally:
                model.atom_recycling, model.refinement_training_loops = old_mode, old_loops
                model.zero_grad(set_to_none=True)
            restored = base.audit.tensor_digest(model.state_dict().items())
            require(restored == initial, "semantic probe did not restore exact model state")
            result["restored_model_sha256"] = restored
    finally:
        trainer.restore_rng_state(caller_rng)
        after_rng = rng_digest(trainer.capture_rng_state())
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
        handles.extend(screen.execution_counter_hooks(model, counts, atom_recycling=model.atom_recycling))
        return model
    def anchor(model, batch, output, config):
        require(dict(counts) == screen.expected_call_counts(2,
            atom_recycling=config["atom_recycling"], training=True), "native preflight atom counters differ")
        observed.append(dict(counts))
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
        "native atom model layout differs before semantic/formal work")
    require(len(observed) == len(native["rows"]) == 8, "incomplete native atom counter rows")
    for row, calls in zip(native["rows"], observed, strict=True):
        row["observed_calls"] = calls
    native.update(atom_recycling=config["atom_recycling"],
                  model_fields={key: config[key] for key in MODEL_FIELDS})
    base.audit.write_json(native_path, native)
    gc.collect()
    semantic_probe(config, native["model_sha256"], args.out / "atom_semantics_verified.json")


def train(context, arm):
    original = scratch.preflight
    scratch.preflight = native_preflight
    try:
        with bindings():
            base.train(context, arm)
    finally:
        scratch.preflight = original


def infer(context):
    import torch
    from mambafold.model.bimamba3 import mamba3_runtime_info

    run = context["run"]
    base.no_outputs([run / "inference_admitted.json", *(base.evaluation(run, arm) for arm in ARMS)])
    device = base.hardware()
    torch.set_num_threads(1)
    admitted = {arm: admitted_training(context, arm) for arm in ARMS}
    require(common_config(admitted["atom_reset"]["config"])
            == common_config(admitted["atom_carry"]["config"]), "actual reset/carry common config differs")
    streams, initials, semantics, runtimes = {}, {}, {}, {}
    runtime = mamba3_runtime_info()
    for arm in ARMS:
        verify_actual_training(context, arm)
        home = Path(context["protocol"]["arms"][arm]["training"])
        initial = read(home / "initialization.json")
        initials[arm] = initial["target_sha256"]
        runtimes[arm] = initial["mamba3_runtime"]
        base.verify_external_runtime(runtimes[arm], runtime)
        semantics[arm] = read(semantic_path(context, arm))
        streams[arm], digest = inputs.read_inputs(home / "audit/inputs.jsonl", 40000)
        require(digest == admitted[arm]["complete"]["input_stream_sha256"], "actual input digest differs")
    require(initials["atom_reset"] == initials["atom_carry"], "formal reset/carry initials are not bitwise identical")
    require(streams["atom_reset"] == streams["atom_carry"], "actual ordered 40000 inputs differ")
    require(semantics["atom_reset"]["input_sha256"] == semantics["atom_carry"]["input_sha256"],
            "semantic probes used different input tensors")
    evidence, arms = {}, {}
    for arm in ARMS:
        home = Path(context["protocol"]["arms"][arm]["training"])
        directory = base.evaluation(run, arm)
        require(base.hardware() == device, "common inference physical GPU changed")
        with base.inference_model_runtime(runtimes[arm]):
            screen.infer(SimpleNamespace(checkpoint=home / "ckpt_0010000.pt", panel=run / "panel",
                                        out=directory, depths=[2]))
        require(base.hardware() == device, "common inference physical GPU changed")
        verify_inference(context, arm, admitted[arm]["final_checkpoint_sha256"])
        arms[arm] = dict(hardware=device, nominal_K=2,
            checkpoint_sha256=admitted[arm]["final_checkpoint_sha256"], ema_sha256=admitted[arm]["final_ema_sha256"])
        write_new(directory / "gpu_binding.json", arms[arm])
        for path in (directory / "sampling.json", directory / "hardware.json",
                     directory / "gpu_binding.json", directory / "K2/pairs/manifest.json"):
            evidence[base.relative(run, path)] = sha(path)
        for target in context["targets"]:
            path = directory / f"K2/pairs/{target}_seed0_pred.pdb"
            evidence[base.relative(run, path)] = sha(path)
        gc.collect()
        torch.cuda.empty_cache()
    base.unchanged(context)
    write_new(run / "inference_admitted.json", dict(admitted=True, **base.context_binding(context),
        hardware=device, sampler=SAMPLER, coverage=len(context["targets"]), arms=arms,
        initial_model_sha256=initials["atom_reset"], identical_initial_weights=True,
        training_receipts={arm: sha(base.training_receipt_path(run, arm)) for arm in ARMS},
        input_comparison=dict(method="byte_exact", microbatches_per_arm=40000), evidence_sha256=evidence))


def admitted_inference(context):
    with bindings():
        value = _BASE_ADMITTED_INFERENCE(context)
    exact(value, dict(identical_initial_weights=True), "missing identical-initial inference proof")
    initials = {read(Path(context["protocol"]["arms"][arm]["training"]) / "initialization.json")["target_sha256"]
                for arm in ARMS}
    require(initials == {value.get("initial_model_sha256")}, "admitted initials differ")
    return value


def score(context, arm=None):
    with bindings():
        return base.score(context, arm)


def paired_metrics(targets, vectors):
    require(set(vectors) == set(ARMS) and targets and all(set(values) == set(targets) for values in vectors.values()),
            "incomplete paired recycling quality coverage")
    require(all(set(row) == set(QUALITY) and all(proof.finite(value) for value in row.values())
        for values in vectors.values() for row in values.values()), "invalid paired recycling metrics")
    entries = {}
    for target, row in targets.items():
        entries.setdefault(row["entry_id"], []).append(target)
    overall = {arm: dict(coverage=len(targets), **{metric: statistics.fmean(values[name][metric]
        for name in sorted(targets)) for metric in QUALITY}) for arm, values in vectors.items()}
    per_entry = {entry: dict(coverage=len(names), arms={arm: {metric: statistics.fmean(values[name][metric]
        for name in names) for metric in QUALITY} for arm, values in vectors.items()})
        for entry, names in sorted(entries.items())}
    effects = {}
    for metric in QUALITY:
        deltas = {name: vectors["atom_carry"][name][metric] - vectors["atom_reset"][name][metric]
                  for name in sorted(targets)}
        entry_deltas = {entry: statistics.fmean(deltas[name] for name in names)
                        for entry, names in sorted(entries.items())}
        effects[metric] = dict(chain_mean_delta=statistics.fmean(deltas.values()),
            equal_entry_mean_delta=statistics.fmean(entry_deltas.values()),
            entry_deltas=entry_deltas, target_deltas=deltas)
    return dict(coverage=dict(targets=len(targets), entries=len(entries)), overall=overall,
        per_entry=per_entry, atom_carry_minus_reset=effects,
        all_zero_all_atom=all(row["lddt"] == 0 for values in vectors.values() for row in values.values()))


def geometry_diagnostics(directory, targets, vectors, scorer):
    """Count OST violation entries; these are descriptive, unnormalised counts."""
    fields = tuple(f"{role}_{kind}" for role in ("model", "reference")
                   for kind in ("clashes", "bad_bonds", "bad_angles"))
    rows, evidence = {}, {}
    require(set(vectors) == set(targets), "geometry quality coverage differs")
    raw_directory = directory / "K2/scores/raw"
    for target in sorted(targets):
        raw_path = raw_directory / f"{target}_seed0.json"
        identity_path = raw_directory / f"{target}_seed0.inputs.json"
        raw, identity = read(raw_path), read(identity_path)
        require(raw.get("status") == "SUCCESS", "geometry raw score was not successful")
        for metric in QUALITY:
            require(proof.finite(raw.get(metric)) and math.isclose(raw[metric], vectors[target][metric],
                rel_tol=1e-12, abs_tol=1e-15), "geometry raw quality differs from canonical summary")
        exact(identity, dict(schema_version=1,
            prediction_sha256=sha(directory / f"K2/pairs/{target}_seed0_pred.pdb"),
            reference_sha256=targets[target]["reference_sha256"],
            openstructure_version=scorer["version"], compare_args=scorer["compare_args"]),
            "geometry scorer/input identity differs")
        require(all(isinstance(raw.get(field), list) for field in fields),
                "missing or malformed canonical geometry arrays")
        rows[target] = {field: len(raw[field]) for field in fields}
        for path in (raw_path, identity_path):
            evidence[str(path)] = sha(path)
    return dict(coverage_targets=len(rows), unit="OST violation entries; unnormalised counts",
                per_target=rows, totals={field: sum(row[field] for row in rows.values()) for field in fields},
                evidence_sha256=evidence)


def report(context):
    import csv

    run = context["run"]
    base.no_outputs([run / "comparison.json", run / "comparison.csv"])
    inference = admitted_inference(context)
    vectors, costs, scorers, geometry = {}, {}, [], {}
    for arm in ARMS:
        value = read(run / f"scores/{arm}.json")
        exact(value, dict(admitted=True, arm=arm, coverage=len(context["targets"]),
            inference_admission_sha256=sha(run / "inference_admitted.json"), **base.context_binding(context)),
            "atom score admission differs")
        directory = base.evaluation(run, arm)
        path = directory / "K2/scores/summary.json"
        require(sha(path) == value["summary_sha256"], "canonical atom summary changed")
        summary = read(path)
        vectors[arm] = base.verify_canonical(summary, directory, context)
        require(vectors[arm] == value["metrics"] and summary["openstructure"] == value["scorer"],
                "canonical atom score vectors/scorer differ")
        require(sha(context["protocol"]["ost"]) == value["ost_executable_sha256"], "OST executable changed")
        scorers.append(value["scorer"])
        geometry[arm] = geometry_diagnostics(directory, context["targets"], vectors[arm], value["scorer"])
        receipt = admitted_training(context, arm)
        training = receipt["complete"]
        measured = read(directory / "sampling.json")
        costs[arm] = dict(parameters=PARAMETERS[arm], training=dict(
            steady_mean_step_ms=training["steady_mean_step_ms"],
            all_update_weighted_mean_step_ms=receipt["all_update_weighted_mean_step_ms"],
            peak_allocated_gib=training["peak_allocated_gib"], peak_reserved_gib=training["peak_reserved_gib"],
            timing_scope=training["timing_scope"]), inference=dict(coverage=len(measured),
            mean_seconds=statistics.fmean(row["seconds"] for row in measured),
            peak_allocated_gib=max(row["peak_allocated_gib"] for row in measured),
            seconds_by_target={row["target"]: row["seconds"] for row in measured}))
    require(scorers[0] == scorers[1], "canonical scorer differs across arms")
    result = dict(benchmark_role=ROLE, nominal_K=2, sampler=SAMPLER, **base.context_binding(context),
        panel_coverage=read(run / "panel_admitted.json")["coverage"], **paired_metrics(context["targets"], vectors),
        costs=costs, geometry=geometry, inference_hardware=inference["hardware"], identical_initial_weights=True,
        parameters_matched=True, architecture_selected=False,
        limits="Single seed, fixed scratch WSD10000 applied updates. Descriptive declared sequence-homology-excluded "
            "whole-target results; not remote-homology/coordinate/ESMC-pretraining exclusion. Identical trainable "
            "modules/initial weights and fixed sampler; only incoming hidden atom state reset/carry differs. "
            "Full BPTT within each velocity call. CA is canonical OST bb_lddt. Measured instrumented sampler "
            "includes final CPU copy, excludes loading/PDB/scoring; single observation per target. No auto-selection.")
    base.unchanged(context)
    write_new(run / "comparison.json", result)
    with (run / "comparison.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["arm", "coverage", *QUALITY, "parameters", "steady_train_ms", "all_update_train_ms",
                         "train_allocated_gib", "train_reserved_gib", "inference_seconds", "inference_allocated_gib"])
        for arm in ARMS:
            cost = costs[arm]
            writer.writerow([arm, len(context["targets"]), *(result["overall"][arm][key] for key in QUALITY),
                PARAMETERS[arm], cost["training"]["steady_mean_step_ms"],
                cost["training"]["all_update_weighted_mean_step_ms"], cost["training"]["peak_allocated_gib"],
                cost["training"]["peak_reserved_gib"], cost["inference"]["mean_seconds"],
                cost["inference"]["peak_allocated_gib"]])
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
