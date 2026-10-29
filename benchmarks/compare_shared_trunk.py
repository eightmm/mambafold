#!/usr/bin/env python3
"""Paired, bounded architecture screens from one production EMA checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from benchmarks.compare_full_refinement import tensor_digest, write_json  # noqa: E402

ARMS = ("control", "shared", "hierarchy")
COARSE_PREFIX = "latent_residual.stack.coarse_"


def expected_call_counts(depth, *, atom_recycling="none", training=False, network=1, hierarchy=False):
    if atom_recycling not in ("none", "reset", "carry"):
        raise ValueError("unknown atom recycling execution policy")
    calls = dict(network=network, decoder=network * (depth + 1 if training else 1),
                 residue=network * depth, latent=network * depth)
    if not depth:
        calls.pop("residue")
        calls.pop("latent")
    if hierarchy and depth:
        calls["coarse"] = network * depth
    if atom_recycling != "none":
        decoder = depth + int(training) if depth else 1
        calls.update(decoder=network * decoder, encoder=network,
                     encoder_slot=network * (depth + 1), encoder_cross=network * (depth + 1),
                     decoder_slot=network * decoder, decoder_cross=network * decoder,
                     head=network * (depth + 1 if training else 1))
    return calls


def execution_counter_hooks(model, calls, *, atom_recycling="none", depth=2):
    modules = dict(network=model, decoder=model.atom_decoder)
    if depth:
        modules.update(residue=model.latent_residual.residue_stack, latent=model.latent_residual.stack)
        if hasattr(model.latent_residual.stack, "coarse_stack"):
            modules["coarse"] = model.latent_residual.stack.coarse_stack
    if atom_recycling != "none":
        modules.update(encoder=model.atom_encoder, encoder_slot=model.atom_encoder.mamba,
                       encoder_cross=model.atom_encoder.cross, decoder_slot=model.atom_decoder.mamba,
                       decoder_cross=model.atom_decoder.cross, head=model.atom_decoder.out)
    if any(module is None for module in modules.values()):
        raise ValueError("declared execution counter module is absent")
    return [module.register_forward_hook(lambda *args, key=key: calls.update({key: 1}))
            for key, module in modules.items()]


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def migrate_ema(source, target, prelude, shared):
    """Validate every target tensor before copying; donor suffix is explicit.

    The first suffix group is reused twice: indices3..8 for3+6, or9..11 for9+3.
    The remaining original suffix and the narrow residue stack are discarded.
    This changes the teacher function and requires initial structure admission.
    """
    result, mapping, fresh = {}, {}, []
    for name, expected in target.items():
        donor = name
        if shared and name.startswith("latent_residual.residue_stack.layers."):
            suffix = name.removeprefix("latent_residual.residue_stack.layers.")
            index, rest = suffix.split(".", 1)
            if int(index) >= shared:
                raise ValueError("unexpected shared donor index")
            donor = f"residue_trunk.layers.{prelude + int(index)}.{rest}"
        if name.startswith(COARSE_PREFIX):
            fresh.append(name)
            result[name] = expected
            continue
        if donor not in source or (source[donor].shape != expected.shape
                                  or source[donor].dtype != expected.dtype):
            raise ValueError(f"missing or incompatible donor for {name}")
        mapping[name] = donor
        result[name] = source[donor]
    used = set(mapping.values())
    excluded = sorted(set(source) - used)
    if not shared and (excluded or fresh or source.keys() != target.keys()):
        raise ValueError("control must preserve the complete source EMA")
    if shared and any(not (name.startswith("latent_residual.residue_stack.")
                          or name.startswith("residue_trunk.layers.")) for name in excluded):
        raise ValueError("migration would discard an unrelated learned component")
    for name in fresh:
        if name.startswith(COARSE_PREFIX + "up.") and torch.count_nonzero(result[name]).item():
            raise ValueError("coarse restoration must start at zero")
    return result, dict(mapping=mapping, fresh=fresh, excluded_source=excluded,
                        function_preserved=not bool(shared),
                        hierarchy_initially_preserves_converted_single_level=True)


def verify_budget(initial, complete, steps, accumulation):
    if initial["parameters"] != initial["trainable_parameters"]:
        raise ValueError("all backbone parameters must remain trainable")
    expected = dict(updates=steps, attempts=steps, skipped_updates=0,
                    microbatches=steps * accumulation, finite_losses=steps * accumulation)
    if any(complete.get(key) != value for key, value in expected.items()):
        raise ValueError("incomplete/interrupted/retried training budget")


def velocity_summary(steps, valid_mask):
    """Describe the actual LiFT anchor without letting padding dominate scale."""
    if len(steps) < 2 or not valid_mask.any():
        raise ValueError('u0 probe requires recurrent readouts and observed atoms')
    values = [value.detach().float()[valid_mask].reshape(-1) for value in steps]
    if not all(torch.isfinite(value).all() for value in values):
        raise ValueError('nonfinite u0 or recurrent velocity')
    anchor, final = values[0], values[-1]
    anchor_rms = float(anchor.square().mean().sqrt())
    final_rms = float(final.square().mean().sqrt())
    delta = float((final - anchor).square().mean().sqrt())
    denominator = float(anchor.norm() * final.norm())
    return dict(u0_rms=anchor_rms, final_rms=final_rms, final_minus_u0_rms=delta,
                relative_update=delta / anchor_rms if anchor_rms > 1e-8 else None,
                u0_final_cosine=float(anchor.dot(final)) / denominator
                if denominator > 1e-12 else None,
                observed_atoms=int(valid_mask.sum()), readouts=len(steps))


def probe(args):
    from benchmarks.compare_full_refinement import batch_digest
    from mambafold.data.loader import build_dataloaders
    from mambafold.train.engine import allatom_loss_surface
    from mambafold.train.trainer import build_model, seed_all

    state = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    checkpoint_step = state['step']
    expected_step = getattr(args, 'expected_step', 0)
    if checkpoint_step != expected_step:
        raise ValueError('u0 probe checkpoint step differs from its declared source')
    config = state['args']
    model = build_model(config, 'cuda').train()
    model.load_state_dict(state['ema'], strict=True)
    initial_model_sha256 = tensor_digest(model.state_dict().items())
    del state
    seed_all(config['seed'])
    loader, _, _ = build_dataloaders(SimpleNamespace(**config), False)
    loader.generator = torch.Generator().manual_seed(config['seed'])
    batches = getattr(args, 'batches', 4)
    if type(batches) is not int or batches < 1:
        raise ValueError('probe batches must be a positive integer')
    expected = [json.loads(line)['sha256']
                for line in (args.audit/'inputs.jsonl').read_text().splitlines()][:batches]
    if len(expected) != batches:
        raise ValueError('missing paired initial training input audit')
    rows = []
    iterator = iter(loader)
    for index in range(batches):
        batch = next(iterator)
        identity = batch_digest(batch)
        if identity != expected[index]:
            raise ValueError('u0 probe does not replay the audited initial training inputs')
        batch = batch.to('cuda')
        with torch.enable_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(batch)
        validation = (args.validate_output(model, batch, output, config)
                      if getattr(args, 'validate_output', None) else {})
        loss, metrics = allatom_loss_surface(
            output, batch, alpha_mode=config['alpha_mode'],
            use_rigid_align=config['use_rigid_align'], w_fm=config['w_fm'],
            w_lddt_atom=config['w_lddt_atom'],
            lddt_pair_chunk_size=config['lddt_pair_chunk_size'])
        row = dict(microbatch=index + 1, input_sha256=identity,
                   t_mean=float(batch.t.mean()), loss=float(loss.detach()),
                   **velocity_summary(output['v_atom_steps'], batch.valid_mask),
                   **validation,
                   **{key: metrics[key] for key in metrics if key.startswith(
                       ('fm_prelude', 'fm_endpoint_', 'fm_trajectory', 'fm_objective',
                        'velocity_delta_'))})
        if not all(math.isfinite(value) for value in row.values() if isinstance(value, float)):
            raise ValueError('nonfinite initial LiFT diagnostic')
        rows.append(row)
        del output, loss, batch
    result = dict(rows=rows, updates=0, checkpoint_step=checkpoint_step,
                  weight_source='transplanted initial EMA' if checkpoint_step == 0
                  else 'checkpoint EMA; no probe updates',
                  checkpoint_sha256=sha256(args.checkpoint),
                  initial_model_sha256=initial_model_sha256,
                  mean={key: statistics.fmean(row[key] for row in rows if row[key] is not None)
                        for key, value in rows[0].items() if isinstance(value, float)},
                  alignment='One final-based alignment within each model, as in actual LiFT; '
                  'not a shared cross-model target frame.',
                  u0_semantics='Pre-core atom velocity; detached anchor in trajectory target, '
                  'also trained by the existing0.01 prelude loss. No probe optimizer updates.')
    write_json(args.out/'u0.json', result)


def train(args):
    from benchmarks import compare_full_refinement as audit
    from mambafold.model.bimamba3 import mamba3_runtime_info
    from mambafold.train import trainer

    cfg = yaml.safe_load(args.config.read_text())
    if cfg.get("resume") or cfg.get("initialize_refiner_from"):
        raise ValueError("screen uses fresh EMA/optimizer/data state, not a resume interface")
    args.out.mkdir(parents=True, exist_ok=False)
    original_build = trainer.build_model
    original_prewarm = trainer.prewarm_sequence_kernels

    def build(config, device):
        model = original_build(config, device)
        state = torch.load(args.source, map_location="cpu", mmap=True, weights_only=False)
        source_step = getattr(args, 'source_step', 10000)
        same_layout = getattr(args, 'same_layout', False)
        if state["step"] != source_step:
            raise ValueError("source checkpoint step differs from the declared initialization")
        runtime = mamba3_runtime_info()
        if any(state.get("mamba3_runtime", {}).get(key) != runtime[key]
               for key in ("revision", "module_sha256", "decay")):
            raise ValueError("source and current Mamba runtime differ")
        architecture_keys = (
            "d_res", "n_trunk", "d_latent", "n_latent_layers", "latent_compression",
            "latent_loops", "latent_supervision", "d_temb", "d_plm", "d_plm_proj",
            "d_state", "mimo_rank", "expand", "headdim", "bimamba_share",
            "d_atom", "n_atom_layers", "atom_mixer", "atom_d_state", "atom_mimo_rank",
            "n_atom_cross_layers", "n_backbone_streams", "d_ca_emb", "d_res_type",
            "d_res_pos", "use_plm", "residue_refiner", "latent_loop_mode", "latent_position",
            "self_conditioning", "atom_recurrence",
        )
        if same_layout:
            architecture_keys += ('shared_trunk_layers', 'latent_hierarchy')
        declared = set(getattr(args, 'architecture_changes', ()))
        supported = {'n_trunk', 'n_latent_layers', 'latent_compression', 'latent_loops'}
        if not declared <= supported:
            raise ValueError('unsupported declared architecture change')
        if any(state["args"].get(key) != config.get(key) for key in architecture_keys
               if key not in declared):
            raise ValueError("source architecture differs outside declared sharing/hierarchy")
        if same_layout:
            values = state['ema']
            receipt = dict(function_preserved=True, initialization='same-layout EMA continuation')
        else:
            migration = getattr(args, 'migration', migrate_ema)
            values, receipt = migration(state["ema"], model.state_dict(),
                                        model.residue_trunk.n_layers, model.shared_trunk_layers)
        model.load_state_dict(values, strict=True)
        receipt.update(source_sha256=sha256(args.source), source_step=source_step,
                       weight_source="ema", optimizer="fresh", data_cursor="fresh",
                       target_sha256=tensor_digest(model.state_dict().items()),
                       source_ema_sha256=tensor_digest(state["ema"].items()),
                       shared_trunk_layers=model.shared_trunk_layers,
                       hierarchy=bool(config.get("latent_hierarchy", False)))
        write_json(args.out / "initialization.json", receipt)
        # Constructor RNG consumption differs by architecture. Worker RNG is
        # separately isolated by the existing audit's loader generator.
        trainer.seed_all(config["seed"])
        return model

    def prewarm(*a, **kw):
        warmed = original_prewarm(*a, **kw)
        expected = math.ceil(kw['max_length'] / kw['length_bin'])
        if len(warmed) != expected:
            raise ValueError("not all training lengths fit during kernel admission")
        return warmed

    trainer.build_model = build
    trainer.prewarm_sequence_kernels = prewarm
    try:
        audit.train(SimpleNamespace(config=args.config, out=args.out))
    finally:
        trainer.build_model = original_build
        trainer.prewarm_sequence_kernels = original_prewarm
    initial = json.loads((args.out / "audit/initial.json").read_text())
    complete = json.loads((args.out / "audit/complete.json").read_text())
    verify_budget(initial, complete, cfg["total_steps"], cfg["grad_accum_steps"])
    final = args.out / f"ckpt_{cfg['total_steps']:07d}.pt"
    state = torch.load(final, map_location="cpu", mmap=True, weights_only=False)
    if state["step"] != cfg["total_steps"] or (args.out / "preempted.json").exists():
        raise ValueError("final checkpoint does not certify the requested budget")
    write_json(args.out / "budget_verified.json", dict(steps=state["step"], skipped=0))


def infer(args):
    n_steps = getattr(args, "n_steps", 500)
    if type(n_steps) is not int or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    from mambafold.sampling import prepare_inference_batch, sample
    from mambafold.structure_io import write_pdb
    from mambafold.train.distributed import enable_cuda_perf_flags
    from mambafold.train.trainer import build_model
    from scripts.rollout import benchmark_seeds, example_from_sequence

    enable_cuda_perf_flags()
    state = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    model = build_model(state["args"], "cuda").eval()
    model.load_state_dict(state["ema"], strict=True)
    checkpoint_step = state["step"]
    del state
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / 'hardware.json', dict(
        gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
        torch=torch.__version__, cuda=torch.version.cuda,
        timing='Warm sampler; actual SDE calls audited; includes final CPU copy, '
               'excludes model/input loading, PDB writing and scoring.'))
    panel = json.loads((args.panel / "panel.json").read_text())["targets"]
    rows, warmed = [], set()
    branch = model.latent_residual
    for index, target in enumerate(panel):
        with np.load(args.panel / "inputs" / f"{target['target']}.npz") as packed:
            example = example_from_sequence(target["sequence"],
                                           torch.from_numpy(packed["esm"].copy()))
        if torch.count_nonzero(example.coords):
            raise ValueError("reference coordinates must not enter inference")
        batch = prepare_inference_batch([example], "cuda", length_bin=64)
        order = args.depths[index % len(args.depths):] + args.depths[:index % len(args.depths)]
        for depth in order:
            model.latent_residual = branch if depth else None
            if depth:
                branch.n_loops = depth
            seeds = [benchmark_seeds(0, [target["target"]])[0]]
            kwargs = dict(n_steps=n_steps, method="sde", sde_tau=.01, return_trunk_latent=False)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(batch)
                model(batch)
                if depth not in warmed:
                    sample(model, batch, seeds, **kwargs)
                    warmed.add(depth)
            calls = Counter()
            recycling = getattr(model, "atom_recycling", "none")
            handles = execution_counter_hooks(model, calls, atom_recycling=recycling, depth=depth)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            try:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    result = sample(model, batch, seeds, **kwargs)
                torch.cuda.synchronize()
            finally:
                for handle in handles:
                    handle.remove()
                model.latent_residual = branch
            elapsed = time.perf_counter() - started
            expected = expected_call_counts(depth, atom_recycling=recycling,
                network=calls["network"], hierarchy=hasattr(branch.stack, "coarse_stack"))
            if calls["network"] < 1 or dict(calls) != expected:
                raise ValueError("incorrect actual sampling execution counts")
            name = target["target"] + "_seed0"
            pairs = args.out / f"K{depth}/pairs"
            pairs.mkdir(parents=True, exist_ok=True)
            write_pdb(result.final_aa[0, :target["length"]].numpy(), example.res_type.numpy(),
                      example.atom_mask.numpy(), np.zeros((target["length"], 14)),
                      np.zeros(target["length"], dtype=np.int64), pairs / f"{name}_pred.pdb")
            (pairs / f"{name}_gt.pdb").symlink_to(
                (args.panel / "references" / f"{target['target']}.pdb").resolve())
            rows.append(dict(target=target["target"], K=depth, checkpoint_step=checkpoint_step,
                             seconds=elapsed, observed_calls=dict(calls),
                             prediction_sha256=sha256(pairs / f"{name}_pred.pdb"),
                             reference_sha256=sha256(pairs / f"{name}_gt.pdb"),
                             peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                             peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
            if recycling != "none":
                rows[-1]["atom_recycling"] = recycling
            if getattr(model, "recycle_policy", "legacy") != "legacy":
                rows[-1]["recycle_policy"] = model.recycle_policy
            write_json(args.out / "sampling.json", rows)
    for depth in args.depths:
        write_json(args.out / f"K{depth}/pairs/manifest.json", dict(
            expected=len(panel), checkpoint_sha256=sha256(args.checkpoint),
            weight_source="ema", sampler=dict(method="sde", n_steps=n_steps, tau=.01, seed=0),
            rows=[row for row in rows if row["K"] == depth]))


def paired_summary(before, after):
    if not before or before.keys() != after.keys():
        raise ValueError("incomplete paired score coverage")
    delta = {key: after[key] - before[key] for key in before}
    if not all(math.isfinite(value) for value in [*before.values(), *after.values()]):
        raise ValueError("nonfinite structural score")
    entries = {}
    for key, value in delta.items():
        entries.setdefault(key.split("_")[0], []).append(value)
    values = np.array([statistics.fmean(row) for row in entries.values()])
    boot = np.random.default_rng(0).choice(values, (10000, len(values))).mean(1)
    return dict(mean_chain_delta=statistics.fmean(delta.values()),
                mean_entry_delta=float(values.mean()),
                entry_bootstrap95=np.quantile(boot, [.025, .975]).tolist(), chain_delta=delta)


def score_all(run, stage):
    protocol = json.loads((run / "protocol.json").read_text())
    targets = json.loads((Path(protocol["panel"]) / "panel.json").read_text())["targets"]
    expected = {target["target"] for target in targets}
    rows, scores = [], {}
    for sampling in sorted((run / "evaluation" / stage).glob("*/sampling.json")):
        measurements = json.loads(sampling.read_text())
        for depth in sorted({row["K"] for row in measurements}):
            directory = sampling.parent / f"K{depth}"
            subprocess.run([sys.executable, str(ROOT / "benchmarks/score_openstructure.py"),
                            "--in-dir", str(directory / "pairs"), "--out-dir",
                            str(directory / "scores"), "--ost", protocol["ost"],
                            "--expected", str(len(expected))], check=True)
            summary = json.loads((directory / "scores/summary.json").read_text())
            observed = [row["target_id"].removesuffix("_seed0") for row in summary["rows"]]
            timed = [row for row in measurements if row["K"] == depth]
            if (len(observed) != len(expected) or set(observed) != expected
                    or len(timed) != len(expected) or {r['target'] for r in timed} != expected
                    or not summary["evaluation_complete"]):
                raise ValueError("missing, duplicate or failed structural evaluation")
            for row in timed:
                name = row['target'] + '_seed0'
                if (sha256(directory / f'pairs/{name}_pred.pdb') != row['prediction_sha256']
                        or sha256(directory / f'pairs/{name}_gt.pdb') != row['reference_sha256']):
                    raise ValueError("prediction/reference changed after inference")
            values = dict(zip(observed, [row["lddt"] for row in summary["rows"]], strict=True))
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError("nonfinite structural score")
            label = sampling.parent.name
            scores[f"{label}/K{depth}"] = values
            rows.append(dict(label=label, K=depth, lddt=statistics.fmean(values.values()),
                             bb_lddt=summary["metrics"]["bb_lddt"]["mean"],
                             mean_seconds=statistics.fmean(row["seconds"] for row in timed),
                             peak_allocated_gib=max(row["peak_allocated_gib"] for row in timed)))
    write_json(run / f"{stage}_scores.json", dict(rows=rows, scores=scores))
    return scores, rows


def record_initial_u0(args, scores, rows):
    if (args.run/'protocol.json').is_file() and json.loads(
            (args.run/'protocol.json').read_text()).get('require_u0_probe', False):
        probes = {arm: json.loads((args.run/f'u0/{arm}/u0.json').read_text()) for arm in ARMS}
        if len({tuple(row['input_sha256'] for row in p['rows']) for p in probes.values()}) != 1:
            raise ValueError('u0 diagnostics used different inputs')
        if any(probe['updates'] != 0 for probe in probes.values()):
            raise ValueError('u0 was measured after training')
        for arm in ARMS:
            probe = probes[arm]
            initial = json.loads((args.run/f'preflight/{arm}/audit/initial.json').read_text())
            if (len(probe['rows']) != 4
                    or probe['initial_model_sha256'] != initial['model_sha256']
                    or probe['checkpoint_sha256'] != sha256(
                        args.run/f'preflight/{arm}/initial_ema.pt')):
                raise ValueError('u0 diagnosis is not bound to the actual initial weights')
            if f'{arm}/K0' not in scores:
                raise ValueError('missing actual pre-core u0 structure evaluation')
        write_json(args.run/'initial_u0_comparison.json', dict(
            probes=probes, structure_rows=rows,
            prelude_vs_control={arm: paired_summary(scores['control/K0'], scores[f'{arm}/K0'])
                                for arm in ('shared', 'hierarchy')},
            gain_over_u0={arm: paired_summary(scores[f'{arm}/K0'], scores[f'{arm}/K2'])
                          for arm in ARMS}))


def gate(args):
    scores, rows = score_all(args.run, "initial")
    # Preserve the u0 diagnosis even when the final-quality guard rejects migration.
    record_initial_u0(args, scores, rows)
    baseline = scores["control/K2"]
    # Exploratory failure guard, not a statistical noninferiority claim.
    for arm in ("shared", "hierarchy"):
        effect = paired_summary(baseline, scores[f"{arm}/K2"])
        if effect["mean_chain_delta"] < -.05:
            write_json(args.run / "admission_failed.json", dict(arm=arm, effect=effect))
            raise ValueError("conversion loses over0.05 mean lDDT; stop before10k training")
    if scores["shared/K2"] != scores["hierarchy/K2"]:
        raise ValueError("zero coarse insertion changed initial sampled structures/scores")
    initial_samples = [json.loads((args.run / f"evaluation/initial/{arm}/sampling.json")
                                 .read_text()) for arm in ("shared", "hierarchy")]
    hashes = [{row['target']: row['prediction_sha256'] for row in samples if row['K'] == 2}
              for samples in initial_samples]
    if hashes[0] != hashes[1]:
        raise ValueError("zero coarse insertion changed initial predicted structures")
    audits = [json.loads((args.run/f"preflight/{arm}/audit/complete.json").read_text())
              for arm in ARMS]
    if len({row["input_stream_sha256"] for row in audits}) != 1:
        raise ValueError("preflight delivered data streams differ")
    write_json(args.run / "admitted.json", dict(rows=rows, initial_drop_guard=.05))


def report(args):
    scores, rows = score_all(args.run, "trained")
    protocol = json.loads((args.run / "protocol.json").read_text())
    initial_scores = json.loads((args.run / 'initial_scores.json').read_text())['scores']
    configs = [yaml.safe_load((args.run/f'configs/{arm}-train.yaml').read_text()) for arm in ARMS]
    differences = {'shared_trunk_layers', 'latent_hierarchy', 'wandb_name'}
    if any({k: v for k, v in config.items() if k not in differences}
           != {k: v for k, v in configs[0].items() if k not in differences}
           for config in configs[1:]):
        raise ValueError('training configs differ outside the declared architecture changes')
    audits = []
    for arm in ARMS:
        initial = json.loads((args.run/f"training/{arm}/audit/initial.json").read_text())
        complete = json.loads((args.run/f"training/{arm}/audit/complete.json").read_text())
        verify_budget(initial, complete, 10000, protocol["grad_accum_steps"])
        receipt = json.loads((args.run/f'training/{arm}/initialization.json').read_text())
        if (receipt['source_sha256'] != protocol['source_sha256']
                or receipt['target_sha256'] != initial['model_sha256']
                or arm == 'control' and receipt['source_ema_sha256'] != receipt['target_sha256']):
            raise ValueError('actual starting weights differ from the declared EMA migration')
        audits.append(complete)
    if len({row['input_stream_sha256'] for row in audits}) != 1:
        raise ValueError("architecture arms consumed different delivered data")
    comparisons = {}
    for step in (2000, 5000, 10000):
        for arm in ARMS:
            comparisons[f'{arm}/progress/{step}'] = paired_summary(
                initial_scores[f'{arm}/K2'], scores[f'{arm}-{step}/K2'])
        for before, after in (("control", "shared"), ("shared", "hierarchy")):
            comparisons[f"{after}-{before}/{step}"] = paired_summary(
                scores[f"{before}-{step}/K2"], scores[f"{after}-{step}/K2"])
    for arm in ARMS:
        comparisons[f"{arm}/K2-K1"] = paired_summary(
            scores[f"{arm}-10000/K1"], scores[f"{arm}-10000/K2"])
    result = dict(rows=rows, comparisons=comparisons, training=dict(zip(ARMS, audits, strict=True)),
                  limits="10k-step single-seed architecture screen;12 development chains/8 entries "
                  "with training homologs. Not a generalization or final convergence result. "
                  "Single-GPU matched batch differs from production's8-GPU batch.")
    if (args.run/'initial_u0_comparison.json').exists():
        result['initial_u0'] = json.loads((args.run/'initial_u0_comparison.json').read_text())
    write_json(args.run / "comparison.json", result)
    lines = ['# Bounded10k architecture screen', '',
             '| Model/added steps | K | lDDT | seconds | allocated GiB |',
             '| --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f"|{row['label']}|{row['K']}|{row['lddt']:.5f}|"
                     f"{row['mean_seconds']:.2f}|{row['peak_allocated_gib']:.3f}|")
    lines.extend(['', result['limits'], '',
                  'Paired effects and input/update audits: comparison.json'])
    (args.run / 'comparison.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    training = sub.add_parser("train")
    for name in ("config", "out", "source"):
        training.add_argument(f"--{name}", type=Path, required=True)
    training.add_argument('--source-step', type=int, default=10000)
    training.add_argument('--same-layout', action='store_true')
    inference = sub.add_parser("infer")
    for name in ("checkpoint", "out", "panel"):
        inference.add_argument(f"--{name}", type=Path, required=True)
    inference.add_argument("--depths", type=int, nargs="+", choices=(0, 1, 2), required=True)
    probing = sub.add_parser('probe')
    for name in ('checkpoint', 'out', 'audit'):
        probing.add_argument(f'--{name}', type=Path, required=True)
    probing.add_argument('--expected-step', type=int, default=0)
    probing.add_argument('--batches', type=int, default=4)
    for name in ("gate", "report"):
        stage = sub.add_parser(name)
        stage.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.stage in ("train", "infer", "probe") and not torch.cuda.is_available():
        parser.error("training/inference requires a Slurm GPU allocation")
    {"train": train, "infer": infer, "probe": probe, "gate": gate, "report": report}[
        args.stage](args)


if __name__ == "__main__":
    main()
