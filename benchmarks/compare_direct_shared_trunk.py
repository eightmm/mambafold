#!/usr/bin/env python3
"""Teacher-free ground-truth LiFT screens with an audited student prelude anchor."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]

from benchmarks import compare_shared_trunk as screen  # noqa: E402
from benchmarks.compare_full_refinement import tensor_digest, write_json  # noqa: E402
from mambafold.train import engine  # noqa: E402

ARMS = ('shared', 'hierarchy')


def validate_config(cfg):
    required = dict(n_trunk=15, shared_trunk_layers=6, latent_loops=2,
                    latent_compression=4, latent_supervision='lift', residue_refiner=True,
                    latent_loop_sampling=False, atom_recurrence=False, w_fm=1., w_lddt_atom=1.,
                    grad_accum_steps=4, copies_per_protein=4)
    if (any(cfg.get(key) != value for key, value in required.items())
            or cfg.get('resume') or cfg.get('initialize_refiner_from')
            or any(cfg.get(key, 0) for key in ('w_bond', 'w_angle', 'w_clash'))):
        raise ValueError('direct comparison requires the declared teacher-free GT LiFT recipe')
    if engine.LIFT_PRELUDE_WEIGHT != .01:
        raise ValueError('LiFT prelude auxiliary must retain the declared0.01 weight')


def verify_gt_metrics(rows):
    """Confirm the executed objective, not merely the config's supervision label."""
    if not rows:
        raise ValueError('missing actual ground-truth training diagnostics')
    for row in rows:
        if any(key.startswith('kd_') for key in row):
            raise ValueError('teacher distillation leaked into the direct training phase')
        expected = row['fm_trajectory'] + .01 * row['fm_prelude']
        if (not math.isclose(row['fm_objective'], expected, rel_tol=1e-5, abs_tol=1e-6)
                or not math.isclose(row['loss'], expected + row['lddt_atom_weighted'],
                                    rel_tol=1e-5, abs_tol=1e-6)
                or not math.isclose(row['fm_atom'], row['fm_endpoint_k2'],
                                    rel_tol=1e-5, abs_tol=1e-6)
                or row['update_applied'] != 1.):
            raise ValueError('actual updates/objective differ from ground-truth LiFT')


def validate_student_anchor(model, batch, output, cfg):
    """Validate real K0 routing and target-only stop-gradient on actual inputs."""
    steps = output['v_atom_steps']
    if (model.residue_trunk.n_layers != 3 or model.shared_trunk_layers != 6
            or len(steps) != 3 or output['latent_depths'].shape != (batch.t.shape[0], 2)):
        raise ValueError('u0 must precede both complete shared-core passes')
    branch, training = model.latent_residual, model.training
    try:
        model.latent_residual = None
        model.eval()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16,
                                             enabled=batch.device.type == 'cuda'):
            bypass = model(batch)['v_atom']
        torch.testing.assert_close(steps[0].detach(), bypass, rtol=1e-4, atol=1e-5)
    finally:
        model.latent_residual = branch
        model.train(training)

    # Isolated readout leaves distinguish the direct anchor gradient from the
    # legitimate later-pass gradients through the prelude/shared hidden state.
    leaves = tuple(value.detach().float().requires_grad_() for value in steps)
    leaf_output = dict(v_atom=leaves[-1], v_atom_steps=leaves,
                       latent_depths=output['latent_depths'].detach())
    loss, _ = engine.allatom_loss_surface(
        leaf_output, batch, use_rigid_align=cfg['use_rigid_align'], w_lddt_atom=0)
    gradients = torch.autograd.grad(loss, leaves)
    with torch.no_grad():
        clean = batch.x_clean.float()
        if cfg['use_rigid_align']:
            clean = engine.weighted_rigid_align(
                clean, engine._recon_atom(batch.x_t.float(), batch.t, leaves[-1].detach()),
                batch.valid_mask)
    aux, _ = engine.allatom_loss_surface(
        dict(v_atom=leaves[0]), replace(batch, x_clean=clean),
        use_rigid_align=False, w_lddt_atom=0)
    expected, = torch.autograd.grad(.01 * aux, leaves[0])
    torch.testing.assert_close(gradients[0], expected, rtol=1e-4, atol=1e-7)
    if not all(torch.isfinite(g).all() and g[batch.valid_mask].abs().sum() > 0
               for g in gradients):
        raise ValueError('missing or nonfinite GT gradients for u0/recurrent readouts')
    return dict(prelude_layers=3, core_passes=2, u0_matches_actual_K0=True,
                detached_anchor_gradient_verified=True)


def train(args):
    cfg = yaml.safe_load(args.config.read_text())
    validate_config(cfg)
    # The normal trainer calls the original engine directly. No teacher model
    # is constructed and no loss function is overridden in this runner.
    screen.train(args)
    rows = [json.loads(line) for line in (args.out/'audit/metrics.jsonl').read_text().splitlines()]
    verify_gt_metrics(rows)
    write_json(args.out/'ground_truth_verified.json', dict(
        teacher_models=0, optimizer_updates=cfg['total_steps'],
        objective='GT LiFT + final soft-lDDT',
        anchor_source='this student after3 prelude blocks, before core pass1',
        prelude_weight=.01, core_passes=2, ground_truth_metrics_verified=True))


def probe(args):
    args.validate_output = validate_student_anchor
    screen.probe(args)


def report(args):
    run = args.run
    protocol = json.loads((run/'protocol.json').read_text())
    complete = {}
    for arm in ARMS:
        d = run/f'training/{arm}'
        initial = json.loads((d/'audit/initial.json').read_text())
        budget = json.loads((d/'audit/complete.json').read_text())
        screen.verify_budget(initial, budget, 10000, 4)
        receipt = json.loads((d/'initialization.json').read_text())
        verified = json.loads((d/'ground_truth_verified.json').read_text())
        if (receipt['source_sha256'] != protocol['source_sha256']
                or receipt['target_sha256'] != initial['model_sha256']
                or verified['teacher_models'] != 0):
            raise ValueError('direct comparison initialization/objective identity mismatch')
        smoke = json.loads((run/f'preflight/{arm}/audit/initial.json').read_text())
        if smoke['model_sha256'] != initial['model_sha256']:
            raise ValueError('training did not restart from its audited unadapted initialization')
        verify_gt_metrics([json.loads(line) for line in (d/'audit/metrics.jsonl')
                          .read_text().splitlines()])
        cfg = yaml.safe_load((run/f'configs/{arm}-train.yaml').read_text())
        validate_config(cfg)
        complete[arm] = budget
    configs = [yaml.safe_load((run/f'configs/{arm}-train.yaml').read_text()) for arm in ARMS]
    ignored = {'latent_hierarchy', 'wandb_name'}
    if ({k:v for k,v in configs[0].items() if k not in ignored}
            != {k:v for k,v in configs[1].items() if k not in ignored}
            or len({v['input_stream_sha256'] for v in complete.values()}) != 1):
        raise ValueError('architecture arms did not share the same recipe/delivered inputs')
    scores, rows = screen.score_all(run, 'trained')
    probes, predictions = {}, {}
    for arm in ARMS:
        probes[arm] = {}
        for step in (0, 5000, 10000):
            p = json.loads((run/f'u0/{arm}-{step}/u0.json').read_text())
            checkpoint = (run/f'preflight/{arm}/initial_ema.pt' if step == 0
                          else run/f'training/{arm}/ckpt_{step:07d}.pt')
            state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
            if (p['checkpoint_sha256'] != screen.sha256(checkpoint)
                    or p['initial_model_sha256'] != tensor_digest(state['ema'].items())
                    or p['checkpoint_step'] != step or p['updates'] != 0
                    or len(p['rows']) != 16
                    or not all(row['u0_matches_actual_K0']
                               and row['detached_anchor_gradient_verified'] for row in p['rows'])):
                raise ValueError('u0 probe weight/routing/gradient identity mismatch')
            del state
            probes[arm][str(step)] = p
            predictions[f'{arm}/{step}'] = [v['input_sha256'] for v in p['rows']]
        for step in (2000, 5000, 10000):
            identity = screen.sha256(run/f'training/{arm}/ckpt_{step:07d}.pt')
            for depth in ((0,1,2) if step == 10000 else (2,)):
                path = run/f'evaluation/trained/{arm}-{step}/K{depth}/pairs/manifest.json'
                if json.loads(path.read_text())['checkpoint_sha256'] != identity:
                    raise ValueError('structure evaluation uses a different checkpoint')
    if len({tuple(v) for v in predictions.values()}) != 1:
        raise ValueError('u0 probes were not evaluated on the same replayed inputs')
    comparisons = {f'hierarchy-shared/{step}': screen.paired_summary(
        scores[f'shared-{step}/K2'], scores[f'hierarchy-{step}/K2'])
        for step in (2000, 5000, 10000)}
    for arm in ARMS:
        for depth in (0,1):
            comparisons[f'{arm}/K2-K{depth}'] = screen.paired_summary(
                scores[f'{arm}-10000/K{depth}'], scores[f'{arm}-10000/K2'])
    baseline = json.loads((run/'baseline/initial_scores.json').read_text())['scores']
    for arm in ARMS:
        comparisons[f'{arm}/versus_original10k'] = screen.paired_summary(
            baseline['control/K2'], scores[f'{arm}-10000/K2'])
    result = dict(rows=rows, comparisons=comparisons, training=complete, probes=probes,
                  limits=protocol['limits'])
    write_json(run/'comparison.json', result)
    lines = ['# Direct ground-truth10k architecture comparison', '',
             '| Model/updates | K | all-atom lDDT | backbone lDDT | seconds | allocated GiB |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f"|{row['label']}|{row['K']}|{row['lddt']:.5f}|{row['bb_lddt']:.5f}|"
                     f"{row['mean_seconds']:.2f}|{row['peak_allocated_gib']:.3f}|")
    lines.extend(['', protocol['limits'], '', 'Full u0 and paired effects:comparison.json'])
    (run/'comparison.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    training = sub.add_parser('train')
    for name in ('config', 'out', 'source'):
        training.add_argument(f'--{name}', type=Path, required=True)
    probing = sub.add_parser('probe')
    for name in ('checkpoint', 'out', 'audit'):
        probing.add_argument(f'--{name}', type=Path, required=True)
    probing.add_argument('--expected-step', type=int, required=True)
    probing.add_argument('--batches', type=int, default=16)
    reporting = sub.add_parser('report')
    reporting.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.stage != 'report' and not torch.cuda.is_available():
        parser.error('training/probing requires a Slurm GPU allocation')
    {'train': train, 'probe': probe, 'report': report}[args.stage](args)
