#!/usr/bin/env python3
"""Bounded frozen-teacher adaptation, followed by per-model recovery admission."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from benchmarks import compare_shared_trunk as screen  # noqa: E402
from benchmarks.compare_full_refinement import tensor_digest, write_json  # noqa: E402
from mambafold.train.engine import _masked_mse_per_example, _metric_floats  # noqa: E402

ARMS = ('shared', 'hierarchy')


@torch.no_grad()
def teacher_readouts(teacher, batch):
    """Read pre-core/final velocities in one eval trunk pass, without RNG draws.

    The decoder input hook supplies the encoder atom skip and time embedding;
    the branch input is precisely the original LiFT u0 residue boundary. Only
    the cheap pre-core decoder is repeated. No teacher activation graph survives.
    """
    if (teacher.training or any(p.requires_grad for p in teacher.parameters())
            or teacher.latent_position != 'post' or teacher.atom_recurrence
            or teacher.latent_residual is None):
        raise ValueError('teacher requires frozen eval, post-core and no atom recurrence')
    captured = {}

    def pre_core(module, args):
        captured['residue'] = args[0]

    def decoder_inputs(module, args):
        captured['atoms'], captured['temb'] = args[1], args[3]

    handles = [teacher.latent_residual.register_forward_pre_hook(pre_core),
               teacher.atom_decoder.register_forward_pre_hook(decoder_inputs)]
    try:
        output = teacher(batch)
    finally:
        for handle in handles:
            handle.remove()
    anchor = teacher._decode(batch, captured['residue'], captured['atoms'],
                             captured['temb'])['v_atom']
    return anchor.detach(), output['v_atom'].detach()


def distillation_loss(output, anchor, endpoint, mask):
    """Equal weight to u0 recovery and the mean teacher-anchored trajectory.

    All targets share the actual noisy input coordinate frame. No rigid
    alignment or student self-anchor is used during adaptation. Each protein
    has equal weight regardless of observed atom count.
    """
    steps, depths = output['v_atom_steps'], output['latent_depths']
    if (len(steps) < 2 or depths.shape != (anchor.shape[0], len(steps) - 1)
            or any(value.shape != anchor.shape for value in (*steps, endpoint))):
        raise ValueError('teacher/student readout layouts differ')
    valid = mask.unsqueeze(-1)
    anchor = anchor.detach().float().masked_fill(~valid, 0)
    endpoint = endpoint.detach().float().masked_fill(~valid, 0)

    def mse(value, target):
        return _masked_mse_per_example(value.float().masked_fill(~valid, 0),
                                       target, mask).mean()

    u0_loss = mse(steps[0], anchor)
    losses = []
    for i, value in enumerate(steps[1:]):
        depth = depths[:, i].detach().float().reshape(-1, 1, 1, 1)
        losses.append(mse(value, (1 - depth) * anchor + depth * endpoint))
    trajectory = torch.stack(losses).mean()
    total = .5 * (u0_loss + trajectory)
    return total, dict(kd_u0=u0_loss.detach(), kd_trajectory=trajectory.detach(),
                       kd_endpoint=mse(steps[-1].detach(), endpoint).detach())


def adapt(args):
    from mambafold.model.bimamba3 import mamba3_runtime_info
    from mambafold.train import engine, trainer

    cfg = yaml.safe_load(args.config.read_text())
    if (cfg['latent_supervision'] != 'lift' or cfg['latent_loops'] != 2
            or cfg.get('self_conditioning') or cfg.get('self_condition_prob')
            or cfg.get('atom_recurrence') or cfg.get('latent_loop_sampling')
            or cfg['shared_trunk_layers'] != 6):
        raise ValueError('adaptation is limited to the declared fixed-K2 3+6 models')
    source = torch.load(args.source, map_location='cpu', mmap=True, weights_only=False)
    runtime = mamba3_runtime_info()
    if source['step'] != 10000 or any(
            source.get('mamba3_runtime', {}).get(k) != runtime[k]
            for k in ('revision', 'module_sha256', 'decay')):
        raise ValueError('teacher step/runtime differs from the pinned production EMA')
    teacher = trainer.build_model(source['args'], 'cuda').eval().requires_grad_(False)
    teacher.load_state_dict(source['ema'], strict=True)
    teacher_identity = tensor_digest(teacher.state_dict().items())
    del source
    original_forward = engine.allatom_forward_and_loss

    def forward(model, batch, **kwargs):
        model.train()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=kwargs.get('use_amp', True)):
            anchor, endpoint = teacher_readouts(teacher, batch)
            output = model(batch)
        loss, metrics = distillation_loss(output, anchor, endpoint, batch.valid_mask)
        # Existing trainer console fields remain explicit: FM/soft-lDDT are
        # disabled in this phase; kd_* metrics carry the actual objective.
        zero = loss.detach().new_zeros(())
        metrics.update(loss=loss.detach(), fm_atom=zero, lddt_atom=zero,
                       alpha=zero + 1, t_mean=batch.t.mean())
        return loss, dict(_metric_floats(metrics), self_cond=0.)

    engine.allatom_forward_and_loss = forward
    try:
        screen.train(args)
    finally:
        engine.allatom_forward_and_loss = original_forward
    unchanged = teacher_identity == tensor_digest(teacher.state_dict().items())
    if not unchanged or any(p.grad is not None for p in teacher.parameters()):
        raise ValueError('frozen teacher changed or accumulated a gradient')
    write_json(args.out / 'teacher_verified.json', dict(
        teacher_sha256=screen.sha256(args.source), teacher_model_sha256=teacher_identity,
        teacher_unchanged=True, teacher_gradients=0, teacher_mode='eval/no_grad',
        teacher_trunk_passes_per_microbatch=1, teacher_decoder_calls_per_microbatch=2,
        objective='.5*MSE(student_u0,teacher_u0)+.5*mean_s MSE(student_us, '
                  '(1-s)*teacher_u0+s*teacher_final)',
        alignment='same input frame; teacher targets detached; no student self-anchor',
        ground_truth_fm_weight=0, soft_lddt_weight=0))


def recovery_decision(baseline_scores, scores, baseline_probe, probe, arm, protocol):
    """Predeclared damage guards, not statistical noninferiority margins."""
    checks, effects = {}, {}
    for depth in (0, 2):
        effect = screen.paired_summary(baseline_scores[f'control/K{depth}'],
                                       scores[f'{arm}/K{depth}'])
        effects[f'K{depth}'] = effect
        checks[f'K{depth}_lddt'] = effect['mean_chain_delta'] >= -protocol['lddt_drop_guard']
    reference = baseline_probe['mean']['fm_prelude']
    measured = probe['mean']['fm_prelude']
    if not (math.isfinite(reference) and reference > 0 and math.isfinite(measured)
            and measured >= 0):
        raise ValueError('invalid u0 FM recovery metric')
    ratio = measured / reference
    checks['u0_fm'] = ratio <= protocol['u0_fm_ratio_guard']
    return dict(passed=all(checks.values()), checks=checks, effects=effects,
                u0_fm_ratio=ratio,
                note='FM uses each model final-based alignment; actual K0/K2 lDDT '
                     'are independent structural guards on the development panel.')


def verify_phase(run, arm, phase, steps):
    directory = run / phase / arm
    initial = json.loads((directory / 'audit/initial.json').read_text())
    complete = json.loads((directory / 'audit/complete.json').read_text())
    receipt = json.loads((directory / 'initialization.json').read_text())
    screen.verify_budget(initial, complete, steps, 4)
    if initial['model_sha256'] != receipt['target_sha256']:
        raise ValueError('initialization differs from the actual optimizer starting weights')
    if (directory / 'preempted.json').exists():
        raise ValueError('interrupted phase cannot certify an uninterrupted screen')
    return initial, complete, receipt


def gate(args):
    run, arm = args.run, args.arm
    protocol = json.loads((run / 'protocol.json').read_text())
    baseline = json.loads((run / 'baseline/initial_scores.json').read_text())
    baseline_probe = json.loads((run / 'baseline/u0.json').read_text())
    scores, rows = screen.score_all(run, f'recovery-{arm}')
    directory = run / 'adaptation' / arm
    checkpoint = directory / f"ckpt_{protocol['adaptation_steps']:07d}.pt"
    probe = json.loads((run / f'u0/{arm}/u0.json').read_text())
    initial, complete, receipt = verify_phase(run, arm, 'adaptation', protocol['adaptation_steps'])
    verified = json.loads((directory / 'teacher_verified.json').read_text())
    smoke_initial, smoke_complete, _ = verify_phase(run, arm, 'preflight', 8)
    expected_inputs = [row['input_sha256'] for row in baseline_probe['rows']]
    if (receipt['source_sha256'] != protocol['source_sha256']
            or receipt['source_ema_sha256'] != verified['teacher_model_sha256']
            or verified['teacher_sha256'] != protocol['source_sha256']
            or not verified['teacher_unchanged'] or verified['teacher_gradients'] != 0
            or initial['model_sha256'] != smoke_initial['model_sha256']
            or probe['checkpoint_sha256'] != screen.sha256(checkpoint)
            or probe['checkpoint_step'] != protocol['adaptation_steps']
            or probe['updates'] != 0 or len(probe['rows']) != 4
            or [r['input_sha256'] for r in probe['rows']] != expected_inputs):
        raise ValueError('adaptation/teacher/probe identity mismatch')
    state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    model_hash = tensor_digest(state['ema'].items())
    if (state['step'] != protocol['adaptation_steps']
            or probe['initial_model_sha256'] != model_hash):
        raise ValueError('probe differs from the actual adapted EMA')
    del state
    samples = json.loads((run / f'evaluation/recovery-{arm}/{arm}/sampling.json').read_text())
    for depth in (0, 2):
        pairs = run / f'evaluation/recovery-{arm}/{arm}/K{depth}/pairs'
        manifest = json.loads((pairs / 'manifest.json').read_text())
        if (manifest['checkpoint_sha256'] != probe['checkpoint_sha256']
                or any(r['checkpoint_step'] != protocol['adaptation_steps'] for r in samples)):
            raise ValueError('actual inference does not use the admitted adapted checkpoint')
    result = recovery_decision(baseline['scores'], scores, baseline_probe, probe, arm, protocol)
    result.update(arm=arm, rows=rows, probe=probe, adaptation=complete,
                  preflight=smoke_complete, checkpoint_sha256=probe['checkpoint_sha256'],
                  model_sha256=model_hash, teacher=verified)
    write_json(run / f'admission/{arm}.json', result)
    if not result['passed']:
        raise ValueError(f'{arm} did not recover; cancel its dependent10k trial')


def continue_training(args):
    protocol = json.loads((args.run / 'protocol.json').read_text())
    admission = json.loads((args.run / f'admission/{args.arm}.json').read_text())
    if not admission['passed']:
        raise ValueError('continuation requires independent recovery admission')
    source = args.run / f"adaptation/{args.arm}/ckpt_{protocol['adaptation_steps']:07d}.pt"
    if screen.sha256(source) != admission['checkpoint_sha256']:
        raise ValueError('admitted adapted EMA changed')
    screen.train(SimpleNamespace(
        config=args.run / f'configs/{args.arm}-train.yaml',
        out=args.run / f'training/{args.arm}', source=source,
        same_layout=True, source_step=protocol['adaptation_steps']))


def report(args):
    run = args.run
    protocol = json.loads((run / 'protocol.json').read_text())
    baseline = json.loads((run / 'baseline/initial_scores.json').read_text())['scores']
    results, trained_scores, all_rows = {}, {}, []
    for arm in ARMS:
        path = run / f'admission/{arm}.json'
        if not path.exists():
            results[arm] = dict(status='no completed recovery admission')
            continue
        admission = json.loads(path.read_text())
        results[arm] = dict(admission=admission, status='recovery rejected')
        if not admission['passed']:
            continue
        budget = run / f'training/{arm}/budget_verified.json'
        sampling = run / f'evaluation/trained-{arm}/{arm}-10000/sampling.json'
        if not budget.exists() or not sampling.exists():
            results[arm]['status'] = 'admitted;10k/evaluation not certified complete'
            continue
        initial, complete, receipt = verify_phase(run, arm, 'training', 10000)
        if (receipt['source_sha256'] != admission['checkpoint_sha256']
                or receipt['source_ema_sha256'] != admission['model_sha256']
                or initial['model_sha256'] != admission['model_sha256']):
            raise ValueError('10k trial did not continue from the admitted adapted EMA')
        scores, rows = screen.score_all(run, f'trained-{arm}')
        for step in (2000, 5000, 10000):
            checkpoint = run / f'training/{arm}/ckpt_{step:07d}.pt'
            for depth in ((0, 1, 2) if step == 10000 else (2,)):
                manifest = json.loads((run / f'evaluation/trained-{arm}/{arm}-{step}'
                                       / f'K{depth}/pairs/manifest.json').read_text())
                if manifest['checkpoint_sha256'] != screen.sha256(checkpoint):
                    raise ValueError('trained inference checkpoint identity differs')
        trained_scores.update(scores)
        all_rows.extend(rows)
        results[arm].update(status='complete', training=complete,
                           K2_minus_K1=screen.paired_summary(
                               scores[f'{arm}-10000/K1'], scores[f'{arm}-10000/K2']),
                           versus_frozen_teacher=screen.paired_summary(
                               baseline['control/K2'], scores[f'{arm}-10000/K2']))
    paired = None
    if all(results[arm]['status'] == 'complete' for arm in ARMS):
        if len({results[arm]['training']['input_stream_sha256'] for arm in ARMS}) != 1:
            raise ValueError('10k arms consumed different input streams')
        if len({results[arm]['admission']['adaptation']['input_stream_sha256']
                for arm in ARMS}) != 1:
            raise ValueError('adaptation arms consumed different input streams')
        paired = screen.paired_summary(trained_scores['shared-10000/K2'],
                                        trained_scores['hierarchy-10000/K2'])
    result = dict(arms=results, rows=all_rows, hierarchy_minus_shared=paired,
                  adaptation_steps=protocol['adaptation_steps'], added_LiFT_steps=10000,
                  limits='One seed,12 development chains/eight entries with training homologs. '
                         'Each candidate receives5k extra teacher updates before10k LiFT. '
                         'Frozen teacher is not a matched continued-training control. '
                         'Single-GPU batch16 differs from production8-GPU batch128.')
    write_json(run / 'comparison.json', result)
    lines = ['# Teacher-adapted architecture screen', '',
             '| Model/added LiFT steps | K | lDDT | seconds | allocated GiB |',
             '| --- | ---: | ---: | ---: | ---: |']
    for row in all_rows:
        lines.append(f"|{row['label']}|{row['K']}|{row['lddt']:.5f}|"
                     f"{row['mean_seconds']:.2f}|{row['peak_allocated_gib']:.3f}|")
    lines.extend(['', *(f"{arm}: {value['status']}" for arm, value in results.items()),
                  '', result['limits']])
    (run / 'comparison.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    training = sub.add_parser('adapt')
    for name in ('config', 'source', 'out'):
        training.add_argument(f'--{name}', type=Path, required=True)
    for name in ('gate', 'continue'):
        command = sub.add_parser(name)
        command.add_argument('--run', type=Path, required=True)
        command.add_argument('--arm', choices=ARMS, required=True)
    reporting = sub.add_parser('report')
    reporting.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.stage in ('adapt', 'continue') and not torch.cuda.is_available():
        parser.error('training requires a Slurm GPU allocation')
    {'adapt': adapt, 'gate': gate, 'continue': continue_training, 'report': report}[
        args.stage](args)


if __name__ == '__main__':
    main()
