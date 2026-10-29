from __future__ import annotations

import math
import os
import re
import subprocess
import sys

import numpy as np
import pytest

from benchmarks import train_batch_confound as probe


def _template():
    return dict(batch_size=1, copies_per_protein=4, grad_accum_steps=4, seed=probe.SEED, num_workers=3,
                wandb_tags=["scratch", "max10k", "single-GPU-comparison"], wandb_name="old", config="old.yaml",
                out_dir="old")


def test_config_changes_only_the_declared_keys_and_keeps_sixteen_samples(tmp_path):
    template = _template()
    config = probe.derive_config(template, tmp_path)
    changed = {key for key in config if config[key] != template.get(key)}
    assert changed == {"config", "out_dir", "wandb_name", "wandb_tags", "grad_accum_steps"}
    assert config["wandb_tags"] == ["scratch", "max10k", "ddp4-c4a1-batch-confound"]
    per_update = config["batch_size"] * config["copies_per_protein"] * config["grad_accum_steps"]
    assert per_update * probe.WORLD == 16 == template["batch_size"] * template["copies_per_protein"] * 4
    with pytest.raises(ValueError):
        probe.derive_config(dict(template, wandb_tags=["scratch"]), tmp_path)


def _metrics(interval=100):
    rows, previous = [], 0
    for step in range(interval, probe.STEPS + 1, interval):
        factor = np.mean([probe.H.multiplier(s) for s in range(previous + 1, step + 1)])
        # The DDP trainer packs interval means into float32 before averaging ranks.
        rows.append(dict(step=step, update_applied=1.0, perf_train_step_ms_max=900.0 if step > interval else 5000.0,
                         perf_data_wait_ms_max=2.0, **{f"lr_{kind}": float(np.float32(peak * factor))
                                                     for kind, peak in probe.PEAKS.items()}))
        previous = step
    return rows


def test_metric_admission_requires_every_update_and_the_wsd_rates():
    probe.bind()
    rows = _metrics()
    probe.verify_metrics(rows, 100)
    for broken in (rows[:-1], [dict(rows[0], update_applied=0.99), *rows[1:]],
                   [*rows[:50], dict(rows[50], lr_muon=rows[50]["lr_muon"] * 1.001), *rows[51:]]):
        with pytest.raises(ValueError):
            probe.verify_metrics(broken, 100)


def test_metric_admission_accepts_the_real_single_gpu_reference_trainer_output():
    """Dry run of the by-value admission on the trainer-written metrics of the three reference seeds."""
    probe.bind()
    training = probe.long.reference_binding(probe.DEFAULT_EVAL)["training_run"]
    for arm in probe.FLAT_ARMS:
        rows = probe.proof.lines(f"{training}/training/{arm}/metrics.jsonl")
        probe.verify_metrics(rows, 100)
        assert 1000 < probe.step_costs(rows)["all_update_ms"] < 2000


def test_step_costs_weight_by_updates_and_drop_the_first_interval_for_steady():
    costs = probe.step_costs(_metrics())
    assert math.isclose(costs["all_update_ms"], (5000.0 + 99 * 900.0) / 100)
    assert costs["steady_ms"] == 900.0 and costs["data_wait_ms"] == 2.0


def test_gpu_chain_requests_no_cpu_or_memory_and_fits_its_qos(tmp_path):
    chain = probe.submit_chain(tmp_path / "run", tmp_path, cpu_job="123")
    lines = [line for line in chain.splitlines() if "sbatch" in line]
    gpu = [line for line in lines if "-p 6000ada" in line]
    assert len(gpu) == 2 and all(not re.search(r"\s(-c|--cpus-per-task|--mem)[\s=]", line) for line in gpu)
    assert all("--kill-on-invalid-dep=yes" in line for line in lines)
    train = next(line for line in gpu if "train.sh" in line)
    assert "--gres=gpu:4" in train and "--nodes=1" in train and "-q short" in train and "--time=10:00:00" in train
    assert "afterok:123" in train
    cpu = [line for line in lines if "-p cpu_only" in line]
    assert len(cpu) == 1 and "--gres" not in cpu[0]


def test_prepared_run_freezes_reference_model_source_and_validates(tmp_path):
    if not probe.DEFAULT_EVAL.is_dir():
        pytest.fail("reference seed-replicate evaluation is required on this cluster")
    prepared = probe.prepare(tmp_path, stamp="test")
    run = tmp_path / "batch-confound-ddp4-test"
    assert prepared["run"] == str(run.resolve())
    subprocess.run([sys.executable, str(run / "verify_frozen.py")], check=True)
    train = (run / "train.sh").read_text()
    assert "NCCL_P2P_DISABLE=1" in train and "gloo" not in train
    # Validate with the run's own frozen modules, as the Slurm stages do.
    code = run / "code"
    check = ("import sys; from pathlib import Path; from benchmarks import train_batch_confound as p; "
             "c = p.validate_run(Path(sys.argv[1]), admitted=False); "
             "assert Path(p.__file__).resolve() == Path(sys.argv[1]).resolve() / 'code/benchmarks/train_batch_confound.py'; "
             "assert c['config']['grad_accum_steps'] == 1 and len(c['targets']) == 15")
    env = dict(os.environ, PYTHONPATH=f"{code}:{code / 'src'}")
    subprocess.run([sys.executable, "-c", check, str(run)], check=True, cwd=code, env=env)
    # A changed recipe must not validate.
    config = run / f"configs/{probe.ARM}.yaml"
    config.write_text(config.read_text().replace("grad_accum_steps: 1", "grad_accum_steps: 2"))
    assert subprocess.run([sys.executable, "-c", check, str(run)], cwd=code, env=env).returncode != 0


def test_report_math_on_the_admitted_reference_reproduces_its_seed_statistics():
    """Dry run of summarize/reference_costs on the reference admission another run already wrote."""
    import json
    from pathlib import Path

    probe.bind()
    reference = json.loads((probe.repo_root() / "out/flat-fine6-long-30k-20261028T005516Z/reference_admitted.json")
                           .read_text())
    panel = json.loads((Path(json.loads((probe.DEFAULT_EVAL / "protocol.json").read_text())["panel"])
                        / "panel.json").read_text())["targets"]
    targets = {row["target"]: row for row in panel}
    assert sorted(targets) == reference["targets"]
    values = reference["arms"][probe.REF_ARM]["per_target"]
    stats = probe.summarize(targets, values, reference)
    aa = stats["versus_three_seed_10k"]["lddt"]["chain_mean"]
    assert math.isclose(aa["value"], 0.16606666666666667) and math.isclose(aa["reference_mean"], 0.15302222222222223)
    assert math.isclose(aa["delta_over_sd"], (aa["value"] - aa["reference_mean"]) / aa["reference_sd"])
    assert stats["paired_minus_reference_seed"][probe.REF_ARM]["lddt"]["chain_mean_delta"] == 0
    assert set(stats["per_target"]) == set(targets)
    costs = probe.reference_costs(reference)
    assert set(costs) == set(probe.FLAT_ARMS)
    assert all(1000 < row["all_update_ms"] < 2000 and 2 < row["gpu_hours"] < 6 for row in costs.values())
