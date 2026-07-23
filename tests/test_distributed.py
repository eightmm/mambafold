import os

import pytest

from mambafold.train.distributed import resolve_dataloader_workers


def test_workers_are_capped_per_rank_from_slurm(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "32")

    workers, cpus, source = resolve_dataloader_workers(16, world_size=8)

    assert workers == 3
    assert cpus == 32
    assert source == "SLURM_CPUS_ON_NODE"


def test_slurm_repetition_suffix_is_accepted(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "64(x2)")

    workers, cpus, _ = resolve_dataloader_workers(16, world_size=8)

    assert workers == 7
    assert cpus == 64


def test_affinity_is_fallback(monkeypatch):
    monkeypatch.delenv("SLURM_CPUS_ON_NODE", raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: set(range(12)))

    workers, cpus, source = resolve_dataloader_workers(8, world_size=4)

    assert workers == 2
    assert cpus == 12
    assert source == "sched_getaffinity"


@pytest.mark.parametrize("requested,world_size", [(-1, 1), (1, 0)])
def test_worker_inputs_are_validated(requested, world_size):
    with pytest.raises(ValueError):
        resolve_dataloader_workers(requested, world_size)
