import os

import pytest

from mambafold.train.distributed import resolve_dataloader_workers


def test_workers_are_capped_per_rank_from_slurm(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "32")
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)

    workers, cpus, source = resolve_dataloader_workers(16, world_size=8)

    assert workers == 3
    assert cpus == 32
    assert source == "SLURM_CPUS_ON_NODE"


def test_slurm_repetition_suffix_is_accepted(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "64(x2)")
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)

    workers, cpus, _ = resolve_dataloader_workers(16, world_size=8)

    assert workers == 7
    assert cpus == 64


def test_affinity_is_fallback(monkeypatch):
    monkeypatch.delenv("SLURM_CPUS_ON_NODE", raising=False)
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: set(range(12)))

    workers, cpus, source = resolve_dataloader_workers(8, world_size=4)

    assert workers == 2
    assert cpus == 12
    assert source == "sched_getaffinity"


def test_multinode_workers_use_local_world_size(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "16")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")

    workers, cpus, source = resolve_dataloader_workers(16, world_size=8)

    assert workers == 3
    assert cpus == 16
    assert source == "SLURM_CPUS_ON_NODE"


@pytest.mark.parametrize("requested,world_size", [(-1, 1), (1, 0)])
def test_worker_inputs_are_validated(requested, world_size):
    with pytest.raises(ValueError):
        resolve_dataloader_workers(requested, world_size)
