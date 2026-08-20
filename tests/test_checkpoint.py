import random
import sys
from itertools import islice
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mambafold.data.loader import inf_loader
from mambafold.train.ema import EMA
from mambafold.train.trainer import (
    capture_rng_state,
    restore_rng_state,
    save_checkpoint,
    seed_all,
    validate_data_resume_state,
)


def test_rng_state_round_trip():
    seed_all(17)
    state = capture_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(3))

    random.random()
    np.random.rand()
    torch.rand(3)
    restore_rng_state(state)

    actual = (random.random(), np.random.rand(), torch.rand(3))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2])


def test_checkpoint_contains_resume_state(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    model = torch.nn.Linear(4, 2)
    ema = EMA(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    loss = model(torch.ones(2, 4)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()

    rng_states = [capture_rng_state()]
    data_state = {
        "micro_batches_consumed": 7,
        "world_size": 1,
        "grad_accum_steps": 7,
        "batches_per_epoch": 11,
    }
    save_checkpoint(
        tmp_path,
        3,
        model,
        ema,
        optimizer,
        scheduler,
        SimpleNamespace(
            config="test.yaml",
            keep_last_checkpoints=2,
            keep_checkpoint_steps=[],
        ),
        rng_states=rng_states,
        data_state=data_state,
    )

    latest = tmp_path / "ckpt_latest.pt"
    checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
    assert latest.resolve() == tmp_path / "ckpt_0000003.pt"
    assert checkpoint["step"] == 3
    assert len(checkpoint["rng_states"]) == 1
    assert checkpoint["data_state"] == data_state
    assert checkpoint["wandb_run_id"] is None
    assert not any(key.startswith("module.") for key in checkpoint["model"])

    for step in (4, 5):
        save_checkpoint(
            tmp_path,
            step,
            model,
            ema,
            optimizer,
            scheduler,
            SimpleNamespace(
                config="test.yaml",
                keep_last_checkpoints=2,
                keep_checkpoint_steps=[3],
            ),
        )
    assert sorted(path.name for path in tmp_path.glob("ckpt_*.pt")) == [
        "ckpt_0000003.pt",
        "ckpt_0000004.pt",
        "ckpt_0000005.pt",
        "ckpt_latest.pt",
    ]


def test_inf_loader_resumes_epoch_and_batch():
    class Sampler:
        def __init__(self):
            self.epochs = []

        def set_epoch(self, epoch):
            self.epochs.append(epoch)

    sampler = Sampler()
    values = list(
        islice(
            inf_loader(
                ["batch-0", "batch-1", "batch-2"],
                sampler,
                start_epoch=2,
                start_batch=1,
            ),
            4,
        )
    )

    assert values == ["batch-1", "batch-2", "batch-0", "batch-1"]
    assert sampler.epochs == [2, 3]


def test_inf_loader_fast_forwards_before_loading():
    class Sampler:
        def __init__(self):
            self.epoch = 0
            self.start_batch = 0

        def set_epoch(self, epoch):
            self.epoch = epoch

        def set_start_batch(self, start_batch):
            self.start_batch = start_batch

    class Loader:
        def __init__(self, sampler):
            self.sampler = sampler
            self.loaded = []

        def __iter__(self):
            for batch_idx in range(self.sampler.start_batch, 3):
                self.loaded.append((self.sampler.epoch, batch_idx))
                yield f"epoch-{self.sampler.epoch}-batch-{batch_idx}"

    sampler = Sampler()
    loader = Loader(sampler)
    values = list(
        islice(
            inf_loader(loader, sampler, start_epoch=2, start_batch=2),
            3,
        )
    )

    assert values == [
        "epoch-2-batch-2",
        "epoch-3-batch-0",
        "epoch-3-batch-1",
    ]
    assert loader.loaded == [(2, 2), (3, 0), (3, 1)]


def test_validate_data_resume_state_accepts_matching_contract():
    validate_data_resume_state(
        {
            "world_size": 8,
            "batch_size": 9,
            "grad_accum_steps": 7,
            "batches_per_epoch": 15338,
            "dataset_size": 1104386,
            "sampler_type": "LengthBucketedDistributedBatchSampler",
            "seed": 0,
        },
        {},
        world_size=8,
        batch_size=9,
        grad_accum_steps=7,
        batches_per_epoch=15338,
        dataset_size=1104386,
        sampler_type="LengthBucketedDistributedBatchSampler",
        seed=0,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("world_size", 4),
        ("batch_size", 8),
        ("grad_accum_steps", 8),
        ("batches_per_epoch", 15337),
        ("dataset_size", 1104385),
        ("sampler_type", "DistributedSampler"),
        ("seed", 1),
    ],
)
def test_validate_data_resume_state_rejects_mismatch(field, value):
    saved = {
        "world_size": 8,
        "batch_size": 9,
        "grad_accum_steps": 7,
        "batches_per_epoch": 15338,
        "dataset_size": 1104386,
        "sampler_type": "LengthBucketedDistributedBatchSampler",
        "seed": 0,
    }
    saved[field] = value

    with pytest.raises(RuntimeError, match=field):
        validate_data_resume_state(
            saved,
            {},
            world_size=8,
            batch_size=9,
            grad_accum_steps=7,
            batches_per_epoch=15338,
            dataset_size=1104386,
            sampler_type="LengthBucketedDistributedBatchSampler",
            seed=0,
        )
