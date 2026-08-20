"""Tests for rank-aligned length-bucketed DDP sampling."""

from itertools import islice

import pytest
from torch.utils.data import DataLoader, Dataset

from mambafold.data.length_sampler import LengthBucketedDistributedBatchSampler
from mambafold.data.loader import inf_loader


def _sampler(rank: int, world_size: int = 4):
    # Many examples per length make replacement collisions across a global
    # batch unlikely while retaining all eight production padding bins.
    index_lengths = {i: i % 1024 + 1 for i in range(8192)}
    return LengthBucketedDistributedBatchSampler(
        index_lengths=index_lengths,
        batch_size=4,
        rank=rank,
        world_size=world_size,
        num_samples_per_rank=512,
        seed=17,
        megabatch_mult=16,
    )


def test_length_bucketed_sampler_aligns_rank_steps():
    samplers = [_sampler(rank) for rank in range(4)]
    rank_batches = [list(sampler) for sampler in samplers]

    assert {len(batches) for batches in rank_batches} == {128}
    for step_batches in zip(*rank_batches):
        assert all(len(batch) == 4 for batch in step_batches)
        lengths = [index % 1024 + 1 for batch in step_batches for index in batch]
        # Each DDP step comes from one length-sorted global batch, not four
        # independent locations in the epoch's shuffled batch list.
        assert max(lengths) - min(lengths) <= 192


def test_length_bucketed_sampler_is_epoch_deterministic():
    first = _sampler(rank=2)
    second = _sampler(rank=2)
    assert list(first) == list(second)

    second.set_epoch(1)
    assert list(first) != list(second)


def test_length_bucketed_sampler_fast_forward_preserves_epoch_suffix():
    baseline = _sampler(rank=1)
    expected = list(baseline)

    resumed = _sampler(rank=1)
    resumed.set_start_batch(37)

    assert len(resumed) == len(expected)
    assert list(resumed) == expected[37:]

    resumed.set_epoch(1)
    resumed.set_start_batch(0)
    epoch_one = _sampler(rank=1)
    epoch_one.set_epoch(1)
    assert list(resumed) == list(epoch_one)


@pytest.mark.parametrize("start_batch", [0, 37, 128])
def test_length_bucketed_sampler_fast_forward_boundaries(start_batch):
    expected = list(_sampler(rank=0))
    resumed = _sampler(rank=0)
    resumed.set_start_batch(start_batch)
    assert list(resumed) == expected[start_batch:]


@pytest.mark.parametrize("start_batch", [-1, 129])
def test_length_bucketed_sampler_rejects_invalid_fast_forward(start_batch):
    sampler = _sampler(rank=0)
    with pytest.raises(ValueError, match="start_batch"):
        sampler.set_start_batch(start_batch)


def test_fast_forward_does_not_fetch_skipped_dataset_items():
    class CountingDataset(Dataset):
        def __init__(self):
            self.calls = []

        def __len__(self):
            return 8192

        def __getitem__(self, index):
            self.calls.append(index)
            return index

    expected_sampler = _sampler(rank=0)
    expected_batch = list(expected_sampler)[37]
    sampler = _sampler(rank=0)
    dataset = CountingDataset()
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)

    actual_batch = next(
        islice(
            inf_loader(loader, sampler, start_epoch=0, start_batch=37),
            1,
        )
    )

    assert actual_batch.tolist() == expected_batch
    assert dataset.calls == expected_batch
