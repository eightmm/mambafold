"""Tests for rank-aligned length-bucketed DDP sampling."""

from mambafold.data.length_sampler import LengthBucketedDistributedBatchSampler


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
