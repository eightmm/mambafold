"""DDP length-bucketed batch sampler.

The training loop pads every rank up to one global sequence length so the
TileLang Mamba-3 kernels, which specialize on length, see a single shape per
step. With independent per-rank draws that global maximum is the max of
`world_size` random chain lengths: measured over this corpus it averages 624
residues at 8 ranks against a mean own length of 312, so every rank does 2x the
necessary work. Drawing all ranks from one length-sorted global batch brings
that back to 1.0x. That, not intra-batch padding, is why bucketing is on at
batch_size 1.

The draw itself is uniform over valid indices. A length-balance weighting
(`w = clip((L/200)**0.5, clip_min, clip_max)`) used to sit here and was removed:
it was never declared in any config, `length_balanced_sampling: false` read as
"no length weighting", and it was nonetheless applied by this sampler, giving
chains of 450+ residues 1.5x the draw rate of chains under 200. That is a
different training measure from the one the corpus documents describe, and from
SimpleFold's, which draws each entry once.
"""

from __future__ import annotations

import math
from typing import Iterator

import torch
from torch.utils.data import Sampler


class LengthBucketedDistributedBatchSampler(Sampler[list[int]]):
    """Per-rank *batch* sampler that groups near-equal-length proteins together.

    Yields whole batches whose members have similar length, so the collator pads
    each batch to ~its own longest sequence instead of the global max, cutting
    padding waste across the atom and residue sequence paths.

    Operates on a precomputed `index_lengths` map (valid dataset index → true
    example length, from `length_cache`). Only those indices are emitted, so
    `RCSBDataset.__getitem__` returns `files[idx]` directly (no skip-to-next),
    keeping each batch's real content aligned with the length it was bucketed by.

    Per epoch, every rank deterministically reconstructs the same global draw:
      1. draw `num_samples_per_rank * world_size` valid indices uniformly
         (with replacement).
      2. split into global megabatches, sort by true length, and chunk into
         `batch_size * world_size` global batches.
      3. shard each global batch across ranks and shuffle the shared batch order.

    Aligning ranks to one length-sorted global batch is important for JIT-backed
    sequence kernels: independent per-rank draws can make one rank compile a new
    long-sequence kernel while another rank reaches a DDP collective, eventually
    timing out. The training loop still synchronizes the final padded length to
    cover the rare global batch that straddles a padding-bin boundary.

    Megabatches (not a single global sort) keep epoch-to-epoch stochasticity
    while still grouping similar lengths. `drop_last` happens inside each
    megabatch (a partial trailing batch is dropped).
    """

    def __init__(
        self,
        index_lengths: dict[int, int],
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        num_samples_per_rank: int | None = None,
        seed: int = 0,
        megabatch_mult: int = 50,
    ):
        if world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {world_size}")
        if not (0 <= rank < world_size):
            raise ValueError(f"rank={rank} out of range for world_size={world_size}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if not index_lengths:
            raise ValueError("index_lengths is empty (no valid files for bucketing)")

        # `self.valid[pos]` is the dataset index; weights/lengths are aligned to pos.
        self.valid = torch.tensor(sorted(index_lengths.keys()), dtype=torch.long)
        lens = [index_lengths[int(i)] for i in self.valid]
        self.lengths = torch.tensor(lens, dtype=torch.long)
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.num_samples = (
            num_samples_per_rank
            if num_samples_per_rank is not None
            else math.ceil(len(self.valid) / world_size)
        )
        self.seed = seed
        self.epoch = 0
        self._start_batch = 0
        self.megabatch_mult = max(1, megabatch_mult)

        # Exact per-rank batch count after globally aligned drop_last.
        global_batch = self.batch_size * self.world_size
        mb = global_batch * self.megabatch_mult
        global_samples = self.num_samples * self.world_size
        full_mb, rem = divmod(global_samples, mb)
        self._n_batches = full_mb * self.megabatch_mult + rem // global_batch

        self._n_valid = len(self.valid)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_start_batch(self, start_batch: int) -> None:
        """Skip batch indices before dataset workers receive any samples.

        The offset changes only which suffix of the deterministic epoch is
        yielded.  ``__len__`` deliberately remains the full epoch length so
        checkpoint accounting stays independent of a one-time resume offset.
        """
        start_batch = int(start_batch)
        if not 0 <= start_batch <= self._n_batches:
            raise ValueError(f"start_batch={start_batch} outside [0, {self._n_batches}]")
        self._start_batch = start_batch

    def __iter__(self) -> Iterator[list[int]]:
        g = torch.Generator()
        # All ranks must build the same global draw and batch order. Rank only
        # selects its non-overlapping slice from each global batch below.
        g.manual_seed(self.seed + self.epoch)
        global_samples = self.num_samples * self.world_size
        # Uniform draw with replacement over valid indices.
        pos = torch.randint(
            len(self.valid), (global_samples,), generator=g, dtype=torch.long
        )
        lens = self.lengths[pos]
        global_batch = self.batch_size * self.world_size
        mb = global_batch * self.megabatch_mult

        batches: list[list[int]] = []
        for s in range(0, len(pos), mb):
            chunk = pos[s : s + mb]
            order = torch.argsort(lens[s : s + mb])  # sort this megabatch by length
            chunk = chunk[order]
            n_full = len(chunk) // global_batch
            for b in range(n_full):
                sel = chunk[b * global_batch : (b + 1) * global_batch]
                # Striding gives every rank samples spanning the same narrow
                # length interval instead of assigning low lengths to rank 0
                # and high lengths to the last rank.
                local = sel[self.rank :: self.world_size]
                batches.append(self.valid[local].tolist())  # positions → dataset indices

        # Shuffle batch order so the model doesn't see length-monotonic batches.
        perm = torch.randperm(len(batches), generator=g).tolist()
        return iter([batches[i] for i in perm[self._start_batch :]])

    def __len__(self) -> int:
        return self._n_batches

    def __repr__(self) -> str:
        return (
            f"LengthBucketedDistributedBatchSampler("
            f"valid={self._n_valid}, rank={self.rank}/{self.world_size}, bs={self.batch_size}, "
            f"batches/epoch={self._n_batches}, megabatch_mult={self.megabatch_mult}, "
            f"len_min={int(self.lengths.min())}, len_max={int(self.lengths.max())})"
        )
