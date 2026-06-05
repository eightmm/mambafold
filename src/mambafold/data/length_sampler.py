"""DDP-compatible length-balanced sampler for RCSB protein training.

Default `DistributedSampler(shuffle=True)` samples every protein with equal
probability. The dataset audit showed our train set is 90% L<500 (median
sum-chain-length ≈ 217) — uniform sampling spends ~90% of training compute
on short monomers, contributing to long-chain quality drop at L=512-1024
(0.80 → 0.72).

`LengthBalancedDistributedSampler` upweights longer proteins via a configurable
exponent (default L^0.5, "moderate"). Each DDP rank draws its own
WeightedRandomSampler-style indices with a rank-offset seed so the union
across ranks is approximately balanced and ranks don't duplicate work
heavily.

For length stats, the sampler reads `data/splits/metadata.tsv` once and
matches it to `dataset.files`. Files missing from metadata fall back to
weight 1.0.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Sequence

import torch
from torch.utils.data import Sampler


def _load_protein_lengths(metadata_path: Path) -> dict[str, int]:
    """pdb_id (lowercase) → sum of standard-residue lengths across protein chains."""
    chains: dict[str, int] = defaultdict(int)
    with open(metadata_path) as f:
        header = f.readline().rstrip("\n").split("\t")
        idx_id = header.index("pdb_id")
        idx_len = header.index("n_standard")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                continue
            pid = parts[idx_id].lower()
            try:
                n = int(parts[idx_len])
            except ValueError:
                continue
            if n > 0:
                chains[pid] += n
    return dict(chains)


def _weight_from_length(L: int, mode: str, exponent: float, clip_min: float, clip_max: float) -> float:
    """Map a protein's sum-chain length to a sampling weight.

    mode="power": w = max(clip_min, min(clip_max, (L / 200) ** exponent))
        — moderate upweight for longer proteins (default exponent=0.5).
    mode="linear_clip": w = clip(L / 200, clip_min, clip_max)
        — proportional, clipped to a comfortable range.
    """
    base = max(L, 1) / 200.0
    if mode == "power":
        w = base ** exponent
    elif mode == "linear_clip":
        w = base
    else:
        raise ValueError(f"unknown length-balance mode: {mode!r}")
    return float(max(clip_min, min(clip_max, w)))


class LengthBalancedDistributedSampler(Sampler[int]):
    """Per-rank weighted sampler over a flat protein list.

    Args:
        dataset_files: list of Paths or strings matching `dataset.files`
            (the order MUST match `dataset[idx]` order).
        metadata_path: Path to `data/splits/metadata.tsv`.
        rank, world_size: DDP coordinates. Default rank=0, world_size=1 (single-GPU).
        num_samples_per_rank: per-rank sample count per epoch. Default =
            `ceil(len(files) / world_size)`.
        mode, exponent, clip_min, clip_max: see `_weight_from_length`.
        seed: RNG seed; per-epoch RNG = seed + epoch*world_size + rank.
    """

    def __init__(
        self,
        dataset_files: Sequence,
        metadata_path: str | Path,
        rank: int = 0,
        world_size: int = 1,
        num_samples_per_rank: int | None = None,
        mode: str = "power",
        exponent: float = 0.5,
        clip_min: float = 1.0,
        clip_max: float = 1.5,
        seed: int = 0,
    ):
        # Note: torch.utils.data.Sampler in recent PyTorch no longer accepts
        # data_source — skip super().__init__ (Sampler is just an iterable protocol).
        if world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {world_size}")
        if not (0 <= rank < world_size):
            raise ValueError(f"rank={rank} out of range for world_size={world_size}")

        lengths = _load_protein_lengths(Path(metadata_path))
        n = len(dataset_files)
        weights = []
        n_default = 0
        for f in dataset_files:
            stem = Path(f).stem.lower()
            L = lengths.get(stem)
            if L is None:
                weights.append(1.0)
                n_default += 1
            else:
                weights.append(_weight_from_length(L, mode, exponent, clip_min, clip_max))
        self.weights = torch.tensor(weights, dtype=torch.double)
        self.n_default = n_default

        self.rank = rank
        self.world_size = world_size
        self.num_samples = (
            num_samples_per_rank if num_samples_per_rank is not None
            else math.ceil(n / world_size)
        )
        self.seed = seed
        self.epoch = 0

        # Stats for logging
        self._n_total = n
        self._mode = mode
        self._exponent = exponent
        self._clip = (clip_min, clip_max)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch * self.world_size + self.rank)
        idx = torch.multinomial(
            self.weights, self.num_samples, replacement=True, generator=g,
        )
        return iter(idx.tolist())

    def __len__(self) -> int:
        return self.num_samples

    def __repr__(self) -> str:
        w = self.weights
        return (
            f"LengthBalancedDistributedSampler("
            f"n={self._n_total}, rank={self.rank}/{self.world_size}, "
            f"per_rank={self.num_samples}, mode={self._mode}, exp={self._exponent}, "
            f"clip={self._clip}, default_weight_n={self.n_default}, "
            f"w_min={float(w.min()):.2f}, w_max={float(w.max()):.2f}, "
            f"w_mean={float(w.mean()):.2f})"
        )
