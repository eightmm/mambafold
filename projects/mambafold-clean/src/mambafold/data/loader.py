"""DataLoader utilities."""

from bisect import bisect_right
from pathlib import Path

import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from mambafold.data.collate import ProteinCollator
from mambafold.data.dataset import AFDBDataset, RCSBDataset
from mambafold.data.length_cache import index_lengths_for_dataset
from mambafold.data.length_sampler import LengthBucketedDistributedBatchSampler


class MixedRCSBDataset:
    """Concatenate multiple Boltz-style NPZ sources with per-source ESM dirs."""

    def __init__(self, datasets: list[RCSBDataset], names: list[str]):
        if not datasets:
            raise ValueError("MixedRCSBDataset requires at least one source")
        self.datasets = datasets
        self.names = names
        self.cum = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cum.append(total)

    def __len__(self) -> int:
        return self.cum[-1]

    def _loc(self, idx: int) -> tuple[int, int]:
        if idx < 0:
            idx += len(self)
        if not (0 <= idx < len(self)):
            raise IndexError(idx)
        source = bisect_right(self.cum, idx)
        prev = 0 if source == 0 else self.cum[source - 1]
        return source, idx - prev

    def __getitem__(self, idx: int):
        source, local_idx = self._loc(idx)
        return self.datasets[source][local_idx]

    def index_lengths(self, num_workers: int) -> dict[int, int]:
        out: dict[int, int] = {}
        offset = 0
        for ds in self.datasets:
            if ds.extract_monomer_chains and ds.chain_index is not None:
                local = {i: ds.chain_index[i][2] for i in range(len(ds))}
            else:
                local = index_lengths_for_dataset(ds, num_workers=num_workers)
            out.update({offset + i: L for i, L in local.items()})
            offset += len(ds)
        return out

    def summary(self) -> str:
        parts = [
            f"{name}: n={len(ds)} files={len(ds.files)} esm={ds.esm_dir}"
            for name, ds in zip(self.names, self.datasets)
        ]
        return "MixedRCSBDataset(" + "; ".join(parts) + ")"


class RepeatBatchSampler:
    """Yield each batch of indices `repeat` times in a row.

    This is what makes gradient accumulation over the *same* protein identical
    to running all of its noise copies in one pass. With the loss divided by
    `grad_accum`, accumulating `copies` copies over `accum` micro-steps sums to
    exactly the same gradient as `copies * accum` copies in a single step — the
    same realisation, not merely the same expectation, because every term
    carries weight 1/(copies*accum) either way. It is how SimpleFold's
    `multiplicity: 16` is reachable when 16 copies of a 1024-residue chain do
    not fit in memory at once.

    The repeat has to happen here, on the indices, and not on a collated batch:
    each micro-step must re-enter the collator so that it draws its own time,
    its own noise and its own rotation. Repeating the finished batch would give
    the same 8 samples twice and halve the effective multiplicity while looking
    like it had doubled.
    """

    def __init__(self, base, repeat: int):
        self.base = base
        self.repeat = max(1, int(repeat))

    def __iter__(self):
        for batch in self.base:
            for _ in range(self.repeat):
                yield batch

    def __len__(self) -> int:
        return len(self.base) * self.repeat

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.base, "set_epoch"):
            self.base.set_epoch(epoch)

    def set_start_batch(self, batch: int) -> None:
        # The caller counts micro-steps; the wrapped sampler counts draws.
        if hasattr(self.base, "set_start_batch"):
            self.base.set_start_batch(batch // self.repeat)

    def __repr__(self) -> str:
        return f"RepeatBatchSampler({self.base!r}, repeat={self.repeat})"


def inf_loader(loader, sampler=None, *, start_epoch: int = 0, start_batch: int = 0):
    """DataLoader를 무한 반복하는 제너레이터.

    DistributedSampler 사용 시 epoch마다 set_epoch()을 호출해 셔플링을 보장함.
    Resume can restore the deterministic sampler epoch and skip the already
    consumed batches from the first resumed epoch.
    """
    if start_epoch < 0 or start_batch < 0:
        raise ValueError("start_epoch and start_batch must be non-negative")
    epoch = start_epoch
    while True:
        resume_batch = start_batch if epoch == start_epoch else 0
        sampler_fast_forward = sampler is not None and hasattr(sampler, "set_start_batch")
        if sampler is not None:
            sampler.set_epoch(epoch)
            if sampler_fast_forward:
                # Skip at the batch-index source.  Skipping after DataLoader
                # iteration would still read, decompress, collate, and discard
                # every previously consumed sample.
                sampler.set_start_batch(resume_batch)
        for batch_idx, batch in enumerate(loader):
            if not sampler_fast_forward and batch_idx < resume_batch:
                continue
            yield batch
        epoch += 1


def _has_files(root: Path, pattern: str) -> bool:
    return root.exists() and next(root.rglob(pattern), None) is not None


def _file_list_has_entries(file_list: str | None) -> bool:
    if file_list is None:
        return False
    path = Path(file_list)
    if not path.exists():
        raise FileNotFoundError(f"file_list does not exist: {file_list}")
    return any(line.strip() for line in path.read_text().splitlines())


def _check_esm_dir(esm_dir: str | None) -> None:
    if not esm_dir:
        return
    esm_path = Path(esm_dir)
    if not esm_path.exists():
        raise FileNotFoundError(f"esm_dir does not exist: {esm_dir}")
    # ESMC caches are content-addressed under by_sequence/<prefix>/<sha>.npy.
    # A recursive glob over the full cache can take minutes on the shared FS and
    # has caused Slurm preflight DataLoader timeouts.  Keep this as a cheap
    # layout sanity check; per-sample missing embeddings are still rejected by
    # RCSBDataset._canonicalize.
    if not (esm_path / "by_sequence").is_dir() and next(esm_path.glob("*.npy"), None) is None:
        raise FileNotFoundError(
            f"esm_dir has no *.npy files: {esm_dir} (run scripts/precompute_esm.py first)"
        )


def _build_rcsb_dataset(
    *,
    data_dir: str,
    max_length: int,
    min_obs_ratio: float,
    file_list: str | None,
    esm_dir: str | None,
    single_chain_only: bool,
    extract_monomer_chains: bool,
    dedup_homomer_chains: bool,
    chain_index_workers: int,
) -> RCSBDataset:
    _check_esm_dir(esm_dir)
    data_path = Path(data_dir)
    if not _file_list_has_entries(file_list) and not _has_files(data_path, "*.npz"):
        raise ValueError(f"RCSB-style source has no *.npz files: {data_dir}")
    return RCSBDataset(
        data_dir=data_dir,
        max_length=max_length,
        min_obs_ratio=min_obs_ratio,
        file_list=file_list,
        esm_dir=esm_dir,
        single_chain_only=single_chain_only,
        extract_monomer_chains=extract_monomer_chains,
        dedup_homomer_chains=dedup_homomer_chains,
        chain_index_workers=chain_index_workers,
    )


def _build_train_dataset(args, single_chain_only: bool):
    obs_ratio = float(getattr(args, "min_obs_ratio", 0.0))
    train_sources = getattr(args, "train_sources", None)
    if train_sources:
        datasets = []
        names = []
        for i, src in enumerate(train_sources):
            if not isinstance(src, dict):
                raise TypeError(f"train_sources[{i}] must be a mapping, got {type(src).__name__}")
            data_dir = src["data_dir"]
            esm_dir = src.get("esm_dir", getattr(args, "esm_dir", None))
            ds = _build_rcsb_dataset(
                data_dir=data_dir,
                max_length=args.max_length,
                min_obs_ratio=obs_ratio,
                file_list=src.get("file_list"),
                esm_dir=esm_dir,
                single_chain_only=single_chain_only,
                extract_monomer_chains=bool(getattr(args, "extract_monomer_chains", False)),
                dedup_homomer_chains=bool(getattr(args, "dedup_homomer_chains", False)),
                chain_index_workers=getattr(args, "length_cache_workers", 8),
            )
            datasets.append(ds)
            names.append(src.get("name", Path(data_dir).name))
        return MixedRCSBDataset(datasets, names)

    data_path = Path(args.data_dir)
    esm_dir = getattr(args, "esm_dir", None)
    if _has_files(data_path, "*.npz"):
        return _build_rcsb_dataset(
            data_dir=args.data_dir,
            max_length=args.max_length,
            min_obs_ratio=obs_ratio,
            file_list=getattr(args, "file_list", None),
            esm_dir=esm_dir,
            single_chain_only=single_chain_only,
            extract_monomer_chains=bool(getattr(args, "extract_monomer_chains", False)),
            dedup_homomer_chains=bool(getattr(args, "dedup_homomer_chains", False)),
            chain_index_workers=getattr(args, "length_cache_workers", 8),
        )
    return AFDBDataset(data_dir=args.data_dir, max_length=args.max_length)


def build_dataloaders(args, is_dist: bool):
    """Build the training DataLoader from args.

    There is no validation loader. This project selects nothing during
    training: it runs a fixed number of steps and reports the last EMA, and
    the only progress signal is a rollout against an external benchmark.

    Returns:
        (train_loader, train_sampler, dataset)
    """
    esm_dir = getattr(args, "esm_dir", None)
    single_chain_only = bool(getattr(args, "single_chain_only", False))
    num_workers = int(getattr(args, "num_workers", 0))
    loader_timeout = float(getattr(args, "loader_timeout", 0.0)) if num_workers else 0.0
    prefetch_factor = int(getattr(args, "prefetch_factor", 1))
    if num_workers > 0 and prefetch_factor < 1:
        raise ValueError(f"prefetch_factor must be >= 1, got {prefetch_factor}")

    # Fail loud if esm_dir is configured but missing/empty. The model also
    # fails if use_plm=True and a batch arrives without ESM features.
    _check_esm_dir(esm_dir)
    dataset = _build_train_dataset(args, single_chain_only)
    if (not is_dist) or dist.get_rank() == 0:
        summary = (
            dataset.summary()
            if isinstance(dataset, MixedRCSBDataset)
            else (f"{type(dataset).__name__}(n={len(dataset)})")
        )
        print(f"[loader] train {summary}", flush=True)

    collator = ProteinCollator(
        augment=True,
        copies_per_protein=getattr(args, "copies_per_protein", 1),
        t_schedule=getattr(args, "t_schedule", "uniform"),
        t_uniform_weight=getattr(args, "t_uniform_weight", 0.02),
        max_length=args.max_length,
        length_bin=getattr(args, "length_bin", 0),
    )
    is_rcsb_like = isinstance(dataset, (RCSBDataset, MixedRCSBDataset))

    # Length bucketing groups near-equal-length proteins per batch so the
    # collator pads to ~batch_max instead of the global max. This reduces
    # padding and shape-specialization waste, and needs batch_size > 1.
    rank = dist.get_rank() if is_dist else 0
    world_size = dist.get_world_size() if is_dist else 1

    # Bucketing is about the *global* batch, not the local one. The training
    # loop pads every rank up to `distributed_max_int(batch.max_len)` so the
    # TileLang kernels, which specialize on sequence length, see one shape per
    # step. With independent per-rank draws that global max is the maximum of
    # `world_size` random chain lengths: measured over this corpus it averages
    # 624 residues at 8 ranks against a mean own length of 312, so every rank
    # does 2x the necessary work. Drawing all ranks from one length-sorted group
    # brings that back to 1.0x. So this stays on even at batch_size 1, where
    # there is no intra-batch padding to save.
    use_bucketing = (
        bool(getattr(args, "length_bucketing", False))
        and is_rcsb_like
        # The cross-rank argument below needs more than one sequence in the
        # global batch, but the batch sampler itself is also what
        # `accum_same_protein` repeats indices through: without it, a single-GPU
        # run raises "accum_same_protein needs a batch sampler" and cannot start
        # at all. Measured on a one-rank smoke before this clause existed. So
        # bucketing also turns on whenever accumulation is repeating a protein,
        # where it costs nothing and buys a runnable single-GPU configuration
        # for debugging.
        and (
            args.batch_size * world_size > 1
            or (
                int(getattr(args, "grad_accum_steps", 1)) > 1
                and bool(getattr(args, "accum_same_protein", True))
            )
        )
    )
    sampler = None  # per-item sampler (None when batch_sampler is used)
    batch_sampler = None
    if use_bucketing:
        # True per-example lengths → bucket by actual length, not metadata sum.
        if isinstance(dataset, MixedRCSBDataset):
            idx_len = dataset.index_lengths(num_workers=getattr(args, "length_cache_workers", 8))
        elif dataset.extract_monomer_chains:
            # Chain-level dataset: lengths come straight from the chain index.
            idx_len = {i: dataset.chain_index[i][2] for i in range(len(dataset))}
        else:
            idx_len = index_lengths_for_dataset(
                dataset,
                num_workers=getattr(args, "length_cache_workers", 8),
            )
        batch_sampler = LengthBucketedDistributedBatchSampler(
            index_lengths=idx_len,
            batch_size=args.batch_size,
            rank=rank,
            world_size=world_size,
            seed=getattr(args, "seed", 0),
        )
        if rank == 0:
            print(f"[loader] {batch_sampler}")
    elif is_dist:
        sampler = DistributedSampler(dataset, shuffle=True)

    accum = max(1, int(getattr(args, "grad_accum_steps", 1)))
    if accum > 1 and bool(getattr(args, "accum_same_protein", True)):
        if batch_sampler is None:
            raise ValueError(
                "accum_same_protein needs a batch sampler so the repeat happens on "
                "indices; enable length_bucketing or set accum_same_protein=False."
            )
        batch_sampler = RepeatBatchSampler(batch_sampler, accum)
        if rank == 0:
            print(f"[loader] {batch_sampler}")

    if batch_sampler is not None:
        # batch_sampler is mutually exclusive with batch_size/shuffle/sampler/drop_last.
        worker_kwargs = (
            {
                "persistent_workers": True,
                "prefetch_factor": prefetch_factor,
            }
            if num_workers > 0
            else {}
        )
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=True,
            timeout=loader_timeout,
            **worker_kwargs,
        )
    else:
        worker_kwargs = (
            {
                "persistent_workers": True,
                "prefetch_factor": prefetch_factor,
            }
            if num_workers > 0
            else {}
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=True,
            timeout=loader_timeout,
            drop_last=True,
            **worker_kwargs,
        )

    epoch_sampler = batch_sampler if batch_sampler is not None else sampler
    return loader, epoch_sampler, dataset
