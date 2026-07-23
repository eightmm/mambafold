"""DataLoader utilities."""

from bisect import bisect_right
from pathlib import Path

import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from mambafold.data.collate import ProteinCollator
from mambafold.data.dataset import AFDBDataset, RCSBDataset
from mambafold.data.length_cache import index_lengths_for_dataset
from mambafold.data.length_sampler import (
    LengthBalancedDistributedSampler,
    LengthBucketedDistributedBatchSampler,
)


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


def inf_loader(loader, sampler=None):
    """DataLoader를 무한 반복하는 제너레이터.

    DistributedSampler 사용 시 epoch마다 set_epoch()을 호출해 셔플링을 보장함.
    """
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        yield from loader
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
    file_list: str | None,
    esm_dir: str | None,
    single_chain_only: bool,
    extract_monomer_chains: bool,
    chain_index_workers: int,
) -> RCSBDataset:
    _check_esm_dir(esm_dir)
    data_path = Path(data_dir)
    if not _file_list_has_entries(file_list) and not _has_files(data_path, "*.npz"):
        raise ValueError(f"RCSB-style source has no *.npz files: {data_dir}")
    return RCSBDataset(
        data_dir=data_dir,
        max_length=max_length,
        file_list=file_list,
        esm_dir=esm_dir,
        single_chain_only=single_chain_only,
        extract_monomer_chains=extract_monomer_chains,
        chain_index_workers=chain_index_workers,
    )


def _build_train_dataset(args, single_chain_only: bool):
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
                file_list=src.get("file_list"),
                esm_dir=esm_dir,
                single_chain_only=single_chain_only,
                extract_monomer_chains=bool(getattr(args, "extract_monomer_chains", False)),
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
            file_list=getattr(args, "file_list", None),
            esm_dir=esm_dir,
            single_chain_only=single_chain_only,
            extract_monomer_chains=bool(getattr(args, "extract_monomer_chains", False)),
            chain_index_workers=getattr(args, "length_cache_workers", 8),
        )
    return AFDBDataset(data_dir=args.data_dir, max_length=args.max_length)


def build_dataloaders(args, is_dist: bool):
    """Build train (and optionally val) DataLoaders from args.

    Returns:
        (train_loader, train_sampler, val_loader, dataset)
    """
    esm_dir = getattr(args, "esm_dir", None)
    single_chain_only = bool(getattr(args, "single_chain_only", False))
    num_workers = int(getattr(args, "num_workers", 0))
    loader_timeout = float(getattr(args, "loader_timeout", 0.0)) if num_workers else 0.0

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
        max_length=args.max_length,
        length_bin=getattr(args, "length_bin", 0),
    )
    # Length-balanced sampler — upweights longer proteins to fight the
    # short-tail bias in PDB. Only applies to train; val stays uniform.
    use_length_balance = bool(getattr(args, "length_balanced_sampling", False))
    is_rcsb_like = isinstance(dataset, (RCSBDataset, MixedRCSBDataset))
    if use_length_balance and not is_rcsb_like:
        print(
            "[loader] length_balanced_sampling=True but dataset is not RCSBDataset; "
            "falling back to DistributedSampler"
        )
        use_length_balance = False

    # Length bucketing groups near-equal-length proteins per batch so the
    # collator pads to ~batch_max instead of the global max — the main lever
    # against the O(L²) pair-stack padding waste. Needs RCSBDataset metadata
    # and only helps when batch_size > 1.
    use_bucketing = (
        bool(getattr(args, "length_bucketing", False)) and is_rcsb_like and args.batch_size > 1
    )

    rank = dist.get_rank() if is_dist else 0
    world_size = dist.get_world_size() if is_dist else 1
    meta_path = getattr(args, "metadata_path", "data/splits/metadata.tsv")
    lb_kw = dict(
        mode=getattr(args, "length_balance_mode", "power"),
        exponent=getattr(args, "length_balance_exponent", 0.5),
        clip_min=getattr(args, "length_balance_clip_min", 1.0),
        clip_max=getattr(args, "length_balance_clip_max", 5.0),
        seed=getattr(args, "seed", 0),
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
            **lb_kw,
        )
        if rank == 0:
            print(f"[loader] {batch_sampler}")
    elif use_length_balance:
        if isinstance(dataset, MixedRCSBDataset):
            raise ValueError(
                "length_balanced_sampling without length_bucketing is not "
                "supported for train_sources"
            )
        sampler = LengthBalancedDistributedSampler(
            dataset_files=dataset.files,
            metadata_path=meta_path,
            rank=rank,
            world_size=world_size,
            **lb_kw,
        )
        if rank == 0:
            print(f"[loader] {sampler}")
    elif is_dist:
        sampler = DistributedSampler(dataset, shuffle=True)

    if batch_sampler is not None:
        # batch_sampler is mutually exclusive with batch_size/shuffle/sampler/drop_last.
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            timeout=loader_timeout,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            timeout=loader_timeout,
            drop_last=True,
        )

    val_loader = None
    val_dir = getattr(args, "val_data_dir", None) or args.data_dir
    if getattr(args, "val_file_list", None) and getattr(args, "eval_interval", 0) > 0:
        val_path = Path(val_dir)
        if _has_files(val_path, "*.npz"):
            val_ds = RCSBDataset(
                data_dir=val_dir,
                max_length=args.max_length,
                file_list=args.val_file_list,
                esm_dir=esm_dir,
                single_chain_only=single_chain_only,
            )
        else:
            val_ds = AFDBDataset(data_dir=val_dir, max_length=args.max_length)
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=ProteinCollator(augment=False, max_length=args.max_length),
            num_workers=min(2, num_workers),
            timeout=loader_timeout if num_workers else 0.0,
            drop_last=False,
        )

    # Return whichever object carries set_epoch (inf_loader calls it each epoch).
    epoch_sampler = batch_sampler if batch_sampler is not None else sampler
    return loader, epoch_sampler, val_loader, dataset
