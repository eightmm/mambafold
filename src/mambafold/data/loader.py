"""DataLoader utilities."""

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


def build_dataloaders(args, is_dist: bool):
    """Build train (and optionally val) DataLoaders from args.

    Returns:
        (train_loader, train_sampler, val_loader, dataset)
    """
    data_path = Path(args.data_dir)
    esm_dir = getattr(args, "esm_dir", None)
    single_chain_only = bool(getattr(args, "single_chain_only", False))

    # Fail loud if esm_dir is configured but missing/empty. The model also
    # fails if use_plm=True and a batch arrives without ESM features.
    if esm_dir:
        esm_path = Path(esm_dir)
        if not esm_path.exists():
            raise FileNotFoundError(f"esm_dir does not exist: {esm_dir}")
        if next(esm_path.rglob("*.npy"), None) is None:
            raise FileNotFoundError(
                f"esm_dir has no *.npy files: {esm_dir} "
                f"(run scripts/precompute_esm.py first)"
            )

    if _has_files(data_path, "*.npz"):
        dataset = RCSBDataset(data_dir=args.data_dir, max_length=args.max_length,
                              file_list=getattr(args, "file_list", None), esm_dir=esm_dir,
                              single_chain_only=single_chain_only,
                              extract_monomer_chains=bool(getattr(args, "extract_monomer_chains", False)),
                              chain_index_workers=getattr(args, "length_cache_workers", 8))
    else:
        dataset = AFDBDataset(data_dir=args.data_dir, max_length=args.max_length)

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
    if use_length_balance and not isinstance(dataset, RCSBDataset):
        print("[loader] length_balanced_sampling=True but dataset is not RCSBDataset; "
              "falling back to DistributedSampler")
        use_length_balance = False

    # Length bucketing groups near-equal-length proteins per batch so the
    # collator pads to ~batch_max instead of the global max — the main lever
    # against the O(L²) pair-stack padding waste. Needs RCSBDataset metadata
    # and only helps when batch_size > 1.
    use_bucketing = (
        bool(getattr(args, "length_bucketing", False))
        and isinstance(dataset, RCSBDataset)
        and args.batch_size > 1
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

    sampler = None          # per-item sampler (None when batch_sampler is used)
    batch_sampler = None
    if use_bucketing:
        # True per-example lengths → bucket by actual length, not metadata sum.
        if dataset.extract_monomer_chains:
            # Chain-level dataset: lengths come straight from the chain index.
            idx_len = {i: dataset.chain_index[i][2] for i in range(len(dataset))}
        else:
            idx_len = index_lengths_for_dataset(
                dataset, num_workers=getattr(args, "length_cache_workers", 8),
            )
        batch_sampler = LengthBucketedDistributedBatchSampler(
            index_lengths=idx_len, batch_size=args.batch_size,
            rank=rank, world_size=world_size, **lb_kw,
        )
        if rank == 0:
            print(f"[loader] {batch_sampler}")
    elif use_length_balance:
        sampler = LengthBalancedDistributedSampler(
            dataset_files=dataset.files, metadata_path=meta_path,
            rank=rank, world_size=world_size, **lb_kw,
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
            num_workers=getattr(args, "num_workers", 0),
            pin_memory=True,
            persistent_workers=(getattr(args, "num_workers", 0) > 0),
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            collate_fn=collator,
            num_workers=getattr(args, "num_workers", 0),
            pin_memory=True,
            persistent_workers=(getattr(args, "num_workers", 0) > 0),
            drop_last=True,
        )

    val_loader = None
    val_dir = getattr(args, "val_data_dir", None) or args.data_dir
    if getattr(args, "val_file_list", None) and getattr(args, "eval_interval", 0) > 0:
        val_path = Path(val_dir)
        if _has_files(val_path, "*.npz"):
            val_ds = RCSBDataset(data_dir=val_dir, max_length=args.max_length,
                                 file_list=args.val_file_list, esm_dir=esm_dir,
                                 single_chain_only=single_chain_only)
        else:
            val_ds = AFDBDataset(data_dir=val_dir, max_length=args.max_length)
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=ProteinCollator(augment=False, max_length=args.max_length),
            num_workers=2, drop_last=False,
        )

    # Return whichever object carries set_epoch (inf_loader calls it each epoch).
    epoch_sampler = batch_sampler if batch_sampler is not None else sampler
    return loader, epoch_sampler, val_loader, dataset
