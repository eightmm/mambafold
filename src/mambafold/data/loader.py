"""DataLoader utilities."""

from pathlib import Path

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from mambafold.data.collate import ProteinCollator
from mambafold.data.dataset import AFDBDataset, RCSBDataset


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

    if _has_files(data_path, "*.npz"):
        dataset = RCSBDataset(data_dir=args.data_dir, max_length=args.max_length,
                              file_list=getattr(args, "file_list", None), esm_dir=esm_dir)
    else:
        dataset = AFDBDataset(data_dir=args.data_dir, max_length=args.max_length)

    collator = ProteinCollator(
        augment=True,
        copies_per_protein=getattr(args, "copies_per_protein", 1),
        gamma_schedule=getattr(args, "gamma_schedule", "logit_normal"),
        max_length=args.max_length,
    )
    sampler = DistributedSampler(dataset, shuffle=True) if is_dist else None
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
                                 file_list=args.val_file_list, esm_dir=esm_dir)
        else:
            val_ds = AFDBDataset(data_dir=val_dir, max_length=args.max_length)
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=ProteinCollator(augment=False, max_length=args.max_length),
            num_workers=2, drop_last=False,
        )

    return loader, sampler, val_loader, dataset
