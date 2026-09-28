"""Link-farm helpers shared by the corpus materialisation stages.

The corpus is never bulk-copied. The Boltz records and the ESMC-6B embeddings
already exist on this filesystem, so structures become symlinks and embeddings
become hard links; both cost a directory entry and no data blocks.

Both helpers make the destination equal its input exactly. Pruning matters as
much as linking: when an admission rule is corrected, a record the new rule
rejects has to leave the farm, otherwise the directory still offers it to a
training run and the corpus is only as clean as whichever rule ran last.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mambafold.data.sequence_cache import sequence_embedding_path  # noqa: E402


def read_ids(path: Path) -> list[str]:
    return [
        Path(token.strip()).stem
        for token in path.read_text().splitlines()
        if token.strip()
    ]


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    chunks: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(chunks)))
            header = line[1:].split()[0]
            chunks = []
        else:
            chunks.append(line)
    if header is not None:
        records.append((header, "".join(chunks)))
    return records


def sync_farm(ids: list[str], src_dir: Path, dst_dir: Path) -> dict[str, int]:
    """Make ``dst_dir`` hold exactly one symlink per id, and nothing else."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    wanted = {f"{entry_id}.npz" for entry_id in ids}

    pruned = 0
    for present in list(dst_dir.iterdir()):
        if present.name not in wanted:
            present.unlink()
            pruned += 1

    linked = existing = missing = 0
    for entry_id in ids:
        src = src_dir / f"{entry_id}.npz"
        dst = dst_dir / f"{entry_id}.npz"
        if dst.is_symlink() or dst.exists():
            existing += 1
            continue
        if not src.exists():
            missing += 1
            continue
        os.symlink(src, dst)
        linked += 1
    return {
        "linked": linked,
        "existing": existing,
        "pruned": pruned,
        "missing_source": missing,
    }


def sync_embeddings(
    records: list[tuple[str, str]], src_root: Path, dst_root: Path
) -> tuple[dict[str, int], list[str]]:
    """Hard-link the embedding each sequence needs; report the ones absent.

    Returns ``(stats, missing_ids)``. A missing embedding is reported rather
    than skipped silently: those sequences are exactly the ones a later ESMC-6B
    pass has to compute before they can be trained on.
    """
    wanted: set[Path] = set()
    seen: set[str] = set()
    linked = existing = 0
    missing: list[str] = []

    for chain_id, sequence in records:
        dst = sequence_embedding_path(dst_root, sequence)
        wanted.add(dst)
        if dst.name in seen:
            continue
        seen.add(dst.name)
        if dst.exists():
            existing += 1
            continue
        src = sequence_embedding_path(src_root, sequence)
        if not src.exists():
            missing.append(chain_id)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.link(src, dst)
        linked += 1

    pruned = 0
    cache_root = dst_root / "by_sequence"
    if cache_root.exists():
        for present in cache_root.rglob("*.npy"):
            if present not in wanted:
                present.unlink()
                pruned += 1

    return (
        {
            "chain_records": len(records),
            "unique_sequences": len(seen),
            "linked": linked,
            "existing": existing,
            "pruned": pruned,
            "missing": len(missing),
        },
        missing,
    )
