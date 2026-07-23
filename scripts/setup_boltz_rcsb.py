#!/usr/bin/env python
"""Build a local MambaFold view over an existing Boltz RCSB dataset.

The source Boltz directory is flat (``<pdb_id>.npz``), while the frozen
MambaFold split files use sharded relative paths. This script preserves the
frozen memberships, intersects them with the available Boltz snapshot, writes
explicit flat derived split files, and creates a symlink view without copying
structure data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

DEFAULT_SPLITS = (
    ("train", Path("data/splits/train.txt")),
    ("val", Path("data/splits/val.txt")),
    ("val_casp", Path("data/splits/val_casp.txt")),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_ids(path: Path) -> list[str]:
    return [
        Path(line.strip()).stem.lower() for line in path.read_text().splitlines() if line.strip()
    ]


def _write_lines(path: Path, lines: list[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(f"{line}\n" for line in lines))
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--view-dir", type=Path, default=Path("data/rcsb_boltz_official_full"))
    parser.add_argument("--alias-dir", type=Path, default=Path("data/rcsb"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--tag", default="boltz_official_full")
    args = parser.parse_args()

    if not args.tag.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"Invalid tag: {args.tag}")
    manifest_path = args.manifest or args.view_dir / "manifest.json"

    source_dir = args.source_dir.resolve()
    source_files = {path.stem.lower(): path.resolve() for path in source_dir.glob("*.npz")}
    if not source_files:
        raise ValueError(f"No Boltz NPZ files found: {source_dir}")

    split_ids = {name: _read_ids(path) for name, path in DEFAULT_SPLITS}
    split_sets = {name: set(ids) for name, ids in split_ids.items()}
    for left, right in (("train", "val"), ("train", "val_casp"), ("val", "val_casp")):
        overlap = split_sets[left] & split_sets[right]
        if overlap:
            raise ValueError(f"Frozen split overlap: {left}/{right} ({len(overlap)} IDs)")

    args.view_dir.mkdir(parents=True, exist_ok=True)
    derived: dict[str, dict[str, object]] = {}
    linked_ids: set[str] = set()
    for name, source_split in DEFAULT_SPLITS:
        matched = [pdb_id for pdb_id in split_ids[name] if pdb_id in source_files]
        output_split = source_split.with_name(f"{source_split.stem}_{args.tag}.txt")
        _write_lines(output_split, [f"{pdb_id}.npz" for pdb_id in matched])
        linked_ids.update(matched)
        derived[name] = {
            "source_split": str(source_split),
            "source_sha256": _sha256(source_split),
            "output_split": str(output_split),
            "input_count": len(split_ids[name]),
            "matched_count": len(matched),
        }

    for pdb_id in sorted(linked_ids):
        source = source_files[pdb_id]
        destination = args.view_dir / source.name
        if destination.is_symlink():
            if destination.resolve() != source:
                raise FileExistsError(f"Symlink target mismatch: {destination}")
            continue
        if destination.exists():
            raise FileExistsError(f"Refusing to replace existing file: {destination}")
        os.symlink(source, destination)

    alias_target = args.view_dir.resolve()
    if args.alias_dir.is_symlink():
        if args.alias_dir.resolve() != alias_target:
            raise FileExistsError(f"Alias target mismatch: {args.alias_dir}")
    elif args.alias_dir.exists():
        raise FileExistsError(f"Refusing to replace existing alias path: {args.alias_dir}")
    else:
        relative_target = os.path.relpath(alias_target, args.alias_dir.parent.resolve())
        os.symlink(relative_target, args.alias_dir)

    manifest = {
        "schema": "mambafold-boltz-rcsb-view-v1",
        "tag": args.tag,
        "source_dir": str(source_dir),
        "source_npz_count": len(source_files),
        "linked_npz_count": len(linked_ids),
        "alias_dir": str(args.alias_dir),
        "splits": derived,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
