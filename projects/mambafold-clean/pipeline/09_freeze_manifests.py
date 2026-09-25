#!/usr/bin/env python
"""Stage 09 — freeze the corpus with checksum manifests.

The AFDB v4 records are hashed in full: they were produced here from an archive
that the live AlphaFold API no longer serves, so the checksum is the only thing
that makes the corpus verifiable later. The RCSB side is a symlink farm over an
external frozen snapshot, so its manifest records the link targets and the id
list rather than re-hashing files this project does not own.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument(
        "--rehash",
        action="store_true",
        help="recompute the AFDB checksums even when the manifest still matches",
    )
    args = ap.parse_args()
    root = Path(args.root).resolve()
    audit = root / "data" / "audit"
    audit.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {}

    v4_npz = root / "data" / "afdb_swissprot_v4" / "npz"
    if v4_npz.exists():
        out = root / "data" / "afdb_swissprot_v4" / "sha256_manifest.tsv"
        paths = sorted(v4_npz.rglob("*.npz"))

        # Hashing the corpus takes hours, so a rerun re-reads it only when the
        # manifest no longer describes what is on disk. Size is part of the
        # comparison: a record rewritten to the same length would otherwise be
        # taken as unchanged.
        current = False
        if not args.rehash and out.exists():
            recorded = {}
            for index, line in enumerate(out.read_text().splitlines()):
                if index == 0:
                    continue
                fields = line.split("\t")
                if len(fields) == 3:
                    recorded[fields[0]] = fields[2]
            on_disk = {str(p.relative_to(v4_npz)): str(p.stat().st_size) for p in paths}
            current = recorded == on_disk

        if current:
            summary["afdb_v4_records"] = len(paths)
            summary["afdb_v4_manifest"] = str(out)
            summary["afdb_v4_manifest_reused"] = True
            print(
                f"manifest already covers {len(paths)} AFDB v4 records; "
                "pass --rehash to recompute",
                flush=True,
            )
        else:
            with out.open("w") as handle:
                handle.write("relpath\tsha256\tbytes\n")
                for path in paths:
                    handle.write(
                        f"{path.relative_to(v4_npz)}\t{sha256(path)}\t{path.stat().st_size}\n"
                    )
            summary["afdb_v4_records"] = len(paths)
            summary["afdb_v4_manifest"] = str(out)
            print(f"hashed {len(paths)} AFDB v4 records", flush=True)

    farm = root / "data" / "rcsb_train"
    if farm.exists():
        out = root / "data" / "splits" / "rcsb_train_link_targets.tsv"
        entries = sorted(farm.iterdir())
        with out.open("w") as handle:
            handle.write("entry\ttarget\n")
            for entry in entries:
                target = os.readlink(entry) if entry.is_symlink() else ""
                handle.write(f"{entry.name}\t{target}\n")
        summary["rcsb_train_records"] = len(entries)
        summary["rcsb_train_manifest"] = str(out)
        print(f"recorded {len(entries)} RCSB link targets", flush=True)

    for name in ("rcsb-admitted.fasta", "afdb-v4-training.fasta"):
        path = audit / name
        if path.exists():
            summary[f"{name}.sha256"] = sha256(path)

    (audit / "freeze_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
