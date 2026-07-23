#!/usr/bin/env python
"""Validate an extracted official Boltz RCSB archive without loading its full manifest."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path


def iter_json_array(path: Path, chunk_size: int = 4 * 1024 * 1024) -> Iterator[object]:
    decoder = json.JSONDecoder()
    buffer = ""
    eof = False
    started = False

    with path.open(encoding="utf-8") as handle:
        while True:
            if not eof and len(buffer) < chunk_size:
                chunk = handle.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True

            buffer = buffer.lstrip()
            if not started:
                if not buffer:
                    if eof:
                        raise ValueError(f"Empty JSON manifest: {path}")
                    continue
                if buffer[0] != "[":
                    raise ValueError(f"Expected a top-level JSON array: {path}")
                buffer = buffer[1:]
                started = True
                continue

            buffer = buffer.lstrip()
            if buffer.startswith(","):
                buffer = buffer[1:]
                continue
            if buffer.startswith("]"):
                return
            if not buffer:
                if eof:
                    raise ValueError(f"Unterminated JSON array: {path}")
                continue

            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if eof:
                    raise
                chunk = handle.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
                continue

            yield value
            buffer = buffer[end:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--archive-bytes", type=int, required=True)
    parser.add_argument("--min-count", type=int, default=200_000)
    args = parser.parse_args()

    manifest_path = args.root / "manifest.json"
    structures_dir = args.root / "structures"
    if not manifest_path.is_file() or not structures_dir.is_dir():
        raise SystemExit(
            f"Expected manifest.json and structures/ under extracted root: {args.root}"
        )

    unmatched_structures = {
        path.stem.lower() for path in structures_dir.glob("*.npz") if path.is_file()
    }
    structure_count = len(unmatched_structures)
    manifest_count = 0
    duplicate_ids: list[str] = []
    missing_structures: list[str] = []
    seen_ids: set[str] = set()

    for record in iter_json_array(manifest_path):
        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
            raise SystemExit(f"Invalid manifest record at index {manifest_count}")
        target_id = record["id"].lower()
        manifest_count += 1
        if target_id in seen_ids:
            if len(duplicate_ids) < 10:
                duplicate_ids.append(target_id)
            continue
        seen_ids.add(target_id)
        if target_id not in unmatched_structures:
            if len(missing_structures) < 10:
                missing_structures.append(target_id)
        else:
            unmatched_structures.remove(target_id)

    if manifest_count < args.min_count:
        raise SystemExit(
            f"Official manifest unexpectedly contains only {manifest_count} records"
        )
    if duplicate_ids or missing_structures or unmatched_structures:
        raise SystemExit(
            "manifest/structure mismatch: "
            f"manifest_count={manifest_count} structure_count={structure_count} "
            f"duplicates={duplicate_ids} missing_structures={missing_structures} "
            f"unmatched_structure_count={len(unmatched_structures)} "
            f"unmatched_structure_examples={sorted(unmatched_structures)[:10]}"
        )

    output = {
        "schema": "mambafold-boltz-rcsb-download-v2",
        "source_url": args.source_url,
        "remote_last_modified": args.snapshot,
        "archive_bytes": args.archive_bytes,
        "manifest_count": manifest_count,
        "structure_count": structure_count,
        "paired_id_count": manifest_count,
    }
    output_path = args.root / "download_manifest.json"
    tmp_path = output_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(output, indent=2) + "\n")
    tmp_path.replace(output_path)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
