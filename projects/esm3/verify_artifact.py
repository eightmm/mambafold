#!/usr/bin/env python3
"""Verify the immutable ESM3 v1 checkpoint artifact without loading it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(checkpoint: Path) -> dict:
    """Verify *checkpoint* against the frozen manifest and return it."""
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise ValueError(f"checkpoint does not exist: {checkpoint}")

    manifest = json.loads(MANIFEST.read_text())
    expected_size = int(manifest["checkpoint"]["bytes"])
    expected_digest = manifest["checkpoint"]["sha256"]
    if checkpoint.stat().st_size != expected_size:
        raise ValueError(
            f"checkpoint size mismatch: expected={expected_size} actual={checkpoint.stat().st_size}"
        )

    actual_digest = sha256(checkpoint)
    if actual_digest != expected_digest:
        raise ValueError(
            f"checkpoint SHA-256 mismatch: expected={expected_digest} actual={actual_digest}"
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    try:
        manifest = verify_checkpoint(args.checkpoint)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"verified {manifest['project_id']}: {args.checkpoint.resolve()}")


if __name__ == "__main__":
    main()
