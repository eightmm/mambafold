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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint does not exist: {checkpoint}")

    manifest = json.loads(MANIFEST.read_text())
    expected_size = int(manifest["checkpoint"]["bytes"])
    expected_digest = manifest["checkpoint"]["sha256"]
    if checkpoint.stat().st_size != expected_size:
        raise SystemExit(
            f"checkpoint size mismatch: expected={expected_size} actual={checkpoint.stat().st_size}"
        )

    actual_digest = sha256(checkpoint)
    if actual_digest != expected_digest:
        raise SystemExit(
            f"checkpoint SHA-256 mismatch: expected={expected_digest} actual={actual_digest}"
        )
    print(f"verified {manifest['project_id']}: {checkpoint}")


if __name__ == "__main__":
    main()
