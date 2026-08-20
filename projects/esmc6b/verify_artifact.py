#!/usr/bin/env python3
"""Verify the provisional ESMC-6B 170k EMA artifact without loading it."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest.json"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest of *path*."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_integrity(manifest: dict) -> tuple[int, str]:
    checkpoint = manifest.get("checkpoint", {})
    expected_size = checkpoint.get("bytes")
    expected_digest = checkpoint.get("sha256")
    if (
        not isinstance(expected_size, int)
        or expected_size <= 0
        or not isinstance(expected_digest, str)
        or SHA256_PATTERN.fullmatch(expected_digest) is None
    ):
        raise ValueError(
            "manifest checkpoint bytes/SHA-256 are placeholders; "
            "patch them from the final EMA export before verification or publication"
        )
    return expected_size, expected_digest


def verify_checkpoint(checkpoint: Path, manifest_path: Path = MANIFEST) -> dict:
    """Verify *checkpoint* against *manifest_path* and return the manifest."""
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise ValueError(f"checkpoint does not exist: {checkpoint}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_size, expected_digest = _expected_integrity(manifest)
    actual_size = checkpoint.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"checkpoint size mismatch: expected={expected_size} actual={actual_size}")

    actual_digest = sha256(checkpoint)
    if actual_digest != expected_digest:
        raise ValueError(
            f"checkpoint SHA-256 mismatch: expected={expected_digest} actual={actual_digest}"
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST,
        help="manifest to verify against (default: projects/esmc6b/manifest.json)",
    )
    args = parser.parse_args()

    try:
        manifest = verify_checkpoint(args.checkpoint, args.manifest)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"verified {manifest['project_id']}: {args.checkpoint.resolve()}")


if __name__ == "__main__":
    main()
