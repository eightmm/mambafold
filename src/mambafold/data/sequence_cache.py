"""Content-addressed paths for protein language-model embeddings."""

from __future__ import annotations

import hashlib
from pathlib import Path

SEQUENCE_CACHE_DIR = "by_sequence"


def sequence_digest(sequence: str) -> str:
    """Return the stable cache identity for one canonical full sequence."""
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def sequence_embedding_path(root: str | Path, sequence: str) -> Path:
    """Map a full canonical sequence to its content-addressed ``.npy`` path."""
    digest = sequence_digest(sequence)
    return Path(root) / SEQUENCE_CACHE_DIR / digest[:2] / f"{digest}.npy"
