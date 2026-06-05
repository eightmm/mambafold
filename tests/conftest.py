"""Shared fixtures + sys.path / TMPDIR setup for all tests."""

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# TileLang JIT needs an exec-capable tmpdir; /tmp is often noexec on HPC hosts.
if "TMPDIR" not in os.environ:
    fallback = ROOT / ".cache" / "tmp"
    fallback.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(fallback)
    tempfile.tempdir = str(fallback)
