#!/usr/bin/env python3
"""Export a deterministic EMA-only checkpoint for MambaFold inference.

The source must be a trusted local training checkpoint.  PyTorch checkpoints
use pickle for non-tensor values, so this script must not be used on untrusted
downloads.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

EXPORT_SCHEMA_VERSION = 1
HASH_CHUNK_SIZE = 16 * 1024 * 1024


class ExportError(RuntimeError):
    """Raised when an inference artifact cannot be exported safely."""


def sha256_file(path: Path, chunk_size: int = HASH_CHUNK_SIZE) -> str:
    """Return the SHA-256 digest of *path* without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _checkpoint_args(value: Any) -> Any:
    if isinstance(value, Mapping):
        return value
    try:
        vars(value)
    except TypeError as exc:
        raise ExportError("checkpoint args must be a mapping or namespace") from exc
    return value


def _ema_state_and_metadata(value: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not isinstance(value, Mapping) or not value:
        raise ExportError("checkpoint has no non-empty EMA state")

    state: dict[str, torch.Tensor] = {}
    tensors: list[dict[str, Any]] = []
    dtype_value_counts: dict[str, int] = {}
    state_value_count = 0
    tensor_bytes = 0

    for key in sorted(value):
        if not isinstance(key, str):
            raise ExportError(f"EMA state key must be a string, got {key!r}")
        tensor = value[key]
        if not isinstance(tensor, torch.Tensor):
            raise ExportError(f"EMA state entry {key!r} is not a tensor")
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            raise ExportError(f"EMA floating tensor {key!r} must be FP32, got {tensor.dtype}")
        if tensor.device.type != "cpu":
            raise ExportError(f"EMA tensor {key!r} was not loaded on CPU")

        numel = tensor.numel()
        nbytes = numel * tensor.element_size()
        dtype = str(tensor.dtype)
        state[key] = tensor
        state_value_count += numel
        tensor_bytes += nbytes
        dtype_value_counts[dtype] = dtype_value_counts.get(dtype, 0) + numel
        tensors.append(
            {
                "name": key,
                "shape": list(tensor.shape),
                "dtype": dtype,
                "numel": numel,
                "bytes": nbytes,
            }
        )

    return state, {
        "tensor_count": len(tensors),
        "state_value_count": state_value_count,
        "tensor_bytes": tensor_bytes,
        "dtype_value_counts": dict(sorted(dtype_value_counts.items())),
        "tensors": tensors,
    }


def _write_torch_temp(path: Path, payload: Mapping[str, Any]) -> Path:
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            # Passing a file object gives the zip archive a stable internal
            # name; passing a temporary path would make the artifact hash
            # depend on that path.
            torch.save(payload, handle, pickle_protocol=4)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _write_text_temp(path: Path, content: str) -> Path:
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _unlink_if_same_file(temporary: Path, published: Path) -> None:
    try:
        if published.exists() and os.path.samefile(temporary, published):
            published.unlink()
    except FileNotFoundError:
        pass


def _publish_pair_no_replace(
    artifact_temp: Path,
    artifact: Path,
    sidecar_temp: Path,
    sidecar: Path,
) -> None:
    artifact_published = False
    sidecar_published = False
    try:
        # A hard link publishes each fully written inode atomically and fails
        # with EEXIST instead of replacing a concurrently created file.
        os.link(artifact_temp, artifact)
        artifact_published = True
        os.link(sidecar_temp, sidecar)
        sidecar_published = True
        directory_fd = os.open(artifact.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as exc:
        if sidecar_published:
            _unlink_if_same_file(sidecar_temp, sidecar)
        if artifact_published:
            _unlink_if_same_file(artifact_temp, artifact)
        raise ExportError(f"refusing to overwrite existing artifact: {exc.filename}") from exc
    except Exception:
        if sidecar_published:
            _unlink_if_same_file(sidecar_temp, sidecar)
        if artifact_published:
            _unlink_if_same_file(artifact_temp, artifact)
        raise


def export_inference_checkpoint(
    checkpoint: str | Path,
    output: str | Path,
    *,
    expected_step: int,
) -> dict[str, Any]:
    """Write an EMA-only inference artifact and its ``SHA256SUMS`` sidecar."""

    if type(expected_step) is not int or expected_step <= 0:
        raise ExportError(f"expected_step must be a positive integer, got {expected_step!r}")

    checkpoint_path = Path(checkpoint).resolve(strict=True)
    if not checkpoint_path.is_file():
        raise ExportError(f"checkpoint is not a regular file: {checkpoint_path}")

    output_path = Path(output).absolute()
    if output_path.suffix != ".pt":
        raise ExportError(f"output must use the .pt suffix: {output_path}")
    sidecar_path = output_path.with_name(f"{output_path.name}.SHA256SUMS")
    if output_path == checkpoint_path:
        raise ExportError("output must differ from the source checkpoint")
    for candidate in (output_path, sidecar_path):
        if _path_exists(candidate):
            raise ExportError(f"refusing to overwrite existing artifact: {candidate}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_stat_before = checkpoint_path.stat()
    try:
        checkpoint_data = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except Exception as exc:
        raise ExportError(f"cannot load trusted checkpoint {checkpoint_path}: {exc}") from exc
    if not isinstance(checkpoint_data, Mapping):
        raise ExportError("checkpoint root must be a mapping")

    step = checkpoint_data.get("step")
    if type(step) is not int or step != expected_step:
        raise ExportError(f"checkpoint step={step!r}, expected {expected_step}")
    if "args" not in checkpoint_data:
        raise ExportError("checkpoint is missing args")
    args = _checkpoint_args(checkpoint_data["args"])
    ema_state, state_metadata = _ema_state_and_metadata(checkpoint_data.get("ema"))
    source_sha256 = sha256_file(checkpoint_path)
    if _stat_identity(checkpoint_path.stat()) != _stat_identity(source_stat_before):
        raise ExportError("source checkpoint changed while it was being read")

    export_metadata = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "format": "mambafold_ema_inference",
        "source": {
            "filename": checkpoint_path.name,
            "size_bytes": source_stat_before.st_size,
            "sha256": source_sha256,
        },
        "state_dict": state_metadata,
    }
    # Deliberately assign the same mapping to both keys.  PyTorch serialization
    # preserves this alias, so loader compatibility does not duplicate tensors.
    payload = {
        "checkpoint_version": checkpoint_data.get("checkpoint_version"),
        "artifact_type": "mambafold_ema_inference",
        "step": step,
        "model": ema_state,
        "ema": ema_state,
        "args": args,
        "export_metadata": export_metadata,
    }

    artifact_temp: Path | None = None
    sidecar_temp: Path | None = None
    try:
        artifact_temp = _write_torch_temp(output_path, payload)
        artifact_sha256 = sha256_file(artifact_temp)
        sidecar_temp = _write_text_temp(
            sidecar_path,
            f"{artifact_sha256}  {output_path.name}\n",
        )
        if _stat_identity(checkpoint_path.stat()) != _stat_identity(source_stat_before):
            raise ExportError("source checkpoint changed during export")
        _publish_pair_no_replace(
            artifact_temp,
            output_path,
            sidecar_temp,
            sidecar_path,
        )
    finally:
        if artifact_temp is not None:
            artifact_temp.unlink(missing_ok=True)
        if sidecar_temp is not None:
            sidecar_temp.unlink(missing_ok=True)

    return {
        "output": output_path.as_posix(),
        "sidecar": sidecar_path.as_posix(),
        "artifact_sha256": artifact_sha256,
        "source_sha256": source_sha256,
        "source_size_bytes": source_stat_before.st_size,
        "step": step,
        "state_dict": state_metadata,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        result = export_inference_checkpoint(
            args.checkpoint,
            args.output,
            expected_step=args.expected_step,
        )
    except ExportError as exc:
        raise SystemExit(f"inference checkpoint export failed: {exc}") from exc
    print(
        "EMA inference checkpoint exported: "
        f"step={result['step']} state_values={result['state_dict']['state_value_count']} "
        f"sha256={result['artifact_sha256']} output={result['output']}"
    )
    print(f"SHA256SUMS: {result['sidecar']}")


if __name__ == "__main__":
    main()
