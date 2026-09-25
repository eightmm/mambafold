"""Portable, confidence-only checkpoint schema.

The serialized payload contains only tensors and plain Python scalar/container
types supported by ``torch.load(..., weights_only=True)``.  In particular, it
does not include folding weights, optimizer objects, namespaces, or local paths.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from mambafold.confidence.model import PLDDTHead, PLDDTHeadConfig

PLDDT_CHECKPOINT_SCHEMA_VERSION = 1
PLDDT_CHECKPOINT_ARTIFACT_TYPE = "mambafold_plddt_head"
_ROOT_KEYS = {
    "schema_version",
    "artifact_type",
    "head_config",
    "state_dict",
    "temperature",
    "step",
    "metrics",
    "provenance",
}


class PLDDTCheckpointError(ValueError):
    """Raised when a standalone pLDDT checkpoint violates its schema."""


def build_plddt_checkpoint(
    head: PLDDTHead,
    *,
    temperature: float = 1.0,
    step: int = 0,
    metrics: Mapping[str, float] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a validated, CPU-portable confidence checkpoint payload."""

    if not isinstance(head, PLDDTHead):
        raise TypeError(f"head must be PLDDTHead, got {type(head).__name__}")
    temperature = _validate_temperature(temperature)
    step = _validate_step(step)
    clean_metrics = _validate_metrics({} if metrics is None else metrics)
    clean_provenance = _validate_provenance({} if provenance is None else provenance)
    state_dict = {
        key: value.detach().to(device="cpu").contiguous().clone()
        for key, value in head.state_dict().items()
    }
    payload: dict[str, Any] = {
        "schema_version": PLDDT_CHECKPOINT_SCHEMA_VERSION,
        "artifact_type": PLDDT_CHECKPOINT_ARTIFACT_TYPE,
        "head_config": head.config.to_dict(),
        "state_dict": state_dict,
        "temperature": temperature,
        "step": step,
        "metrics": clean_metrics,
        "provenance": clean_provenance,
    }
    _validate_payload(payload)
    return payload


def save_plddt_checkpoint(
    path: str | Path,
    head: PLDDTHead,
    *,
    temperature: float = 1.0,
    step: int = 0,
    metrics: Mapping[str, float] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> None:
    """Atomically save a standalone pLDDT head checkpoint."""

    destination = Path(path)
    if destination.name in {"", ".", ".."} or destination.suffix != ".pt":
        raise PLDDTCheckpointError(f"checkpoint path must name a .pt file: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = build_plddt_checkpoint(
        head,
        temperature=temperature,
        step=step,
        metrics=metrics,
        provenance=provenance,
    )

    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def load_plddt_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    expected_config: PLDDTHeadConfig | None = None,
) -> tuple[PLDDTHead, float, dict[str, Any]]:
    """Safely load and strictly validate a standalone pLDDT checkpoint.

    Returns:
        ``(head, temperature, metadata)``.  ``metadata`` contains the validated
        ``step``, ``metrics``, and ``provenance`` dictionaries.  The reconstructed
        model is in evaluation mode.
    """

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise PLDDTCheckpointError(f"checkpoint is not a file: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except Exception as exc:
        raise PLDDTCheckpointError(f"cannot safely load pLDDT checkpoint: {exc}") from exc

    config, state_dict, temperature, metadata = _validate_payload(payload)
    if expected_config is not None:
        if not isinstance(expected_config, PLDDTHeadConfig):
            raise TypeError("expected_config must be PLDDTHeadConfig or None")
        if config != expected_config:
            raise PLDDTCheckpointError(
                f"checkpoint config {config.to_dict()} does not match expected "
                f"{expected_config.to_dict()}"
            )
    head = PLDDTHead(config)
    try:
        head.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise PLDDTCheckpointError(f"state_dict is incompatible with config: {exc}") from exc
    head.eval()
    return head, temperature, metadata


def _validate_payload(
    payload: Any,
) -> tuple[PLDDTHeadConfig, dict[str, Tensor], float, dict[str, Any]]:
    if not isinstance(payload, dict):
        raise PLDDTCheckpointError(f"checkpoint root must be a dict, got {type(payload).__name__}")
    actual_keys = set(payload)
    if actual_keys != _ROOT_KEYS or not all(isinstance(key, str) for key in payload):
        missing = sorted(_ROOT_KEYS - actual_keys)
        extra = sorted(actual_keys - _ROOT_KEYS, key=repr)
        raise PLDDTCheckpointError(f"invalid checkpoint keys: missing={missing}, extra={extra}")
    if type(payload["schema_version"]) is not int:
        raise PLDDTCheckpointError("schema_version must be an integer")
    if payload["schema_version"] != PLDDT_CHECKPOINT_SCHEMA_VERSION:
        raise PLDDTCheckpointError(
            f"unsupported schema_version {payload['schema_version']!r}; "
            f"expected {PLDDT_CHECKPOINT_SCHEMA_VERSION}"
        )
    if payload["artifact_type"] != PLDDT_CHECKPOINT_ARTIFACT_TYPE:
        raise PLDDTCheckpointError(f"invalid artifact_type {payload['artifact_type']!r}")
    try:
        config = PLDDTHeadConfig.from_dict(payload["head_config"])
    except (TypeError, ValueError) as exc:
        raise PLDDTCheckpointError(f"invalid config: {exc}") from exc
    temperature = _validate_temperature(payload["temperature"])
    step = _validate_step(payload["step"])
    metrics = _validate_metrics(payload["metrics"])
    provenance = _validate_provenance(payload["provenance"])
    state_dict = _validate_state_dict(payload["state_dict"])
    metadata = {"step": step, "metrics": metrics, "provenance": provenance}
    return config, state_dict, temperature, metadata


def _validate_state_dict(value: Any) -> dict[str, Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise PLDDTCheckpointError("state_dict must be a non-empty mapping")
    state_dict: dict[str, Tensor] = {}
    for key, tensor in value.items():
        if not isinstance(key, str) or not key:
            raise PLDDTCheckpointError(f"state_dict key must be a non-empty string, got {key!r}")
        if not isinstance(tensor, Tensor):
            raise PLDDTCheckpointError(f"state_dict entry {key!r} must be a tensor")
        if tensor.layout is not torch.strided:
            raise PLDDTCheckpointError(f"state_dict entry {key!r} must use strided layout")
        if not tensor.is_floating_point():
            raise PLDDTCheckpointError(f"state_dict entry {key!r} must be floating point")
        if not bool(torch.isfinite(tensor).all()):
            raise PLDDTCheckpointError(f"state_dict entry {key!r} contains non-finite values")
        state_dict[key] = tensor
    return state_dict


def _validate_temperature(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PLDDTCheckpointError(f"temperature must be a positive finite number, got {value!r}")
    temperature = float(value)
    if not math.isfinite(temperature) or temperature <= 0:
        raise PLDDTCheckpointError(f"temperature must be a positive finite number, got {value!r}")
    return temperature


def _validate_step(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise PLDDTCheckpointError(f"step must be a non-negative integer, got {value!r}")
    return value


def _validate_metrics(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        raise PLDDTCheckpointError(f"metrics must be a plain dict, got {type(value).__name__}")
    metrics: dict[str, float] = {}
    for key, metric in value.items():
        if not isinstance(key, str) or not key:
            raise PLDDTCheckpointError(f"metric key must be a non-empty string, got {key!r}")
        if isinstance(metric, bool) or not isinstance(metric, (int, float)):
            raise PLDDTCheckpointError(f"metric {key!r} must be numeric, got {metric!r}")
        metric_value = float(metric)
        if not math.isfinite(metric_value):
            raise PLDDTCheckpointError(f"metric {key!r} must be finite, got {metric!r}")
        metrics[key] = metric_value
    return metrics


def _validate_provenance(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PLDDTCheckpointError(
            f"provenance must be a plain dict, got {type(value).__name__}"
        )
    if not all(isinstance(key, str) and key for key in value):
        raise PLDDTCheckpointError("provenance keys must be non-empty strings")
    return {key: _plain_provenance(item, key) for key, item in value.items()}


def _plain_provenance(value: Any, location: str) -> Any:
    """Copy JSON-like provenance while rejecting local absolute paths."""

    if isinstance(value, str):
        if value.startswith(("/", "~/", "file://")) or (
            len(value) >= 3 and value[1:3] in {":/", ":\\"}
        ):
            raise PLDDTCheckpointError(
                f"provenance {location!r} must not contain a private absolute path"
            )
        return value
    if value is None or type(value) in {bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise PLDDTCheckpointError(f"provenance {location!r} must be finite")
        return value
    if isinstance(value, list):
        return [_plain_provenance(item, f"{location}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise PLDDTCheckpointError(
                    f"provenance key under {location!r} must be a non-empty string"
                )
            clean[key] = _plain_provenance(item, f"{location}.{key}")
        return clean
    raise PLDDTCheckpointError(
        f"provenance {location!r} uses unsupported type {type(value).__name__}"
    )
