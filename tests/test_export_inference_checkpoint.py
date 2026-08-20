from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from scripts.export_inference_checkpoint import (
    ExportError,
    export_inference_checkpoint,
)

EXPECTED_STEP = 4


def _write_tiny_checkpoint(
    path: Path,
    *,
    step: int = EXPECTED_STEP,
    include_ema: bool = True,
) -> tuple[dict, dict[str, torch.Tensor]]:
    model = torch.nn.Linear(3, 2)
    ema_state = {
        "bias": torch.tensor([10.0, 20.0], dtype=torch.float32),
        "weight": torch.arange(6, dtype=torch.float32).reshape(2, 3) + 30.0,
    }
    args = {
        "config": "configs/tiny.yaml",
        "use_plm": False,
        "nested": {"values": [1, 2, 3], "enabled": True},
    }
    checkpoint = {
        "checkpoint_version": 2,
        "step": step,
        "model": {key: torch.zeros_like(value) for key, value in model.state_dict().items()},
        "optimizer": {"large_training_state": torch.ones(5)},
        "scheduler": {"last_epoch": step},
        "args": args,
        "wandb_run_id": "tiny-run",
    }
    if include_ema:
        checkpoint["ema"] = ema_state
    torch.save(checkpoint, path)
    return checkpoint, ema_state


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_success_is_deterministic_and_writes_metadata_and_sidecar(tmp_path):
    source = tmp_path / f"ckpt_{EXPECTED_STEP:07d}.pt"
    checkpoint, ema_state = _write_tiny_checkpoint(source)
    first_output = tmp_path / "first" / "inference.pt"
    second_output = tmp_path / "second" / "inference.pt"

    first = export_inference_checkpoint(source, first_output, expected_step=EXPECTED_STEP)
    second = export_inference_checkpoint(source, second_output, expected_step=EXPECTED_STEP)

    assert first["artifact_sha256"] == second["artifact_sha256"]
    assert first_output.read_bytes() == second_output.read_bytes()
    exported = torch.load(first_output, map_location="cpu", weights_only=False)
    assert exported["args"] == checkpoint["args"]
    assert exported["model"] is exported["ema"]
    assert "optimizer" not in exported
    assert "scheduler" not in exported
    assert "wandb_run_id" not in exported
    for key, expected in ema_state.items():
        assert exported["ema"][key].dtype == torch.float32
        torch.testing.assert_close(exported["ema"][key], expected)

    metadata = exported["export_metadata"]
    assert metadata["source"] == {
        "filename": source.name,
        "size_bytes": source.stat().st_size,
        "sha256": _sha256(source),
    }
    assert metadata["state_dict"]["tensor_count"] == 2
    assert metadata["state_dict"]["state_value_count"] == 8
    assert metadata["state_dict"]["dtype_value_counts"] == {"torch.float32": 8}
    assert [item["name"] for item in metadata["state_dict"]["tensors"]] == [
        "bias",
        "weight",
    ]

    sidecar = first_output.with_name(f"{first_output.name}.SHA256SUMS")
    assert sidecar.read_text(encoding="utf-8") == f"{_sha256(first_output)}  inference.pt\n"
    assert not list(first_output.parent.glob("*.tmp"))


def test_rejects_wrong_step_without_writing_output(tmp_path):
    source = tmp_path / "source.pt"
    _write_tiny_checkpoint(source, step=EXPECTED_STEP - 1)
    output = tmp_path / "inference.pt"

    with pytest.raises(ExportError, match=r"checkpoint step=3, expected 4"):
        export_inference_checkpoint(source, output, expected_step=EXPECTED_STEP)

    assert not output.exists()
    assert not output.with_name(f"{output.name}.SHA256SUMS").exists()


@pytest.mark.parametrize("existing", ["artifact", "sidecar"])
def test_refuses_to_overwrite_artifact_or_sidecar(tmp_path, existing):
    source = tmp_path / "source.pt"
    _write_tiny_checkpoint(source)
    output = tmp_path / "inference.pt"
    sidecar = output.with_name(f"{output.name}.SHA256SUMS")
    occupied = output if existing == "artifact" else sidecar
    occupied.write_bytes(b"keep-me")

    with pytest.raises(ExportError, match="refusing to overwrite"):
        export_inference_checkpoint(source, output, expected_step=EXPECTED_STEP)

    assert occupied.read_bytes() == b"keep-me"
    if existing == "sidecar":
        assert not output.exists()


def test_rejects_missing_ema(tmp_path):
    source = tmp_path / "source.pt"
    _write_tiny_checkpoint(source, include_ema=False)
    output = tmp_path / "inference.pt"

    with pytest.raises(ExportError, match="no non-empty EMA state"):
        export_inference_checkpoint(source, output, expected_step=EXPECTED_STEP)

    assert not output.exists()


def test_exported_keys_follow_load_from_checkpoint_contract_without_mamba(tmp_path):
    source = tmp_path / "source.pt"
    _, ema_state = _write_tiny_checkpoint(source)
    output = tmp_path / "inference.pt"
    export_inference_checkpoint(source, output, expected_step=EXPECTED_STEP)

    checkpoint = torch.load(output, map_location="cpu", weights_only=False)
    assert {"args", "model", "ema", "step"} <= set(checkpoint)
    for use_ema in (False, True):
        state_key = "ema" if use_ema and "ema" in checkpoint else "model"
        model = torch.nn.Linear(3, 2)
        model.load_state_dict(checkpoint[state_key], strict=True)
        torch.testing.assert_close(model.weight, ema_state["weight"])
        torch.testing.assert_close(model.bias, ema_state["bias"])
