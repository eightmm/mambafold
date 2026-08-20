"""CPU-only contract tests for the provisional ESMC-6B release package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from projects.esmc6b.predict_fasta import read_fasta, sequence_example
from projects.esmc6b.verify_artifact import verify_checkpoint

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "esmc6b"


def test_manifest_marks_prerelease_and_external_pinned_plm() -> None:
    manifest = json.loads((PROJECT / "manifest.json").read_text())

    assert manifest["status"] == "provisional_prerelease_baseline"
    assert manifest["release_status"] == "verified_prerelease"
    assert manifest["source_tag"] == "esmc6b-170k-preview.1"
    assert manifest["checkpoint"]["filename"] == "mambafold-esmc6b-170k-ema.pt"
    assert manifest["checkpoint"]["step"] == 170000
    assert manifest["checkpoint"]["model_parameter_count"] == 404856302
    assert manifest["checkpoint"]["state_value_count"] == 404856326
    assert manifest["checkpoint"]["integrity_status"] == "verified_local_export"
    assert manifest["checkpoint"]["bytes"] == 1619662835
    assert manifest["checkpoint"]["sha256"] == (
        "465ddb7d873479e51487a79b39d2a871a10b3b54be178adcd76afe7f86665a02"
    )
    assert manifest["conditioning"] == {
        "model": "biohub/ESMC-6B",
        "runtime_name": "esmc-6b",
        "revision": "45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a",
        "embedding_dimensions": 2560,
        "sequence_only": True,
        "weights_bundled": False,
        "download_script": "scripts/download_esmc6b.sh",
    }
    assert manifest["ongoing_work"]["geometry_finetuning_status"] == "training_in_progress"
    assert manifest["evaluation"]["status"] == ("provisional_retrospective_engineering_evidence")
    assert manifest["evaluation"]["n_seeds"] == 1
    assert manifest["evaluation"]["target_list_sha256"] == (
        "4309a73bcb42b90a1573f12ca4a2f3635b99a3adf4065f8de42f8a5293056df1"
    )
    assert manifest["evaluation"]["result_summary_sha256"] == (
        "f8ec8894b6dacc20b73e4d33065c79ba945cf504513ae0d98f7acda89009d3dd"
    )


def test_saved_training_config_has_release_contract_and_declared_digest() -> None:
    config_path = PROJECT / "training_config.json"
    config = json.loads(config_path.read_text())
    manifest = json.loads((PROJECT / "manifest.json").read_text())

    assert config["total_steps"] == 170000
    assert config["batch_size"] == 9
    assert config["grad_accum_steps"] == 7
    assert config["d_res"] == 1024
    assert config["d_state"] == 64
    assert config["d_plm"] == 2560
    assert config["n_trunk"] == 18
    assert config["d_atom"] == 256
    assert config["n_atom_layers"] == 2
    assert config["use_pair_stack"] is False
    assert config["trunk_attn_every"] == 6
    assert config["requested_num_workers"] == 16
    assert (
        hashlib.sha256(config_path.read_bytes()).hexdigest()
        == (manifest["training_provenance"]["saved_config_sha256"])
    )


def test_fasta_and_feature_contract_without_loading_esmc(tmp_path: Path) -> None:
    fasta = tmp_path / "input.fasta"
    fasta.write_text(">example one\nACDEFGHIKL\n")

    assert read_fasta(fasta) == [("example", "ACDEFGHIKL")]
    example = sequence_example("ACDEFGHIKL", torch.zeros(10, 2560))
    assert example.seq_len == 10
    assert tuple(example.esm.shape) == (10, 2560)
    assert example.is_nterm.tolist() == [True] + [False] * 9
    assert example.is_cterm.tolist() == [False] * 9 + [True]

    invalid = tmp_path / "invalid.fasta"
    invalid.write_text(">bad\nACDEFGHIKX\n")
    with pytest.raises(ValueError, match="non-standard"):
        read_fasta(invalid)
    with pytest.raises(ValueError, match="expected ESMC-6B embeddings"):
        sequence_example("ACDEFGHIKL", torch.zeros(10, 1536))


def test_verifier_rejects_placeholder_then_accepts_ready_manifest(tmp_path: Path) -> None:
    artifact = tmp_path / "mambafold-esmc6b-170k-ema.pt"
    artifact.write_bytes(b"small CPU-only verifier fixture")

    manifest = json.loads((PROJECT / "manifest.json").read_text())
    manifest["checkpoint"]["bytes"] = "PLACEHOLDER_AFTER_EMA_EXPORT"
    manifest["checkpoint"]["sha256"] = "PLACEHOLDER_AFTER_EMA_EXPORT"
    placeholder_manifest = tmp_path / "placeholder-manifest.json"
    placeholder_manifest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="placeholders"):
        verify_checkpoint(artifact, placeholder_manifest)

    manifest["checkpoint"]["bytes"] = artifact.stat().st_size
    manifest["checkpoint"]["sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest["checkpoint"]["integrity_status"] = "verified_for_test"
    ready_manifest = tmp_path / "manifest.json"
    ready_manifest.write_text(json.dumps(manifest))

    verified = verify_checkpoint(artifact, ready_manifest)
    assert verified["project_id"] == "mambafold-esmc6b-170k"

    artifact.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="size mismatch"):
        verify_checkpoint(artifact, ready_manifest)
