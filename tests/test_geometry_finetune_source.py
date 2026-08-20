from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from scripts.validate_geometry_finetune_source import (
    ARCHITECTURE_KEYS,
    DATA_KEYS,
    SourceValidationError,
    validate_geometry_finetune_source,
)

EXPECTED_STEP = 4


def _base_config() -> dict:
    values = {
        "max_length": 16,
        "d_res": 8,
        "n_trunk": 2,
        "d_res_type": 4,
        "d_res_pos": 4,
        "d_plm": 12,
        "d_plm_proj": 4,
        "d_ca_emb": 4,
        "d_state": 2,
        "mimo_rank": 1,
        "expand": 2,
        "headdim": 2,
        "bimamba_share": False,
        "trunk_attn_layers": None,
        "trunk_attn_every": 2,
        "n_attn_heads": 2,
        "trunk_time_film": True,
        "trunk_adaln_zero": True,
        "use_pair_stack": False,
        "d_pair": 4,
        "n_pair_blocks": 0,
        "n_pair_heads": 1,
        "pair_mult_c": 2,
        "pair_use_cueq": False,
        "d_atom": 4,
        "n_atom_layers": 1,
        "use_plm": True,
        "data_dir": "data/train",
        "val_data_dir": "data/val",
        "file_list": "splits/train.txt",
        "val_file_list": "splits/val.txt",
        "train_sources": [
            {
                "name": "rcsb",
                "data_dir": "data/train",
                "file_list": "splits/train.txt",
                "esm_dir": "data/esm",
            }
        ],
        "single_chain_only": True,
        "extract_monomer_chains": True,
        "esm_dir": "data/esm",
        "copies_per_protein": 1,
        "length_bin": 8,
        "length_balanced_sampling": True,
        "metadata_path": "splits/metadata.tsv",
        "length_balance_mode": "power",
        "length_balance_exponent": 0.5,
        "length_balance_clip_min": 1.0,
        "length_balance_clip_max": 1.5,
        "length_bucketing": True,
        "crop_schedule": [{"weights": {16: 1.0}}],
        "total_steps": EXPECTED_STEP,
        "batch_size": 2,
        "grad_accum_steps": 3,
        "seed": 7,
    }
    assert set(ARCHITECTURE_KEYS) <= set(values)
    assert set(DATA_KEYS) <= set(values)
    return values


def _write_case(tmp_path: Path) -> dict[str, Path | dict]:
    source_config = _base_config()
    finetune_config = dict(source_config)
    finetune_config.update({"total_steps": 2, "lr": 1e-5, "w_ost_clash": 0.5})

    source_config_path = tmp_path / "source.yaml"
    finetune_config_path = tmp_path / "finetune.yaml"
    source_config_path.write_text(yaml.safe_dump(source_config), encoding="utf-8")
    finetune_config_path.write_text(yaml.safe_dump(finetune_config), encoding="utf-8")

    checkpoint_args = dict(source_config)
    checkpoint_args.update(
        {
            "config": source_config_path.name,
            "out_dir": str(tmp_path),
        }
    )
    checkpoint = {
        "checkpoint_version": 2,
        "step": EXPECTED_STEP,
        "model": {
            "linear.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
            "linear.bias": torch.ones(2),
        },
        "ema": {
            "linear.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
            "linear.bias": torch.ones(2),
        },
        "optimizer": {"state": {}, "param_groups": [{"lr": 1e-4}]},
        "scheduler": {
            "base_lrs": [1e-4],
            "last_epoch": EXPECTED_STEP,
            "_step_count": EXPECTED_STEP + 1,
            "_last_lr": [5e-5],
        },
        "args": checkpoint_args,
        "wandb_run_id": "test-run",
        "rng_states": [{"rank": 0}],
        "data_state": {
            "micro_batches_consumed": EXPECTED_STEP * source_config["grad_accum_steps"],
            "world_size": 1,
            "batch_size": source_config["batch_size"],
            "grad_accum_steps": source_config["grad_accum_steps"],
            "batches_per_epoch": 5,
            "dataset_size": 10,
            "sampler_type": "TestSampler",
            "seed": source_config["seed"],
        },
    }
    checkpoint_path = tmp_path / f"ckpt_{EXPECTED_STEP:07d}.pt"
    torch.save(checkpoint, checkpoint_path)
    latest = tmp_path / "ckpt_latest.pt"
    latest.symlink_to(checkpoint_path.name)
    code_path = tmp_path / "train.py"
    code_path.write_text("VALUE = 1\n", encoding="utf-8")
    provenance = tmp_path / "run" / "source_provenance.json"
    return {
        "checkpoint": checkpoint,
        "checkpoint_path": checkpoint_path,
        "latest": latest,
        "source_config": source_config,
        "source_config_path": source_config_path,
        "finetune_config": finetune_config,
        "finetune_config_path": finetune_config_path,
        "code_path": code_path,
        "provenance": provenance,
    }


def _validate(case: dict[str, Path | dict]) -> dict:
    return validate_geometry_finetune_source(
        checkpoint=case["checkpoint_path"],
        latest=case["latest"],
        source_config=case["source_config_path"],
        finetune_config=case["finetune_config_path"],
        expected_step=EXPECTED_STEP,
        expected_world_size=1,
        provenance_out=case["provenance"],
        repo_root=Path(case["checkpoint_path"]).parent,
        code_paths=[case["code_path"]],
    )


def _resave(case: dict[str, Path | dict]) -> None:
    torch.save(case["checkpoint"], case["checkpoint_path"])


def test_valid_source_writes_atomic_provenance(tmp_path):
    case = _write_case(tmp_path)
    payload = _validate(case)

    provenance_path = Path(case["provenance"])
    written = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert written == payload
    assert written["source"]["step"] == EXPECTED_STEP
    assert (
        written["source"]["checkpoint_sha256"]
        == hashlib.sha256(Path(case["checkpoint_path"]).read_bytes()).hexdigest()
    )
    assert written["finetune"]["initial_weights"] == "ema"
    assert written["finetune"]["optimizer_scheduler"] == "fresh"
    assert written["contract"]["model_ema"]["state_key_count"] == 2
    assert written["code"]["files"] == {"train.py": hashlib.sha256(b"VALUE = 1\n").hexdigest()}
    assert not list(provenance_path.parent.glob("*.tmp"))

    with pytest.raises(SourceValidationError, match="refusing to overwrite"):
        _validate(case)


def test_rejects_latest_symlink_to_another_checkpoint(tmp_path):
    case = _write_case(tmp_path)
    other = tmp_path / "ckpt_0000003.pt"
    torch.save(case["checkpoint"], other)
    Path(case["latest"]).unlink()
    Path(case["latest"]).symlink_to(other.name)

    with pytest.raises(SourceValidationError, match="latest checkpoint resolves"):
        _validate(case)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(checkpoint_version=1), "checkpoint_version"),
        (lambda value: value.update(step=3), "checkpoint step"),
        (
            lambda value: value["scheduler"].update(last_epoch=3),
            "scheduler last_epoch",
        ),
        (
            lambda value: value["data_state"].update(world_size=2),
            "data_state world_size",
        ),
    ],
)
def test_rejects_incomplete_training_state(tmp_path, mutation, message):
    case = _write_case(tmp_path)
    mutation(case["checkpoint"])
    _resave(case)

    with pytest.raises(SourceValidationError, match=message):
        _validate(case)


def test_rejects_checkpoint_source_architecture_mismatch(tmp_path):
    case = _write_case(tmp_path)
    case["checkpoint"]["args"]["d_state"] = 4
    _resave(case)

    with pytest.raises(SourceValidationError, match="checkpoint/source architecture"):
        _validate(case)


def test_rejects_finetune_data_mismatch(tmp_path):
    case = _write_case(tmp_path)
    config = dict(case["finetune_config"])
    config["file_list"] = "splits/other.txt"
    Path(case["finetune_config_path"]).write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(SourceValidationError, match="fine-tune/source data"):
        _validate(case)


def test_rejects_model_ema_shape_mismatch(tmp_path):
    case = _write_case(tmp_path)
    case["checkpoint"]["ema"]["linear.weight"] = torch.ones(3, 2)
    _resave(case)

    with pytest.raises(SourceValidationError, match="model/EMA shape mismatch"):
        _validate(case)
