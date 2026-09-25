"""Portable pLDDT rollout storage and batching contracts."""

from __future__ import annotations

import bisect
import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

import mambafold.confidence.rollout as rollout_module
import scripts.train_plddt as train_plddt_module
from mambafold.confidence import (
    PLDDTHead,
    PLDDTHeadConfig,
    load_plddt_checkpoint,
    macro_per_protein_cross_entropy,
    soft_adjacent_bin_labels,
)
from mambafold.confidence.rollout import (
    ROLLOUT_SHARD_SCHEMA,
    PLDDTRolloutDataset,
    ShardGroupedSampler,
    collate_plddt_rollouts,
    sequence_sha256,
    write_rollout_manifest,
    write_rollout_shard,
)
from mambafold.data.dataset import RCSBDataset
from scripts.generate_plddt_rollouts import (
    _dataset_target,
    _load_deterministic_example,
    _polymer_geometry_quality,
)
from scripts.train_plddt import (
    _rollout_contract_and_split_inventory,
    _run_rank0_checked,
)


def _record(target_id: str, length: int, width: int = 4, seed: int = 0):
    target_mask = torch.ones(length, dtype=torch.bool)
    if length:
        target_mask[-1] = False
    target = torch.linspace(0.1, 0.9, length)
    target[~target_mask] = 0.0
    return {
        "target_id": target_id,
        "sequence_sha256": sequence_sha256("A" * length),
        "seed": seed,
        "trunk_latent": torch.randn(length, width),
        "pred_ca_A": torch.randn(length, 3),
        "true_ca_A": torch.randn(length, 3),
        "plddt_target": target,
        "target_mask": target_mask,
    }


def _manifest(tmp_path: Path, shard_specs: list[list[dict]], *, n_steps: int = 2):
    tmp_path.mkdir(parents=True, exist_ok=True)
    entries = []
    for index, records in enumerate(shard_specs):
        entries.append(write_rollout_shard(tmp_path / f"shard-{index}.pt", records))
    manifest_path = tmp_path / "manifest.json"
    write_rollout_manifest(
        manifest_path,
        shards=entries,
        provenance={
            "checkpoint": {
                "basename": "fold.pt",
                "sha256": "a" * 64,
                "step": 170000,
                "weights": "ema",
            },
            "config": {
                "sampler": "sde",
                "n_steps": n_steps,
                "sde_tau": 0.01,
                "sde_eps": 0.01,
                "sde_w_cutoff": 0.99,
                "sde_log_timesteps": True,
                "geometry_guidance": None,
                "label": "hard_lDDT-Ca",
                "coordinate_unit": "Angstrom",
            },
            "conditioning": {
                "model": "biohub/ESMC-6B",
                "revision": "45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a",
                "embedding_dimensions": 2560,
            },
            "split": {"file_list_basename": "split.txt", "file_list_sha256": "b" * 64},
        },
        selection={"rank": 0, "world_size": 1, "start_index": 0, "end_index": 2},
        sequence_sha256s=[
            record["sequence_sha256"] for records in shard_specs for record in records
        ],
    )
    return manifest_path


def test_rollout_roundtrip_is_weights_only_and_collates_variable_lengths(tmp_path):
    first = _record("target-a", 3, seed=11)
    second = _record("target-b", 5, seed=12)
    manifest_path = _manifest(tmp_path, [[first, second]])

    raw = torch.load(tmp_path / "shard-0.pt", map_location="cpu", weights_only=True)
    assert raw["schema"] == ROLLOUT_SHARD_SCHEMA
    assert raw["records"][0]["trunk_latent"].dtype == torch.bfloat16

    dataset = PLDDTRolloutDataset([manifest_path], verify_hashes=True)
    batch = collate_plddt_rollouts([dataset[0], dataset[1]])
    assert len(dataset) == 2
    assert batch["trunk_latent"].shape == (2, 5, 4)
    assert batch["trunk_latent"].dtype == torch.bfloat16
    assert batch["residue_mask"].tolist() == [
        [True, True, True, False, False],
        [True, True, True, True, True],
    ]
    assert not batch["target_mask"][0, 3:].any()
    assert batch["pred_ca_A"].dtype == torch.float32


def test_writer_refuses_overwrite_and_private_manifest_path(tmp_path):
    shard_path = tmp_path / "shard.pt"
    write_rollout_shard(shard_path, [_record("target-a", 3)])
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_rollout_shard(shard_path, [_record("target-a", 3)])

    with pytest.raises(ValueError, match="absolute path"):
        write_rollout_manifest(
            tmp_path / "bad-manifest.json",
            shards=[
                {
                    "path": shard_path.name,
                    "sha256": rollout_module.sha256_file(shard_path),
                    "bytes": shard_path.stat().st_size,
                    "records": 1,
                    "sequence_sha256s": [sequence_sha256("AAA")],
                }
            ],
            provenance={"local_checkpoint": str(shard_path.resolve())},
            selection={"rank": 0},
            sequence_sha256s=[sequence_sha256("AAA")],
        )


def test_invalid_target_outside_mask_is_rejected(tmp_path):
    record = _record("target-a", 3)
    record["plddt_target"][-1] = 0.5
    with pytest.raises(ValueError, match="zero outside target_mask"):
        write_rollout_shard(tmp_path / "bad.pt", [record])


def test_manifest_hash_detects_modified_shard(tmp_path):
    manifest_path = _manifest(tmp_path, [[_record("target-a", 3)]])
    shard_path = tmp_path / "shard-0.pt"
    with shard_path.open("r+b") as handle:
        handle.seek(-1, 2)
        value = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([value[0] ^ 1]))

    dataset = PLDDTRolloutDataset([manifest_path], verify_hashes=True)
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        _ = dataset[0]


def test_dataset_rejects_manifest_sequence_inventory_not_present_in_shard(tmp_path):
    manifest_path = _manifest(tmp_path, [[_record("target-a", 3)]])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    false_inventory = [sequence_sha256("CCCC")]
    manifest["sequence_sha256s"] = false_inventory
    manifest["shards"][0]["sequence_sha256s"] = false_inventory
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    dataset = PLDDTRolloutDataset([manifest_path], verify_hashes=True)
    with pytest.raises(ValueError, match="canonical-sequence inventory mismatch"):
        _ = dataset[0]


def test_shard_grouped_sampler_visits_all_records_with_grouped_access(tmp_path):
    manifest_path = _manifest(
        tmp_path,
        [
            [_record("a0", 3), _record("a1", 4)],
            [_record("b0", 3), _record("b1", 4)],
            [_record("c0", 3), _record("c1", 4)],
        ],
    )
    dataset = PLDDTRolloutDataset([manifest_path])
    sampler = ShardGroupedSampler(dataset, shuffle=True, seed=17)
    indices = list(sampler)

    assert sorted(indices) == list(range(len(dataset)))
    shard_ids = [bisect.bisect_right(dataset.cumulative_records, index) for index in indices]
    transitions = sum(left != right for left, right in zip(shard_ids, shard_ids[1:]))
    assert transitions == len(dataset.shards) - 1

    validation_parts = [
        list(
            ShardGroupedSampler(
                dataset,
                shuffle=False,
                rank=rank,
                world_size=4,
                even_divisible=False,
            )
        )
        for rank in range(4)
    ]
    assert sorted(index for part in validation_parts for index in part) == list(range(len(dataset)))


def test_evicted_shard_is_hashed_only_on_first_load(tmp_path, monkeypatch):
    manifest_path = _manifest(
        tmp_path,
        [[_record("a", 3)], [_record("b", 3)]],
    )
    real_sha256 = rollout_module.sha256_file
    calls = []

    def counting_sha256(path, *args, **kwargs):
        calls.append(Path(path).name)
        return real_sha256(path, *args, **kwargs)

    monkeypatch.setattr(rollout_module, "sha256_file", counting_sha256)
    dataset = PLDDTRolloutDataset([manifest_path], cache_size=1)
    _ = dataset[0]
    _ = dataset[1]
    _ = dataset[0]
    assert calls.count("shard-0.pt") == 1
    assert calls.count("shard-1.pt") == 1


class _RandomCropDataset:
    def __init__(self, path: Path):
        self.files = [path]

    def __getitem__(self, index):
        del index
        return int(torch.randint(0, 1_000_000, (1,)).item())


class _RandomCropChainDataset(_RandomCropDataset):
    def __init__(self, path: Path):
        super().__init__(path)
        self.extract_monomer_chains = True
        self.chain_index = [(0, 3, 17)]


def test_polymer_geometry_quality_separates_valid_and_broken_traces():
    valid = torch.zeros(8, 3)
    valid[:, 0] = torch.arange(8) * 3.8
    median, fraction = _polymer_geometry_quality(valid)
    assert median == pytest.approx(3.8)
    assert fraction == 1.0

    broken = valid.clone()
    broken[::2, 0] += 20.0
    _, broken_fraction = _polymer_geometry_quality(broken)
    assert broken_fraction < 0.8


def test_deterministic_crop_is_independent_of_caller_rng(tmp_path):
    dataset = _RandomCropDataset(tmp_path / "nested" / "target.npz")
    torch.manual_seed(1)
    _, first = _load_deterministic_example(dataset, 0, crop_seed=9, data_dir=tmp_path)
    torch.manual_seed(999)
    _, second = _load_deterministic_example(dataset, 0, crop_seed=9, data_dir=tmp_path)
    assert first == second


def test_chain_target_uses_file_index_and_chain_origin_in_crop_identity(tmp_path):
    dataset = _RandomCropChainDataset(tmp_path / "nested" / "target.npz")
    path, origin = _dataset_target(dataset, 0)
    assert path == dataset.files[0]
    assert origin == 3
    torch.manual_seed(1)
    _, chain_value = _load_deterministic_example(dataset, 0, crop_seed=9, data_dir=tmp_path)

    entry_dataset = _RandomCropDataset(dataset.files[0])
    torch.manual_seed(1)
    _, entry_value = _load_deterministic_example(
        entry_dataset, 0, crop_seed=9, data_dir=tmp_path
    )
    assert chain_value != entry_value


def test_explicit_chain_list_preserves_target_order(tmp_path):
    chain_list = tmp_path / "chains.tsv"
    chain_list.write_text("b.npz\t2\t81\na.npz\t0\t42\n", encoding="utf-8")
    dataset = RCSBDataset(str(tmp_path), chain_list=str(chain_list))

    assert len(dataset) == 2
    assert _dataset_target(dataset, 0) == (tmp_path / "b.npz", 2)
    assert _dataset_target(dataset, 1) == (tmp_path / "a.npz", 0)


def test_collated_rollouts_complete_one_confidence_head_update():
    batch = collate_plddt_rollouts([_record("short", 3), _record("long", 5)])
    config = PLDDTHeadConfig(
        d_model=4,
        n_bins=10,
        n_layers=1,
        n_heads=2,
        ff_mult=2,
        dropout=0.0,
    )
    head = PLDDTHead(config)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
    labels = soft_adjacent_bin_labels(
        batch["plddt_target"],
        batch["target_mask"],
        config.n_bins,
    )

    logits = head(batch["trunk_latent"].float(), batch["residue_mask"])
    loss = macro_per_protein_cross_entropy(logits, labels, batch["target_mask"])
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in head.parameters())


def test_trainer_rejects_train_validation_sequence_overlap(tmp_path):
    train_manifest = _manifest(tmp_path / "train", [[_record("train", 3)]])
    val_manifest = _manifest(tmp_path / "val", [[_record("val", 3)]])

    with pytest.raises(ValueError, match="canonical sequence overlap"):
        _rollout_contract_and_split_inventory([train_manifest], [val_manifest])


def test_trainer_rejects_mixed_sampler_contracts(tmp_path):
    train_manifest = _manifest(tmp_path / "train", [[_record("train", 3)]], n_steps=500)
    val_manifest = _manifest(tmp_path / "val", [[_record("val", 4)]], n_steps=50)

    with pytest.raises(ValueError, match="mixed folding/sampler"):
        _rollout_contract_and_split_inventory([train_manifest], [val_manifest])


def test_trainer_contract_embeds_split_hashes(tmp_path):
    train_manifest = _manifest(tmp_path / "train", [[_record("train", 3)]], n_steps=500)
    val_manifest = _manifest(tmp_path / "val", [[_record("val", 4)]], n_steps=500)

    contract = _rollout_contract_and_split_inventory([train_manifest], [val_manifest])

    assert contract["folding_checkpoint"]["sha256"] == "a" * 64
    assert contract["conditioning"]["embedding_dimensions"] == 2560
    assert contract["n_steps"] == 500
    assert contract["split_manifests"]["train"][0]["sha256"] == rollout_module.sha256_file(
        train_manifest
    )


def test_rank0_checked_mutation_reports_exclusive_create_failure(tmp_path):
    occupied = tmp_path / "occupied"
    occupied.mkdir()

    with pytest.raises(RuntimeError, match="FileExistsError"):
        _run_rank0_checked(
            lambda: occupied.mkdir(exist_ok=False),
            rank=0,
            world_size=1,
            description="create output",
        )


def test_train_script_one_step_cpu_writes_portable_artifacts(
    tmp_path,
    monkeypatch,
):
    train_manifest = _manifest(tmp_path / "train", [[_record("train", 3)]], n_steps=500)
    val_manifest = _manifest(tmp_path / "val", [[_record("val", 4)]], n_steps=500)
    out_dir = tmp_path / "trained"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "head": {
                    "d_model": 4,
                    "n_bins": 8,
                    "n_layers": 1,
                    "n_heads": 2,
                    "ff_mult": 2,
                    "dropout": 0.0,
                },
                "data": {
                    "train_manifests": [str(train_manifest)],
                    "val_manifests": [str(val_manifest)],
                    "verify_hashes": True,
                    "verify_eagerly": False,
                    "shard_cache_size": 1,
                },
                "training": {
                    "out_dir": str(out_dir),
                    "seed": 5,
                    "batch_size": 1,
                    "num_workers": 0,
                    "total_steps": 1,
                    "warmup_steps": 0,
                    "lr": 1e-3,
                    "weight_decay": 0.0,
                    "grad_clip": 1.0,
                    "amp": False,
                    "amp_dtype": "bf16",
                    "log_interval": 1,
                    "eval_interval": 1,
                    "ckpt_interval": 1,
                    "ece_bins": 4,
                    "max_spearman_residues": 100,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_plddt.py", "--config", str(config_path)],
    )

    train_plddt_module.main()

    assert (out_dir / "run_config.json").is_file()
    assert (out_dir / "plddt_head_best.pt").is_file()
    latest = out_dir / "plddt_head_latest.pt"
    head, temperature, metadata = load_plddt_checkpoint(latest)
    assert head.config.d_model == 4
    assert temperature == 1.0
    assert metadata["step"] == 1
    assert metadata["provenance"]["rollout_contract"]["n_steps"] == 500

    with pytest.raises(RuntimeError, match="fresh pLDDT output directory"):
        train_plddt_module.main()
