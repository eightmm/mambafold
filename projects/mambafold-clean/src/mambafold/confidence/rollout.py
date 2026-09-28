"""Portable, integrity-checked storage for offline pLDDT rollouts.

Rollout shards intentionally contain only primitive Python containers and CPU
tensors.  They can therefore be loaded with ``torch.load(...,
weights_only=True)`` and do not depend on pickled project classes.  Manifests
contain relative shard paths and content hashes; dataset/checkpoint locations
are never serialized into a public artifact.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.utils.data import Dataset, Sampler

ROLLOUT_SHARD_SCHEMA = "mambafold.plddt-rollout-shard.v1"
ROLLOUT_MANIFEST_SCHEMA = "mambafold.plddt-rollout-manifest.v1"

_RECORD_KEYS = {
    "target_id",
    "sequence_sha256",
    "seed",
    "trunk_latent",
    "pred_ca_A",
    "true_ca_A",
    "plddt_target",
    "target_mask",
}


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Return the lowercase SHA-256 digest of a file without loading it at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_sha256(sequence: str) -> str:
    """Hash an uppercase canonical amino-acid sequence."""
    canonical = sequence.strip().upper()
    alphabet = frozenset("ACDEFGHIKLMNPQRSTVWY")
    if not canonical or any(aa not in alphabet for aa in canonical):
        raise ValueError("sequence must contain only the 20 canonical amino acids")
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _plain_record(record: Mapping[str, Any]) -> dict[str, Any]:
    missing = sorted(_RECORD_KEYS.difference(record))
    unexpected = sorted(set(record).difference(_RECORD_KEYS))
    if missing or unexpected:
        raise ValueError(f"invalid rollout record keys: missing={missing}, unexpected={unexpected}")

    target_id = str(record["target_id"])
    sequence_hash = str(record["sequence_sha256"])
    seed = int(record["seed"])
    if not target_id or "/" in target_id or "\\" in target_id:
        raise ValueError("target_id must be a non-empty logical identifier, not a path")
    if len(sequence_hash) != 64 or any(c not in "0123456789abcdef" for c in sequence_hash):
        raise ValueError("sequence_sha256 must be a lowercase SHA-256 digest")

    latent_source = torch.as_tensor(record["trunk_latent"]).detach().to(device="cpu")
    if not torch.isfinite(latent_source).all():
        raise ValueError(
            f"trunk_latent contains NaN/Inf before serialization "
            f"(target_id={target_id!r}, seed={seed})"
        )
    # Folding inference runs in BF16, whose exponent range matches FP32. Keep
    # that range in the artifact: FP16 can turn valid activations into Inf.
    latent = latent_source.to(dtype=torch.bfloat16)
    pred_ca = torch.as_tensor(record["pred_ca_A"]).detach().to(device="cpu", dtype=torch.float32)
    true_ca = torch.as_tensor(record["true_ca_A"]).detach().to(device="cpu", dtype=torch.float32)
    target = torch.as_tensor(record["plddt_target"]).detach().to(device="cpu", dtype=torch.float32)
    target_mask = torch.as_tensor(record["target_mask"]).detach().to(device="cpu", dtype=torch.bool)

    if latent.ndim != 2:
        raise ValueError(f"trunk_latent must have shape [L,D], got {tuple(latent.shape)}")
    length = latent.shape[0]
    expected_shapes = {
        "pred_ca_A": (length, 3),
        "true_ca_A": (length, 3),
        "plddt_target": (length,),
        "target_mask": (length,),
    }
    actual_shapes = {
        "pred_ca_A": tuple(pred_ca.shape),
        "true_ca_A": tuple(true_ca.shape),
        "plddt_target": tuple(target.shape),
        "target_mask": tuple(target_mask.shape),
    }
    bad = {
        key: (actual_shapes[key], shape)
        for key, shape in expected_shapes.items()
        if actual_shapes[key] != shape
    }
    if bad:
        raise ValueError(f"rollout tensor shape mismatch (actual, expected): {bad}")
    if not torch.isfinite(latent).all():
        max_abs = float(latent_source.float().abs().max()) if latent_source.numel() else 0.0
        raise ValueError(
            f"trunk_latent overflowed during BF16 serialization "
            f"(target_id={target_id!r}, seed={seed}, max_abs={max_abs:.6g})"
        )
    if not torch.isfinite(pred_ca).all() or not torch.isfinite(true_ca).all():
        raise ValueError("CA coordinates contain NaN/Inf")
    if not torch.isfinite(target).all():
        raise ValueError("plddt_target contains NaN/Inf")
    if target.numel() and (target.min() < 0 or target.max() > 1):
        raise ValueError("pLDDT targets must lie in [0,1]")
    if torch.count_nonzero(target.masked_select(~target_mask)):
        raise ValueError("plddt_target must be zero outside target_mask")

    return {
        "target_id": target_id,
        "sequence_sha256": sequence_hash,
        "seed": seed,
        "trunk_latent": latent.contiguous(),
        "pred_ca_A": pred_ca.contiguous(),
        "true_ca_A": true_ca.contiguous(),
        "plddt_target": target.contiguous(),
        "target_mask": target_mask.contiguous(),
    }


def _atomic_no_clobber(path: Path, write_tmp) -> None:
    """Publish a complete file atomically and fail if the destination exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        write_tmp(tmp_path)
        with tmp_path.open("rb") as handle:
            os.fsync(handle.fileno())
        # A hard-link publish is atomic and, unlike os.replace, cannot clobber
        # a file concurrently created by another rollout rank.
        os.link(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def write_rollout_shard(path: str | Path, records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate and atomically save one safe rollout shard.

    Returns the portable manifest entry for this shard.  The caller must pass a
    path relative to its intended manifest directory to ``write_manifest``.
    """
    output_path = Path(path)
    plain_records = [_plain_record(record) for record in records]
    if not plain_records:
        raise ValueError("refusing to write an empty rollout shard")
    payload = {"schema": ROLLOUT_SHARD_SCHEMA, "records": plain_records}
    _atomic_no_clobber(output_path, lambda tmp: torch.save(payload, tmp))
    return {
        "path": output_path.name,
        "sha256": sha256_file(output_path),
        "bytes": output_path.stat().st_size,
        "records": len(plain_records),
        "sequence_sha256s": sorted({str(record["sequence_sha256"]) for record in plain_records}),
    }


def load_rollout_shard(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
) -> dict[str, Any]:
    """Load and validate a shard through PyTorch's restricted loader."""
    shard_path = Path(path)
    if expected_bytes is not None and shard_path.stat().st_size != int(expected_bytes):
        raise RuntimeError(f"rollout shard byte-size mismatch: {shard_path}")
    if expected_sha256 is not None and sha256_file(shard_path) != expected_sha256:
        raise RuntimeError(f"rollout shard SHA-256 mismatch: {shard_path}")
    payload = torch.load(shard_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != ROLLOUT_SHARD_SCHEMA:
        raise ValueError(f"unsupported rollout shard schema: {shard_path}")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"rollout shard has no records: {shard_path}")
    return {"schema": ROLLOUT_SHARD_SCHEMA, "records": [_plain_record(r) for r in records]}


def _assert_portable(value: Any, location: str = "manifest") -> None:
    """Reject absolute paths and non-JSON values from public provenance."""
    if value is None or isinstance(value, (bool, int, float)):
        return
    if isinstance(value, str):
        is_windows_absolute = len(value) >= 3 and value[1:3] in {":/", ":\\"}
        if Path(value).is_absolute() or value.startswith(("~/", "file://")) or is_windows_absolute:
            raise ValueError(f"absolute path is not portable at {location}: {value}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_portable(item, f"{location}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"manifest key at {location} is not a string")
            _assert_portable(item, f"{location}.{key}")
        return
    raise TypeError(f"non-JSON provenance value at {location}: {type(value).__name__}")


def write_rollout_manifest(
    path: str | Path,
    *,
    shards: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    selection: Mapping[str, Any],
    sequence_sha256s: Sequence[str],
) -> dict[str, Any]:
    """Atomically write a portable rollout manifest and return its payload."""
    if not shards:
        raise ValueError("manifest requires at least one shard")
    clean_shards: list[dict[str, Any]] = []
    shard_sequence_inventory: set[str] = set()
    for shard in shards:
        entry = {
            "path": str(shard["path"]),
            "sha256": str(shard["sha256"]),
            "bytes": int(shard["bytes"]),
            "records": int(shard["records"]),
            "sequence_sha256s": sorted(set(shard["sequence_sha256s"])),
        }
        if (
            Path(entry["path"]).is_absolute()
            or ".." in Path(entry["path"]).parts
            or "\\" in entry["path"]
        ):
            raise ValueError(f"shard path must be local to the manifest: {entry['path']}")
        shard_sequence_inventory.update(entry["sequence_sha256s"])
        clean_shards.append(entry)
    sequence_inventory = sorted(set(str(value) for value in sequence_sha256s))
    if not sequence_inventory:
        raise ValueError("manifest requires at least one canonical sequence SHA-256")
    for value in sequence_inventory:
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"invalid canonical sequence SHA-256: {value!r}")
    if sequence_inventory != sorted(shard_sequence_inventory):
        raise ValueError("manifest sequence inventory does not match its rollout shards")
    payload = {
        "schema": ROLLOUT_MANIFEST_SCHEMA,
        "provenance": dict(provenance),
        "selection": dict(selection),
        "num_shards": len(clean_shards),
        "num_records": sum(entry["records"] for entry in clean_shards),
        "num_sequences": len(sequence_inventory),
        "sequence_sha256s": sequence_inventory,
        "shards": clean_shards,
    }
    _assert_portable(payload)
    serialized = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    _atomic_no_clobber(Path(path), lambda tmp: tmp.write_text(serialized, encoding="utf-8"))
    return payload


def load_rollout_manifest(path: str | Path) -> dict[str, Any]:
    """Read and structurally validate one JSON rollout manifest."""
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != ROLLOUT_MANIFEST_SCHEMA:
        raise ValueError(f"unsupported rollout manifest schema: {manifest_path}")
    _assert_portable(payload)
    shards = payload.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"manifest has no shards: {manifest_path}")
    expected_records = 0
    shard_sequence_inventory: set[str] = set()
    for entry in shards:
        for key in ("path", "sha256", "bytes", "records", "sequence_sha256s"):
            if key not in entry:
                raise ValueError(f"manifest shard entry lacks {key}: {manifest_path}")
        shard_rel = Path(entry["path"])
        if shard_rel.is_absolute() or ".." in shard_rel.parts or "\\" in entry["path"]:
            raise ValueError(f"non-local shard path in manifest: {entry['path']}")
        shard_sequences = entry["sequence_sha256s"]
        if (
            not isinstance(shard_sequences, list)
            or not shard_sequences
            or shard_sequences != sorted(set(shard_sequences))
        ):
            raise ValueError(f"invalid shard sequence inventory: {manifest_path}")
        shard_sequence_inventory.update(shard_sequences)
        expected_records += int(entry["records"])
    if int(payload.get("num_shards", -1)) != len(shards):
        raise ValueError(f"manifest num_shards mismatch: {manifest_path}")
    if int(payload.get("num_records", -1)) != expected_records:
        raise ValueError(f"manifest num_records mismatch: {manifest_path}")
    sequence_inventory = payload.get("sequence_sha256s")
    if (
        not isinstance(sequence_inventory, list)
        or not sequence_inventory
        or sequence_inventory != sorted(set(sequence_inventory))
    ):
        raise ValueError(f"manifest sequence inventory must be sorted and unique: {manifest_path}")
    for value in sequence_inventory:
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError(f"manifest contains invalid sequence SHA-256: {manifest_path}")
    if int(payload.get("num_sequences", -1)) != len(sequence_inventory):
        raise ValueError(f"manifest num_sequences mismatch: {manifest_path}")
    if sequence_inventory != sorted(shard_sequence_inventory):
        raise ValueError(f"manifest/shard sequence inventory mismatch: {manifest_path}")
    return payload


def expand_manifest_paths(values: Sequence[str | Path]) -> list[Path]:
    """Expand explicit paths or shell-independent ``*`` manifest patterns."""
    expanded: list[Path] = []
    for value in values:
        path = Path(value)
        if any(char in str(path) for char in "*?["):
            matches = sorted(path.parent.glob(path.name))
            if not matches:
                raise FileNotFoundError(f"manifest pattern matched no files: {value}")
            expanded.extend(matches)
        else:
            expanded.append(path)
    if not expanded:
        raise ValueError("at least one rollout manifest is required")
    return expanded


class PLDDTRolloutDataset(Dataset):
    """Map-style dataset over one or more rollout manifests.

    Shards are loaded lazily with a small per-process LRU cache.  Integrity can
    be checked on every first load (default) or once eagerly at construction.
    """

    def __init__(
        self,
        manifests: Sequence[str | Path],
        *,
        verify_hashes: bool = True,
        verify_eagerly: bool = False,
        cache_size: int = 2,
    ):
        self.verify_hashes = bool(verify_hashes)
        self.cache_size = max(1, int(cache_size))
        self.shards: list[dict[str, Any]] = []
        self.cumulative_records: list[int] = []
        total = 0
        seen_paths: set[Path] = set()
        for manifest_path in expand_manifest_paths(manifests):
            manifest = load_rollout_manifest(manifest_path)
            for entry in manifest["shards"]:
                shard_path = (manifest_path.parent / entry["path"]).resolve()
                if shard_path in seen_paths:
                    raise ValueError(f"duplicate rollout shard across manifests: {shard_path}")
                seen_paths.add(shard_path)
                full_entry = dict(entry)
                full_entry["resolved_path"] = shard_path
                self.shards.append(full_entry)
                total += int(entry["records"])
                self.cumulative_records.append(total)
        if total == 0:
            raise ValueError("rollout manifests contain no records")
        self._cache: OrderedDict[int, list[dict[str, Any]]] = OrderedDict()
        self._verified_shards: set[int] = set()
        if verify_eagerly:
            for shard_index in range(len(self.shards)):
                self._load_records(shard_index)
            self._cache.clear()

    def __len__(self) -> int:
        return self.cumulative_records[-1]

    def _load_records(self, shard_index: int) -> list[dict[str, Any]]:
        if shard_index in self._cache:
            records = self._cache.pop(shard_index)
            self._cache[shard_index] = records
            return records
        entry = self.shards[shard_index]
        needs_integrity_check = self.verify_hashes and shard_index not in self._verified_shards
        payload = load_rollout_shard(
            entry["resolved_path"],
            expected_sha256=entry["sha256"] if needs_integrity_check else None,
            expected_bytes=entry["bytes"] if needs_integrity_check else None,
        )
        if needs_integrity_check:
            self._verified_shards.add(shard_index)
        records = payload["records"]
        if len(records) != int(entry["records"]):
            raise ValueError(f"shard record-count mismatch: {entry['resolved_path']}")
        actual_sequences = sorted({record["sequence_sha256"] for record in records})
        if actual_sequences != entry["sequence_sha256s"]:
            raise ValueError(
                f"shard canonical-sequence inventory mismatch: {entry['resolved_path']}"
            )
        self._cache[shard_index] = records
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return records

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard_index = bisect.bisect_right(self.cumulative_records, index)
        previous = 0 if shard_index == 0 else self.cumulative_records[shard_index - 1]
        return self._load_records(shard_index)[index - previous]


class ShardGroupedSampler(Sampler[int]):
    """Shuffle rollouts without turning shard loading into random I/O.

    Each epoch shuffles shard order and then record order within each shard.  A
    distributed run gives every rank one contiguous slice of that grouped
    order, padding like :class:`torch.utils.data.DistributedSampler` so all
    ranks execute the same number of optimizer steps.
    """

    def __init__(
        self,
        dataset: PLDDTRolloutDataset,
        *,
        shuffle: bool = True,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        even_divisible: bool = True,
    ):
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError(f"invalid sampler rank/world_size: {rank}/{world_size}")
        self.dataset = dataset
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.even_divisible = bool(even_divisible)
        self.epoch = 0
        if self.even_divisible:
            self.num_samples = (len(dataset) + world_size - 1) // world_size
            self.total_size = self.num_samples * world_size
        else:
            rank_start = len(dataset) * rank // world_size
            rank_end = len(dataset) * (rank + 1) // world_size
            self.num_samples = rank_end - rank_start
            self.total_size = len(dataset)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        shard_order = list(range(len(self.dataset.shards)))
        if self.shuffle:
            permutation = torch.randperm(len(shard_order), generator=generator).tolist()
            shard_order = [shard_order[index] for index in permutation]
        indices: list[int] = []
        for shard_index in shard_order:
            start = 0 if shard_index == 0 else self.dataset.cumulative_records[shard_index - 1]
            end = self.dataset.cumulative_records[shard_index]
            local = list(range(start, end))
            if self.shuffle and len(local) > 1:
                permutation = torch.randperm(len(local), generator=generator).tolist()
                local = [local[index] for index in permutation]
            indices.extend(local)
        if self.even_divisible:
            if len(indices) < self.total_size:
                padding = self.total_size - len(indices)
                repeats = (padding + len(indices) - 1) // len(indices)
                indices.extend((indices * repeats)[:padding])
            rank_start = self.rank * self.num_samples
            rank_end = rank_start + self.num_samples
        else:
            rank_start = len(indices) * self.rank // self.world_size
            rank_end = len(indices) * (self.rank + 1) // self.world_size
        rank_indices = indices[rank_start:rank_end]
        return iter(rank_indices)


def collate_plddt_rollouts(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Right-pad variable-length rollout records into a training batch."""
    if not records:
        raise ValueError("cannot collate an empty rollout batch")
    records = [_plain_record(record) for record in records]
    batch_size = len(records)
    max_length = max(record["trunk_latent"].shape[0] for record in records)
    d_model = records[0]["trunk_latent"].shape[1]
    if any(record["trunk_latent"].shape[1] != d_model for record in records):
        raise ValueError("all rollout records in a batch must share latent width")

    latent = torch.zeros(batch_size, max_length, d_model, dtype=torch.bfloat16)
    pred_ca = torch.zeros(batch_size, max_length, 3, dtype=torch.float32)
    true_ca = torch.zeros_like(pred_ca)
    target = torch.zeros(batch_size, max_length, dtype=torch.float32)
    target_mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
    residue_mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
    for row, record in enumerate(records):
        length = record["trunk_latent"].shape[0]
        latent[row, :length] = record["trunk_latent"]
        pred_ca[row, :length] = record["pred_ca_A"]
        true_ca[row, :length] = record["true_ca_A"]
        target[row, :length] = record["plddt_target"]
        target_mask[row, :length] = record["target_mask"]
        residue_mask[row, :length] = True
    return {
        "target_id": [record["target_id"] for record in records],
        "sequence_sha256": [record["sequence_sha256"] for record in records],
        "seed": torch.tensor([record["seed"] for record in records], dtype=torch.long),
        "trunk_latent": latent,
        "pred_ca_A": pred_ca,
        "true_ca_A": true_ca,
        "plddt_target": target,
        "target_mask": target_mask,
        "residue_mask": residue_mask,
    }


__all__ = [
    "PLDDTRolloutDataset",
    "ROLLOUT_MANIFEST_SCHEMA",
    "ROLLOUT_SHARD_SCHEMA",
    "ShardGroupedSampler",
    "collate_plddt_rollouts",
    "expand_manifest_paths",
    "load_rollout_manifest",
    "load_rollout_shard",
    "sequence_sha256",
    "sha256_file",
    "write_rollout_manifest",
    "write_rollout_shard",
]
