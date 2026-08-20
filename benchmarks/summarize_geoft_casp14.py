#!/usr/bin/env python3
"""Summarize the paired CASP14 geometry-fine-tune checkpoint comparison.

This module only consumes already-produced inference and scoring artifacts.  It
deliberately fails closed on checkpoint/config provenance, sampling settings,
score coverage, and target identity before comparing any metrics.  CASP14 has
already informed development in this project, so the resulting decision is
retrospective engineering evidence rather than an untouched test-set claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

BASE = "base170k"
CONDITIONS = (BASE, "ft250", "ft500", "ft1000", "ft1500", "ft2000")
EXPECTED_TARGETS = 70
DEFAULT_BOOTSTRAP = 20_000
DEFAULT_SEED = 20260818
_EPS = 1e-12

BASE_CONFIG = "direct_allatom_puremamba_attn6_geo_adaln_sf360_esmc6b_gpu8.yaml"
FINETUNE_CONFIG = "direct_allatom_puremamba_attn6_geo_adaln_sf360_esmc6b_gpu8_geoft.yaml"


@dataclass(frozen=True)
class ConditionSpec:
    checkpoint_step: int
    checkpoint_name: str
    config_name: str
    source_provenance_required: bool


SPECS = {
    BASE: ConditionSpec(170_000, "ckpt_0170000.pt", BASE_CONFIG, False),
    "ft250": ConditionSpec(250, "ckpt_0000250.pt", FINETUNE_CONFIG, True),
    "ft500": ConditionSpec(500, "ckpt_0000500.pt", FINETUNE_CONFIG, True),
    "ft1000": ConditionSpec(1_000, "ckpt_0001000.pt", FINETUNE_CONFIG, True),
    "ft1500": ConditionSpec(1_500, "ckpt_0001500.pt", FINETUNE_CONFIG, True),
    "ft2000": ConditionSpec(2_000, "ckpt_0002000.pt", FINETUNE_CONFIG, True),
}

METRICS = (
    "gdt_ts",
    "all_atom_lddt",
    "bb_lddt",
    "tm_score",
    "ost_model_clashes_per_1k_atoms",
    "local_hard_clashes_per_1k_atoms",
    "bond_p95_A",
    "ca_crossings",
)
CONTRACT = {
    "dataset": "CASP14 whole70 exact70",
    "expected_target_count": EXPECTED_TARGETS,
    "paired_conditions": list(CONDITIONS),
    "inference": {
        "sampler": "sde",
        "n_steps": 500,
        "seed": 0,
        "use_ema": True,
        "geometry_guidance_preset": "bond_cleanup",
        "guidance_off": True,
        "inactive_channel_scales": {
            "scale": 0.0,
            "steric_scale": 0.0,
            "vdw_scale": 0.0,
        },
    },
    "promotion_gate": {
        "ost_model_clash_reduction_fraction_min": 0.50,
        "mean_gdt_ts_delta_min": -0.005,
        "mean_all_atom_lddt_delta_min": -0.005,
        "selection": "earliest accuracy-preserving fine-tune checkpoint",
        "failure": "no_promotion",
    },
    "evidence_role": (
        "retrospective development evidence; CASP14 is not an untouched confirmatory test set"
    ),
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing required artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON artifact: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    except FileNotFoundError as exc:
        raise ValueError(f"missing provenance artifact: {path}") from exc
    return digest.hexdigest()


def _resolve_path(root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}: expected a non-empty path string")
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    root_relative = (root / path).resolve()
    if root_relative.exists():
        return root_relative
    return (Path.cwd() / path).resolve()


def _cached_sha256(path: Path, cache: dict[Path, str]) -> str:
    digest = cache.get(path)
    if digest is None:
        digest = _sha256(path)
        cache[path] = digest
    return digest


def _finite(value: Any, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: expected a finite number, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label}: expected a finite number, got {value!r}")
    return number


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label}: expected a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: expected a positive integer") from exc
    if number <= 0 or number != _finite(value, label=label):
        raise ValueError(f"{label}: expected a positive integer")
    return number


def _index_rows(rows: Any, *, key: str, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError(f"{label}: rows must be a list")
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or key not in row:
            raise ValueError(f"{label}: every row must contain {key!r}")
        target = str(row[key]).strip().lower()
        if not target:
            raise ValueError(f"{label}: empty target ID")
        if target in indexed:
            raise ValueError(f"{label}: duplicate target {target}")
        indexed[target] = row
    return indexed


def _same_file(left: Path, right: Path) -> bool:
    try:
        return left.samefile(right)
    except (FileNotFoundError, OSError):
        return left.resolve() == right.resolve()


def _load_ids(root: Path, manifests: dict[str, dict[str, Any]]) -> list[str]:
    values = [manifests[condition].get("ids_file") for condition in CONDITIONS]
    if any(value != values[0] for value in values[1:]):
        raise ValueError("condition manifest mismatch for ids_file")
    ids_path = _resolve_path(root, values[0], label="ids_file")
    try:
        ids = [token.strip().lower() for token in ids_path.read_text().split() if token.strip()]
    except FileNotFoundError as exc:
        raise ValueError(f"missing IDs file: {ids_path}") from exc
    if len(ids) != EXPECTED_TARGETS or len(set(ids)) != EXPECTED_TARGETS:
        raise ValueError(
            f"IDs file must contain exactly {EXPECTED_TARGETS} unique targets, got {len(ids)}"
        )
    return sorted(ids)


def _validate_manifest(
    root: Path,
    condition: str,
    *,
    hash_cache: dict[Path, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = root / "conditions" / condition / "manifest.json"
    manifest = _read_json(manifest_path)
    spec = SPECS[condition]

    if manifest.get("condition") != condition:
        raise ValueError(f"{condition}: manifest condition must be exactly {condition!r}")
    if manifest.get("checkpoint_step") != spec.checkpoint_step:
        raise ValueError(f"{condition}: checkpoint_step must be exactly {spec.checkpoint_step}")

    checkpoint = _resolve_path(root, manifest.get("checkpoint"), label=f"{condition} checkpoint")
    if checkpoint.name != spec.checkpoint_name:
        raise ValueError(
            f"{condition}: checkpoint must be exactly {spec.checkpoint_name}, got {checkpoint.name}"
        )
    actual_checkpoint_sha = _cached_sha256(checkpoint, hash_cache)
    if manifest.get("checkpoint_sha256") != actual_checkpoint_sha:
        raise ValueError(f"{condition}: checkpoint SHA-256 mismatch")

    config = _resolve_path(root, manifest.get("config"), label=f"{condition} config")
    if config.name != spec.config_name:
        raise ValueError(
            f"{condition}: config must be exactly {spec.config_name}, got {config.name}"
        )
    actual_config_sha = _cached_sha256(config, hash_cache)
    if manifest.get("config_sha256") != actual_config_sha:
        raise ValueError(f"{condition}: config SHA-256 mismatch")

    provenance_path: Path | None = None
    provenance_sha: str | None = None
    if spec.source_provenance_required:
        provenance_path = _resolve_path(
            root,
            manifest.get("source_provenance"),
            label=f"{condition} source_provenance",
        )
        provenance_sha = _cached_sha256(provenance_path, hash_cache)
        if manifest.get("source_provenance_sha256") != provenance_sha:
            raise ValueError(f"{condition}: source provenance SHA-256 mismatch")
    elif (
        "source_provenance" not in manifest
        or "source_provenance_sha256" not in manifest
        or manifest["source_provenance"] is not None
        or manifest["source_provenance_sha256"] is not None
    ):
        raise ValueError(
            f"{condition}: base source_provenance and its SHA must both be explicit null"
        )

    exact_settings = {
        "dataset": "CASP14 whole70 exact70",
        "sampler": "sde",
        "n_steps": 500,
        "seed": 0,
        "seeds": [0],
        "sde_tau": 0.01,
        "sde_eps": 0.01,
        "sde_w_cutoff": 0.99,
        "sde_log_timesteps": True,
        "max_length": 1024,
        "output_format": "both",
        "use_ema": True,
        "single_chain_only": True,
        "n_predicted": EXPECTED_TARGETS,
        "expected_target_count": EXPECTED_TARGETS,
    }
    for key, expected in exact_settings.items():
        if manifest.get(key) != expected:
            raise ValueError(f"{condition}: {key} must be exactly {expected!r}")
    if manifest.get("geometry_guidance_preset") != "bond_cleanup":
        raise ValueError(f"{condition}: geometry_guidance_preset must be exactly 'bond_cleanup'")
    if manifest.get("guidance_off") is not True:
        raise ValueError(f"{condition}: guidance_off must be derived and exactly true")
    guidance = manifest.get("geometry_guidance")
    if not isinstance(guidance, dict):
        raise ValueError(f"{condition}: missing explicit geometry_guidance object")
    for channel in ("scale", "steric_scale", "vdw_scale"):
        if (
            channel not in guidance
            or _finite(guidance[channel], label=f"{condition}.geometry_guidance.{channel}") != 0.0
        ):
            raise ValueError(f"{condition}: geometry_guidance.{channel} must be exactly zero")

    manifest_rows = _index_rows(manifest.get("rows"), key="pdb_id", label=f"{condition} manifest")
    if len(manifest_rows) != EXPECTED_TARGETS:
        raise ValueError(
            f"{condition}: manifest coverage must be exactly {EXPECTED_TARGETS}, "
            f"got {len(manifest_rows)}"
        )
    for target, row in manifest_rows.items():
        if _positive_int(row.get("L"), label=f"{condition}/{target} length") > 1024:
            raise ValueError(f"{condition}/{target}: length exceeds 1024")
        if row.get("n_chains") != 1 or row.get("n_seeds_ok") != 1:
            raise ValueError(f"{condition}/{target}: expected one chain and one seed")

    metadata = {
        "manifest": str(manifest_path.resolve()),
        "checkpoint": str(checkpoint),
        "checkpoint_step": spec.checkpoint_step,
        "checkpoint_sha256": actual_checkpoint_sha,
        "config": str(config),
        "config_sha256": actual_config_sha,
        "source_provenance": None if provenance_path is None else str(provenance_path),
        "source_provenance_sha256": provenance_sha,
    }
    return manifest, {"rows": manifest_rows, "metadata": metadata}


def _validate_shared_provenance(
    root: Path,
    manifests: dict[str, dict[str, Any]],
    manifest_details: dict[str, dict[str, Any]],
) -> None:
    base = manifest_details[BASE]["metadata"]
    ft_paths = {
        manifest_details[condition]["metadata"]["source_provenance"] for condition in CONDITIONS[1:]
    }
    ft_shas = {
        manifest_details[condition]["metadata"]["source_provenance_sha256"]
        for condition in CONDITIONS[1:]
    }
    if len(ft_paths) != 1 or len(ft_shas) != 1:
        raise ValueError("fine-tune conditions must share one exact source provenance")
    provenance_path = Path(next(iter(ft_paths)))
    provenance = _read_json(provenance_path)
    if provenance.get("schema_version") != 1:
        raise ValueError("source provenance schema_version must be exactly 1")

    source = provenance.get("source")
    finetune = provenance.get("finetune")
    if not isinstance(source, dict) or not isinstance(finetune, dict):
        raise ValueError("source provenance must contain source and finetune objects")
    if source.get("step") != 170_000:
        raise ValueError("source provenance step must be exactly 170000")
    if source.get("checkpoint_sha256") != base["checkpoint_sha256"]:
        raise ValueError("source provenance checkpoint SHA does not match base170k")
    source_checkpoint = _resolve_path(
        root, source.get("checkpoint"), label="source provenance checkpoint"
    )
    if not _same_file(source_checkpoint, Path(base["checkpoint"])):
        raise ValueError("source provenance checkpoint does not resolve to base170k")
    if source.get("source_config_sha256") != base["config_sha256"]:
        raise ValueError("source provenance config SHA does not match base170k")
    source_config = _resolve_path(
        root, source.get("source_config"), label="source provenance source_config"
    )
    if not _same_file(source_config, Path(base["config"])):
        raise ValueError("source provenance source_config does not resolve to base config")

    ft_config_paths = {
        manifest_details[condition]["metadata"]["config"] for condition in CONDITIONS[1:]
    }
    ft_config_shas = {
        manifest_details[condition]["metadata"]["config_sha256"] for condition in CONDITIONS[1:]
    }
    if len(ft_config_paths) != 1 or len(ft_config_shas) != 1:
        raise ValueError("fine-tune conditions must use one exact config")
    if finetune.get("config_sha256") != next(iter(ft_config_shas)):
        raise ValueError("source provenance fine-tune config SHA mismatch")
    provenance_ft_config = _resolve_path(
        root, finetune.get("config"), label="source provenance finetune config"
    )
    if not _same_file(provenance_ft_config, Path(next(iter(ft_config_paths)))):
        raise ValueError("source provenance fine-tune config path mismatch")
    if (
        finetune.get("initial_weights") != "ema"
        or finetune.get("start_step") != 0
        or finetune.get("optimizer_scheduler") != "fresh"
    ):
        raise ValueError("source provenance fine-tune initialization contract mismatch")

    paired_keys = (
        "sampler",
        "n_steps",
        "seed",
        "use_ema",
        "single_chain_only",
        "ids_file",
        "sde_tau",
        "sde_eps",
        "sde_w_cutoff",
        "sde_log_timesteps",
        "max_length",
        "output_format",
        "geometry_guidance_preset",
        "geometry_guidance",
        "guidance_off",
    )
    for key in paired_keys:
        values = [manifests[condition].get(key) for condition in CONDITIONS]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"condition manifest mismatch for {key}")


def _load_scores(
    root: Path,
    condition: str,
    manifest_rows: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    score_dir = root / "scores" / condition
    ost_summary = _read_json(score_dir / "openstructure" / "summary.json")
    local = _read_json(score_dir / "local_geometry.json")
    ost_rows = _index_rows(
        ost_summary.get("rows"), key="target", label=f"{condition} OpenStructure summary"
    )
    local_rows = _index_rows(local.get("rows"), key="pdb_id", label=f"{condition} local geometry")
    expected = set(manifest_rows)
    if set(ost_rows) != expected or set(local_rows) != expected:
        raise ValueError(f"{condition}: target mismatch among manifest/OpenStructure/local scores")
    if len(expected) != EXPECTED_TARGETS:
        raise ValueError(f"{condition}: score coverage must be exactly {EXPECTED_TARGETS}")
    if ost_summary.get("target_count") != EXPECTED_TARGETS:
        raise ValueError(f"{condition}: OpenStructure target_count must be 70")
    if ost_summary.get("success_count") != EXPECTED_TARGETS:
        raise ValueError(f"{condition}: OpenStructure success_count must be 70")
    if local.get("n") != EXPECTED_TARGETS:
        raise ValueError(f"{condition}: local geometry n must be 70")
    if _finite(local.get("clash_threshold_A"), label=f"{condition} clash threshold") != 1.5:
        raise ValueError(f"{condition}: local clash threshold must be exactly 1.5 Å")
    ca_definition = local.get("nonlocal_ca_metric_definition")
    expected_definition = {
        "sequence_separation_gt": 12,
        "point_penetration_floor_A": 3.6,
        "segment_penetration_floor_A": 2.5,
        "segment_max_edge_A": 6.0,
    }
    if ca_definition != expected_definition:
        raise ValueError(f"{condition}: nonlocal C-alpha metric definition mismatch")

    rows: dict[str, dict[str, float]] = {}
    for target in sorted(expected):
        summary_row = ost_rows[target]
        local_row = local_rows[target].get("pred")
        if not isinstance(local_row, dict):
            raise ValueError(f"{condition}/{target}: missing local prediction score")
        raw_path = score_dir / "openstructure" / f"{target}.json"
        raw = _read_json(raw_path)
        if raw.get("status") != "SUCCESS":
            raise ValueError(f"{condition}/{target}: OpenStructure status is not SUCCESS")
        raw_clashes = raw.get("model_clashes")
        if not isinstance(raw_clashes, list):
            raise ValueError(f"{condition}/{target}: model_clashes must be a list")
        n_atoms = _positive_int(
            local_row.get("n_atoms"), label=f"{condition}/{target} predicted atoms"
        )
        raw_atom_count = raw.get("model_n_atoms", raw.get("n_atoms"))
        if (
            raw_atom_count is not None
            and _positive_int(
                raw_atom_count, label=f"{condition}/{target} OpenStructure predicted atoms"
            )
            != n_atoms
        ):
            raise ValueError(f"{condition}/{target}: predicted atom count mismatch")

        source_metrics = {
            "gdt_ts": "oligo_gdtts",
            "all_atom_lddt": "lddt",
            "bb_lddt": "bb_lddt",
            "tm_score": "tm_score",
        }
        row: dict[str, float] = {}
        for output_key, source_key in source_metrics.items():
            raw_value = _finite(raw.get(source_key), label=f"{condition}/{target} {source_key}")
            summary_value = _finite(
                summary_row.get(source_key),
                label=f"{condition}/{target} summary {source_key}",
            )
            if not math.isclose(raw_value, summary_value, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    f"{condition}/{target}: OpenStructure raw/summary mismatch for {source_key}"
                )
            row[output_key] = raw_value
        row.update(
            {
                "ost_model_clashes_per_1k_atoms": len(raw_clashes) * 1000.0 / n_atoms,
                "local_hard_clashes_per_1k_atoms": _finite(
                    local_row.get("clashes_per_1k_atoms"),
                    label=f"{condition}/{target} local hard clashes",
                ),
                "bond_p95_A": _finite(
                    local_row.get("bond_p95_A"),
                    label=f"{condition}/{target} bond p95",
                ),
                "ca_crossings": _finite(
                    local_row.get("nonlocal_ca_segment_clashes_lt_2p5A"),
                    label=f"{condition}/{target} CA crossings",
                ),
                "n_atoms": float(n_atoms),
            }
        )
        if row["local_hard_clashes_per_1k_atoms"] < 0 or row["bond_p95_A"] < 0:
            raise ValueError(f"{condition}/{target}: negative local geometry metric")
        if row["ca_crossings"] < 0 or not row["ca_crossings"].is_integer():
            raise ValueError(f"{condition}/{target}: CA crossings must be a non-negative integer")
        rows[target] = row

    score_contract = {
        "openstructure": ost_summary.get("openstructure"),
        "clash_threshold_A": local["clash_threshold_A"],
        "nonlocal_ca_metric_definition": ca_definition,
    }
    return rows, score_contract


def _bootstrap_cis(
    deltas: dict[str, np.ndarray], *, n_resamples: int, seed: int
) -> dict[str, list[float]]:
    if n_resamples < 1:
        raise ValueError("bootstrap must be positive")
    target_count = len(next(iter(deltas.values())))
    rng = np.random.default_rng(seed)
    samples: dict[str, np.ndarray] = {
        metric: np.empty(n_resamples, dtype=np.float64) for metric in deltas
    }
    for start in range(0, n_resamples, 1_000):
        count = min(1_000, n_resamples - start)
        indices = rng.integers(0, target_count, size=(count, target_count))
        for metric, values in deltas.items():
            samples[metric][start : start + count] = values[indices].mean(axis=1)
    return {
        metric: [float(value) for value in np.percentile(samples[metric], [2.5, 97.5])]
        for metric in deltas
    }


def summarize(
    root: Path, *, bootstrap: int = DEFAULT_BOOTSTRAP, seed: int = DEFAULT_SEED
) -> dict[str, Any]:
    """Load, validate, and summarize all six paired CASP14 conditions."""
    root = root.resolve()
    hash_cache: dict[Path, str] = {}
    manifests: dict[str, dict[str, Any]] = {}
    manifest_details: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        manifest, details = _validate_manifest(root, condition, hash_cache=hash_cache)
        manifests[condition] = manifest
        manifest_details[condition] = details
    _validate_shared_provenance(root, manifests, manifest_details)
    expected_targets = _load_ids(root, manifests)

    rows: dict[str, dict[str, dict[str, float]]] = {}
    score_contracts: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        condition_rows, score_contract = _load_scores(
            root, condition, manifest_details[condition]["rows"]
        )
        if sorted(condition_rows) != expected_targets:
            raise ValueError(f"{condition}: score targets do not exactly match the IDs file")
        rows[condition] = condition_rows
        score_contracts[condition] = score_contract

    base_lengths = {target: int(row["L"]) for target, row in manifest_details[BASE]["rows"].items()}
    for condition in CONDITIONS[1:]:
        lengths = {
            target: int(row["L"]) for target, row in manifest_details[condition]["rows"].items()
        }
        if lengths != base_lengths:
            raise ValueError(f"{condition}: target lengths differ from base170k")
        if score_contracts[condition] != score_contracts[BASE]:
            raise ValueError(f"{condition}: scoring contract differs from base170k")

    condition_results: dict[str, dict[str, Any]] = {}
    base_means = {
        metric: statistics.fmean(rows[BASE][target][metric] for target in expected_targets)
        for metric in METRICS
    }
    for condition_index, condition in enumerate(CONDITIONS):
        means = {
            metric: statistics.fmean(rows[condition][target][metric] for target in expected_targets)
            for metric in METRICS
        }
        result: dict[str, Any] = {
            **manifest_details[condition]["metadata"],
            "coverage": EXPECTED_TARGETS,
            "means": means,
            "pooled_ca_crossings": int(
                sum(rows[condition][target]["ca_crossings"] for target in expected_targets)
            ),
        }
        if condition != BASE:
            delta_arrays = {
                metric: np.asarray(
                    [
                        rows[condition][target][metric] - rows[BASE][target][metric]
                        for target in expected_targets
                    ],
                    dtype=np.float64,
                )
                for metric in METRICS
            }
            cis = _bootstrap_cis(
                delta_arrays,
                n_resamples=bootstrap,
                seed=seed + condition_index,
            )
            deltas = {
                metric: {
                    "candidate_minus_base_mean": float(values.mean()),
                    "paired_target_bootstrap_95pct_ci": cis[metric],
                }
                for metric, values in delta_arrays.items()
            }
            base_clash = base_means["ost_model_clashes_per_1k_atoms"]
            clash_reduction = (
                None
                if base_clash <= 0.0
                else 1.0 - means["ost_model_clashes_per_1k_atoms"] / base_clash
            )
            criteria = {
                "ost_model_clash_reduction": {
                    "observed": clash_reduction,
                    "threshold": 0.50,
                    "operator": ">=",
                    "passed": clash_reduction is not None and clash_reduction + _EPS >= 0.50,
                },
                "mean_gdt_ts_delta": {
                    "observed": deltas["gdt_ts"]["candidate_minus_base_mean"],
                    "threshold": -0.005,
                    "operator": ">=",
                    "passed": deltas["gdt_ts"]["candidate_minus_base_mean"] + _EPS >= -0.005,
                },
                "mean_all_atom_lddt_delta": {
                    "observed": deltas["all_atom_lddt"]["candidate_minus_base_mean"],
                    "threshold": -0.005,
                    "operator": ">=",
                    "passed": deltas["all_atom_lddt"]["candidate_minus_base_mean"] + _EPS >= -0.005,
                },
            }
            result["deltas_vs_base"] = deltas
            result["gate"] = {
                "criteria": criteria,
                "passed": all(item["passed"] for item in criteria.values()),
            }
        condition_results[condition] = result

    eligible = [
        condition for condition in CONDITIONS[1:] if condition_results[condition]["gate"]["passed"]
    ]
    selected = eligible[0] if eligible else None
    return {
        "schema_version": 1,
        "experiment": "geoft_casp14_whole70_paired_v1",
        "contract": CONTRACT,
        "bootstrap": {"resamples": bootstrap, "seed": seed, "unit": "paired target"},
        "coverage": {
            "expected": EXPECTED_TARGETS,
            "paired": EXPECTED_TARGETS,
            "target_ids": expected_targets,
            "complete": True,
        },
        "conditions": condition_results,
        "decision": {
            "status": "promote" if selected is not None else "no_promotion",
            "selected_condition": selected,
            "eligible_conditions": eligible,
            "selection_rule": "earliest eligible fine-tune checkpoint",
        },
        "per_target": {
            target: {condition: rows[condition][target] for condition in CONDITIONS}
            for target in expected_targets
        },
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _fmt_delta_ci(result: dict[str, Any], metric: str) -> str:
    row = result["deltas_vs_base"][metric]
    low, high = row["paired_target_bootstrap_95pct_ci"]
    return f"{_fmt(row['candidate_minus_base_mean'])} [{_fmt(low)}, {_fmt(high)}]"


def render_markdown(summary: dict[str, Any]) -> str:
    decision = summary["decision"]
    lines = [
        "# CASP14 geometry fine-tune comparison",
        "",
        (
            "This is a paired, guidance-off comparison on the same 70 CASP14 "
            "single-chain targets. CASP14 has already been used during development, "
            "so these results are **retrospective engineering evidence**, not an "
            "untouched confirmatory test-set result."
        ),
        "",
        "## Decision",
        "",
        f"- Status: **{decision['status']}**",
        f"- Selected condition: **{decision['selected_condition'] or 'none'}**",
        (
            "- Gate: at least 50% mean OpenStructure clash-rate reduction, with "
            "mean GDT-TS and all-atom lDDT deltas each at least -0.005."
        ),
        "- Eligible candidates are ordered by fine-tune step; the earliest is preferred.",
        "",
        "## Equal-target means",
        "",
        (
            "| Condition | GDT-TS | all-atom lDDT | bb-lDDT | TM | "
            "OST clashes/1k | local hard clashes/1k | bond p95 Å | CA crossings | Gate |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for condition in CONDITIONS:
        result = summary["conditions"][condition]
        means = result["means"]
        gate = "—" if condition == BASE else ("PASS" if result["gate"]["passed"] else "FAIL")
        lines.append(
            f"| {condition} | {_fmt(means['gdt_ts'])} | "
            f"{_fmt(means['all_atom_lddt'])} | {_fmt(means['bb_lddt'])} | "
            f"{_fmt(means['tm_score'])} | "
            f"{_fmt(means['ost_model_clashes_per_1k_atoms'])} | "
            f"{_fmt(means['local_hard_clashes_per_1k_atoms'])} | "
            f"{_fmt(means['bond_p95_A'])} | {_fmt(means['ca_crossings'], 2)} | {gate} |"
        )

    lines.extend(
        [
            "",
            "## Paired deltas versus base170k",
            "",
            (
                "| Condition | ΔGDT-TS [95% CI] | Δall-atom lDDT [95% CI] | "
                "Δbb-lDDT [95% CI] | ΔTM [95% CI] | ΔOST clash/1k [95% CI] | "
                "Clash reduction | Δlocal hard clash/1k [95% CI] | "
                "Δbond p95 Å [95% CI] | ΔCA crossings [95% CI] |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for condition in CONDITIONS[1:]:
        result = summary["conditions"][condition]
        reduction = result["gate"]["criteria"]["ost_model_clash_reduction"]["observed"]
        reduction_text = "n/a" if reduction is None else f"{100.0 * reduction:.1f}%"
        lines.append(
            f"| {condition} | {_fmt_delta_ci(result, 'gdt_ts')} | "
            f"{_fmt_delta_ci(result, 'all_atom_lddt')} | "
            f"{_fmt_delta_ci(result, 'bb_lddt')} | "
            f"{_fmt_delta_ci(result, 'tm_score')} | "
            f"{_fmt_delta_ci(result, 'ost_model_clashes_per_1k_atoms')} | "
            f"{reduction_text} | "
            f"{_fmt_delta_ci(result, 'local_hard_clashes_per_1k_atoms')} | "
            f"{_fmt_delta_ci(result, 'bond_p95_A')} | "
            f"{_fmt_delta_ci(result, 'ca_crossings')} |"
        )

    lines.extend(
        [
            "",
            (
                f"Coverage: {summary['coverage']['paired']}/{summary['coverage']['expected']} "
                "targets in every condition. Each delta in geoft_comparison.json also includes a "
                "deterministic paired-target bootstrap 95% confidence interval."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    result = summarize(args.root, bootstrap=args.bootstrap, seed=args.seed)
    json_path = args.out_json or args.root / "geoft_comparison.json"
    md_path = args.out_md or args.root / "geoft_comparison.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result["decision"], indent=2))


if __name__ == "__main__":
    main()
