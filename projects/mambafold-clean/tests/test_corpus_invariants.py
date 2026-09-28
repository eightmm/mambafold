"""Check the built corpus against the contract, independently of what built it.

These assertions re-derive every rule from the Boltz manifest and the files on
disk. They deliberately do not import the pipeline: a bug in stage 02 that
mis-applies a filter would still produce a self-consistent `report.json`, so the
report is not evidence. The manifest is.

Skips itself when the corpus has not been built, so it is safe to run in a
fresh checkout.

    pytest -q tests/test_corpus_invariants.py
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "splits"
AUDIT = ROOT / "data" / "audit"
TRAIN_FARM = ROOT / "data" / "rcsb_train"
HELDOUT_FARM = ROOT / "data" / "rcsb_heldout_post_cutoff"

RELEASE_CUTOFF = "2020-05-01"
MAX_CHAINS = 300
SAMPLE = int(os.environ.get("CORPUS_TEST_SAMPLE", "300"))


def _require(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"corpus not built: {missing[0]}")


def _ids(name: str) -> set[str]:
    return {
        line.strip().removesuffix(".npz")
        for line in (SPLITS / name).read_text().splitlines()
        if line.strip()
    }


def _boltz_manifest_path() -> Path:
    env = ROOT / "config" / "paths.env"
    _require(env)
    for line in env.read_text().splitlines():
        line = line.strip()
        if line.startswith("BOLTZ_ROOT="):
            root = Path(line.split("=", 1)[1].strip().strip('"'))
            return root / "manifest.json"
    pytest.skip("BOLTZ_ROOT not declared in config/paths.env")


@pytest.fixture(scope="module")
def manifest() -> dict[str, dict]:
    path = _boltz_manifest_path()
    _require(path)
    records = json.loads(path.read_text())
    return {str(r["id"]).lower(): r for r in records}


@pytest.fixture(scope="module")
def splits() -> dict[str, set[str]]:
    _require(SPLITS / "admitted_all.txt")
    return {
        name: _ids(f"{name}.txt")
        for name in (
            "admitted_all",
            "heldout_post_cutoff",
            "excluded_size",
            "excluded_non_protein",
        )
    }


def test_admitted_all_is_the_whole_training_set(splits):
    """No validation split exists; nothing is carved out of the admitted pool."""
    assert not (SPLITS / "val.txt").exists(), "a validation split reappeared"
    assert not (SPLITS / "train.txt").exists(), "a train/val carve-out reappeared"
    assert splits["admitted_all"]


def test_buckets_are_mutually_disjoint(splits):
    buckets = ["admitted_all", "heldout_post_cutoff", "excluded_size", "excluded_non_protein"]
    for i, a in enumerate(buckets):
        for b in buckets[i + 1 :]:
            assert splits[a] & splits[b] == set(), f"{a} overlaps {b}"


def test_every_manifest_record_is_accounted_for(splits, manifest):
    union = set().union(*(splits[k] for k in (
        "admitted_all", "heldout_post_cutoff", "excluded_size", "excluded_non_protein"
    )))
    assert union == set(manifest), (
        f"corpus accounts for {len(union)} of {len(manifest)} manifest records"
    )


def test_admitted_entries_satisfy_every_admission_rule(splits, manifest):
    """Re-derive the rule from the manifest rather than trusting report.json."""
    for entry_id in splits["admitted_all"]:
        record = manifest[entry_id]
        structure = record["structure"]
        chains = record.get("chains") or []

        num_chains = structure.get("num_chains")
        assert num_chains is None or num_chains <= MAX_CHAINS, entry_id
        assert sum(1 for c in chains if c.get("valid")) >= 1, entry_id
        assert sum(1 for c in chains if c.get("mol_type") == 0) >= 1, entry_id

        date = structure.get("released") or structure.get("deposited")
        assert date, entry_id
        assert date <= RELEASE_CUTOFF, f"{entry_id} released {date}"


def test_heldout_pool_is_strictly_after_the_cutoff(splits, manifest):
    for entry_id in splits["heldout_post_cutoff"]:
        structure = manifest[entry_id]["structure"]
        date = structure.get("released") or structure.get("deposited")
        assert date > RELEASE_CUTOFF, f"{entry_id} released {date}"


def test_farms_mirror_their_lists_and_never_overlap(splits):
    _require(TRAIN_FARM, HELDOUT_FARM)
    train_farm = {p.name.removesuffix(".npz") for p in TRAIN_FARM.iterdir()}
    heldout_farm = {p.name.removesuffix(".npz") for p in HELDOUT_FARM.iterdir()}
    assert train_farm == splits["admitted_all"]
    assert heldout_farm == splits["heldout_post_cutoff"]
    assert train_farm & heldout_farm == set()


def test_training_symlinks_resolve(splits):
    _require(TRAIN_FARM)
    entries = sorted(TRAIN_FARM.iterdir())
    picks = random.Random(0).sample(entries, min(SAMPLE, len(entries)))
    dangling = [p.name for p in picks if not p.resolve().exists()]
    assert not dangling, f"{len(dangling)} dangling symlinks, e.g. {dangling[:3]}"


def test_every_admitted_sequence_has_an_embedding():
    """The training loader resolves embeddings by sequence hash, so a missing
    one silently drops the example rather than failing loudly."""
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from mambafold.data.sequence_cache import sequence_embedding_path

    fasta = AUDIT / "rcsb-admitted.fasta"
    cache = ROOT / "data" / "rcsb_esmc6b"
    _require(fasta, cache)

    sequences: list[str] = []
    chunks: list[str] = []
    for line in fasta.read_text().splitlines():
        if line.startswith(">"):
            if chunks:
                sequences.append("".join(chunks))
                chunks = []
        elif line.strip():
            chunks.append(line.strip())
    if chunks:
        sequences.append("".join(chunks))

    picks = random.Random(0).sample(sequences, min(SAMPLE, len(sequences)))
    missing = [s[:20] for s in picks if not sequence_embedding_path(cache, s).exists()]
    assert not missing, f"{len(missing)} sampled sequences have no embedding"


def test_afdb_corpus_matches_the_published_list():
    sequences = ROOT / "data" / "afdb_swissprot_v4" / "sequences.tsv"
    id_list = ROOT / "data" / "external" / "simplefold_swissprot_list.csv"
    _require(sequences, id_list)

    published = {
        line.strip().split(",")[0]
        for line in id_list.read_text().splitlines()
        if line.strip().startswith("AF-")
    }
    converted = {
        line.split("\t", 1)[0]
        for index, line in enumerate(sequences.read_text().splitlines())
        if index and line.strip()
    }
    assert converted <= published, "converted records outside the published list"
    assert converted == published, f"{len(published - converted)} published ids not converted"
