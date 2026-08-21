#!/usr/bin/env python
"""Give the AFDB v4 corpus a sequence-addressed ESMC-6B cache.

The embeddings were computed once against the v6 corpus and stored under
occurrence names, ``AF-<accession>-F1-model_v6_ch0.npy``. An AlphaFold DB version
change alters the prediction, not the UniProt sequence, so the same embedding is
correct for the v4 record of the same accession whenever the sequence is
unchanged — and the ESMC-6B cache is addressed by the SHA-256 of that sequence,
so re-keying the existing files is all that is required.

Re-deriving the mapping by re-reading the 268,977 v6 ``.npz`` records is
I/O-bound and slow. The v4 conversion already emits ``sequences.tsv``, so this
script joins on the accession instead and never opens a structure record.

Sequence agreement is enforced through the cached array itself: an embedding has
one row per residue, capped at ``--max_length``, so a v4 sequence whose length
disagrees with the cached row count cannot be the sequence that produced it.
That catches the realistic failure — a UniProt canonical revision between
releases — and those accessions are written to ``needs_new_embedding.txt``
instead of being linked. ``--verify_samples`` additionally re-reads a random
sample of v6 records and compares sequences character by character, which
detects a same-length substitution that the row-count check cannot see.

Usage:
    uv run --no-sync python scripts/link_afdb_esm_by_v4_sequence.py \
      --v4_sequences data/afdb_swissprot_v4/sequences.tsv \
      --legacy_esm_dir "$LEGACY_ESMC_AFDB" \
      --dest_esm_dir data/afdb_esmc6b \
      --v6_npz_dir "$LEGACY_AFDB_V6_NPZ" \
      --verify_samples 2000 \
      --out_dir data/afdb_swissprot_v4/sequence_audit
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pipeline" / "lib"))

from precompute_esm import get_protein_chains  # noqa: E402

from mambafold.data.sequence_cache import sequence_embedding_path  # noqa: E402


def accession(entry_id: str) -> str:
    parts = entry_id.split("-")
    return parts[1] if len(parts) > 1 else entry_id


def read_v4_sequences(path: Path) -> dict[str, tuple[str, str]]:
    """accession -> (entry_id, sequence)"""
    out: dict[str, tuple[str, str]] = {}
    for line_number, line in enumerate(path.read_text().splitlines()):
        if line_number == 0 and line.startswith("entry_id\t"):
            continue
        entry_id, _, sequence = line.partition("\t")
        if not entry_id or not sequence:
            continue
        out[accession(entry_id)] = (entry_id, sequence)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4_sequences", required=True)
    ap.add_argument("--legacy_esm_dir", required=True)
    ap.add_argument("--dest_esm_dir", required=True)
    ap.add_argument("--legacy_suffix", default="-F1-model_v6_ch0.npy")
    ap.add_argument("--legacy_prefix", default="AF-")
    ap.add_argument("--v6_npz_dir", default=None)
    ap.add_argument("--verify_samples", type=int, default=0)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--embedding_dim", type=int, default=2560)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    v4 = read_v4_sequences(Path(args.v4_sequences))
    legacy_dir = Path(args.legacy_esm_dir)
    dest_dir = Path(args.dest_esm_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"v4 records: {len(v4)}", flush=True)

    # An embedding stores one row per residue up to the cap, so the row count
    # pins the sequence length only while that length is below the cap. At or
    # above it every sequence yields exactly `max_length` rows and the check
    # says nothing, so those accessions are compared against the v6 record
    # itself. They are a small minority, which is why this is affordable.
    capped = {acc for acc, (_eid, seq) in v4.items() if len(seq) >= args.max_length}
    v6_index: dict[str, Path] = {}
    if capped and args.v6_npz_dir:
        for path in Path(args.v6_npz_dir).rglob("*.npz"):
            acc = accession(path.stem)
            if acc in capped:
                v6_index[acc] = path
        print(
            f"capped-length accessions: {len(capped)}, "
            f"{len(v6_index)} resolvable against v6 records",
            flush=True,
        )

    linked = existing = no_cache = length_mismatch = bad_shape = 0
    capped_verified = capped_unverifiable = capped_mismatch = 0
    needs_embedding: list[str] = []
    seen_dest: set[str] = set()

    for index, (acc, (entry_id, sequence)) in enumerate(sorted(v4.items())):
        dest = sequence_embedding_path(dest_dir, sequence)
        if str(dest) in seen_dest or dest.exists():
            existing += 1
            continue

        legacy = legacy_dir / f"{args.legacy_prefix}{acc}{args.legacy_suffix}"
        if not legacy.exists():
            no_cache += 1
            needs_embedding.append(entry_id)
            continue

        try:
            array = np.load(legacy, mmap_mode="r")
        except Exception:  # noqa: BLE001
            bad_shape += 1
            needs_embedding.append(entry_id)
            continue

        expected_rows = min(len(sequence), args.max_length)
        if array.ndim != 2 or array.shape[1] != args.embedding_dim:
            bad_shape += 1
            needs_embedding.append(entry_id)
            continue
        if array.shape[0] != expected_rows:
            length_mismatch += 1
            needs_embedding.append(entry_id)
            continue

        if len(sequence) >= args.max_length:
            v6_path = v6_index.get(acc)
            if v6_path is None:
                capped_unverifiable += 1
                needs_embedding.append(entry_id)
                continue
            chains = list(get_protein_chains(v6_path, strict=False))
            if not chains or chains[0] != sequence:
                capped_mismatch += 1
                needs_embedding.append(entry_id)
                continue
            capped_verified += 1

        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(legacy, dest)
            linked += 1
            seen_dest.add(str(dest))
        except FileExistsError:
            existing += 1

        if (index + 1) % 25000 == 0:
            print(
                f"  {index + 1}/{len(v4)} linked={linked} existing={existing} "
                f"no_cache={no_cache} length_mismatch={length_mismatch}",
                flush=True,
            )

    sampled = sample_mismatch = sample_missing = 0
    sample_mismatched_ids: list[str] = []
    if args.verify_samples and args.v6_npz_dir:
        v6_by_accession: dict[str, Path] = {}
        for path in Path(args.v6_npz_dir).rglob("*.npz"):
            v6_by_accession[accession(path.stem)] = path

        rng = random.Random(args.seed)
        candidates = sorted(set(v4) & set(v6_by_accession))
        picks = rng.sample(candidates, min(args.verify_samples, len(candidates)))
        for acc in picks:
            entry_id, v4_sequence = v4[acc]
            chains = list(get_protein_chains(v6_by_accession[acc], strict=False))
            if not chains:
                sample_missing += 1
                continue
            sampled += 1
            if chains[0] != v4_sequence:
                sample_mismatch += 1
                sample_mismatched_ids.append(entry_id)

    (out_dir / "needs_new_embedding.txt").write_text(
        "\n".join(sorted(needs_embedding)) + ("\n" if needs_embedding else "")
    )
    report = {
        "v4_records": len(v4),
        "linked": linked,
        "already_present": existing,
        "no_legacy_cache": no_cache,
        "length_mismatch": length_mismatch,
        "unreadable_or_bad_shape": bad_shape,
        "capped_length_verified_against_v6": capped_verified,
        "capped_length_mismatch": capped_mismatch,
        "capped_length_unverifiable": capped_unverifiable,
        "needs_new_embedding": len(needs_embedding),
        "verification": {
            "sampled": sampled,
            "sequence_mismatch": sample_mismatch,
            "v6_record_unreadable": sample_missing,
            "mismatched_ids": sample_mismatched_ids[:50],
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if sample_mismatch:
        print(
            "WARNING: sampled sequences disagree between v4 and v6; "
            "the accession join is not safe and embeddings must be recomputed",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
