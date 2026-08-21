#!/usr/bin/env python
"""Stage 02 — reproduce SimpleFold's RCSB admission rule over the Boltz records.

SimpleFold applies three dynamic filters to ``rcsb_protein``. Two are declared
per-dataset in ``configs/data/pdb.yaml``:

    ResolutionFilter(resolution=5.0)
    DateFilter(date="2020-05-01", ref="released")

and one is declared in the same file's top-level ``filters:`` block, which
``datasets/train_datamodule.py`` applies to every dataset *before* the
per-dataset list:

    SizeFilter(min_chains=1, max_chains=300)

All three read ``Record`` fields that the Boltz preprocessing already wrote into
``manifest.json`` beside the structures, so this script reads that manifest —
the same object the reference implementation filters on — rather than
re-deriving the fields from another source.

Filter semantics, from the reference implementation:

* ``SizeFilter``  ``num_chains <= max_chains and num_valid >= min_chains``,
  where ``num_valid`` counts every chain with ``valid`` true, of any molecule
  type.
* ``ResolutionFilter``  ``structure.resolution <= resolution``, with no
  missing-value branch.
* ``DateFilter``  uses ``released``, falling back to ``deposited`` when it is
  empty; a record with neither is rejected.

A note on resolution. Every record in this Boltz archive carries
``resolution: 0.0``, including the X-ray entries, so ``ResolutionFilter`` admits
the entire corpus in the reference pipeline — it is a no-op there, not a quality
gate. ``--resolution_source boltz`` (the default) reproduces that. ``rcsb_api``
instead uses the real resolutions fetched by ``fetch_rcsb_entry_metadata.py``,
which is a stricter corpus than SimpleFold trained on and therefore a
deliberate deviation; the report always records what the other choice would
have decided so the difference is never silent.

Our own corpus additionally sets ``single_chain_only`` and
``extract_monomer_chains``, which SimpleFold does not. That happens in the
dataset at load time, not in this entry-level rule.

Outputs (in ``--out_dir``):
    admitted_all.txt          every admitted entry; the whole training set
    heldout_post_cutoff.txt   entries excluded by the date rule (clean eval pool)
    excluded_size.txt         entries excluded by the size rule
    excluded_resolution.txt   entries excluded by the resolution rule
    excluded_no_date.txt      entries with no usable date
    report.json               counts, rule parameters, and input digests

Usage:
    python pipeline/02_build_admission_splits.py \
      --npz_dir data/rcsb_boltz_official_full \
      --boltz_manifest "$BOLTZ_MANIFEST" \
      --entry_meta data/splits/rcsb_entry_metadata.tsv \
      --out_dir data/splits/simplefold_matched
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

MAX_CHAINS = 300
MIN_CHAINS = 1


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_boltz_manifest(path: Path) -> dict[str, dict]:
    """entry_id -> {resolution, method, deposited, released, num_chains, num_valid}"""
    with path.open() as handle:
        records = json.load(handle)
    out: dict[str, dict] = {}
    for record in records:
        entry_id = str(record.get("id", "")).lower()
        if not entry_id:
            continue
        structure = record.get("structure") or {}
        chains = record.get("chains") or []
        out[entry_id] = {
            "resolution": structure.get("resolution"),
            "method": structure.get("method") or "",
            "deposited": structure.get("deposited") or "",
            "released": structure.get("released") or "",
            "num_chains": structure.get("num_chains"),
            "num_valid": sum(1 for chain in chains if chain.get("valid")),
            # Boltz encodes protein as molecule type 0; the dataset loader uses
            # the same test when it enumerates trainable chains.
            "num_protein": sum(1 for chain in chains if chain.get("mol_type") == 0),
        }
    return out


def read_entry_meta(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    lines = path.read_text().splitlines()
    if not lines:
        raise ValueError(f"empty metadata file: {path}")
    header = lines[0].split("\t")
    expected = ["pdb_id", "deposit_date", "release_date", "resolution", "experimental_method"]
    if header != expected:
        raise ValueError(f"unexpected header in {path}: {header}")
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) != len(expected):
            continue
        rows[parts[0].lower()] = dict(zip(expected, parts, strict=True))
    return rows


def read_id_list(path: Path) -> set[str]:
    return {
        Path(token.strip()).stem.lower()
        for token in path.read_text().splitlines()
        if token.strip()
    }


def write_list(path: Path, ids: list[str]) -> None:
    body = "\n".join(f"{i}.npz" for i in sorted(ids))
    path.write_text(body + "\n" if body else "")


def resolution_excludes(
    record: dict, api_record: dict[str, str] | None, source: str, max_resolution: float
) -> bool:
    """The per-dataset ResolutionFilter decision for one entry.

    One implementation, used both by the admission loop and by the
    counterfactual bookkeeping, so the two can never drift apart.
    """
    if source == "boltz":
        resolution = record["resolution"]
        return resolution is not None and resolution > max_resolution
    raw = (api_record or {}).get("resolution", "")
    return bool(raw) and float(raw) > max_resolution


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--boltz_manifest", required=True)
    ap.add_argument(
        "--entry_meta",
        default=None,
        help="RCSB Data API metadata, for the resolution cross-report",
    )
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--release_cutoff", default="2020-05-01")
    ap.add_argument("--max_resolution", type=float, default=5.0)
    ap.add_argument(
        "--resolution_source",
        choices=["boltz", "rcsb_api"],
        default="boltz",
        help="'boltz' reproduces the reference pipeline exactly; 'rcsb_api' is stricter",
    )
    ap.add_argument(
        "--skip_size_filter",
        action="store_true",
        help="diagnostic only; not a matched corpus",
    )
    ap.add_argument(
        "--allow_non_protein",
        action="store_true",
        help="admit entries with no protein chain; they carry no trainable example",
    )
    args = ap.parse_args()

    npz_dir = Path(args.npz_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entry_ids = sorted({p.stem.lower() for p in npz_dir.glob("*.npz")})
    manifest = read_boltz_manifest(Path(args.boltz_manifest))
    api_meta = read_entry_meta(Path(args.entry_meta)) if args.entry_meta else {}

    admitted: list[str] = []
    post_cutoff: list[str] = []
    excluded_size: list[str] = []
    excluded_resolution: list[str] = []
    excluded_no_date: list[str] = []
    missing_manifest: list[str] = []

    excluded_non_protein: list[str] = []
    size_oversized = size_no_valid = 0
    date_from_deposit = 0
    api_resolution_would_exclude: list[str] = []
    api_resolution_unknown = 0
    # The protein requirement is this project's addition, not one of SimpleFold's
    # filters, so the corpus has two defensible sizes. Both are reported: the
    # entries this rule drops are counted against the cutoff here rather than
    # left to be reconstructed by hand, because the SimpleFold-exact figure is
    # what the reference's published "~160K" corresponds to.
    non_protein_pre_cutoff = non_protein_post_cutoff = 0

    for entry_id in entry_ids:
        record = manifest.get(entry_id)
        if record is None:
            missing_manifest.append(entry_id)
            continue

        # 1. global SizeFilter, applied first by the reference datamodule
        if not args.skip_size_filter:
            num_chains = record["num_chains"]
            oversized = num_chains is not None and num_chains > MAX_CHAINS
            no_valid = record["num_valid"] < MIN_CHAINS
            if oversized or no_valid:
                size_oversized += int(oversized)
                size_no_valid += int(no_valid and not oversized)
                excluded_size.append(entry_id)
                continue

        # 2. protein requirement. Not one of SimpleFold's filters, but its RCSB
        # dataset is `rcsb_protein` and ours is protein-only by design: an entry
        # with no protein chain yields no trainable example, contributes nothing
        # to the training FASTA, and would only pad the corpus counts.
        if not args.allow_non_protein and record["num_protein"] < 1:
            excluded_non_protein.append(entry_id)
            # Record where this entry would have landed had the rule not been
            # applied, so the SimpleFold-exact corpus size is a pipeline output
            # rather than an arithmetic footnote.
            if not resolution_excludes(
                record, api_meta.get(entry_id), args.resolution_source, args.max_resolution
            ):
                date = record["released"] or record["deposited"]
                if date and date <= args.release_cutoff:
                    non_protein_pre_cutoff += 1
                elif date:
                    non_protein_post_cutoff += 1
            continue

        # 3. per-dataset ResolutionFilter
        if resolution_excludes(
            record, api_meta.get(entry_id), args.resolution_source, args.max_resolution
        ):
            excluded_resolution.append(entry_id)
            continue

        # cross-report: what the other resolution source would have decided
        raw_api = (api_meta.get(entry_id) or {}).get("resolution", "")
        if not raw_api:
            api_resolution_unknown += 1
        elif float(raw_api) > args.max_resolution:
            api_resolution_would_exclude.append(entry_id)

        # 4. per-dataset DateFilter
        date = record["released"] or record["deposited"]
        if not date:
            excluded_no_date.append(entry_id)
            continue
        if not record["released"]:
            date_from_deposit += 1

        if date <= args.release_cutoff:
            admitted.append(entry_id)
        else:
            post_cutoff.append(entry_id)

    # No validation split. This project selects nothing during training, so every
    # admitted entry trains and `admitted_all.txt` is the whole training set.
    # Anything that screens the training corpus must screen this list.
    admitted_set = set(admitted)
    write_list(out_dir / "admitted_all.txt", sorted(admitted_set))
    write_list(out_dir / "heldout_post_cutoff.txt", post_cutoff)
    write_list(out_dir / "excluded_size.txt", excluded_size)
    write_list(out_dir / "excluded_non_protein.txt", excluded_non_protein)
    write_list(out_dir / "excluded_resolution.txt", excluded_resolution)
    write_list(out_dir / "excluded_no_date.txt", excluded_no_date)
    write_list(out_dir / "missing_manifest.txt", missing_manifest)
    write_list(out_dir / "api_resolution_would_exclude.txt", api_resolution_would_exclude)

    report = {
        "rule": {
            "order": "SizeFilter (global) -> protein requirement -> ResolutionFilter -> DateFilter",
            "require_protein_chain": not args.allow_non_protein,
            "size_filter": {
                "min_chains": MIN_CHAINS,
                "max_chains": MAX_CHAINS,
                "skipped": args.skip_size_filter,
            },
            "release_cutoff": args.release_cutoff,
            "date_ref": "released, falling back to deposited",
            "max_resolution": args.max_resolution,
            "resolution_source": args.resolution_source,
            "resolution_note": (
                "every record in this Boltz archive carries resolution 0.0, so the "
                "reference ResolutionFilter is a no-op on this corpus"
            ),
            "not_applied_here": (
                "single_chain_only / extract_monomer_chains are dataset-time settings, "
                "not part of this entry-level rule"
            ),
        },
        "inputs": {
            "npz_dir": str(npz_dir),
            "npz_entries": len(entry_ids),
            "boltz_manifest": str(args.boltz_manifest),
            "boltz_manifest_records": len(manifest),
            "entry_meta": args.entry_meta,
        },
        "counts": {
            "admitted_total": len(admitted),
            "heldout_post_cutoff": len(post_cutoff),
            "excluded_size": len(excluded_size),
            "excluded_size_over_max_chains": size_oversized,
            "excluded_size_no_valid_chain": size_no_valid,
            "excluded_non_protein": len(excluded_non_protein),
            "excluded_resolution": len(excluded_resolution),
            "excluded_no_date": len(excluded_no_date),
            "missing_from_boltz_manifest": len(missing_manifest),
            "date_taken_from_deposit_fallback": date_from_deposit,
        },
        "resolution_cross_report": {
            "meaning": (
                "entries that survived the active resolution rule but that the other "
                "source would have excluded at the same threshold"
            ),
            "rcsb_api_would_exclude": len(api_resolution_would_exclude),
            "rcsb_api_no_value": api_resolution_unknown,
        },
        "protein_filter_cross_report": {
            "meaning": (
                "what the corpus would be without the protein requirement. "
                "SimpleFold's filters carry no molecule-type rule, so "
                "simplefold_exact_admitted is the figure its published '~160K' "
                "corresponds to; the extra entries yield no trainable example, "
                "so the two corpora train identically"
            ),
            "simplefold_exact_admitted": len(admitted) + non_protein_pre_cutoff,
            "simplefold_exact_heldout_post_cutoff": len(post_cutoff) + non_protein_post_cutoff,
            "non_protein_pre_cutoff": non_protein_pre_cutoff,
            "non_protein_post_cutoff": non_protein_post_cutoff,
        },
    }
    if args.entry_meta:
        report["inputs"]["entry_meta_sha256"] = file_digest(Path(args.entry_meta))
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({**report["counts"], **report["resolution_cross_report"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
