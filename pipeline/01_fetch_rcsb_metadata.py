#!/usr/bin/env python
"""Fetch RCSB entry metadata (release date, resolution, method) via the Data API.

The Boltz-processed ``.npz`` records carry no date or resolution field, and the
local mmCIF mirror used by ``scripts/extract_deposit_dates.py`` is no longer on
disk. This script pulls the fields required to reproduce the SimpleFold RCSB
admission rule:

    ResolutionFilter(resolution=5.0)
    DateFilter(date="2020-05-01", ref="released")

``DateFilter`` uses the release date and falls back to the deposition date when
the release date is missing, so both are recorded here.

Output TSV columns:
    pdb_id, deposit_date, release_date, resolution, experimental_method

Resumable: an existing output file is read first and only missing ids are
requested.

Usage:
    uv run --no-sync python scripts/fetch_rcsb_entry_metadata.py \
      --npz_dir data/rcsb_boltz_official_full \
      --out_tsv data/splits/rcsb_entry_metadata.tsv
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ENDPOINT = "https://data.rcsb.org/graphql"

QUERY = """
query ($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    rcsb_accession_info { deposit_date initial_release_date }
    rcsb_entry_info { resolution_combined experimental_method }
  }
}
"""


def read_ids(npz_dir: Path | None, id_file: Path | None) -> list[str]:
    ids: set[str] = set()
    if npz_dir is not None:
        for path in npz_dir.rglob("*.npz"):
            ids.add(path.stem.lower())
    if id_file is not None:
        for line in id_file.read_text().splitlines():
            token = line.strip()
            if not token:
                continue
            ids.add(Path(token).stem.lower())
    return sorted(ids)


def load_existing(out_tsv: Path) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    if not out_tsv.exists():
        return rows
    for line_number, line in enumerate(out_tsv.read_text().splitlines()):
        if line_number == 0 and line.startswith("pdb_id\t"):
            continue
        parts = line.split("\t")
        if len(parts) != 5:
            continue
        rows[parts[0]] = parts
    return rows


def post(ids: list[str], timeout: int, retries: int) -> list[dict]:
    payload = json.dumps({"query": QUERY, "variables": {"ids": [i.upper() for i in ids]}})
    request = urllib.request.Request(
        ENDPOINT,
        data=payload.encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            if "errors" in body and not body.get("data"):
                raise RuntimeError(str(body["errors"])[:400])
            entries = (body.get("data") or {}).get("entries") or []
            return [e for e in entries if e]
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < retries:
                time.sleep(2.0 * (attempt + 1))
    assert last_err is not None
    raise last_err


def _date(value: str | None) -> str:
    if not value:
        return ""
    return str(value)[:10]


def to_row(entry: dict) -> list[str]:
    pdb_id = str(entry.get("rcsb_id", "")).lower()
    accession = entry.get("rcsb_accession_info") or {}
    info = entry.get("rcsb_entry_info") or {}
    resolutions = info.get("resolution_combined") or []
    resolution = ""
    numeric = [float(r) for r in resolutions if r is not None]
    if numeric:
        resolution = f"{min(numeric):.3f}"
    return [
        pdb_id,
        _date(accession.get("deposit_date")),
        _date(accession.get("initial_release_date")),
        resolution,
        str(info.get("experimental_method") or ""),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", default=None)
    ap.add_argument("--id_file", default=None)
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--batch_size", type=int, default=250)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--sleep", type=float, default=0.1)
    args = ap.parse_args()

    if not args.npz_dir and not args.id_file:
        ap.error("one of --npz_dir or --id_file is required")

    ids = read_ids(
        Path(args.npz_dir) if args.npz_dir else None,
        Path(args.id_file) if args.id_file else None,
    )
    out_tsv = Path(args.out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    rows = load_existing(out_tsv)
    todo = [i for i in ids if i not in rows]
    print(
        f"total ids: {len(ids)}  cached: {len(ids) - len(todo)}  to fetch: {len(todo)}",
        flush=True,
    )

    missing: list[str] = []
    started = time.time()
    for start in range(0, len(todo), args.batch_size):
        chunk = todo[start : start + args.batch_size]
        entries = post(chunk, timeout=args.timeout, retries=args.retries)
        returned = set()
        for entry in entries:
            row = to_row(entry)
            if not row[0]:
                continue
            rows[row[0]] = row
            returned.add(row[0])
        missing.extend(i for i in chunk if i not in returned)

        done = start + len(chunk)
        if (start // args.batch_size) % 20 == 0 or done >= len(todo):
            elapsed = time.time() - started
            print(
                f"  {done}/{len(todo)} fetched  {elapsed:.0f}s  missing={len(missing)}",
                flush=True,
            )
            _write(out_tsv, rows)
        if args.sleep:
            time.sleep(args.sleep)

    _write(out_tsv, rows)
    print(f"wrote {len(rows)} rows -> {out_tsv}")
    if missing:
        missing_path = out_tsv.with_suffix(".missing.txt")
        missing_path.write_text("\n".join(sorted(missing)) + "\n")
        print(f"WARNING: {len(missing)} ids returned no entry -> {missing_path}")
    return 0


def _write(out_tsv: Path, rows: dict[str, list[str]]) -> None:
    tmp = out_tsv.with_suffix(out_tsv.suffix + ".tmp")
    with tmp.open("w") as handle:
        handle.write("pdb_id\tdeposit_date\trelease_date\tresolution\texperimental_method\n")
        for key in sorted(rows):
            handle.write("\t".join(rows[key]) + "\n")
    tmp.replace(out_tsv)


if __name__ == "__main__":
    raise SystemExit(main())
