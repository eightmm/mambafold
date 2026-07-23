#!/usr/bin/env python
"""Download AlphaFold DB structures from SimpleFold ID lists and convert to npz.

The SimpleFold public lists contain IDs like `AF-Q87AD2-F1-model_v4`.  The
current AlphaFold DB website serves newer model versions, so this script resolves
the UniProt accession through the AlphaFold API, downloads the current CIF, and
converts it to the MambaFold Boltz-style npz layout.

Usage:
    PYTHONPATH=src uv run python scripts/download_afdb_simplefold.py \
      --id_list data/external/simplefold/swissprot_list.csv \
      --out_dir data/afdb_swissprot/npz \
      --cif_dir data/external/afdb_swissprot_cif \
      --manifest data/afdb_swissprot/manifest.tsv \
      --limit 100 --workers 8
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pdb_to_npz import convert, parse_cif  # noqa: E402

ID_RE = re.compile(r"^AF-([A-Za-z0-9]+)-F(\d+)-model_v(\d+)$", re.IGNORECASE)


def _read_text(url: str, timeout: int, retries: int) -> str:
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.read().decode("utf-8")
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    assert last_err is not None
    raise last_err


def parse_simplefold_id(raw: str) -> tuple[str, str, str, str]:
    source_id = raw.strip()
    m = ID_RE.match(source_id)
    if not m:
        raise ValueError(f"bad SimpleFold AFDB id: {source_id}")
    acc, fragment, version = m.groups()
    return source_id, acc.upper(), f"F{fragment}", f"v{version}"


def resolve_cif_url(accession: str, fragment: str, timeout: int, retries: int) -> tuple[str, str]:
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{accession}"
    rows = json.loads(_read_text(api_url, timeout=timeout, retries=retries))
    if not rows:
        raise ValueError("empty AlphaFold API response")
    prefix = f"AF-{accession}-{fragment}".upper()
    for row in rows:
        entry_id = str(row.get("entryId", "")).upper()
        cif_url = row.get("cifUrl")
        if entry_id == prefix and cif_url:
            return str(cif_url), str(row.get("modelCreatedDate", ""))
    row = rows[0]
    cif_url = row.get("cifUrl")
    if not cif_url:
        raise ValueError("AlphaFold API row has no cifUrl")
    return str(cif_url), str(row.get("modelCreatedDate", ""))


def shard_for(accession: str) -> str:
    return accession[:2].lower()


def process_one(raw_id: str, args) -> tuple[str, list[str]]:
    source_id, accession, fragment, source_version = parse_simplefold_id(raw_id)
    cif_url, model_created = resolve_cif_url(accession, fragment, args.timeout, args.retries)
    model_id = Path(cif_url).stem
    shard = shard_for(accession)
    out_path = Path(args.out_dir) / shard / f"{model_id}.npz"
    cif_path = Path(args.cif_dir) / shard / f"{model_id}.cif" if args.cif_dir else None

    if out_path.exists() and out_path.stat().st_size > 0 and not args.overwrite:
        return "skip", [
            source_id, accession, fragment, source_version, model_id, model_created,
            str(out_path), str(cif_path or ""), "skip", "",
        ]

    cif_text = ""
    if cif_path and cif_path.exists() and cif_path.stat().st_size > 0 and not args.overwrite:
        cif_text = cif_path.read_text()
    else:
        cif_text = _read_text(cif_url, timeout=args.timeout, retries=args.retries)
        if cif_path:
            cif_path.parent.mkdir(parents=True, exist_ok=True)
            cif_path.write_text(cif_text)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = convert(parse_cif(cif_text), model_id, verbose=False)
    np.savez_compressed(out_path, **arrays)

    return "ok", [
        source_id, accession, fragment, source_version, model_id, model_created,
        str(out_path), str(cif_path or ""), "ok", "",
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id_list", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cif_dir", default=None)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_in_flight", type=int, default=0,
                    help="Cap queued futures. Default: workers * 4.")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    ids = [line.strip() for line in Path(args.id_list).read_text().splitlines() if line.strip()]
    if args.start:
        ids = ids[args.start:]
    if args.limit:
        ids = ids[: args.limit]

    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "source_id", "accession", "fragment", "source_version", "model_id",
        "model_created", "npz_path", "cif_path", "status", "error",
    ]

    max_in_flight = args.max_in_flight or max(1, args.workers * 4)
    ok = skip = fail = 0
    t0 = time.time()
    with manifest.open("w") as fh, ThreadPoolExecutor(max_workers=args.workers) as ex:
        fh.write("\t".join(header) + "\n")
        fh.flush()
        pending = {}
        next_idx = 0
        done_count = 0

        def submit_until_full() -> None:
            nonlocal next_idx
            while next_idx < len(ids) and len(pending) < max_in_flight:
                raw_id = ids[next_idx]
                pending[ex.submit(process_one, raw_id, args)] = raw_id
                next_idx += 1

        submit_until_full()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                raw_id = pending.pop(fut)
                done_count += 1
                try:
                    status, row = fut.result()
                except Exception as exc:  # noqa: BLE001
                    status = "fail"
                    try:
                        source_id, accession, fragment, version = parse_simplefold_id(raw_id)
                    except Exception:
                        source_id, accession, fragment, version = raw_id, "", "", ""
                    row = [source_id, accession, fragment, version, "", "", "", "", "fail",
                           f"{type(exc).__name__}: {exc}"]
                fh.write("\t".join(row) + "\n")
                if status == "ok":
                    ok += 1
                elif status == "skip":
                    skip += 1
                else:
                    fail += 1
                if done_count % 100 == 0 or done_count == len(ids):
                    fh.flush()
                    rate = done_count / max(time.time() - t0, 1e-6)
                    print(f"[{done_count}/{len(ids)}] ok={ok} skip={skip} fail={fail} "
                          f"rate={rate:.2f}/s", flush=True)
            submit_until_full()

    print(f"Done. ok={ok} skip={skip} fail={fail}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
