"""Pull per-entry deposit dates from local mmCIF.gz files.

mmCIF exposes the deposit date via:
    _pdbx_database_status.recvd_initial_deposition_date   YYYY-MM-DD

We grep the field as text to avoid BioPython's slow full parse.
Output: TSV with (pdb_id, deposit_date), and a holdout_ids.txt of entries
whose deposit_date >= --cutoff.
"""

import argparse
import gzip
import multiprocessing as mp
import re
import time
from pathlib import Path

# Matches both loop form:
#   _pdbx_database_status.recvd_initial_deposition_date    2022-07-15
# and column form inside a loop_.
DATE_RE = re.compile(
    r"_pdbx_database_status\.recvd_initial_deposition_date\s+['\"]?(\d{4}-\d{2}-\d{2})",
)


def _scan_one(path: Path):
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            head = f.read(64 * 1024)        # first 64 KB — deposit date is always in header
        m = DATE_RE.search(head)
        if m:
            return path.stem.removesuffix(".cif").lower(), m.group(1)
        # fallback: some entries have the date further in
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        m = DATE_RE.search(text)
        return path.stem.removesuffix(".cif").lower(), m.group(1) if m else ""
    except Exception:
        return path.stem.removesuffix(".cif").lower(), ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif_dir", required=True)
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--out_holdout", required=True)
    ap.add_argument("--cutoff", default="2022-07-01",
                    help="YYYY-MM-DD; entries deposited on/after go to holdout")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2))
    args = ap.parse_args()

    paths = sorted(Path(args.cif_dir).rglob("*.cif.gz"))
    print(f"Scanning {len(paths)} cif.gz files with {args.workers} workers")

    t0 = time.time()
    pairs: list[tuple[str, str]] = []
    missing = 0
    with mp.Pool(args.workers) as pool:
        for i, rec in enumerate(pool.imap_unordered(_scan_one, paths, chunksize=64), 1):
            pairs.append(rec)
            if not rec[1]:
                missing += 1
            if i % 20000 == 0 or i == len(paths):
                dt = time.time() - t0
                print(f"[{i}/{len(paths)}] rate={i/dt:.0f}/s missing={missing}",
                      flush=True)

    pairs.sort()
    out_tsv = Path(args.out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_tsv, "w") as f:
        f.write("pdb_id\tdeposit_date\n")
        for pid, date in pairs:
            f.write(f"{pid}\t{date}\n")

    holdout = [pid for pid, date in pairs if date and date >= args.cutoff]
    Path(args.out_holdout).write_text("\n".join(sorted(holdout)) + "\n")

    print(f"\nWrote {out_tsv} (n={len(pairs)})")
    print(f"Wrote {args.out_holdout} "
          f"(n={len(holdout)} entries deposited on/after {args.cutoff})")
    if missing:
        print(f"WARN: {missing} entries had no deposit-date field")


if __name__ == "__main__":
    main()
