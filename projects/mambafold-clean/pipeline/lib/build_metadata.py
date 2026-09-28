"""Scan a directory of MambaFold .npz files and emit:

1. metadata.tsv  — one row per .npz (per-chain fields joined)
2. sequences.fasta — one record per protein chain, id = `<pdb_id>_<chain_name>`

Used by make_val_split.py for MMseqs2 clustering, and by the dataloader for
quick filtering (`file_list` + seq-length stratification).

When ``--file_list`` is provided, only those coordinate-training records are
exported. Output order is deterministic so the FASTA can be content-addressed
in benchmark leakage reports.
"""

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mambafold.data.constants import AA_3TO1, AA_TO_ID  # noqa: E402

MOL_TYPE_PROTEIN = 0


def canonical_chain(residues: np.ndarray) -> tuple[str, int]:
    """Mirror the training loader's canonical protein-residue filter."""
    kept_names = []
    observed = 0
    for residue in residues:
        name = str(residue["name"])
        if not bool(residue["is_standard"]) or name not in AA_TO_ID or name == "UNK":
            continue
        kept_names.append(name)
        observed += int(bool(residue["is_present"]))
    return "".join(AA_3TO1[name] for name in kept_names), observed


def resolve_paths(npz_dir: Path, file_list: Path | None) -> list[Path]:
    if file_list is None:
        return sorted(npz_dir.rglob("*.npz"))
    relative_paths = [
        line.strip()
        for line in file_list.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(relative_paths) != len(set(relative_paths)):
        raise ValueError(f"duplicate entries in file list: {file_list}")
    paths = [npz_dir / relative_path for relative_path in relative_paths]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        preview = missing[:10]
        raise FileNotFoundError(
            f"{len(missing)} file-list entries are missing under {npz_dir}: {preview}"
        )
    return paths


def _scan_one(npz_path: Path):
    try:
        npz = np.load(npz_path, allow_pickle=False)
        chains = npz["chains"]
        residues = npz["residues"]
    except Exception as e:  # noqa: BLE001
        return {
            "pdb_id": npz_path.stem,
            "error": f"{type(e).__name__}: {e}",
            "rows": [],
            "fasta": [],
        }

    rows = []
    fasta = []
    pdb_id = npz_path.stem
    for ch in chains:
        if int(ch["mol_type"]) != MOL_TYPE_PROTEIN:
            continue
        chain_name = str(ch["name"]).strip()
        r0 = int(ch["res_idx"])
        rN = r0 + int(ch["res_num"])
        slc = residues[r0:rN]
        seq, n_obs = canonical_chain(slc)
        n_std = len(seq)
        rows.append(
            {
                "pdb_id": pdb_id,
                "chain": chain_name,
                "seq_len": len(seq),
                "n_standard": n_std,
                "n_observed": n_obs,
            }
        )
        if len(seq) >= 10:
            fasta.append(f">{pdb_id}_{chain_name}\n{seq}\n")
    return {"pdb_id": pdb_id, "error": "", "rows": rows, "fasta": fasta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--file_list", type=Path, default=None)
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--out_fasta", required=True)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2))
    ap.add_argument("--fail_on_error", action="store_true")
    args = ap.parse_args()

    npz_dir = Path(args.npz_dir)
    paths = resolve_paths(npz_dir, args.file_list)
    print(f"Scanning {len(paths)} npz files ...")

    out_tsv = Path(args.out_tsv)
    out_fasta = Path(args.out_fasta)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    out_fasta.parent.mkdir(parents=True, exist_ok=True)

    n_chains = n_err = 0
    t0 = time.time()
    with mp.Pool(args.workers) as pool, open(out_tsv, "w") as ftsv, open(out_fasta, "w") as ffa:
        ftsv.write("pdb_id\tchain\tseq_len\tn_standard\tn_observed\n")
        for i, rec in enumerate(pool.imap(_scan_one, paths, chunksize=32), 1):
            if rec["error"]:
                n_err += 1
                continue
            for row in rec["rows"]:
                ftsv.write(
                    "\t".join(
                        str(row[k])
                        for k in ("pdb_id", "chain", "seq_len", "n_standard", "n_observed")
                    )
                    + "\n"
                )
                n_chains += 1
            for r in rec["fasta"]:
                ffa.write(r)
            if i % 10000 == 0 or i == len(paths):
                dt = time.time() - t0
                print(
                    f"[{i}/{len(paths)}] chains={n_chains} err={n_err} rate={i / dt:.0f}/s",
                    flush=True,
                )
    print(f"\nDone. n_protein_chains={n_chains}, n_errors={n_err}")
    print(f"  {out_tsv}\n  {out_fasta}")
    if args.fail_on_error and n_err:
        raise SystemExit(f"metadata export incomplete: n_errors={n_err}")


if __name__ == "__main__":
    main()
