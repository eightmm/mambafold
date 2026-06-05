"""Scan a directory of MambaFold .npz files and emit:

1. metadata.tsv  — one row per .npz (per-chain fields joined)
2. sequences.fasta — one record per protein chain, id = `<pdb_id>_<chain_name>`

Used by make_val_split.py for MMseqs2 clustering, and by the dataloader for
quick filtering (`file_list` + seq-length stratification).
"""

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mambafold.data.constants import AA_3TO1  # noqa: E402

MOL_TYPE_PROTEIN = 0


def _scan_one(npz_path: Path):
    try:
        npz = np.load(npz_path, allow_pickle=False)
        chains = npz["chains"]
        residues = npz["residues"]
    except Exception as e:  # noqa: BLE001
        return {"pdb_id": npz_path.stem, "error": f"{type(e).__name__}: {e}",
                "rows": [], "fasta": []}

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
        seq = "".join(AA_3TO1.get(str(r["name"]), "X") for r in slc)
        n_std = int(sum(r["is_standard"] for r in slc))
        n_obs = int(sum(r["is_present"] for r in slc))
        rows.append({
            "pdb_id":       pdb_id,
            "chain":        chain_name,
            "seq_len":      len(seq),
            "n_standard":   n_std,
            "n_observed":   n_obs,
        })
        if len(seq) >= 10:
            fasta.append(f">{pdb_id}_{chain_name}\n{seq}\n")
    return {"pdb_id": pdb_id, "error": "", "rows": rows, "fasta": fasta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--out_tsv",  required=True)
    ap.add_argument("--out_fasta", required=True)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2))
    args = ap.parse_args()

    npz_dir = Path(args.npz_dir)
    paths = sorted(npz_dir.rglob("*.npz"))
    print(f"Scanning {len(paths)} npz files ...")

    out_tsv = Path(args.out_tsv)
    out_fasta = Path(args.out_fasta)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    out_fasta.parent.mkdir(parents=True, exist_ok=True)

    n_chains = n_err = 0
    t0 = time.time()
    with mp.Pool(args.workers) as pool, \
         open(out_tsv, "w") as ftsv, \
         open(out_fasta, "w") as ffa:
        ftsv.write("pdb_id\tchain\tseq_len\tn_standard\tn_observed\n")
        for i, rec in enumerate(pool.imap_unordered(_scan_one, paths, chunksize=32), 1):
            if rec["error"]:
                n_err += 1
                continue
            for row in rec["rows"]:
                ftsv.write("\t".join(str(row[k]) for k in
                    ("pdb_id", "chain", "seq_len", "n_standard", "n_observed")) + "\n")
                n_chains += 1
            for r in rec["fasta"]:
                ffa.write(r)
            if i % 10000 == 0 or i == len(paths):
                dt = time.time() - t0
                print(f"[{i}/{len(paths)}] chains={n_chains} err={n_err} "
                      f"rate={i/dt:.0f}/s", flush=True)
    print(f"\nDone. n_protein_chains={n_chains}, n_errors={n_err}")
    print(f"  {out_tsv}\n  {out_fasta}")


if __name__ == "__main__":
    main()
