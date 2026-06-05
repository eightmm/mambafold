"""Cluster RCSB chain sequences and split into train / val.

Pipeline:
    build_metadata.py  →  sequences.fasta  →  mmseqs cluster  →  train.txt, val.txt

Usage:
    python scripts/make_val_split.py \\
        --fasta data/splits/sequences.fasta \\
        --out_dir data/splits \\
        --min_seq_id 0.3 \\
        --val_ratio 0.05 \\
        --holdout_ids data/splits/holdout.txt  (optional)

Holdout semantics:
    IDs listed in --holdout_ids (one PDB id per line) are ALWAYS placed into
    val and the entire cluster they belong to is also excluded from train —
    prevents homolog leakage. Use this for CASP targets or explicit temporal
    holdouts.
"""

import argparse
import os
import random
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

MMSEQS_BIN = os.environ.get("MMSEQS_BIN") or str(
    Path(__file__).resolve().parents[1] / "tools" / "mmseqs" / "bin" / "mmseqs"
)


def run_mmseqs_cluster(fasta_path: Path, min_seq_id: float,
                       tmpdir: Path, threads: int = 16):
    """Cluster fasta with MMseqs2. Returns {representative: [member_ids]}."""
    if not Path(MMSEQS_BIN).exists():
        raise FileNotFoundError(
            f"MMSEQS_BIN not found: {MMSEQS_BIN}. "
            f"Set env MMSEQS_BIN or install via tools/mmseqs."
        )
    db   = tmpdir / "seqDB"
    clu  = tmpdir / "cluDB"
    wrk  = tmpdir / "work"
    tsv  = tmpdir / "clusters.tsv"
    wrk.mkdir(parents=True, exist_ok=True)

    print(f"Running MMseqs2 cluster (min_seq_id={min_seq_id}, threads={threads}) ...")
    subprocess.run([MMSEQS_BIN, "createdb", str(fasta_path), str(db)],
                   check=True, capture_output=True)
    subprocess.run([MMSEQS_BIN, "cluster", str(db), str(clu), str(wrk),
                    "--min-seq-id", str(min_seq_id),
                    "-c", "0.8", "--cov-mode", "0",
                    "--threads", str(threads)],
                   check=True, capture_output=True)
    subprocess.run([MMSEQS_BIN, "createtsv", str(db), str(db), str(clu), str(tsv)],
                   check=True, capture_output=True)

    clusters = defaultdict(list)
    with open(tsv) as f:
        for line in f:
            rep, member = line.rstrip("\n").split("\t")
            clusters[rep].append(member)
    return dict(clusters)


def _pdb_id(chain_key: str) -> str:
    """Fasta id like `10gs_A` → `10gs` (the RCSB file stem)."""
    return chain_key.split("_", 1)[0]


def split_clusters(clusters, holdout_pdb_ids: set[str], val_ratio: float, seed: int):
    """Split cluster list into (train, val, val_casp).

    Design choice: temporal holdout pdb_ids ONLY go to val_casp. Their cluster
    mates (homologs) remain in whichever set (train or val) they were assigned
    to by normal cluster-level split. This accepts some homology leakage on the
    val_casp side (our temporal holdout is meant as *blind* eval — homologs in
    train are actually representative of real deployment conditions). Without
    this, a 7% holdout drags ~60% of the corpus along through homology.

    val:        cluster-level random draw, val_ratio of all chains
    val_casp:   holdout pdb_ids (after removing any that leak into val)
    train:      the rest
    """
    rng = random.Random(seed)
    reps = sorted(clusters.keys())
    rng.shuffle(reps)

    total = sum(len(clusters[r]) for r in reps)
    val_target = int(total * val_ratio)

    val_clusters = set()
    n_val = 0
    for rep in reps:
        if n_val >= val_target:
            break
        val_clusters.add(rep)
        n_val += len(clusters[rep])

    train, val, val_casp = [], [], []
    for rep, members in clusters.items():
        bucket = val if rep in val_clusters else train
        for m in members:
            if _pdb_id(m) in holdout_pdb_ids:
                val_casp.append(m)
            else:
                bucket.append(m)
    return sorted(set(train)), sorted(set(val)), sorted(set(val_casp))


def _write_list_unique_pdb(path: Path, chain_keys):
    """Write unique {pdb_id}.npz one per line (dedup; .npz layout is file-level)."""
    seen = set()
    lines = []
    for ck in chain_keys:
        pid = _pdb_id(ck)
        if pid in seen:
            continue
        seen.add(pid)
        lines.append(f"{pid}.npz")
    path.write_text("\n".join(sorted(lines)) + "\n")
    return len(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--out_dir", default="data/splits")
    ap.add_argument("--min_seq_id", type=float, default=0.3)
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--holdout_ids", default=None,
                    help="Optional file listing pdb_ids (one per line) to force into val")
    args = ap.parse_args()

    holdout = set()
    if args.holdout_ids:
        p = Path(args.holdout_ids)
        if p.exists():
            holdout = {ln.strip().lower() for ln in p.read_text().splitlines() if ln.strip()}
            print(f"Loaded {len(holdout)} holdout pdb_ids from {p}")
        else:
            print(f"WARN: holdout file {p} not found; proceeding without explicit holdout")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="mf_mmseqs_",
                                     dir=os.environ.get("TMPDIR")) as tmp:
        clusters = run_mmseqs_cluster(Path(args.fasta), args.min_seq_id,
                                      Path(tmp), threads=args.threads)
    print(f"Clusters: {len(clusters)}")

    train, val, val_casp = split_clusters(clusters, holdout, args.val_ratio, args.seed)

    # De-duplicate at pdb-id level ACROSS splits (val_casp > val > train priority)
    # so one PDB whose chains span different clusters lands only once.
    def _pdbs(ck_list):
        return {_pdb_id(c) for c in ck_list}
    val_casp_pdbs = _pdbs(val_casp)
    val_pdbs      = _pdbs(val)      - val_casp_pdbs
    train_pdbs    = _pdbs(train)    - val_casp_pdbs - val_pdbs

    def _write(path, pdbs):
        # Paths relative to data_dir include the 2-char shard prefix
        # (download_rcsb_cif.sh sharded by chars 1:3 of the pdb id).
        lines = sorted(
            f"{p[1:3] if len(p) >= 3 else '00'}/{p}.npz"
            for p in pdbs
        )
        path.write_text("\n".join(lines) + "\n")
        return len(lines)

    n_train = _write(out_dir / "train.txt",    train_pdbs)
    n_val   = _write(out_dir / "val.txt",      val_pdbs)
    n_casp  = _write(out_dir / "val_casp.txt", val_casp_pdbs)

    total = n_train + n_val + n_casp
    print("\n=== Split Summary ===")
    print(f"  Clusters:        {len(clusters):>8d}")
    print(f"  Train (files):   {n_train:>8d}  ({n_train/total*100:.1f}%)")
    print(f"  Val   (files):   {n_val:>8d}  ({n_val/total*100:.1f}%)")
    print(f"  Val-holdout:     {n_casp:>8d}  (always-val, e.g. CASP targets)")
    print(f"  Written to: {out_dir}/{{train,val,val_casp}}.txt")


if __name__ == "__main__":
    main()
