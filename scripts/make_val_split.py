"""Cluster RCSB npz dataset by sequence identity and split into train/val.

Usage:
    python scripts/make_val_split.py \
        --data_dir data/rcsb \
        --out_dir data/splits \
        --min_seq_id 0.3 \
        --val_ratio 0.05 \
        --seed 42

Steps:
    1. Extract protein sequences from .npz files → FASTA
    2. Cluster with MMseqs2 at --min_seq_id threshold
    3. Assign clusters to train/val (cluster-level, no homolog leakage)
    4. Write train.txt / val.txt file lists
"""

import argparse
import random
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mambafold.data.constants import AA_3TO1 as AA3TO1

MMSEQS_BIN = Path.home() / "mmseqs" / "bin" / "mmseqs"


def extract_sequences(data_dir: Path, min_length: int = 20):
    """Extract protein sequences from npz files.

    Returns:
        dict[str, str]: {npz_relative_path: concatenated_protein_sequence}
    """
    sequences = {}
    npz_files = sorted(data_dir.rglob("*.npz"))
    print(f"Scanning {len(npz_files)} npz files...")

    for i, f in enumerate(npz_files):
        if i % 10000 == 0:
            print(f"  {i}/{len(npz_files)}", flush=True)
        try:
            data = np.load(f)
            chains = data["chains"]
            residues = data["residues"]

            seq_parts = []
            for ch in chains:
                if ch["mol_type"] != 0:  # protein only
                    continue
                r_start = int(ch["res_idx"])
                r_end = r_start + int(ch["res_num"])
                for r in residues[r_start:r_end]:
                    if r["is_standard"]:
                        seq_parts.append(AA3TO1.get(str(r["name"]), "X"))

            seq = "".join(seq_parts)
            if len(seq) >= min_length:
                key = str(f.relative_to(data_dir))
                sequences[key] = seq
        except Exception:
            continue

    print(f"Extracted {len(sequences)} sequences (min_length={min_length})")
    return sequences


def run_mmseqs_cluster(sequences: dict, min_seq_id: float, tmpdir: Path):
    """Cluster sequences with MMseqs2.

    Returns:
        dict[str, list[str]]: {representative: [member_ids]}
    """
    fasta_path = tmpdir / "sequences.fasta"
    db_path = tmpdir / "seqDB"
    clu_path = tmpdir / "cluDB"
    tsv_path = tmpdir / "clusters.tsv"

    # Write FASTA
    with open(fasta_path, "w") as f:
        for name, seq in sequences.items():
            f.write(f">{name}\n{seq}\n")

    # MMseqs2 pipeline
    print(f"\nRunning MMseqs2 clustering (min_seq_id={min_seq_id})...")
    subprocess.run(
        [str(MMSEQS_BIN), "createdb", str(fasta_path), str(db_path)],
        check=True, capture_output=True,
    )
    subprocess.run(
        [str(MMSEQS_BIN), "cluster", str(db_path), str(clu_path), str(tmpdir),
         "--min-seq-id", str(min_seq_id),
         "-c", "0.8",            # coverage threshold
         "--cov-mode", "0",      # bidirectional coverage
         "--threads", "8"],
        check=True, capture_output=True,
    )
    subprocess.run(
        [str(MMSEQS_BIN), "createtsv", str(db_path), str(db_path),
         str(clu_path), str(tsv_path)],
        check=True, capture_output=True,
    )

    # Parse clusters
    clusters = defaultdict(list)
    with open(tsv_path) as f:
        for line in f:
            rep, member = line.strip().split("\t")
            clusters[rep].append(member)

    print(f"Found {len(clusters)} clusters from {len(sequences)} sequences")
    return dict(clusters)


def split_clusters(clusters: dict, val_ratio: float, seed: int):
    """Split clusters into train/val sets.

    Returns:
        (train_files, val_files): lists of npz relative paths
    """
    rng = random.Random(seed)
    cluster_ids = sorted(clusters.keys())
    rng.shuffle(cluster_ids)

    total = sum(len(v) for v in clusters.values())
    val_target = int(total * val_ratio)

    val_files = []
    train_files = []
    val_count = 0

    for cid in cluster_ids:
        members = clusters[cid]
        if val_count < val_target:
            val_files.extend(members)
            val_count += len(members)
        else:
            train_files.extend(members)

    return sorted(train_files), sorted(val_files)


def main():
    parser = argparse.ArgumentParser(description="Cluster-based train/val split")
    parser.add_argument("--data_dir", type=str, default="data/rcsb")
    parser.add_argument("--out_dir", type=str, default="data/splits")
    parser.add_argument("--min_seq_id", type=float, default=0.3)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--min_length", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract sequences
    sequences = extract_sequences(data_dir, min_length=args.min_length)

    # Step 2: Cluster
    with tempfile.TemporaryDirectory(prefix="mmseqs_") as tmpdir:
        clusters = run_mmseqs_cluster(sequences, args.min_seq_id, Path(tmpdir))

    # Step 3: Split
    train_files, val_files = split_clusters(clusters, args.val_ratio, args.seed)

    # Step 4: Write file lists
    train_path = out_dir / "train.txt"
    val_path = out_dir / "val.txt"
    train_path.write_text("\n".join(train_files) + "\n")
    val_path.write_text("\n".join(val_files) + "\n")

    print(f"\n=== Split Summary ===")
    print(f"Clusters: {len(clusters)}")
    print(f"Train: {len(train_files)} ({len(train_files)/len(sequences)*100:.1f}%)")
    print(f"Val:   {len(val_files)} ({len(val_files)/len(sequences)*100:.1f}%)")
    print(f"Written: {train_path}, {val_path}")


if __name__ == "__main__":
    main()
