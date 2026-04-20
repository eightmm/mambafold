#!/usr/bin/env python
"""Pre-compute ESM embeddings per protein chain and save as .npy.

Deduplication: identical sequences share one ESM forward pass.
Output: {out_dir}/{stem}_ch{j}.npy  for j = 0, 1, ... (protein chain index)

Phase 1: scan all npz files → build {seq: [(path, chain_idx), ...]}
Phase 2: for each unique seq, run ESM once → write to all matching files
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mambafold.data.constants import AA_3TO1, AA_TO_ID
from mambafold.data.esm import ESMEmbedder


def get_protein_chains(npz_path: Path) -> list[str]:
    """Return one sequence string per protein chain (same filter as RCSBDataset)."""
    try:
        data = np.load(npz_path)
        residues = data["residues"]
        chains = data["chains"]
        result = []
        for ch in chains:
            if ch["mol_type"] != 0:
                continue
            r_start = int(ch["res_idx"])
            r_end = r_start + int(ch["res_num"])
            seq = []
            for i in range(r_start, r_end):
                res = residues[i]
                name = str(res["name"])
                if not res["is_standard"] or name not in AA_TO_ID or name == "UNK":
                    continue
                seq.append(AA_3TO1.get(name, "X"))
            if seq:
                result.append("".join(seq))
        return result
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--esm_model", default="esm3-open")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--file_list", default=None)
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--max_length", type=int, default=2048)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.file_list:
        files = [data_dir / l.strip() for l in Path(args.file_list).read_text().splitlines() if l.strip()]
    else:
        files = sorted(data_dir.rglob("*.npz"))
    print(f"Found {len(files)} files", flush=True)

    # ── Phase 1: collect unique sequences ────────────────────────────────────
    print("Phase 1: scanning sequences...", flush=True)
    # seq -> list of output paths that need this embedding
    seq_to_paths: dict[str, list[Path]] = defaultdict(list)
    n_scan_err = 0
    for i, path in enumerate(files):
        chains = get_protein_chains(path)
        if not chains:
            n_scan_err += 1
            continue
        for j, seq in enumerate(chains):
            seq = seq[:args.max_length]
            out_path = out_dir / f"{path.stem}_ch{j}.npy"
            if args.skip_existing and out_path.exists():
                continue
            seq_to_paths[seq].append(out_path)
        if (i + 1) % 10000 == 0:
            print(f"  scanned {i+1}/{len(files)}, unique seqs so far: {len(seq_to_paths)}", flush=True)

    n_unique = len(seq_to_paths)
    n_total_files = sum(len(v) for v in seq_to_paths.values())
    print(f"Phase 1 done: {n_unique} unique seqs → {n_total_files} files to write  (scan_err={n_scan_err})", flush=True)

    if n_unique == 0:
        print("Nothing to do.", flush=True)
        return

    # ── Phase 2: ESM inference + write ───────────────────────────────────────
    print("Phase 2: running ESM...", flush=True)
    embedder = ESMEmbedder(model_name=args.esm_model, device=args.device)

    n_done = 0
    n_err = 0
    for k, (seq, out_paths) in enumerate(seq_to_paths.items()):
        try:
            emb = embedder([seq])                              # [1, L, d_esm]
            arr = emb[0, :len(seq)].cpu().numpy().astype(np.float32)
            for out_path in out_paths:
                np.save(out_path, arr)
            n_done += len(out_paths)
        except Exception as e:
            print(f"Error seq[{k}] len={len(seq)}: {e}", flush=True)
            n_err += 1
            torch.cuda.empty_cache()

        if (k + 1) % 100 == 0:
            torch.cuda.empty_cache()
            print(f"  [{k+1}/{n_unique}] written={n_done} err={n_err}", flush=True)

    print(f"Done. written={n_done} err={n_err}", flush=True)


if __name__ == "__main__":
    main()
