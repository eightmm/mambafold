#!/usr/bin/env python
"""Pre-compute sequence-deduplicated ESM embeddings and save as ``.npy``.

The default layout stores exactly one embedding per full canonical sequence at
``{out_dir}/by_sequence/{sha256[:2]}/{sha256}.npy``.  The sequence hash is the
chain-to-embedding mapping, so repeated chains and structures do not duplicate
the large embedding array.  ``occurrence`` layout remains available only for
legacy cache compatibility.

Phase 1: scan inputs and collect full canonical sequences.
Phase 2: shard globally unique sequences and write one embedding per sequence.
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
from mambafold.data.sequence_cache import sequence_embedding_path


def read_fasta(path: Path) -> list[tuple[str, str]]:
    """Read FASTA records as ``(header, sequence)`` pairs."""
    records: list[tuple[str, str]] = []
    header: str | None = None
    chunks: list[str] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(chunks)))
            header = line[1:].split()[0]
            chunks = []
        elif header is None:
            raise ValueError(f"Sequence before FASTA header: {path}")
        else:
            chunks.append(line)
    if header is not None:
        records.append((header, "".join(chunks)))
    return records


def get_protein_chains(npz_path: Path, *, strict: bool = False) -> list[str]:
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
        if strict:
            raise
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--esm_model", default="esm3-open")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--file_list", default=None)
    parser.add_argument(
        "--cache_layout",
        choices=["sequence", "occurrence"],
        default="sequence",
        help="Store one file per unique full sequence (default), or retain the "
             "legacy <stem>_ch<index>.npy occurrence layout.",
    )
    parser.add_argument(
        "--single_chain_fasta",
        default=None,
        help="Optional FASTA produced by build_metadata.py for a monomer-only "
             "dataset. Headers must be <npz_stem>_<chain_name>; embeddings are "
             "written as <npz_stem>_ch0.npy without opening each NPZ.",
    )
    parser.add_argument("--limit", type=int, default=0,
                        help="Debug: process only the first N input files.")
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "float32"],
                        help="Storage dtype for embeddings on disk. "
                             "float16 halves disk use with negligible quality cost.")
    parser.add_argument("--shard_idx", type=int, default=0,
                        help="When running N processes across M GPUs, set each "
                             "process's shard index (0..shard_count-1).")
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--shard_files", action="store_true",
                        help="Partition input files before scanning. This avoids "
                             "each GPU rereading the full NPZ corpus, at the cost "
                             "of deduplicating identical sequences only within a shard.")
    parser.add_argument("--scan_only", action="store_true",
                        help="Build and report the sequence index without loading ESM.")
    parser.add_argument("--fail_on_error", action="store_true",
                        help="Exit non-zero if any sequence embedding fails.")
    args = parser.parse_args()
    assert 0 <= args.shard_idx < args.shard_count
    if args.cache_layout == "sequence" and args.shard_files and args.shard_count > 1:
        raise ValueError(
            "sequence cache layout requires global sequence sharding; "
            "disable --shard_files so identical sequences are computed once"
        )
    np_dtype = np.float16 if args.dtype == "float16" else np.float32

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.file_list:
        lines = Path(args.file_list).read_text().splitlines()
        files = [data_dir / line.strip() for line in lines if line.strip()]
    else:
        files = sorted(data_dir.rglob("*.npz"))
    if args.limit:
        files = files[:args.limit]
    if args.shard_files and args.shard_count > 1:
        files = files[args.shard_idx::args.shard_count]
    print(f"Found {len(files)} files", flush=True)

    # ── Phase 1: collect unique sequences ────────────────────────────────────
    print("Phase 1: scanning sequences...", flush=True)
    # Full canonical sequence -> output paths. Sequence layout always has one
    # content-addressed path; occurrence layout may have many legacy paths.
    seq_to_paths: dict[str, list[Path]] = defaultdict(list)
    n_scan_err = 0

    def add_sequence(seq: str, occurrence_path: Path) -> None:
        if not seq:
            return
        out_path = (
            sequence_embedding_path(out_dir, seq)
            if args.cache_layout == "sequence"
            else occurrence_path
        )
        if args.skip_existing and out_path.exists():
            return
        if args.cache_layout == "sequence":
            # Multiple occurrences map to the same path and must not inflate
            # either storage or the written counter.
            seq_to_paths.setdefault(seq, [out_path])
        else:
            seq_to_paths[seq].append(out_path)

    if args.single_chain_fasta:
        paths_by_stem = {path.stem: path for path in files}
        seen_stems: set[str] = set()
        for header, seq in read_fasta(Path(args.single_chain_fasta)):
            stem, separator, _chain_name = header.rpartition("_")
            if not separator or stem not in paths_by_stem:
                continue
            if stem in seen_stems:
                raise ValueError(f"Multiple FASTA records for monomer stem: {stem}")
            seen_stems.add(stem)
            out_path = out_dir / f"{stem}_ch0.npy"
            add_sequence(seq, out_path)
        n_scan_err = len(paths_by_stem) - len(seen_stems)
    else:
        for i, path in enumerate(files):
            chains = get_protein_chains(path)
            if not chains:
                n_scan_err += 1
                continue
            for j, seq in enumerate(chains):
                out_path = out_dir / f"{path.stem}_ch{j}.npy"
                add_sequence(seq, out_path)
            if (i + 1) % 10000 == 0:
                print(
                    f"  scanned {i+1}/{len(files)}, "
                    f"unique seqs so far: {len(seq_to_paths)}",
                    flush=True,
                )

    # Shard unique sequences across workers so 8 GPUs can run concurrently.
    all_seqs = sorted(seq_to_paths.keys())
    if args.shard_files:
        my_seqs = all_seqs
    else:
        my_seqs = [s for i, s in enumerate(all_seqs) if i % args.shard_count == args.shard_idx]
    n_unique = len(all_seqs)
    n_my = len(my_seqs)
    n_total_outputs = sum(len(v) for v in seq_to_paths.values())
    print(f"Phase 1 done: {n_unique} unique seqs → {n_total_outputs} outputs to write  "
          f"(scan_err={n_scan_err})", flush=True)
    print(f"Shard {args.shard_idx}/{args.shard_count}: {n_my} seqs", flush=True)

    if args.fail_on_error and n_scan_err:
        raise SystemExit(f"Sequence scan failures: {n_scan_err}")

    if args.scan_only:
        return
    if n_my == 0:
        print("Nothing to do for this shard.", flush=True)
        return

    # ── Phase 2: ESM inference + write ───────────────────────────────────────
    print("Phase 2: running ESM...", flush=True)
    embedder = ESMEmbedder(model_name=args.esm_model, device=args.device)

    n_done = 0
    n_err = 0
    for k, seq in enumerate(my_seqs):
        out_paths = seq_to_paths[seq]
        try:
            model_seq = seq[:args.max_length]
            emb = embedder([model_seq])                        # [1, L, d_esm]
            arr = emb[0, :len(model_seq)].cpu().numpy().astype(np_dtype)
            for out_path in out_paths:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(out_path, arr)
            n_done += len(out_paths)
        except Exception as e:
            print(f"Error seq[{k}] len={len(seq)}: {e}", flush=True)
            n_err += 1
            torch.cuda.empty_cache()

        if (k + 1) % 100 == 0:
            torch.cuda.empty_cache()
            print(f"  [{k+1}/{n_my}] written={n_done} err={n_err}", flush=True)

    print(f"Done. written={n_done} err={n_err}", flush=True)
    if args.fail_on_error and n_err:
        raise SystemExit(f"Embedding failures: {n_err}")


if __name__ == "__main__":
    main()
