#!/usr/bin/env python
"""Compute ESMC-6B embeddings for the external benchmark FASTAs.

The rollout is the only evidence this project produces, and it cannot run
without embeddings for the CASP targets. They are absent by construction: the
cache is sequence-addressed and populated from the training corpus, while the
benchmark targets are exactly the sequences the corpus must not contain. A
count against the two training caches finds 0 of 70, 1 of 22 and 2 of 21 — the
three that do hit are coincidental matches, not leakage of the structures.

Written into a separate `data/casp_esmc6b` rather than the training caches, so
a benchmark sequence can never be mistaken for a training record. The legacy
directory name now includes CAMEO22 as well as CASP; changing it would only
duplicate a large sequence-addressed cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mambafold.data.sequence_cache import sequence_embedding_path  # noqa: E402


def read_fasta(path: Path) -> dict[str, str]:
    seqs: dict[str, str] = {}
    name: str | None = None
    buf: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                seqs[name] = "".join(buf)
            name = line[1:].split()[0]
            buf = []
        elif line.strip():
            buf.append(line.strip())
    if name is not None:
        seqs[name] = "".join(buf)
    return seqs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta_dir", default="benchmarks/external_testsets")
    ap.add_argument("--out_dir", default="data/casp_esmc6b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--report", default="data/audit/casp_embeddings.json")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    fastas = sorted((ROOT / args.fasta_dir).glob("*.fasta"))
    if not fastas:
        print(f"no FASTA under {args.fasta_dir}", file=sys.stderr)
        return 2
    out_dir = ROOT / args.out_dir

    wanted: dict[str, str] = {}
    per_set: dict[str, list[str]] = {}
    for fasta in fastas:
        seqs = read_fasta(fasta)
        per_set[fasta.name] = sorted(seqs)
        wanted.update(seqs)
    todo = {
        name: seq
        for name, seq in wanted.items()
        if args.force or not sequence_embedding_path(out_dir, seq).exists()
    }
    print(f"{len(wanted)} target sequences across {len(fastas)} sets; "
          f"{len(todo)} to embed", flush=True)

    written = 0
    if todo:
        from mambafold.data.esm import ESMEmbedder

        embedder = ESMEmbedder(device=args.device)
        started = time.time()
        for index, (name, seq) in enumerate(sorted(todo.items()), 1):
            emb = embedder([seq])[0][: len(seq)].to(torch.float32).cpu().numpy()
            if emb.shape[0] != len(seq):
                raise RuntimeError(
                    f"{name}: got {emb.shape[0]} embedding rows for {len(seq)} residues"
                )
            path = sequence_embedding_path(out_dir, seq)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".npy.tmp")
            # Write through an open handle. `np.save` appends ".npy" to any
            # filename that does not already end in it, so `np.save(tmp, ...)`
            # with a ".npy.tmp" name silently creates ".npy.tmp.npy" and the
            # rename below then fails on a file that was never written.
            with open(tmp, "wb") as fh:
                np.save(fh, emb)
            tmp.replace(path)
            written += 1
            if index % 10 == 0 or index == len(todo):
                print(f"  {index}/{len(todo)}  {time.time()-started:.0f}s", flush=True)

    report = {"out_dir": str(out_dir), "sets": {}}
    for fasta in fastas:
        seqs = read_fasta(fasta)
        present = {n: sequence_embedding_path(out_dir, s).exists() for n, s in seqs.items()}
        report["sets"][fasta.name] = {
            "targets": len(seqs),
            "embedded": sum(present.values()),
            "missing": sorted(n for n, ok in present.items() if not ok),
            "lengths": {"min": min(map(len, seqs.values())), "max": max(map(len, seqs.values()))},
            "fasta_sha256": hashlib.sha256(fasta.read_bytes()).hexdigest(),
        }
        s = report["sets"][fasta.name]
        print(f"  {fasta.name:34} {s['embedded']}/{s['targets']} embedded")
    report["written_this_run"] = written
    out = ROOT / args.report
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"report -> {out}")
    return 0 if all(v["embedded"] == v["targets"] for v in report["sets"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
