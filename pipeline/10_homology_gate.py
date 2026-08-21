#!/usr/bin/env python
"""Stage 10 — MMseqs2 homology screen, the second and binding leakage gate.

Stage 08 checks exact sequence identity. That is necessary and nowhere near
sufficient: a training sequence one substitution away from a benchmark target is
not an exact match and is very much leakage. This stage runs the screen
`benchmarks/BENCHMARK_POLICY.md` predeclares, and no generalization claim stands
without it.

The rule, frozen before any model score is looked at: exclude a target when
MMseqs2 finds at least 30% sequence identity over at least 80% of the
*benchmark-query* sequence, searched against the union of both training sources.

Coverage is measured on the query only (`--cov-mode 2`). Reciprocal coverage
would be wrong here: training crops contiguous windows, so a short target buried
inside a long training chain is just as available to the model as a
length-matched one, and requiring the training sequence to be covered would let
exactly that case through.

Everything needed to re-run or audit the screen is recorded — MMseqs2 version,
the SHA-256 of every input FASTA, the exact command, and the per-hit identity
and coverage behind each exclusion.

Usage:
    python pipeline/10_homology_gate.py --mmseqs "$MMSEQS_BIN"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ["casp14_70", "casp15_single_chain_22", "casp16_single_chain_21"]

# Frozen before looking at any model score. Do not tune to improve a result.
MIN_IDENTITY = 0.30
MIN_QUERY_COVERAGE = 0.80


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_records(path: Path) -> int:
    return sum(1 for line in path.read_text().splitlines() if line.startswith(">"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--mmseqs", required=True)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--sensitivity", type=float, default=7.5)
    ap.add_argument("--min_identity", type=float, default=MIN_IDENTITY)
    ap.add_argument("--min_query_coverage", type=float, default=MIN_QUERY_COVERAGE)
    ap.add_argument("--keep_tmp", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    audit = root / "data" / "audit"
    mmseqs = Path(args.mmseqs)
    if not mmseqs.is_file():
        raise FileNotFoundError(f"MMseqs2 not found at {mmseqs}")

    training = [audit / "rcsb-admitted.fasta", audit / "afdb-v4-training.fasta"]
    missing = [str(p) for p in training if not p.exists()]
    if missing:
        raise FileNotFoundError(f"training FASTA missing: {missing[0]}; run stages 04 and 08 first")

    version = subprocess.run(
        [str(mmseqs), "version"], capture_output=True, text=True, check=True
    ).stdout.strip()
    print(f"MMseqs2 {version}", flush=True)

    tmp_root = Path(tempfile.mkdtemp(prefix="homology_gate_", dir=str(audit)))
    combined = tmp_root / "training-union.fasta"
    with combined.open("w") as out:
        for path in training:
            out.write(path.read_text())
    print(f"training union: {count_records(combined)} sequences", flush=True)

    summary: dict[str, object] = {
        "rule": {
            "min_identity": args.min_identity,
            "min_query_coverage": args.min_query_coverage,
            "coverage_mode": "query only (--cov-mode 2)",
            "rationale": (
                "training crops contiguous windows, so a target embedded in a longer "
                "training chain is available to the model; requiring reciprocal "
                "coverage would let that case through"
            ),
        },
        "mmseqs_version": version,
        "training_sources": [
            {"path": str(p), "records": count_records(p), "sha256": sha256(p)} for p in training
        ],
        "benchmarks": {},
    }

    for name in BENCHMARKS:
        query = root / "benchmarks" / "external_testsets" / f"{name}.fasta"
        result = tmp_root / f"{name}.m8"
        work = tmp_root / f"work_{name}"
        command = [
            str(mmseqs), "easy-search", str(query), str(combined), str(result), str(work),
            "--min-seq-id", str(args.min_identity),
            "-c", str(args.min_query_coverage),
            "--cov-mode", "2",
            "-s", str(args.sensitivity),
            "--alignment-mode", "3",
            "--max-seqs", "4000",
            "--threads", str(args.threads),
            "--format-output", "query,target,fident,alnlen,qcov,tcov,evalue,bits",
        ]
        print(f"== {name}", flush=True)
        subprocess.run(command, check=True, capture_output=True)

        best: dict[str, dict] = {}
        for line in result.read_text().splitlines():
            fields = line.split("\t")
            if len(fields) < 8:
                continue
            q, target = fields[0], fields[1]
            fident, alnlen, qcov = float(fields[2]), int(fields[3]), float(fields[4])
            if fident < args.min_identity or qcov < args.min_query_coverage:
                continue
            prior = best.get(q)
            if prior is None or fident > prior["identity"]:
                best[q] = {
                    "target": target,
                    "identity": round(fident, 4),
                    "query_coverage": round(qcov, 4),
                    "alignment_length": alnlen,
                }

        targets = count_records(query)
        excluded = sorted(best)
        summary["benchmarks"][name] = {  # type: ignore[index]
            "query_fasta": str(query),
            "query_sha256": sha256(query),
            "targets": targets,
            "excluded": len(excluded),
            "admitted": targets - len(excluded),
            "command": " ".join(command),
            "hits": {q: best[q] for q in excluded},
        }
        print(f"   {len(excluded)} of {targets} excluded by homology", flush=True)
        for q in excluded:
            hit = best[q]
            print(
                f"     {q}  id={hit['identity']:.3f} "
                f"qcov={hit['query_coverage']:.3f}  <- {hit['target']}"
            )

        shutil.copy2(result, audit / f"{name}-homology.m8")

    out_path = audit / "homology_gate.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"report -> {out_path}")

    if not args.keep_tmp:
        shutil.rmtree(tmp_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
