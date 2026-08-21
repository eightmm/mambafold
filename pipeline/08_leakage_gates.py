#!/usr/bin/env python
"""Stage 08 — exact-sequence leakage gate over the full training corpus.

Runs the published benchmarks against the union of both training sources. An
RCSB-only pass is not the gate: the AFDB distillation stream is training data
too, and a target matching a SwissProt sequence is just as exposed.

This checks exact identity only. It is the first of two gates; the MMseqs2
homology screen in ``benchmarks/BENCHMARK_POLICY.md`` is the second, and no
generalization claim stands on this one alone.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ["casp14_70", "casp15_single_chain_22", "casp16_single_chain_21"]


def afdb_fasta_from_sequences(src: Path, dst: Path) -> int:
    """Write one FASTA record per converted entry.

    ``sequences.tsv`` is appended to, so a resumed conversion can record the
    same entry twice — identical content, duplicate row. The readers elsewhere
    key by entry id and collapse that silently; this writer would emit both and
    produce a FASTA the audit tool rejects outright. Collapse here too, keeping
    the last row for an id, which is the one the latest conversion wrote.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    records: dict[str, str] = {}
    with src.open() as handle:
        for index, line in enumerate(handle):
            if index == 0 and line.startswith("entry_id\t"):
                continue
            entry_id, _, sequence = line.rstrip("\n").partition("\t")
            if entry_id and sequence:
                records[entry_id] = sequence

    with dst.open("w") as out:
        for entry_id, sequence in records.items():
            out.write(f">{entry_id}\n")
            for start in range(0, len(sequence), 80):
                out.write(sequence[start:start + 80] + "\n")
    return len(records)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args()

    root = Path(args.root).resolve()
    audit = root / "data" / "audit"
    training = [audit / "rcsb-admitted.fasta"]

    v4_sequences = root / "data" / "afdb_swissprot_v4" / "sequences.tsv"
    if v4_sequences.exists():
        afdb_fasta = audit / "afdb-v4-training.fasta"
        count = afdb_fasta_from_sequences(v4_sequences, afdb_fasta)
        print(f"AFDB v4 training sequences: {count}", flush=True)
        training.append(afdb_fasta)
    else:
        print("WARNING: AFDB v4 sequences absent; gate covers the RCSB half only", flush=True)

    summary = {}
    for name in BENCHMARKS:
        out = audit / f"{name}-exact-overlap.json"
        out.unlink(missing_ok=True)
        cmd = [
            sys.executable,
            str(root / "benchmarks" / "audit_sequence_overlap.py"),
            "--targets", str(root / "benchmarks" / "external_testsets" / f"{name}.fasta"),
            "--out", str(out),
        ]
        for fasta in training:
            cmd += ["--training", str(fasta)]
        subprocess.run(cmd, check=True)
        result = json.loads(out.read_text())["result"]
        summary[name] = {
            "targets": result["target_records"],
            "training_records": result["training_records"],
            "exact_overlap": result["exact_overlap_targets"],
            "clean": result["exact_clean_targets"],
            "sources": [f.name for f in training],
        }

    (audit / "leakage_gate_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
