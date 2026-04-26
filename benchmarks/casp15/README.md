# CASP15-multimer benchmark

Official CASP15 oligomer targets — for direct comparison against published
AF-Multimer / Boltz-1 / Chai-1 numbers. Complements the in-house holdout
benchmark (`benchmarks/sets/`).

## Setup (one-time)

```bash
# 1. Download CASP15 sources from predictioncenter.org
bash benchmarks/casp15/download.sh
#    → benchmarks/casp15/raw/casp15.seq.txt
#    → benchmarks/casp15/raw/oligo/{H*,T*o}.pdb        (50 GT PDBs)

# 2. Parse oligomer GT → per-target manifest + per-chain FASTA
.venv/bin/python benchmarks/casp15/parse_targets.py
#    → benchmarks/casp15/manifest.tsv                  (39 multimer targets)
#    → benchmarks/casp15/targets_protein_multimer.txt  (id list)
#    → benchmarks/casp15/sequences/<target>.fasta      (one record per chain)
```

## What we get

39 protein multimer targets after filtering:

| metric | count |
|---|---|
| heteromers | 29 |
| homomers   | 10 |
| dimers (2 chains) | 23 |
| trimers (3 chains) | 6 |
| tetramer+ | 10 |

Largest assembly: `H1144` (A1B1C1…Z1, 27 chains). Within-crop targets
(≤ 2048 residues) cover the vast majority of real CASP15 multimer entries.

## Layout

```
benchmarks/casp15/
├── download.sh                       # one-shot fetch from predictioncenter.org
├── parse_targets.py                  # PDB → manifest + per-chain FASTA
├── manifest.tsv                      # target_id, kind, n_chains, stoichiometry, ...
├── targets_protein_multimer.txt      # plain id list (consumed by runner)
├── sequences/<target>.fasta          # per-chain records, header `>{target}|{chain}`
└── raw/                              # gitignored — sources from predictioncenter.org
    ├── casp15.seq.txt
    ├── casp15.targets.oligo.tar.gz
    └── oligo/<id>.pdb                # canonical reference structures
```

## TODO — FASTA-input multi-chain inference adapter

The current `benchmarks/run_inference.py` reads RCSBDataset npz (which
needs ground-truth coordinates and ESM cache). For CASP15 we have only
sequences. A new entry point is needed:

```python
# benchmarks/casp15/run_casp15.py  (NOT YET WRITTEN)
#   for target in targets_protein_multimer.txt:
#       build ProteinExample from per-chain FASTAs (chain_id, entity_id, etc.)
#       no coords / no observed_mask (will be sampled from noise)
#       sample_euler  →  predicted PDB
#   write <target>_pred.pdb / <target>_gt.pdb pairs into benchmarks/results/casp15_<ckpt>/
```

This is a moderate (~100 LOC) adaptation of `scripts/infer_seq.py` (which
is single-chain only) plus the multichain `save_pdb_multichain` helper
already in `benchmarks/run_inference.py`. Once written, scoring uses the
existing `benchmarks/score.py`.

Open question for the adapter: how to handle homomers — feed the same
sequence N times with distinct chain_ids and shared entity_ids, or build
the symmetric assembly explicitly. The latter matches what AF-Multimer
does and is what `RCSBDataset.canonicalise` produces from RCSB-derived
training data.

## Why both this and the holdout benchmark?

The in-house holdout (`benchmarks/sets/`) uses date-cutoff RCSB targets:
much larger N (300+), avoids leakage, fast iteration, and matches the
training distribution. CASP15 is smaller (39 multimers) and exact targets
are publicly scored — the only way to make like-for-like comparisons
against AF-Multimer / Boltz-1 / Chai-1 published numbers, which is what a
paper or release announcement needs.
