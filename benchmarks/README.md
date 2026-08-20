# Benchmarks

These utilities evaluate the active MambaFold-ESMC-6B track on fixed
single-chain inputs. Read [`BENCHMARK_POLICY.md`](BENCHMARK_POLICY.md) before
interpreting a score: the committed FASTAs are stable inputs, not automatically
leakage-free splits.

```text
benchmarks/
├── BENCHMARK_POLICY.md            dataset roles and admission rules
├── audit_sequence_overlap.py    exact coordinate-training overlap gate
├── external_testsets/           fixed public FASTA inputs
├── run_inference.py             checkpoint + PDB IDs → prediction/GT pairs
├── run_eval.sh                  inference + lightweight score wrapper
├── score_simplefold_metrics.py  identity-matched local scorer
├── score_local_geometry.py      bond/clash geometry report
└── sets/                        fixed benchmark ID lists
```

The active comparator roster is SimpleFold-360M, SimpleFold-3B when a matching
scale-reference result is available, ESMFold v1, and DPLM-2 Bit 650M.
OmegaFold is excluded because OOM failures made its coverage incomplete, and
the frozen ESM3 archive is not an active baseline. DPLM-2 emits only backbone
atoms plus CB, so its backbone lDDT is comparable but its all-atom lDDT is not.

## Current evaluation order

1. CASP16 strict single-chain targets, after exact and MMseqs2 homology gates.
2. CASP15 strict single-chain targets under the same gates.
3. CASP14 only for development and preview reproduction.
4. CAMEO22, Apo, and CoDNaS only as overlap/conformational diagnostics.
5. Prospective post-checkpoint RCSB/CAMEO and CASP17 after public references
   for confirmatory evaluation.

## Exact-overlap gate

Export canonical training sequences from every coordinate source, then run:

```bash
PYTHONPATH=. uv run --no-sync python benchmarks/audit_sequence_overlap.py \
  --targets benchmarks/external_testsets/casp16_single_chain_21.fasta \
  --training /path/to/rcsb-training.fasta \
  --training /path/to/afdb-swissprot-training.fasta \
  --out /path/to/casp16-exact-overlap.json \
  --write-exact-clean-fasta /path/to/casp16-exact-clean.fasta \
  --write-exact-clean-ids /path/to/casp16-exact-clean-ids.txt
```

The generated FASTA and ID list are only exact-match-clean. Apply the declared
MMseqs2 homology screen and remove its excluded IDs before passing the final ID
list to `score_external_openstructure.py --target-ids`.

## PDB-ID inference and scoring

`run_inference.py` uses `RCSBDataset(single_chain_only=True)` and requires
processed `.npz` records plus an ESMC-6B embedding cache compatible with the
checkpoint.

```bash
bash benchmarks/run_eval.sh /path/to/esmc6b-checkpoint.pt t1_quick 0
```

Use OpenStructure 2.9.1 for paper-facing CASP/CAMEO metrics. The lightweight
scorer reports TM-score, GDT-TS, all-atom lDDT, Cα lDDT, and RMSD variants.
The exact released SimpleFold CAMEO22, Apo, and CoDNaS artifacts can be staged
with `bash scripts/download_simplefold_testsets.sh`; their heavy references and
baseline predictions remain outside Git.
