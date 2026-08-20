# External benchmark FASTA inputs

These files freeze sequence inputs for single-chain evaluation. They are
inputs, not pre-approved test splits; apply
[`../BENCHMARK_POLICY.md`](../BENCHMARK_POLICY.md) before making a
generalization claim.

| FASTA | Records | Evaluation role |
| --- | ---: | --- |
| `casp16_single_chain_21.fasta` | 21 | first primary candidate after exact and MMseqs2 filtering |
| `casp15_single_chain_22.fasta` | 22 | second primary candidate after exact and MMseqs2 filtering |
| `casp14_70.fasta` | 70 | ESMC-6B preview reproduction/development only |
| `cameo22_183.fasta` | 183 | SimpleFold-set overlap diagnostic; not a generalization test |
| `apo_90.fasta` | 90 | five-sample, two-state conformational diagnostic |
| `codnas_77.fasta` | 77 | five-sample, two-state conformational diagnostic |

Known exact coordinate-training matches are `T1227s1` and `T1243` in CASP16,
`T1106s2` and `T1120` in CASP15, and `T1029`, `T1030`, `T1034`, `T1065s2`,
`T1082`, and `T1092` in CASP14. CAMEO22, Apo, and CoDNaS have extensive
direct training overlap documented in the policy.

Run the exact-overlap gate against canonical FASTAs exported from every
coordinate-training source:

```bash
PYTHONPATH=. uv run --no-sync python benchmarks/audit_sequence_overlap.py \
  --targets benchmarks/external_testsets/casp16_single_chain_21.fasta \
  --training /path/to/rcsb-training.fasta \
  --training /path/to/afdb-swissprot-training.fasta \
  --out /path/to/casp16-exact-overlap.json \
  --write-exact-clean-fasta /path/to/casp16-exact-clean.fasta \
  --write-exact-clean-ids /path/to/casp16-exact-clean-ids.txt
```

The emitted FASTA and ID list still require the policy's independent MMseqs2
homology screen. Pass the resulting admitted ID list to the scorer's
`--target-ids`. Passing both gates supports only a coordinate-training-clean
claim; it does not show that a sequence was absent from ESMC pretraining.

Run the active ESMC-6B preview on an admitted FASTA with:

```bash
PYTHONPATH=src:. uv run --no-sync python -m projects.esmc6b.predict_fasta \
  --fasta /path/to/admitted-targets.fasta \
  --checkpoint /path/to/mambafold-esmc6b-170k-ema.pt \
  --out predictions/admitted-targets \
  --output-format both --n_steps 500 --seed 0
```

Reference structures and official SimpleFold predictions remain outside Git.
Use OpenStructure 2.9.1 for CASP/CAMEO scores and the upstream five-sample,
two-state TM-score contract for Apo/CoDNaS. `manifest.json` records input
counts and hashes; regenerate it from verified local sources with
`PYTHONPATH=src uv run --no-sync python scripts/export_external_test_fastas.py`.
