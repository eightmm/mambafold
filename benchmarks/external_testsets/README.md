# External benchmark FASTA inputs

These files freeze the sequence inputs used by MambaFold's external
single-chain evaluations. They can be passed directly to the public FASTA
inference command; no PDB-ID lookup is required.

| FASTA | Records | Evaluation role |
| --- | ---: | --- |
| `casp14_70.fasta` | 70 | Frozen ESM3 CASP14 result |
| `casp15_single_chain_22.fasta` | 22 | Strict single-chain folding |
| `casp16_single_chain_21.fasta` | 21 | Strict single-chain folding |
| `cameo22_183.fasta` | 183 | Exact SimpleFold CAMEO22 folding set |
| `apo_90.fasta` | 90 | SimpleFold five-sample, two-state set |
| `codnas_77.fasta` | 77 | SimpleFold five-sample, two-state set |

Run a complete FASTA with the frozen ESM3 checkpoint:

```bash
PYTHONPATH=src python projects/esm3/predict_fasta.py \
  --fasta benchmarks/external_testsets/cameo22_183.fasta \
  --checkpoint /path/to/ckpt_0120000.pt \
  --out predictions/cameo22 \
  --output-format both --n_steps 500 --seed 0
```

The FASTA files contain inputs only. Reference structures and official
SimpleFold predictions are deliberately kept out of Git; obtain them with
`scripts/download_simplefold_testsets.sh` or the corresponding CASP preparation
script. Use OpenStructure 2.9.1 for CASP/CAMEO folding scores and the upstream
five-sample, two-state TM-score contract for Apo and CoDNaS.

`manifest.json` records counts and SHA-256 digests. Regenerate this directory
from verified local source artifacts with:

```bash
PYTHONPATH=src python scripts/export_external_test_fastas.py
```

These are external evaluation collections, not automatically leakage-free
splits. The current full RCSB training corpus overlaps many CAMEO22, Apo, and
CoDNaS PDB entries. Report the exact training cutoff or overlap audit whenever
using them to make a generalization claim.
