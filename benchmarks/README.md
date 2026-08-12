# Benchmarks

These utilities evaluate a checkpoint against PDB-ID datasets that include
reference structures. They are separate from
[`projects/esm3/predict_fasta.py`](../projects/esm3/predict_fasta.py), which
accepts sequence-only FASTA input and does not produce a reference structure.

```text
benchmarks/
├── run_inference.py              checkpoint + PDB IDs → prediction/GT PDB pairs
├── run_eval.sh                   inference + lightweight score wrapper
├── external_testsets/            fixed public FASTA model inputs
├── score_simplefold_metrics.py   preferred identity-matched scorer
├── score_local_geometry.py       bond/clash geometry report
├── score.py                      legacy quick scorer
└── sets/                         fixed benchmark ID lists
```

`run_inference.py` filters inputs to single-chain examples through
`RCSBDataset(single_chain_only=True)`. It needs the processed `.npz` records
and PLM embedding cache compatible with the checkpoint.

```bash
bash benchmarks/run_eval.sh /path/to/checkpoint.pt t1_quick 0
```

`score_simplefold_metrics.py` reports TM-score, GDT-TS, all-atom lDDT,
Cα lDDT, Cα RMSD, and all-atom RMSD. It is not the OpenStructure scorer used
for the frozen ESM3 CASP14 record; use the project release protocol for that
comparison.

## Public SimpleFold test sets

The exact released SimpleFold CAMEO22, Apo, and CoDNaS artifacts can be
downloaded and converted to MambaFold inference inputs with:

```bash
bash scripts/download_simplefold_testsets.sh
```

This produces 183 CAMEO22 folding targets, 90 Apo two-state targets, and 77
CoDNaS two-state targets under `data/simplefold_official/testsets/`. The
official SimpleFold predictions remain separated by model size and sample.
Use OpenStructure 2.9.1 for CAMEO22 folding metrics; Apo and CoDNaS follow the
upstream five-sample, two-state TM-score protocol.

These public targets are not automatically excluded from MambaFold training.
Check the exact training split before treating a score as an unseen-test result.

The corresponding sequence inputs, together with CASP14/15/16, are committed
under [`external_testsets`](external_testsets/README.md). Public model inference
consumes those FASTA files directly and can write PDB, mmCIF, or both.
