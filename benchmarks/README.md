# Benchmarks

These utilities evaluate a checkpoint against PDB-ID datasets that include
reference structures. They are separate from
[`projects/esm3/predict_fasta.py`](../projects/esm3/predict_fasta.py), which
accepts sequence-only FASTA input and does not produce a reference structure.

```text
benchmarks/
├── run_inference.py              checkpoint + PDB IDs → prediction/GT PDB pairs
├── run_eval.sh                   inference + lightweight score wrapper
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
