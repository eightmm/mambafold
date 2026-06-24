# Benchmarks

Single-chain benchmark helpers.

```text
benchmarks/
├── run_inference.py        # ckpt + ids -> predicted/GT PDB pairs
├── score_simplefold_metrics.py  # preferred: TM, GDT-TS, lDDT, RMSD
├── score.py                # legacy quick scorer
├── run_eval.sh             # inference + scoring wrapper
└── sets/                   # fixed id lists
```

`run_eval.sh` uses `score_simplefold_metrics.py`, which matches atoms by
identity and reports TM-score, GDT-TS, all-atom lDDT, C-alpha lDDT, C-alpha
RMSD, and all-atom RMSD. `run_inference.py` filters to single-chain examples
through `RCSBDataset(single_chain_only=True)`.
