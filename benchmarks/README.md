# Benchmarks

Single-chain benchmark helpers.

```text
benchmarks/
├── run_inference.py        # ckpt + ids -> predicted/GT PDB pairs
├── run_stage1_ca_eval.py   # Stage 1 CA-only evaluation
├── score.py                # PDB pairs -> scores.json
├── run_eval.sh             # inference + scoring wrapper
└── sets/                   # fixed id lists
```

Scoring reports CA lDDT, TM-score, CA RMSD, and all-atom RMSD. `run_inference.py` filters to single-chain examples through `RCSBDataset(single_chain_only=True)`.
