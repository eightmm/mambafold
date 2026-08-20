# Inference

## Sequence-only FASTA

The active public entrypoint computes pinned ESMC-6B embeddings and samples a
single-chain all-atom structure:

```bash
PYTHONPATH=src:. uv run --no-sync python -m projects.esmc6b.predict_fasta \
  --fasta projects/esmc6b/examples/example.fasta \
  --checkpoint /path/to/mambafold-esmc6b-170k-ema.pt \
  --out predictions/esmc6b-example \
  --output-format both --n_steps 50 --seed 0
```

The output directory contains a manifest and PDB and/or mmCIF files. The
50-step setting is exploratory; recorded CASP14 results use 500-step SDE
sampling.

## Processed PDB-ID benchmark

For a processed RCSB benchmark with a compatible ESMC cache:

```bash
PYTHONPATH=src uv run python benchmarks/run_inference.py \
  --ckpt outputs/train/<run>/ckpt_latest.pt \
  --ids benchmarks/sets/t1_quick.txt \
  --out benchmarks/results/<run>_t1 \
  --esm_dir data/rcsb_esmc6b_official_full \
  --n_steps 50
```

Ground-truth PDBs contain observed atoms only, so missing side-chain atoms are
not represented as coordinates during all-atom scoring. Use OpenStructure
2.9.1 for paper-facing CASP/CAMEO metrics, and apply the overlap policy in
[`../benchmarks/BENCHMARK_POLICY.md`](../benchmarks/BENCHMARK_POLICY.md)
before making a generalization claim.
