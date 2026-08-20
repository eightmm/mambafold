# Documentation

The sole active public model contract is the provisional ESMC-6B step-170k
package in [`projects/esmc6b`](../projects/esmc6b/README.md). The geometry
fine-tune initialized from that EMA remains in training. The ESM3 project is a
frozen legacy archive, not an active research or comparison track.

| File | Scope |
| --- | --- |
| `architecture.md` | active pair-free atom→token→atom architecture |
| `data_pipeline.md` | ESMC-6B RCSB/AFDB Boltz-style data pipeline |
| `inference.md` | ESMC-6B FASTA and PDB-ID benchmark inference |
| `training.md` | active ESMC-6B training contract and status |
| `models/esmc6b.md` | artifact identity, data contract, and reporting gate |
| `models/esm3-legacy.md` | frozen historical context only |
| `pair_module.md` | archived pair-stack ablation; not the active mainline |
| `results/external_dataset_results.md` | active ESMC-only benchmark status and leakage-aware interpretation |
| `results/external_common_results.md` | archived legacy ESM3/OmegaFold snapshot; not an active claim source |
| `results/geometry_guidance_validity.md` | archived mixed-checkpoint CASP14 engineering experiment |

The benchmark admission policy and known coordinate-training overlaps are in
[`benchmarks/BENCHMARK_POLICY.md`](../benchmarks/BENCHMARK_POLICY.md).
