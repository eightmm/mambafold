# Documentation

The public entrypoint for the completed model is
[`projects/esm3/README.md`](../projects/esm3/README.md). The documents here
describe the shared architecture and the in-progress ESMC-6B research path.

| File | Scope |
| --- | --- |
| `architecture.md` | direct all-atom atom→token→atom architecture |
| `data_pipeline.md` | active ESMC-6B RCSB/AFDB Boltz-style data pipeline |
| `inference.md` | PDB-ID benchmark inference and lightweight scoring |
| `training.md` | research training notes; not an ESM3 continuation recipe |
| `models/esm3-legacy.md` | historical ESM3 baseline context; use `projects/esm3/` as the release source of truth |
| `models/esmc6b.md` | ESMC-6B data/training contract and reporting gate |
| `pair_module.md` | archived pair-stack/ablation design; not the active pair-free mainline |

Do not use an ESM3 checkpoint with an ESMC configuration: their PLM projection
dimensions differ.
