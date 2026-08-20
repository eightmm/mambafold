# Documentation

The public entrypoint for the completed model is
[`projects/esm3/README.md`](../projects/esm3/README.md). The verified but
provisional step-170k ESMC-6B package is
[`projects/esmc6b/README.md`](../projects/esmc6b/README.md). The documents here
describe the shared architecture and the continuing ESMC-6B research path.

| File | Scope |
| --- | --- |
| `architecture.md` | direct all-atom atom→token→atom architecture |
| `data_pipeline.md` | active ESMC-6B RCSB/AFDB Boltz-style data pipeline |
| `inference.md` | PDB-ID benchmark inference and lightweight scoring |
| `training.md` | research training notes; not an ESM3 continuation recipe |
| `models/esm3-legacy.md` | historical ESM3 baseline context; use `projects/esm3/` as the release source of truth |
| `models/esmc6b.md` | ESMC-6B data/training contract and reporting gate |
| `pair_module.md` | archived pair-stack/ablation design; not the active pair-free mainline |
| `results/external_dataset_results.md` | canonical six-model results organized by CASP14, CASP15, CASP16, CAMEO22, Apo, and CoDNaS |
| `results/external_common_results.md` | detailed ESM3-focused runtime, CASP14 accuracy, paired statistics, and sampler analysis |
| `results/geometry_guidance_validity.md` | paired retrospective CASP14 validity test of inference-only geometry guidance for the frozen ESM3 and interim ESMC-6B checkpoints |

Do not use an ESM3 checkpoint with an ESMC configuration: their PLM projection
dimensions differ.
