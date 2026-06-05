# MambaFold Docs

Current docs describe the single-chain coarse-to-fine model only.

| File | Purpose |
|---|---|
| `architecture.md` | Stage 1/Stage 2 architecture |
| `training.md` | training phases, losses, launch commands |
| `data_pipeline.md` | RCSB `.npz` and ESM3 cache pipeline |
| `inference.md` | Euler sampling and scoring |
| `pair_module.md` | Stage 1 pair stack details |

```mermaid
flowchart LR
    A[RCSB npz + ESM3] --> B[RCSBDataset single_chain_only]
    B --> C[ProteinBatch + FM corruption]
    C --> D[Stage 1 C-alpha scaffold]
    D --> E[Stage 2 all-atom refiner]
    E --> F[Inference PDB + scores]
```
