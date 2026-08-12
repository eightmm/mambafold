# Model projects

Each directory under `projects/` is a versioned model release contract. It
contains the checkpoint identity, inference boundary, evaluation record, and
minimal entrypoints required to use that release without changing its reported
result.

| Project | Status | Entry point |
| --- | --- | --- |
| [`esm3/`](esm3/) | frozen model artifact; interface release `esm3-v1.1.0` | `predict_fasta.py` for FASTA, `run_casp14.sh` for the fixed benchmark protocol |
| ESMC-6B research track | active, not released | Interim status and evaluation are recorded in [`docs/models/esmc6b.md`](../docs/models/esmc6b.md) |

A new user interface or a future model may receive a new release tag, but the
ESM3 step-120,000 EMA checkpoint, saved training configuration, and CASP14
result must not be modified.
