# Model projects

Each directory under `projects/` is an artifact contract: checkpoint identity,
inference boundary, evaluation record, and the minimal reproduction entrypoint.

| Project | Status | Entry point |
| --- | --- | --- |
| [`esmc6b/`](esmc6b/) | sole active track; provisional step-170k EMA prerelease while geometry fine-tuning continues | `predict_fasta.py` for FASTA and `run_casp14.sh` for retrospective reproduction |
| [`esm3/`](esm3/) | frozen legacy archive; interface release `esm3-v1.1.0` | historical FASTA and CASP14 reproduction only |

New research, benchmarks, and releases target ESMC-6B. The ESM3 directory is
kept immutable for historical reproducibility and must not be used as an
active comparator or resumed with an ESMC configuration.
