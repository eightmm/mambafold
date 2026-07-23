# Frozen model projects

Each subdirectory is a self-contained release contract rather than a mutable
training recipe. A project contains its model artifact identity, saved config,
data and evaluation boundary, score record, and a minimal verification or
inference entrypoint.

| Project | Status | Purpose |
| --- | --- | --- |
| [`esm3/`](esm3/) | frozen at `esm3-v1.1.0` | Completed ESM3-conditioned MambaFold checkpoint and CASP14 evaluation |
| `esmc6b/` | reserved | Future ESMC-6B project; do not create it until a completed checkpoint and frozen evaluation record exist |

Do not add training experiments, new checkpoints, or changed metric results to
a frozen project. Those belong to a new project/version with a new manifest and
Git tag.
