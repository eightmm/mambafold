# MambaFold ESM3 legacy baseline

This document records the completed ESM3-conditioned MambaFold baseline.  It
is a **legacy/reproducibility track**, not the active training path.  New work
uses the sequence-only ESMC-6B track described in
[`esmc6b.md`](esmc6b.md).

## Model identity

| Field | Value |
| --- | --- |
| Conditioning | ESM3-open residue embeddings, 1,536 dimensions |
| Architecture | direct all-atom flow matching; atom-to-token-to-atom Bi-Mamba; sparse gated attention every six residue-trunk blocks; no explicit pair stack |
| Parameters | 422.4M |
| Training hardware | 4 x NVIDIA B200 GPUs |
| Maximum length | 1,024 residues |
| Configuration | `configs/direct_allatom_puremamba_attn6_geo_adaln_sf360.yaml` |
| Evaluated checkpoint | `direct_puremamba_attn6_geo_adaln_sf360_mixed_v1`, step 120,000, EMA weights |
| Sampling | SDE, 500 steps, seed 0 |

The checkpoint itself is not stored in this Git repository.  Its local
provenance path during the original run was
`outputs/train/direct_puremamba_attn6_geo_adaln_sf360_mixed_v1/ckpt_0120000.pt`.
This is a 6.75 GB artifact and must be distributed separately with its SHA-256
and access terms before it can be treated as a downloadable release asset.

## CASP14 evaluation

The model was evaluated on the 70 CASP14 whole single-chain targets in
`data/casp_official/casp14_70_whole_ids_exact.txt`; T1044 was excluded because
it exceeds the 1,024-residue model limit.  Structures were scored with
OpenStructure 2.9.1 `compare-structures` using `--lddt --bb-lddt
--rigid-scores --tm-score`.

| Metric | MambaFold ESM3 | SimpleFold-360M | SimpleFold-3B |
| --- | ---: | ---: | ---: |
| GDT-TS (mean) | **0.670** | 0.585 | 0.639 |
| TM-score (mean) | **0.757** | 0.674 | 0.720 |
| all-atom lDDT (mean) | 0.657 | 0.617 | **0.666** |
| backbone lDDT (mean) | **0.763** | 0.703 | 0.747 |
| RMSD, Angstrom (mean; lower is better) | **6.276** | 9.382 | 7.732 |

The SimpleFold columns are the protocol values recorded with the local
OpenStructure report.  They are useful references, but a claim of exact
cross-project equivalence requires independent reproduction with the same
target preparation, reference structures, and software environment.

## Reproducing inference

Prepare the legacy ESM3 cache (`d_plm: 1536`) and compatible processed
single-chain RCSB/CASP data, then run:

```bash
PYTHONPATH=src uv run python benchmarks/run_inference.py \
  --ckpt /path/to/ckpt_0120000.pt \
  --ids data/casp_official/casp14_70_whole_ids_exact.txt \
  --esm_dir /path/to/rcsb_esm_official_full \
  --out benchmarks/results/esm3_casp14 \
  --max_length 1024 --sampler sde --n_steps 500 --seed_offset 0
```

Then score the generated PDB files:

```bash
tools/scoring_venv/bin/python benchmarks/score_simplefold_metrics.py \
  --in_dir benchmarks/results/esm3_casp14 \
  --out benchmarks/results/esm3_casp14/scores.json
```

For paper-comparable OpenStructure metrics, use the same OpenStructure 2.9.1
command described above.  The lightweight Python scorer and OpenStructure use
different implementations; do not mix their GDT-TS values in a single table.

## Provenance and limitations

- This is a single-chain, standard-amino-acid model. It does not model
  multimers, ligands, nucleic acids, metals, cofactors, waters, or PTMs.
- The legacy ESM3 cache was incomplete (127,946 of 211,742 RCSB entries) and
  is retained only to reproduce this historical checkpoint.
- ESM3 is not used as the active path because its training may include
  structure-aware information.  The ESMC-6B path uses pinned sequence-only
  embeddings and is reported separately when training completes.
