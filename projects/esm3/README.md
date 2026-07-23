# MambaFold ESM3 v1.0.0

**Status: frozen.** This is the complete ESM3-conditioned MambaFold project.
It is limited to inference and evaluation of the retained checkpoint; it does
not accept continued training, fine-tuning, architectural changes, or metric
replacement. The immutable source identifier is Git tag `esm3-v1.0.0`.

## What is frozen

- Checkpoint: step 120,000 EMA artifact, 422.4M parameters.
- Model conditioning: ESM3-open residue embeddings, 1,536 dimensions.
- Architecture: direct all-atom flow matching with atom-to-token-to-atom
  Bi-Mamba, sparse gated attention every six residue-trunk blocks, and no
  explicit pair stack.
- Input boundary: single-chain standard-amino-acid proteins, maximum length
  1,024 residues. Multimers, ligands, nucleic acids, metals, cofactors, water,
  and PTMs are outside this project.
- Evaluation boundary: CASP14 whole 70-target set, EMA, SDE with 500 steps and
  seed 0; T1044 is excluded because it exceeds the length limit.

The machine-readable source of truth is [`manifest.json`](manifest.json).
[`training_config.json`](training_config.json) is the exact configuration saved
alongside the retained checkpoint, not a subsequently edited YAML default.

## Results

OpenStructure 2.9.1 `compare-structures` was used with `--lddt --bb-lddt
--rigid-scores --tm-score` and `--fault-tolerant --min-pep-length 4`.

| Metric | Mean | Median |
| --- | ---: | ---: |
| GDT-TS | 0.670 | 0.697 |
| GDT-HA | 0.532 | 0.517 |
| TM-score | 0.757 | 0.843 |
| all-atom lDDT | 0.657 | 0.732 |
| backbone lDDT | 0.763 | 0.847 |
| RMSD (Å; lower is better) | 6.276 | 3.258 |

The corresponding SimpleFold reference values and exact comparison provenance
are recorded in the manifest. They should not be mixed with the repository's
lightweight Python scorer, whose GDT-TS implementation differs.

## Artifact verification

The 6.3 GiB checkpoint is intentionally not committed to Git. Obtain it from
the project artifact store, then verify it before use:

```bash
python projects/esm3/verify_artifact.py \
  --checkpoint /path/to/ckpt_0120000.pt
```

The verifier requires the SHA-256 in the manifest and fails on a mismatch.

## Reproduce the recorded inference contract

The following runs inference and the repository's lightweight score; it does
not overwrite an existing directory and it does not train or resume anything.

```bash
ESM3_CHECKPOINT=/path/to/ckpt_0120000.pt \
ESM3_DATA_DIR=/path/to/casp14_npz_70 \
ESM3_EMBEDDINGS=/path/to/casp14_esm3_70 \
ESM3_IDS=/path/to/casp14_70_whole_ids_exact.txt \
ESM3_OUT=/path/to/new_output_dir \
bash projects/esm3/run_casp14.sh
```

For the reported OpenStructure metrics, run the command in the manifest against
the generated prediction/reference PDB pairs. Keep the target list, sampler,
OpenStructure version, and fault-tolerance policy unchanged.

## Provenance limitation

The exact checkpoint configuration and artifact digest are retained. The
original training-source commit was not recorded in the checkpoint metadata,
so this release freezes the current compatible source tree and explicitly does
not claim bitwise reproduction of the historical training run. It does support
artifact verification and reproduction of the recorded inference/evaluation
contract.
