# MambaFold

MambaFold is a single-chain, all-atom protein structure generator based on
flow matching and a Bi-Mamba residue trunk. It has two deliberately separate
model tracks:

| Track | Status | What is available |
| --- | --- | --- |
| **ESM3** | frozen | Verified 422.4M-parameter checkpoint, FASTA inference, and CASP14 evaluation record |
| **ESMC-6B** | active research track | Verified step-170k EMA preview plus ongoing geometry fine-tuning |

## Use the frozen ESM3 model

ESM3 remains the only completed model selection in this repository. It is
inference/evaluation only: its checkpoint, saved configuration, and CASP14
result are fixed. The current interface release is
[`esm3-v1.1.0`](https://github.com/eightmm/mambafold/tree/esm3-v1.1.0).

1. Install the project environment and obtain the checkpoint separately. The
   6.3 GiB checkpoint is not stored in Git.

   ```bash
   MAMBA_SKIP_CUDA_BUILD=TRUE uv sync --extra dev
   ```

2. Verify the downloaded checkpoint before use.

   ```bash
   python projects/esm3/verify_artifact.py \
     --checkpoint /path/to/ckpt_0120000.pt
   ```

3. Predict structures from a standard single-chain FASTA file.

   ```bash
   PYTHONPATH=src python projects/esm3/predict_fasta.py \
     --fasta projects/esm3/examples/example.fasta \
     --checkpoint /path/to/ckpt_0120000.pt \
     --out predictions/example \
     --output-format both --n_steps 50 --seed 0
   ```

The output directory contains PDB and mmCIF (`.cif`) structures plus a
prediction manifest. Select `pdb`, `cif`, or `both` with `--output-format`.
Both structure formats store predicted pLDDT as B-factors on a 0–100 scale.
Inputs must use the 20 standard amino-acid letters and have length 10–1,024;
multimers, ligands, nucleic acids, metals, cofactors, waters, and PTMs are out
of scope.

## ESM3 result

The frozen step-120,000 EMA checkpoint was evaluated on the CASP14 70 whole
single-chain targets with SDE (500 steps, seed 0) and OpenStructure 2.9.1.

| Metric | Mean | Median |
| --- | ---: | ---: |
| GDT-TS | 0.670 | 0.697 |
| TM-score | 0.757 | 0.843 |
| all-atom lDDT | 0.657 | 0.732 |
| backbone lDDT | 0.763 | 0.847 |
| RMSD (Å; lower is better) | 6.276 | 3.258 |

### Comparison with SimpleFold

The table below uses mean CASP14 values under the common 70-target,
500-step SDE, OpenStructure 2.9.1 reporting contract. SimpleFold aggregates
are from the [SimpleFold paper](https://arxiv.org/abs/2509.18480). The ESM3 row
is frozen; the ESMC-6B row is the verified step-170k preview and is not a final
model selection.

| Model | Parameters | GDT-TS ↑ | TM-score ↑ | all-atom lDDT ↑ | backbone lDDT ↑ | RMSD (Å) ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SimpleFold-360M | 360M | 0.585 | 0.674 | 0.617 | 0.703 | 9.382 |
| SimpleFold-3B | 2.86B | 0.639 | 0.720 | **0.666** | 0.747 | 7.732 |
| **MambaFold-ESMC-6B, step 170,000 preview** | **404.9M** | **0.682** | **0.761** | 0.646 | **0.769** | **6.146** |
| **MambaFold-ESM3** | **422.4M** | **0.670** | **0.757** | 0.657 | **0.763** | **6.276** |

Against the size-matched SimpleFold-360M, the frozen ESM3 model improves mean
GDT-TS by 0.085, TM-score by 0.083, all-atom lDDT by 0.040, and backbone lDDT
by 0.060,
while reducing mean RMSD by 3.106 Å. Against SimpleFold-3B it is higher on
GDT-TS, TM-score, backbone lDDT, and RMSD, but lower by 0.009 all-atom lDDT.
These are aggregate comparisons, not a paired significance test.

The step-170k ESMC-6B preview is above SimpleFold-360M in GDT-TS (+0.097),
TM-score (+0.087), all-atom lDDT (+0.029), and backbone lDDT (+0.066), while
reducing RMSD by 3.236 Å. Against SimpleFold-3B it is higher in GDT-TS
(+0.043), TM-score (+0.041), backbone lDDT (+0.022), and RMSD (-1.586 Å), but
lower by 0.020 in all-atom lDDT. ESMC-6B sequence pretraining postdates
CASP14, so this row is retrospective engineering evidence rather than a
temporally clean blind-test claim.

The full artifact identity, evaluation protocol, and CASP14 reproduction entry
point are in [projects/esm3](projects/esm3/README.md). FASTA predictions are
not CASP14 scores unless evaluated with that frozen target set and protocol.

## External test FASTA files

Fixed model-input FASTAs are committed under
[`benchmarks/external_testsets`](benchmarks/external_testsets/README.md):
CASP14 (70), strict single-chain CASP15 (22) and CASP16 (21), CAMEO22 (183),
Apo (90), and CoDNaS (77). The model consumes these files through the same
FASTA interface used for user sequences. Reference structures remain separate
evaluation artifacts and are not required to generate a prediction.

The completed snapshot prediction coverage, measured runtime/VRAM, and
currently available structure scores are organized by dataset in
[the external benchmark results](docs/results/external_dataset_results.md).

## Repository layout

```text
projects/esm3/     frozen ESM3 artifact contract, FASTA CLI, and evaluation entrypoint
projects/esmc6b/   provisional step-170k EMA artifact contract and FASTA CLI
src/mambafold/     model, data, sampling, and training package
benchmarks/        PDB-ID benchmark inference and scoring utilities
configs/           research configurations; ESMC-6B is the active future track
docs/              architecture, data, training, and evaluation notes
tests/             focused correctness checks
```

## ESMC-6B research track

The ESMC-6B path is checkpoint-incompatible with ESM3 (2,560-dimensional
sequence-only embeddings versus 1,536-dimensional ESM3 embeddings). It is
trained from scratch and must never resume the frozen ESM3 checkpoint. The
ESMC-6B research program remains active. The verified step-170k EMA is
available as a deliberately provisional prerelease through
[`projects/esmc6b`](projects/esmc6b/README.md); it initializes the ongoing
geometry fine-tuning run and is not a final model selection. Other ESMC values
reported in this repository remain checkpoint-specific snapshots. Detailed
training status and reporting limits are documented in
[docs/models/esmc6b.md](docs/models/esmc6b.md).

## License and model artifacts

The Git tree contains source code and reproducibility metadata; model weights
are distributed separately as release assets. The MambaFold ESMC preview does
not bundle the upstream ESMC-6B parameters. Before redistributing a checkpoint
or running either track, comply with the applicable model and source-data
terms.
Bundled third-party reference data and its upstream license are documented in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
