# MambaFold

MambaFold is a single-chain, all-atom protein structure generator built from
flow matching, an atom-to-token-to-atom Bi-Mamba path, and a pair-free residue
trunk. The sole active public track is conditioned on the frozen,
sequence-only ESMC-6B protein language model.

| Track | Status | Public contract |
| --- | --- | --- |
| **MambaFold-ESMC-6B** | active research; provisional step-170k preview | verified EMA artifact, FASTA inference, and retrospective CASP14 reproduction |
| ESM3 | frozen legacy archive | immutable historical reproduction under `projects/esm3/`; not an active model or comparator |

The 170k EMA preview initializes an ongoing geometry fine-tuning experiment.
It is not the final checkpoint or a frozen paper result.

## Use the ESMC-6B preview

1. Install the environment and download the separately distributed MambaFold
   EMA artifact. The upstream ESMC-6B weights are not bundled.

   ```bash
   MAMBA_SKIP_CUDA_BUILD=TRUE uv sync --extra dev
   bash scripts/download_esmc6b.sh
   gh release download esmc6b-170k-preview.1 \
     --repo eightmm/mambafold \
     --pattern 'mambafold-esmc6b-170k-ema.pt*'
   ```

2. Verify the checkpoint before loading it.

   ```bash
   .venv/bin/python projects/esmc6b/verify_artifact.py \
     --checkpoint /path/to/mambafold-esmc6b-170k-ema.pt
   ```

3. Predict a standard single-chain FASTA.

   ```bash
   PYTHONPATH=src:. uv run --no-sync python -m projects.esmc6b.predict_fasta \
     --fasta projects/esmc6b/examples/example.fasta \
     --checkpoint /path/to/mambafold-esmc6b-170k-ema.pt \
     --out predictions/esmc6b-example \
     --output-format both --n_steps 50 --seed 0
   ```

Inputs must contain only the 20 standard amino acids and be 10--1,024 residues
long. The active contract is single-chain protein folding without ligands,
nucleic acids, metals, cofactors, waters, or post-translational modifications.
See [`projects/esmc6b`](projects/esmc6b/README.md) for the exact model revision,
checksum, and inference boundary.

## Provisional 170k result

The step-170k EMA was evaluated on 70 CASP14 whole-chain targets with seed 0,
500-step SDE sampling, and OpenStructure 2.9.1.

| Model | Parameters | GDT-TS ↑ | TM-score ↑ | all-atom lDDT ↑ | backbone lDDT ↑ | RMSD (Å) ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SimpleFold-360M | 360M | 0.585 | 0.674 | 0.617 | 0.703 | 9.382 |
| SimpleFold-3B | 2.86B | 0.639 | 0.720 | **0.666** | 0.747 | 7.732 |
| **MambaFold-ESMC-6B, step 170k preview** | **404.9M** | **0.682** | **0.761** | 0.646 | **0.769** | **6.146** |

SimpleFold values are published aggregates from the
[SimpleFold paper](https://arxiv.org/abs/2509.18480) under the same 70-target,
500-step SDE, OpenStructure reporting contract. This comparison is
retrospective engineering evidence: CASP14 informed MambaFold development,
six CASP14 sequences exactly match the coordinate-training corpus, and
ESMC-6B pretraining postdates CASP14. It is not a temporally blind or
leakage-controlled test.

## Benchmark plan

The active comparator roster is SimpleFold-360M (primary size-matched model),
SimpleFold-3B when a matching scale-reference result is available, ESMFold v1,
and DPLM-2 Bit 650M. OmegaFold is excluded because its incomplete, OOM-limited
coverage prevents a clean full-set comparison. The ESM3 archive is not an
active baseline.

The next scoreable datasets are strict single-chain CASP16, then CASP15, after
removing exact coordinate-training matches and applying a declared MMseqs2
homology filter. Prospective post-checkpoint RCSB/CAMEO targets and CASP17
after public references become available are the intended confirmatory sets.
Dataset roles, known overlaps, and allowed claims are fixed in
[`benchmarks/BENCHMARK_POLICY.md`](benchmarks/BENCHMARK_POLICY.md).

## Repository layout

```text
projects/esmc6b/   active provisional artifact and FASTA inference contract
projects/esm3/     immutable legacy archive
src/mambafold/     model, data, sampling, and training package
benchmarks/        benchmark inputs, policy, inference, and scoring utilities
configs/           active ESMC-6B and archived research configurations
docs/              architecture, data, training, and evaluation notes
tests/             focused correctness checks
```

## License and model artifacts

The Git tree contains source code and reproducibility metadata; model weights
are distributed separately as release assets. The preview does not bundle
upstream ESMC-6B parameters. Follow the applicable upstream model and
source-data terms. Bundled third-party reference data and licenses are listed
in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
