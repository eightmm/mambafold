# MambaFold ESMC-6B 170k baseline prerelease

**Status: provisional prerelease.** This package describes the step-170,000
EMA baseline used to initialize the ongoing 50,000-step geometry fine-tuning
run. The fine-tuning run is still in progress, so this baseline is not a final
model selection or a frozen result release.

The machine-readable source of truth is [`manifest.json`](manifest.json).
[`training_config.json`](training_config.json) preserves the resolved JSON
configuration saved with the step-170,000 training checkpoint, rather than a
later YAML default. The compatible source is tagged
`esmc6b-170k-preview.1`.

Download the EMA artifact and checksum from the GitHub prerelease:

```bash
gh release download esmc6b-170k-preview.1 \
  --repo eightmm/mambafold \
  --pattern 'mambafold-esmc6b-170k-ema.pt*'
```

## Model boundary

- MambaFold artifact: inference-only step-170,000 EMA state, 404,856,302
  model parameters across 678 state keys (404,856,326 stored tensor values,
  including persistent buffers).
- Conditioning: sequence-only `biohub/ESMC-6B` revision
  `45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a`, with 2,560-dimensional residue
  embeddings.
- Architecture: direct all-atom flow matching with an atom-to-token-to-atom
  Bi-Mamba path, sparse gated attention every six residue-trunk blocks, and no
  explicit pair stack.
- Input boundary: single-chain proteins containing only the 20 standard amino
  acids, from 10 through 1,024 residues.

The MambaFold release asset never bundles the ESMC-6B PLM weights. Download the
pinned external model separately and comply with its upstream access and
license terms:

```bash
bash scripts/download_esmc6b.sh
```

`ESMC_6B_MODEL_DIR` may point to an already downloaded snapshot of that exact
revision. A different ESMC revision is outside this prerelease contract.

The repository does not currently declare a general code or model-weight
license. Publishing this research preview does not relicense the upstream
ESMC-6B model or grant additional redistribution rights.

## Artifact verification

The release asset is named `mambafold-esmc6b-170k-ema.pt` (1,619,662,835
bytes; SHA-256
`465ddb7d873479e51487a79b39d2a871a10b3b54be178adcd76afe7f86665a02`).
It contains the MambaFold EMA inference state, saved model arguments, and
checkpoint step; it excludes optimizer state, training data state, RNG state,
and all ESMC-6B parameters. Verify the artifact before allowing PyTorch to load
it:

```bash
.venv/bin/python projects/esmc6b/verify_artifact.py \
  --checkpoint /path/to/mambafold-esmc6b-170k-ema.pt
```

## Predict from FASTA

The FASTA entry point validates each record before loading either model,
computes pinned ESMC-6B embeddings, samples all-atom coordinates, and writes
PDB, mmCIF, or both. Predicted per-residue pLDDT is stored in atom B-factors on
a 0--100 scale.

```bash
PYTHONPATH=src:. uv run --no-sync python -m projects.esmc6b.predict_fasta \
  --fasta projects/esmc6b/examples/example.fasta \
  --checkpoint /path/to/mambafold-esmc6b-170k-ema.pt \
  --out predictions/esmc6b-example \
  --output-format both --n_steps 50 --seed 0
```

The output directory must not already exist. `--n_steps 50` is an exploratory
default; the recorded retrospective CASP14 comparison uses 500 SDE steps.

## Reproduce the baseline CASP14 contract

The following wrapper runs the 70-target, guidance-off baseline contract and
the repository's lightweight scorer. It does not train or resume a model and
refuses to overwrite an output directory.

```bash
ESMC6B_CHECKPOINT=/path/to/mambafold-esmc6b-170k-ema.pt \
ESMC6B_DATA_DIR=/path/to/casp14_npz_70 \
ESMC6B_EMBEDDINGS=/path/to/casp14_esmc6b_70 \
ESMC6B_IDS=/path/to/casp14_70_whole_ids_exact.txt \
ESMC6B_OUT=/path/to/new_output_dir \
bash projects/esmc6b/run_casp14.sh
```

The OpenStructure values in the manifest use OpenStructure 2.9.1, not the
lightweight Python scorer. CASP14 was used during checkpoint and sampler
development, and ESMC-6B sequence pretraining postdates CASP14. These results
are therefore retrospective engineering evidence, not an untouched
confirmatory benchmark.

## Ongoing work

The 50,000-step geometry fine-tuning experiment starts afresh from this 170k
EMA state. Its optimizer, scheduler, step counter, and EMA are separate, and
its eventual checkpoint is not represented by this artifact or manifest.
