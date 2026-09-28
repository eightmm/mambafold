# Benchmark and provenance record

## Fixed model and inference

- Folding model: Run A final EMA, `out/run-a-mamba3-1024-atom14/ckpt_0300000.pt`, SHA-256 `8ad6079edff800912e3c6b175e896221344cb4bb6b866bd7b3dd7e474b325f1e`, trained for 300,000 steps under `configs/run_a_mamba.yaml` (SHA-256 `a7414eee9e55bd64a61aba3a6048515d7c5bc90a6ffc08323857906a1360aa70`). The current config specifies a 15-block, 1024-wide Mamba-3 residue trunk and reports 370,480,666 folding-model parameters. The frozen ESMC-6B encoder is separate.
- Primary sampler: SDE, 500 steps, tau 0.01, epsilon 0.01, noise cutoff 0.99, logarithmic timesteps, seed 0, EMA weights. This is the configuration used by the existing Run A CASP14 and CAMEO22 rollouts and fixed before CASP15/16 scoring.
- CASP15 and CASP16 rollout jobs: 84044 and 84045; dependent OpenStructure jobs: 84050 and 84051. These are experiment identifiers, not manuscript evidence by themselves. Success requires `sacct` exit 0 and complete score files.
- Original FASTA SHA-256: CASP15 `23f03d952f08da4c08832dd89e388a174bd3e927398beec4ca3ffb5ef9147047`; CASP16 `a5c42094cea41a489512bfd172f139f3c409d42d888acb411319267beb98efba`. See `benchmarks/external_testsets/` for the committed sequences.

## Target selection

The coordinate-training union is the admitted RCSB set and AFDB SwissProt v4. `pipeline/08_leakage_gates.py` checks exact sequence overlap. `pipeline/10_homology_gate.py` uses MMseqs2 at at least 30% identity over at least 80% of the benchmark query, query coverage mode 2. The precomputed audit excludes 8 of 70 CASP14, 3 of 22 CASP15, and 3 of 21 CASP16 targets. CAMEO22 admits 68 of 183 by its full-chain screen. Baseline target IDs are taken from these model-independent lists, never from baseline success.

CASP14 was sampled during method development, so its clean coordinate gate does not make it an untouched validation set. CAMEO22 lacks a frozen domain-level screen. CASP15/16 were scored after the Run A sampler was fixed, but earlier project results on these target families were visible during the broader research program. They therefore are post-freeze checks, not fully blind prospective benchmarks.

## References and scores

CASP15 uses official domain/EU references from the frozen `primary_reference_manifest.tsv`, with mapped-residue-weighted domain means within each target, then an unweighted target mean. CASP16 uses official whole-chain references. CASP14 uses frozen whole-chain references; CAMEO22 uses state-1 references. Blank PDB chain identifiers are normalized to `A` before OpenStructure scoring. `benchmarks/stage_casp_references.py` records SHA-256 hashes of the staged coordinates; its normalization was checked byte-for-byte against the prior SimpleFold evaluation input for T1104-D1.

Reference input SHA-256 parity was verified against the corresponding SimpleFold evaluation pairs on every admitted pair: CASP14 62/62, CASP15 28/28 domain pairs, CASP16 18/18 whole-chain pairs, and CAMEO22 68/68 state-1 pairs.

All scores use OpenStructure 2.9.1 and `compare-structures --fault-tolerant --min-pep-length 4 --lddt --bb-lddt --rigid-scores --tm-score`. The CASP15 aggregation implementation reproduced all six metrics of an independently scored 22-target SimpleFold result to numerical precision (`<1e-12`). `benchmarks/score_openstructure.py` records per-pair input hashes and refuses incomplete expected counts.

The SimpleFold-360M baseline originates from locally generated seed-0 predictions in the prior evaluation archive. Its frozen invocation used the released `simplefold_360M` checkpoint with `--num_steps 500 --tau 0.01 --nsample_per_protein 1 --backend torch --output_format pdb --seed 0`, one canonical sequence per input. The prior archive's run manifest and launch script record this command. CASP14/CAMEO22 predictions can also be obtained from [Apple's released benchmark archives](https://github.com/apple/ml-simplefold#evaluation); CASP15/16 were inferred locally because those archives are not part of Apple's release. The source scores had complete coverage (70/70, 22/22, 21/21, 183/183). `benchmarks/summarize_matched_baseline.py` reaggregates those target rows using the current admitted-ID files. Its output contains source-summary hashes. An older file named `casp14_common62` describes a **different** 62-target set (six IDs differ) and was excluded; this package uses the complete 70-target source before filtering.

The SimpleFold comparison differs in pretrained PLM, corpus (SimpleFold additionally used AFESM), crop/training schedule, and optimization. Its score gap cannot be assigned to the SSM versus attention mixer. We have no matched-transformer control arm.

## Stereochemistry diagnostic

The same 28 admitted CASP15 domain/EU pairs and 18 CASP16 whole-chain pairs were rescored for both models with the same OpenStructure 2.9.1 command plus `--lddt-no-stereochecks`. The reference file SHA-256 matched between models for every pair. The script `benchmarks/diagnose_lddt_stereo.py` verifies that the original scores had stereochemistry checks enabled, verifies each no-check run succeeded, and uses the original mapped-residue weights within CASP15 targets. Its compact result is `data/stereo_diagnosis.json`; full per-pair scorer JSON is under the ignored `outputs/diagnostics/lddt-stereo-{casp15,casp16}/` directories. CPU-only Slurm jobs 86186 and 86187 completed with exit code 0. The diagnostic changes the scoring rule only; it neither fixes the structures nor replaces the reported benchmark lDDT.

The prior geometry fine-tune comparison uses the same 62 CASP14 and 68 CAMEO22 admitted targets, with OpenStructure 2.9.1 standard scores under `outputs/benchmarks/{run-a-final,run-a-geo-clash-v2}/{casp14,cameo22}/scores-admitted/`. The geo checkpoint is `out/run-a-geo-clash-v2/ckpt_0010000.pt`. Violation counts are lengths of each raw scorer's `model_clashes` array, averaged across admitted targets; these are unnormalized counts, so only within-cohort, paired-run comparisons are meaningful.

## Reproduction commands

Run heavy inference on a GPU compute node through Slurm; run scoring on `cpu_only`. The reference root below is the prepared official CASP dataset directory with `primary_reference_manifest.tsv` and `references/`.

```bash
MF_BENCHMARK_SET=casp15 MF_BENCHMARK_STEPS=500 MF_BENCHMARK_SEED=0 \
  MF_BENCHMARK_SDE_TAU=0.01 MF_BENCHMARK_OUT_ROOT=outputs/benchmarks/run-a-final \
  sbatch scripts/slurm_rollout_external.sh

MF_BENCHMARK_SET=casp16 MF_BENCHMARK_STEPS=500 MF_BENCHMARK_SEED=0 \
  MF_BENCHMARK_SDE_TAU=0.01 MF_BENCHMARK_OUT_ROOT=outputs/benchmarks/run-a-final \
  sbatch scripts/slurm_rollout_external.sh

MF_BENCHMARK_SET=casp15 MF_BENCHMARK_RUN=run-a-final \
  MF_BENCHMARK_REFERENCE_ROOT=/path/to/casp15_single_chain \
  sbatch scripts/slurm_score_external.sh

MF_BENCHMARK_SET=casp16 MF_BENCHMARK_RUN=run-a-final \
  MF_BENCHMARK_REFERENCE_ROOT=/path/to/casp16_single_chain \
  sbatch scripts/slurm_score_external.sh
```

The scoring script stages full and admitted sets separately. The two CASP15 `target-summary.json` files are the primary target-level results; CASP16 `summary.json` is already at target level. `docs/paper/data/` contains compact baseline rows and hashes; the large checkpoints, embeddings, raw PDBs, and local absolute paths are intentionally excluded from Git.
