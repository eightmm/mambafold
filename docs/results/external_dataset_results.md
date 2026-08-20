# External benchmark results by dataset

Snapshot: 2026-08-13 (Asia/Seoul)

> **Model-status boundary.** MambaFold-ESM3 step 120,000 is the frozen release
> artifact. Every MambaFold-ESMC-6B value is a provisional checkpoint snapshot
> from an active training program, not a final model or release result.
> "Complete" and "final for this snapshot" below refer only to artifact or
> evaluation completion. CASP14 is retrospective engineering evidence, not an
> untouched confirmatory evaluation.

This document is the dataset-oriented status page for the six frozen external
single-chain FASTA sets. Prediction generation and failure accounting are
final for this snapshot. Reference-based structure accuracy is complete for
CASP14, CASP15, CASP16, and CAMEO22. Apo and CoDNaS retain system measurements
until their five-sample, two-reference-state evaluation is completed.

## Evaluation contract

- Inputs: 463 dataset records representing 462 unique sequences. One
  134-residue sequence occurs in both Apo and CoDNaS.
- Seed-0 contract: one attempt per unique sequence for every model; each
  successful attempt writes one structure.
- Two-state sampling: MambaFold-ESM3, MambaFold-ESMC-6B, and
  SimpleFold-360M additionally generated seeds 1--4 for the 166 unique
  Apo/CoDNaS sequences. Each of these models therefore has 1,126 successful
  prediction records: `462 + 4 x 166`.
- Common subset: the seed-0 intersection successfully predicted by all six
  models. OmegaFold determines this intersection because it OOMed on 46 of
  462 unique sequences.
- Time: synchronized per-target wall time after loading the model once per
  shard. Values include target preparation, folding, and structure writing.
  The first target in a shard can include lazy-kernel/JIT warm-up.
- VRAM: maximum PyTorch CUDA memory allocated during one target, including
  resident model parameters. It is not the process-level `nvidia-smi` peak.
- MambaFold-ESM3 timing includes online ESM3 embedding generation.
  MambaFold-ESMC-6B timing loads a precomputed pinned ESMC-6B embedding, so its
  time and VRAM are folding-head measurements rather than end-to-end ESMC-6B
  pipeline measurements.
- The common runtime tables use seed 0 on RTX 6000 Ada 48 GB, except OmegaFold,
  which used RTX PRO 6000 Blackwell Max-Q 96 GB. Extra ESMC two-state seeds
  used RTX A5000 GPUs and are excluded from the cross-model timing tables.
- MambaFold and SimpleFold used 500-step SDE sampling; ESMFold is
  deterministic, DPLM-2 used `max_iter=100`, and OmegaFold used the disclosed
  reduced FP32 contract of one pseudo-MSA, one cycle, and subbatch size 8.
  OmegaFold's official 10-cycle setting did not fit the available 96 GB GPU.

## Overall completion

Each model cell is `successful / expected` for seed 0. Length is
`minimum / median / maximum` residues.

| Dataset | Targets | Length | Common N | MambaFold ESM3 120k | MambaFold ESMC 119.5k | SimpleFold 360M | ESMFold v1 | DPLM-2 650M | OmegaFold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CASP14 | 70 | 35 / 174.5 / 881 | 62 | 70/70 | 70/70 | 70/70 | 70/70 | 70/70 | 62/70 |
| CASP15 strict single-chain | 22 | 114 / 397 / 653 | 13 | 22/22 | 22/22 | 22/22 | 22/22 | 22/22 | 13/22 |
| CASP16 strict single-chain | 21 | 120 / 291 / 620 | 19 | 21/21 | 21/21 | 21/21 | 21/21 | 21/21 | 19/21 |
| CAMEO22 | 183 | 31 / 252 / 709 | 166 | 183/183 | 183/183 | 183/183 | 183/183 | 183/183 | 166/183 |
| Apo | 90 | 90 / 176.5 / 580 | 85 | 90/90 | 90/90 | 90/90 | 90/90 | 90/90 | 85/90 |
| CoDNaS | 77 | 26 / 221 / 593 | 72 | 77/77 | 77/77 | 77/77 | 77/77 | 77/77 | 72/77 |
| **All unique sequences** | **462** | **26 / 228.5 / 881** | **416** | **462/462** | **462/462** | **462/462** | **462/462** | **462/462** | **416/462** |

The coverage-aware finalizer validated every expected row and every successful
structure file. MambaFold-ESMC-6B has 1,126 PDB and 1,126 mmCIF files. All 46
OmegaFold failures are explicit CUDA OOM records, spanning lengths 489--881;
they are not silently removed from the aggregate.

## CASP14

Role: 70 whole-chain CASP14 targets. Prediction coverage is complete for five
models and 62/70 for OmegaFold.

| Model | Coverage | Time, mean (median) s | Max allocated VRAM GiB |
| --- | ---: | ---: | ---: |
| MambaFold-ESM3, step 120,000 | 70/70 | 31.32 (25.52) | 9.41 |
| MambaFold-ESMC-6B, step 119,500 | 70/70 | 31.64 (35.00) | 2.39 |
| SimpleFold-360M | 70/70 | 11.73 (10.87) | 17.59 |
| ESMFold v1 | 70/70 | 2.87 (1.42) | 9.52 |
| DPLM-2 Bit 650M | 70/70 | 2.72 (2.51) | 4.68 |
| OmegaFold model 2 | 62/70 | 30.63 (15.48) | 92.72 |

Time and VRAM use the 62-target all-model common subset. Structure scores
below were recomputed with OpenStructure 2.9.1 from seed-0 predictions.

### Full 70-target structure accuracy

| Model | N | GDT-TS ↑ | GDT-HA ↑ | TM-score ↑ | all-atom lDDT ↑ | backbone lDDT ↑ | RMSD (Å) ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **MambaFold-ESM3, step 120,000** | **70** | **0.670** | **0.533** | **0.757** | **0.657** | **0.763** | **6.265** |
| MambaFold-ESMC-6B, step 119,500 | 70 | 0.654 | 0.514 | 0.740 | 0.632 | 0.753 | 6.611 |
| ESMFold v1 | 70 | 0.623 | 0.505 | 0.700 | 0.634 | 0.722 | 8.545 |
| SimpleFold-360M | 70 | 0.585 | 0.452 | 0.675 | 0.630 | 0.708 | 9.229 |
| DPLM-2 Bit 650M | 70 | 0.408 | 0.294 | 0.530 | 0.199* | 0.531 | 14.449 |

### Common 62-target structure accuracy

| Model | N | GDT-TS ↑ | GDT-HA ↑ | TM-score ↑ | all-atom lDDT ↑ | backbone lDDT ↑ | RMSD (Å) ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **MambaFold-ESM3, step 120,000** | **62** | **0.687** | **0.553** | **0.757** | **0.665** | **0.767** | **5.947** |
| MambaFold-ESMC-6B, step 119,500 | 62 | 0.675 | 0.537 | 0.746 | 0.644 | 0.760 | 5.812 |
| ESMFold v1 | 62 | 0.641 | 0.523 | 0.707 | 0.638 | 0.724 | 7.581 |
| SimpleFold-360M | 62 | 0.607 | 0.474 | 0.683 | 0.643 | 0.717 | 8.084 |
| OmegaFold model 2 | 62 | 0.576 | 0.443 | 0.644 | 0.569 | 0.664 | 9.494 |
| DPLM-2 Bit 650M | 62 | 0.436 | 0.315 | 0.549 | 0.208* | 0.548 | 12.534 |

`*` DPLM-2 writes only `N`, `CA`, `C`, `O`, and `CB`; its all-atom lDDT is
not directly comparable with full-side-chain outputs. Backbone lDDT is the
more appropriate local-quality metric.

The full-70 and common-62 tables must not be mixed. The common set is
conditioned on OmegaFold inference success.

## CASP15 strict single-chain

Role: 22 strict single-chain targets.

| Model | Coverage | Time, mean (median) s | Max allocated VRAM GiB |
| --- | ---: | ---: | ---: |
| MambaFold-ESM3, step 120,000 | 22/22 | 30.37 (26.83) | 5.10 |
| MambaFold-ESMC-6B, step 119,500 | 22/22 | 30.53 (24.85) | 2.37 |
| SimpleFold-360M | 22/22 | 12.37 (12.56) | 16.53 |
| ESMFold v1 | 22/22 | 3.83 (3.11) | 9.05 |
| DPLM-2 Bit 650M | 22/22 | 2.66 (2.60) | 4.62 |
| OmegaFold model 2 | 13/22 | 45.26 (45.35) | 67.06 |

Time and VRAM use the 13-target common subset. CASP15 supplies 33 official
domain/EU references for 22 targets. Each domain/EU is scored independently;
metrics are mapped-residue-weighted within a target and then averaged equally
over targets. Blank chain IDs in official PDB coordinate records are assigned
`A` in staged scoring copies only; source references and coordinates are not
changed.

### Full 22-target structure accuracy

| Model | N | GDT-TS ↑ | GDT-HA ↑ | TM-score ↑ | all-atom lDDT ↑ | backbone lDDT ↑ | RMSD (Å) ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MambaFold-ESM3, step 120,000 | 22 | 0.635 | 0.500 | 0.710 | 0.685 | 0.787 | 11.992 |
| MambaFold-ESMC-6B, step 119,500 | 22 | 0.602 | 0.463 | 0.679 | 0.646 | 0.772 | 13.407 |
| SimpleFold-360M | 22 | 0.590 | 0.461 | 0.670 | 0.681 | 0.766 | 13.981 |
| ESMFold v1 | 22 | 0.649 | 0.535 | 0.705 | 0.709 | 0.798 | 12.824 |
| DPLM-2 Bit 650M | 22 | 0.441 | 0.323 | 0.546 | 0.236* | 0.607 | 18.348 |

### Common 13-target structure accuracy

| Model | N | GDT-TS ↑ | GDT-HA ↑ | TM-score ↑ | all-atom lDDT ↑ | backbone lDDT ↑ | RMSD (Å) ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MambaFold-ESM3, step 120,000 | 13 | 0.702 | 0.556 | 0.758 | 0.732 | 0.834 | 8.785 |
| MambaFold-ESMC-6B, step 119,500 | 13 | 0.655 | 0.504 | 0.721 | 0.679 | 0.809 | 10.727 |
| SimpleFold-360M | 13 | 0.635 | 0.496 | 0.701 | 0.716 | 0.802 | 11.236 |
| ESMFold v1 | 13 | 0.715 | 0.593 | 0.754 | 0.754 | 0.844 | 8.879 |
| DPLM-2 Bit 650M | 13 | 0.457 | 0.345 | 0.538 | 0.248* | 0.620 | 17.433 |
| OmegaFold model 2 | 13 | 0.630 | 0.488 | 0.693 | 0.664 | 0.767 | 11.296 |

On the full set, ESMFold leads GDT-TS and both lDDT variants, while
MambaFold-ESM3 has the best TM-score and RMSD. MambaFold-ESMC-6B is below the
ESM3 checkpoint here, but exceeds SimpleFold in GDT-TS, GDT-HA, TM-score,
backbone lDDT, and RMSD. The same ordering is broadly retained on the common
13 targets.

## CASP16 strict single-chain

Role: 21 strict single-chain targets.

| Model | Coverage | Time, mean (median) s | Max allocated VRAM GiB |
| --- | ---: | ---: | ---: |
| MambaFold-ESM3, step 120,000 | 21/21 | 25.08 (24.92) | 5.22 |
| MambaFold-ESMC-6B, step 119,500 | 21/21 | 30.14 (34.62) | 2.39 |
| SimpleFold-360M | 21/21 | 12.57 (12.54) | 17.56 |
| ESMFold v1 | 21/21 | 4.15 (3.05) | 9.51 |
| DPLM-2 Bit 650M | 21/21 | 2.76 (2.57) | 4.63 |
| OmegaFold model 2 | 19/21 | 46.66 (44.23) | 91.98 |

Time and VRAM use the 19-target common subset. Structure accuracy uses the
official whole-chain reference for each target.

### Full 21-target structure accuracy

| Model | N | GDT-TS ↑ | GDT-HA ↑ | TM-score ↑ | all-atom lDDT ↑ | backbone lDDT ↑ | RMSD (Å) ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MambaFold-ESM3, step 120,000 | 21 | 0.520 | 0.395 | 0.629 | 0.559 | 0.671 | 12.297 |
| MambaFold-ESMC-6B, step 119,500 | 21 | 0.527 | 0.377 | 0.652 | 0.577 | 0.708 | 11.143 |
| SimpleFold-360M | 21 | 0.530 | 0.406 | 0.638 | 0.616 | 0.700 | 13.387 |
| ESMFold v1 | 21 | 0.622 | 0.500 | 0.706 | 0.668 | 0.755 | 11.133 |
| DPLM-2 Bit 650M | 21 | 0.393 | 0.274 | 0.524 | 0.201* | 0.554 | 17.340 |

### Common 19-target structure accuracy

| Model | N | GDT-TS ↑ | GDT-HA ↑ | TM-score ↑ | all-atom lDDT ↑ | backbone lDDT ↑ | RMSD (Å) ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MambaFold-ESM3, step 120,000 | 19 | 0.529 | 0.404 | 0.635 | 0.551 | 0.662 | 11.396 |
| MambaFold-ESMC-6B, step 119,500 | 19 | 0.547 | 0.395 | 0.663 | 0.574 | 0.703 | 10.649 |
| SimpleFold-360M | 19 | 0.541 | 0.416 | 0.647 | 0.611 | 0.696 | 12.076 |
| ESMFold v1 | 19 | 0.643 | 0.520 | 0.721 | 0.666 | 0.753 | 9.788 |
| DPLM-2 Bit 650M | 19 | 0.414 | 0.288 | 0.546 | 0.200* | 0.552 | 16.045 |
| OmegaFold model 2 | 19 | 0.521 | 0.405 | 0.613 | 0.552 | 0.650 | 14.258 |

ESMFold leads the CASP16 accuracy table. Relative to the ESM3 checkpoint,
MambaFold-ESMC-6B improves GDT-TS, TM-score, both lDDT variants, and RMSD, but
not GDT-HA. Against SimpleFold, ESMC has the higher TM-score and backbone lDDT
and the lower RMSD, whereas SimpleFold has slightly higher GDT and all-atom
lDDT.

## CAMEO22

Role: the 183-target CAMEO22 folding set used by SimpleFold.

| Model | Coverage | Time, mean (median) s | Max allocated VRAM GiB |
| --- | ---: | ---: | ---: |
| MambaFold-ESM3, step 120,000 | 183/183 | 32.30 (35.74) | 9.41 |
| MambaFold-ESMC-6B, step 119,500 | 183/183 | 31.57 (34.93) | 2.39 |
| SimpleFold-360M | 183/183 | 12.22 (11.71) | 17.54 |
| ESMFold v1 | 183/183 | 3.43 (1.79) | 9.50 |
| DPLM-2 Bit 650M | 183/183 | 2.73 (2.56) | 4.67 |
| OmegaFold model 2 | 166/183 | 38.33 (27.84) | 91.61 |

Time and VRAM use the 166-target common subset. Structure accuracy uses each
target's frozen state-1 reference.

### Full 183-target structure accuracy

| Model | N | GDT-TS ↑ | GDT-HA ↑ | TM-score ↑ | all-atom lDDT ↑ | backbone lDDT ↑ | RMSD (Å) ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MambaFold-ESM3, step 120,000 | 183 | 0.853 | 0.731 | 0.886 | 0.804 | 0.892 | 2.883 |
| **MambaFold-ESMC-6B, step 119,500** | **183** | **0.862** | **0.737** | **0.891** | **0.811** | **0.904** | **2.698** |
| SimpleFold-360M | 183 | 0.782 | 0.649 | 0.824 | 0.773 | 0.845 | 4.804 |
| ESMFold v1 | 183 | 0.826 | 0.703 | 0.853 | 0.792 | 0.871 | 3.988 |
| DPLM-2 Bit 650M | 183 | 0.660 | 0.494 | 0.749 | 0.291* | 0.728 | 6.606 |

### Common 166-target structure accuracy

| Model | N | GDT-TS ↑ | GDT-HA ↑ | TM-score ↑ | all-atom lDDT ↑ | backbone lDDT ↑ | RMSD (Å) ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MambaFold-ESM3, step 120,000 | 166 | 0.857 | 0.739 | 0.882 | 0.810 | 0.897 | 2.823 |
| **MambaFold-ESMC-6B, step 119,500** | **166** | **0.866** | **0.746** | **0.887** | **0.818** | **0.909** | **2.708** |
| SimpleFold-360M | 166 | 0.791 | 0.661 | 0.823 | 0.781 | 0.853 | 4.636 |
| ESMFold v1 | 166 | 0.828 | 0.707 | 0.848 | 0.795 | 0.875 | 3.952 |
| DPLM-2 Bit 650M | 166 | 0.675 | 0.509 | 0.750 | 0.297* | 0.739 | 6.301 |
| OmegaFold model 2 | 166 | 0.722 | 0.586 | 0.765 | 0.694 | 0.790 | 6.104 |

Within this provisional checkpoint snapshot, MambaFold-ESMC-6B leads every
reported metric on both the full 183-target set and the all-six 166-target
intersection. MambaFold-ESM3 is second on each metric. These numbers describe
reconstruction on the frozen test sequences, not blind generalization: the
active full RCSB training corpus overlaps many CAMEO22 entries, so a
cutoff/overlap audit is required for that claim.

## Apo

Role: 90 sequences from the SimpleFold two-state benchmark.

| Model | Seed-0 coverage | Time, mean (median) s | Max allocated VRAM GiB |
| --- | ---: | ---: | ---: |
| MambaFold-ESM3, step 120,000 | 90/90 | 24.88 (24.69) | 5.22 |
| MambaFold-ESMC-6B, step 119,500 | 90/90 | 30.02 (24.83) | 2.39 |
| SimpleFold-360M | 90/90 | 11.66 (11.10) | 17.56 |
| ESMFold v1 | 90/90 | 2.38 (1.44) | 9.51 |
| DPLM-2 Bit 650M | 90/90 | 2.62 (2.53) | 4.63 |
| OmegaFold model 2 | 85/90 | 26.84 (14.83) | 91.98 |

Time and VRAM use the 85-target seed-0 common subset. Seeds 0--4 are complete
for both MambaFold variants and SimpleFold. The intended five-sample,
two-reference-state structural evaluation has not yet been run, so diversity
or state-recovery claims are not reported.

## CoDNaS

Role: 77 sequences from the SimpleFold two-state benchmark.

| Model | Seed-0 coverage | Time, mean (median) s | Max allocated VRAM GiB |
| --- | ---: | ---: | ---: |
| MambaFold-ESM3, step 120,000 | 77/77 | 25.56 (24.78) | 9.41 |
| MambaFold-ESMC-6B, step 119,500 | 77/77 | 31.76 (35.26) | 2.39 |
| SimpleFold-360M | 77/77 | 12.06 (11.56) | 17.57 |
| ESMFold v1 | 77/77 | 3.20 (1.60) | 9.51 |
| DPLM-2 Bit 650M | 77/77 | 2.68 (2.55) | 4.67 |
| OmegaFold model 2 | 72/77 | 36.01 (21.01) | 92.35 |

Time and VRAM use the 72-target seed-0 common subset. Seeds 0--4 are complete
for both MambaFold variants and SimpleFold. The intended five-sample,
two-reference-state structural evaluation has not yet been run. One sequence
is shared with Apo and was inferred once per model/seed.

## Aggregate system summary

The following values use the 416 unique seed-0 sequences successfully
predicted by all six models.

| Model | Mean (median) s | Max allocated VRAM GiB | Seed-0 coverage |
| --- | ---: | ---: | ---: |
| MambaFold-ESM3, step 120,000 | 29.10 (25.08) | 9.41 | 462/462 |
| MambaFold-ESMC-6B, step 119,500 | 31.19 (35.00) | 2.39 | 462/462 |
| SimpleFold-360M | 12.03 (11.45) | 17.59 | 462/462 |
| ESMFold v1 | 3.14 (1.67) | 9.52 | 462/462 |
| DPLM-2 Bit 650M | 2.70 (2.54) | 4.68 | 462/462 |
| OmegaFold model 2 | 35.10 (22.43) | 92.72 | 416/462 |

These measurements are not hardware-, embedding-, or algorithm-normalized.
They describe the exact runners used here and are not architecture-only
throughput claims.

## Provenance and limitations

- FASTA inputs and hashes:
  [`benchmarks/external_testsets`](../../benchmarks/external_testsets/README.md).
- Detailed CASP14 statistics, paired comparisons, length stratification, and
  sampler analysis:
  [`external_common_results.md`](external_common_results.md).
- Local raw records:
  `outputs/eval/external_compare_v1_20260812/per_target_metrics.csv`.
- Coverage validation:
  `outputs/eval/external_compare_v1_20260812/coverage_validation_report.json`
  (`validation_passed=true`, Slurm finalizer job 53636).
- CASP14 OpenStructure summaries:
  `outputs/eval/external_compare_v1_20260812/scores/`.
- CASP15, CASP16, and CAMEO22 OpenStructure summaries and per-target rows:
  `outputs/eval/external_compare_v1_20260812/scores/external_accuracy_v2/`.
- Training data and cutoff dates are not normalized across baseline models.
  These are controlled inference-output comparisons, not blind or
  leakage-controlled CASP claims.
