# Archived external benchmark snapshot (legacy ESM3/OmegaFold)

> **Archive only; not an active comparison.** This 2026-08-13 snapshot is
> preserved as historical numerical evidence for runners that included the
> frozen ESM3 artifact and an OOM-limited OmegaFold attempt. Active MambaFold
> development and reporting use ESMC-6B only. OmegaFold is excluded from the
> active comparator roster because it did not complete the full target sets.
> Do not copy the common-subset rankings below into current results.

Snapshot: 2026-08-13 (Asia/Seoul)

> Current dataset roles and ESMC-only reporting are in
> [`external_dataset_results.md`](external_dataset_results.md). This document
> retains the detailed legacy ESM3-focused accuracy and speed analysis. The
> ESM3 artifact is frozen; the ESMC-6B step-119,500 snapshot recorded below was
> provisional and has been superseded by the step-170k preview.

## Scope

This snapshot separates **prediction completion and system measurements** from
**reference-based structure accuracy**. Runtime and VRAM do not imply structural
accuracy.

- Inputs: the six frozen single-chain FASTA sets in
  [`benchmarks/external_testsets`](../../benchmarks/external_testsets/).
- Input cardinality: 463 dataset occurrences mapping to 462 unique sequences.
  One 134-residue sequence is shared by Apo and CoDNaS.
- Common runtime subset: seed 0 targets successfully predicted by all five
  models below. This leaves 416 unique sequences (417 dataset occurrences).
- Timing: synchronized per-target wall time after one model load per shard;
  preprocessing, inference, and structure serialization are included. The first
  target of a shard may include lazy-kernel warm-up.
- VRAM: per-target peak CUDA memory allocated, including resident model
  parameters. Values are GiB (`2^30` bytes).

## Prediction coverage

Each cell is `successful / expected`. The `Common` column is the intersection
used in the runtime and VRAM tables.

| Dataset | Targets | MambaFold-ESM3 120k | SimpleFold-360M | ESMFold v1 | DPLM-2 Bit 650M | OmegaFold model 2 | Common |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CASP14 | 70 | 70/70 | 70/70 | 70/70 | 70/70 | 62/70 | 62 |
| CASP15 single-chain | 22 | 22/22 | 22/22 | 22/22 | 22/22 | 13/22 | 13 |
| CASP16 single-chain | 21 | 21/21 | 21/21 | 21/21 | 21/21 | 19/21 | 19 |
| CAMEO22 | 183 | 183/183 | 183/183 | 183/183 | 183/183 | 166/183 | 166 |
| Apo | 90 | 90/90 | 90/90 | 90/90 | 90/90 | 85/90 | 85 |
| CoDNaS | 77 | 77/77 | 77/77 | 77/77 | 77/77 | 72/77 | 72 |
| **All unique sequences** | **462** | **462/462** | **462/462** | **462/462** | **462/462** | **416/462** | **416** |

OmegaFold failed on 46 sequences with CUDA OOM. The other four models completed
all 462 unique sequences.

## Runtime on the common successful subset

Values are **mean (median) seconds per target**. Apo and CoDNaS use seed 0 here
so every model contributes exactly one prediction per sequence.

| Dataset | Common N | MambaFold-ESM3 120k | SimpleFold-360M | ESMFold v1 | DPLM-2 Bit 650M | OmegaFold model 2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CASP14 | 62 | 31.32 (25.52) | 11.73 (10.87) | 2.87 (1.42) | 2.72 (2.51) | 30.63 (15.48) |
| CASP15 single-chain | 13 | 30.37 (26.83) | 12.37 (12.56) | 3.83 (3.11) | 2.66 (2.60) | 45.26 (45.35) |
| CASP16 single-chain | 19 | 25.08 (24.92) | 12.57 (12.54) | 4.15 (3.05) | 2.76 (2.57) | 46.66 (44.23) |
| CAMEO22 | 166 | 32.30 (35.74) | 12.22 (11.71) | 3.43 (1.79) | 2.73 (2.56) | 38.33 (27.84) |
| Apo | 85 | 24.88 (24.69) | 11.66 (11.10) | 2.38 (1.44) | 2.62 (2.53) | 26.84 (14.83) |
| CoDNaS | 72 | 25.56 (24.78) | 12.06 (11.56) | 3.20 (1.60) | 2.68 (2.55) | 36.01 (21.01) |
| **All unique sequences** | **416** | **29.10 (25.08)** | **12.03 (11.45)** | **3.14 (1.67)** | **2.70 (2.54)** | **35.10 (22.43)** |

### Runtime by sequence length

The same 416-sequence intersection, grouped by canonical sequence length,
shows that the models scale differently. Values remain mean (median) seconds.

| Length | N | MambaFold-ESM3 120k | SimpleFold-360M | ESMFold v1 | DPLM-2 Bit 650M | OmegaFold model 2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1--128 | 107 | 29.79 (24.69) | 10.85 (10.46) | 1.08 (1.07) | 2.60 (2.52) | 5.63 (5.37) |
| 129--256 | 152 | 27.51 (24.87) | 11.08 (10.96) | 1.55 (1.50) | 2.52 (2.51) | 19.23 (17.06) |
| 257--512 | 157 | 30.17 (25.59) | 13.75 (13.49) | 6.09 (5.13) | 2.93 (2.73) | 70.56 (66.67) |

There is no five-model common target above length 512 because OmegaFold failed
on every such input. Its length-stratified completion was:

| Length | Successful / attempted |
| --- | ---: |
| 1--128 | 107/107 |
| 129--256 | 152/152 |
| 257--512 | 157/166 |
| 513--768 | 0/35 |
| 769--1024 | 0/2 |

The longest OmegaFold success was length 484 and the first failure was length
489. This boundary is specific to the tested FP32 configuration and 96 GB GPU;
it is not an architecture-wide maximum length claim.

## Runtime interpretation and optimization headroom

The MambaFold-ESM3 timing above is the measured cost of the then-profiled
500-step runner, not an optimized steady-state lower bound.

| Runtime subset | N | MambaFold-ESM3 120k | SimpleFold-360M | Mean-time ratio |
| --- | ---: | ---: | ---: | ---: |
| Full five-model common subset | 416 | 29.10 (25.08) s | 12.03 (11.45) s | 2.42x |
| No logged Mamba kernel compilation | 294 | 25.24 (24.75) s | 11.86 (11.19) s | 2.13x |

The seed-0 MambaFold logs contain 173 TileLang forward-kernel compilation
events. They took 10.49 seconds on average (10-second median; 9--14-second
range) and affected 152/462 targets, including 122/416 targets in the common
subset. Removing those targets narrows the mean gap, but a 2.13x gap remains;
JIT compilation therefore explains only part of the recorded runtime. The
length dependence is also different: on the single longest sequence
(`L=881`), MambaFold took 25.80 seconds and SimpleFold took 30.46 seconds on the
same RTX 6000 Ada. This one target is diagnostic evidence, not a throughput
claim.

The legacy frozen-ESM3 profiling wrapper also has avoidable runner overhead:

- it leaves `record_trajectory=True`, copying a C-alpha trajectory from GPU to
  CPU at every one of the 500 sampling steps;
- it rebuilds the static `ProteinBatch` fields and transfers them to CUDA at
  every step;
- it calls `torch.cuda.empty_cache()` after every target; and
- raw sequence lengths trigger many distinct TileLang kernel shapes instead of
  a small set of pre-warmed length buckets.

The low-risk optimization order is therefore: use a separate evaluation
wrapper with trajectory recording disabled and one reusable static GPU batch;
bucket/pad lengths and warm each kernel shape once; retain allocator caches
except after OOM; and reuse cached PLM embeddings across seeds or repeated
evaluations. The immutable `projects/esm3/` release should remain unchanged,
with output-equivalence checked in the wrapper.

There is additional algorithmic headroom from the sampler, but it is not yet a
validated default. The following single-target sweep used MambaFold-ESMC-6B
step 88,500 on CASP14 T1061 (`L=881`):

| Solver / schedule | Steps | Time (s) | GDT-TS ↑ | TM-score ↑ | all-atom lDDT ↑ | backbone lDDT ↑ | RMSD (Å) ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SDE / logarithmic | 500 | 40.74 | 0.616 | 0.796 | 0.595 | 0.767 | 13.237 |
| SDE / logarithmic | 200 | 16.34 | 0.584 | **0.815** | **0.616** | **0.775** | **11.910** |
| **SDE / uniform** | **100** | **7.20** | **0.612** | 0.794 | 0.576 | 0.766 | 13.382 |
| ODE / uniform | 100 | 7.23 | 0.403 | 0.672 | 0.330 | 0.656 | 18.158 |
| SDE / logarithmic | 100 | 8.21 | 0.019 | 0.087 | 0.008 | 0.104 | 98.628 |

Uniform 100-step SDE was the snapshot's fast-preset candidate: it was 5.7x faster
than 500-step logarithmic SDE on this target while retaining similar global
scores. The collapse of 100-step logarithmic SDE shows that the time schedule,
not only the step count, is critical. A complete CASP14 sweep is required before
claiming a dataset-wide speedup or changing the default.

## Peak VRAM on the common successful subset

Values are the **maximum peak allocated GiB** observed within each dataset.

| Dataset | Common N | MambaFold-ESM3 120k | SimpleFold-360M | ESMFold v1 | DPLM-2 Bit 650M | OmegaFold model 2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CASP14 | 62 | 9.41 | 17.59 | 9.52 | 4.68 | 92.72 |
| CASP15 single-chain | 13 | 5.10 | 16.53 | 9.05 | 4.62 | 67.06 |
| CASP16 single-chain | 19 | 5.22 | 17.56 | 9.51 | 4.63 | 91.98 |
| CAMEO22 | 166 | 9.41 | 17.54 | 9.50 | 4.67 | 91.61 |
| Apo | 85 | 5.22 | 17.56 | 9.51 | 4.63 | 91.98 |
| CoDNaS | 72 | 9.41 | 17.57 | 9.51 | 4.67 | 92.35 |
| **All unique sequences** | **416** | **9.41** | **17.59** | **9.52** | **4.68** | **92.72** |

## CASP14 structure accuracy

### Local full-70 comparison

MambaFold-ESM3, SimpleFold-360M, ESMFold, and DPLM-2 all completed the full
CASP14 70-target set. Their local seed-0 outputs were rescored from scratch with
the same OpenStructure 2.9.1 command.

| Model | N | GDT-TS ↑ | GDT-HA ↑ | TM-score ↑ | all-atom lDDT ↑ | backbone lDDT ↑ | RMSD (Å) ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **MambaFold-ESM3, step 120,000** | **70** | **0.670** | **0.533** | **0.757** | **0.657** | **0.763** | **6.265** |
| ESMFold v1 | 70 | 0.623 | 0.505 | 0.700 | 0.634 | 0.722 | 8.545 |
| SimpleFold-360M | 70 | 0.585 | 0.452 | 0.675 | 0.630 | 0.708 | 9.229 |
| DPLM-2 Bit 650M | 70 | 0.408 | 0.294 | 0.530 | 0.199† | 0.531 | 14.449 |

† DPLM-2 outputs only `N`, `CA`, `C`, `O`, and `CB`, so its all-atom lDDT is
not directly comparable with full-side-chain outputs. Backbone lDDT is the
more appropriate local-quality column.

This table uses the newly generated local prediction set. The later
checkpoint-reference table retains the frozen ESM3 release aggregate and the
published SimpleFold aggregates. Their small numerical differences must not be
averaged together, and the local rerun does not replace the frozen ESM3 result.

The paired differences below are `MambaFold - baseline`; confidence intervals
are percentile intervals from 20,000 deterministic paired target-bootstrap
resamples. A win is a target on which MambaFold has the higher score.

| Baseline | N | Δ GDT-TS [95% CI] | GDT wins | Δ TM-score [95% CI] | TM wins | Δ backbone lDDT [95% CI] | bb-lDDT wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ESMFold v1 | 70 | +0.047 [−0.001, +0.098] | 38/70 | +0.057 [+0.014, +0.104] | 42/70 | +0.041 [+0.008, +0.078] | 41/70 |
| SimpleFold-360M | 70 | +0.085 [+0.043, +0.131] | 43/70 | +0.082 [+0.042, +0.126] | 46/70 | +0.055 [+0.024, +0.089] | 41/70 |
| DPLM-2 Bit 650M | 70 | +0.263 [+0.204, +0.323] | 59/70 | +0.228 [+0.173, +0.282] | 58/70 | +0.233 [+0.182, +0.283] | 64/70 |
| OmegaFold model 2‡ | 62 | +0.111 [+0.041, +0.182] | 38/62 | +0.113 [+0.048, +0.181] | 41/62 | +0.103 [+0.051, +0.157] | 44/62 |

‡ OmegaFold uses the successful common-62 subset, not the full 70 targets. The
GDT-TS interval against ESMFold narrowly includes zero; the TM-score and
backbone-lDDT intervals do not.

The full-70 result also remains favorable after stratifying by sequence length.
Each cell below is `GDT-TS / TM-score`.

| Length | N | MambaFold-ESM3 120k | ESMFold v1 | SimpleFold-360M | DPLM-2 Bit 650M |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1--256 | 43 | **0.707 / 0.750** | 0.650 / 0.686 | 0.642 / 0.685 | 0.477 / 0.563 |
| 257--512 | 19 | **0.640 / 0.772** | 0.620 / 0.754 | 0.529 / 0.679 | 0.341 / 0.518 |
| 513--1024 | 8 | **0.542 / 0.761** | 0.487 / 0.652 | 0.413 / 0.615 | 0.190 / 0.378 |

### Local common-62 comparison

All five local seed-0 predictions were rescored on the exact same 62 CASP14
targets with OpenStructure 2.9.1. These are the targets for which every model,
including OmegaFold, completed inference.

| Model | N | GDT-TS ↑ | GDT-HA ↑ | TM-score ↑ | all-atom lDDT ↑ | backbone lDDT ↑ | RMSD (Å) ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **MambaFold-ESM3, step 120,000** | **62** | **0.687** | **0.553** | **0.757** | **0.665** | **0.767** | **5.947** |
| ESMFold v1 | 62 | 0.641 | 0.523 | 0.707 | 0.638 | 0.724 | 7.581 |
| SimpleFold-360M | 62 | 0.607 | 0.474 | 0.683 | 0.643 | 0.717 | 8.084 |
| OmegaFold model 2 | 62 | 0.576 | 0.443 | 0.644 | 0.569 | 0.664 | 9.494 |
| DPLM-2 Bit 650M | 62 | 0.436 | 0.315 | 0.549 | 0.208† | 0.548 | 12.534 |

† The DPLM-2 files contain only `N`, `CA`, `C`, `O`, and `CB`; its all-atom
lDDT is therefore not directly comparable with the four full-side-chain
outputs. Its backbone lDDT is the more appropriate local-quality column.

This common subset is conditioned on OmegaFold success and excludes its eight
CASP14 CUDA-OOM targets, so it must not be presented as the full 70-target
score. Training data and cutoff dates were also not normalized across models;
this is a controlled inference-output comparison, not a blind or
leakage-controlled CASP claim.

### Local stereochemical diagnostics

OpenStructure also reports stereochemical issue lists. The table aggregates
their counts over the common 62 targets and normalizes by the number of atoms in
each output. Lower is better. These are OpenStructure diagnostics, not
MolProbity clashscores.

| Structure source | Clashes / 1k atoms ↓ | Bad bonds / 1k atoms ↓ | Bad angles / 1k atoms ↓ |
| --- | ---: | ---: | ---: |
| Experimental references | 0.681 | 0.859 | 0.355 |
| MambaFold-ESM3 120k | 7.719 | 0.168 | 0.079 |
| SimpleFold-360M | **0.611** | 0.197 | **0.049** |
| ESMFold v1 | 0.710 | **0.039** | 3.046 |
| OmegaFold model 2 | 1.853 | **0.039** | 3.766 |
| DPLM-2 Bit 650M† | 19.919 | 2.064 | 2.423 |

MambaFold therefore has the strongest global fold metrics in this comparison,
but more inter-residue steric clashes than the other full-atom baselines. Local
relaxation or a stronger clash objective remains a clear improvement target.

### Confidence-output diagnostic

Mean C-alpha B-factor was compared with the target's OpenStructure all-atom
lDDT on the common 62 targets. Spearman correlation is invariant to whether a
runner writes confidence on a 0--1 or 0--100 scale, but this remains a diagnostic
rather than a cross-model calibration benchmark.

| Model | CA B-factor behavior | Spearman ρ with lDDT ↑ |
| --- | --- | ---: |
| MambaFold-ESM3 120k | range 37.89--54.30 | −0.360 |
| SimpleFold-360M | constant 100 | undefined |
| ESMFold v1 | range 0.20--0.99 | 0.914 |
| OmegaFold model 2 | range 11.30--98.62 | 0.947 |

The MambaFold ESM3 training contract has `w_conf=0.0`; its confidence head was
not trained. The step-119.5k ESMC snapshot also has `w_conf=0.0`. Consequently,
MambaFold B-factors from these runs must **not** be reported as calibrated
pLDDT. The observed negative rank correlation is consistent with that contract.

### Checkpoint and published-reference results

These are mean values on the full 70-target CASP14 whole-chain benchmark,
using OpenStructure 2.9.1. SimpleFold values are published paper aggregates;
MambaFold rows were scored locally with the same reporting contract.

| Model / checkpoint | Parameters | SDE steps | GDT-TS ↑ | TM-score ↑ | all-atom lDDT ↑ | backbone lDDT ↑ | RMSD (Å) ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SimpleFold-360M, paper | 360M | 500 | 0.585 | 0.674 | 0.617 | 0.703 | 9.382 |
| SimpleFold-3B, paper | 2.86B | 500 | 0.639 | 0.720 | **0.666** | 0.747 | 7.732 |
| MambaFold-ESMC-6B, step 50,000 | 404.9M | 500 | 0.596 | 0.687 | 0.577 | 0.719 | 8.178 |
| MambaFold-ESMC-6B, step 88,500 | 404.9M | 200 | 0.625 | 0.709 | 0.609 | 0.734 | 8.894 |
| MambaFold-ESMC-6B, step 88,500 | 404.9M | 500 | 0.629 | 0.718 | 0.612 | 0.738 | 7.275 |
| **MambaFold-ESM3, step 120,000** | **422.4M** | **500** | **0.670** | **0.757** | 0.657 | **0.763** | **6.276** |

From ESMC step 50,000 to 88,500 at 500 sampling steps, mean GDT-TS improved by
0.033, TM-score by 0.031, all-atom lDDT by 0.036, and backbone lDDT by 0.020;
RMSD improved by 0.904 Å. The later checkpoint was better on 51/70 targets by
GDT-TS and 55/70 by all-atom lDDT. Its simple local-geometry diagnostics also
improved: bond MAE fell from 0.0200 Å to 0.0143 Å and clashes fell from 12.46 to
9.54 per 1,000 atoms.

The two step-88,500 evaluations used the same A5000 and pipeline. The full
70-target job took 52:30 at 500 steps and 22:28 at 200 steps, a measured 2.34x
speedup. At 200 steps, GDT-TS changed by only −0.004, TM-score by −0.009,
all-atom lDDT by −0.003, and backbone lDDT by −0.004. The higher mean RMSD
(+1.619 Å), despite a median change of only +0.003 Å, shows that a small number
of severe failures remain important even when the average bounded scores
change little.

At the snapshot date, step 88,500 was the latest **fully scored** ESMC checkpoint. Step 119,500
external inference is complete but has not yet been reference-scored. The
ESMC result is an interim research result rather than a released model. ESM3
and ESMC are not conditioning-equivalent: ESM3 has
multimodal structure-aware pretraining, whereas the pinned ESMC-6B conditioner
is sequence-only. The ESM3 result is therefore a historical upper-bound-style
baseline, not evidence that the ESMC run has already converged or a
leakage-controlled CASP result.

## Protocol and limitations

| Model | Inference setting | GPU used for this snapshot |
| --- | --- | --- |
| MambaFold-ESM3 120k | SDE, 500 steps | RTX 6000 Ada 48 GB |
| SimpleFold-360M | SDE, 500 steps, tau 0.01 | RTX 6000 Ada 48 GB |
| ESMFold v1 | deterministic baseline | RTX 6000 Ada 48 GB |
| DPLM-2 Bit 650M | `max_iter=100` | RTX 6000 Ada 48 GB |
| OmegaFold model 2 | FP32, one pseudo-MSA, one cycle, subbatch 8 | RTX PRO 6000 Blackwell 96 GB |

- Runtime is not hardware-normalized, and the inference algorithms use
  different iteration counts. It should be read as the measured cost of these
  exact configurations, not as architecture-only throughput.
- MambaFold-ESMC-6B step 119,500 inference is complete (1,126/1,126 records)
  but is excluded from the structure-accuracy tables because reference scoring
  for that checkpoint has not yet been completed. Its dataset-level coverage,
  cached-embedding runtime, and VRAM are reported in
  `external_dataset_results.md`.
- The local CASP14 common-62 scores and per-target JSON files are under
  `outputs/eval/external_compare_v1_20260812/scores/casp14_common62/`.
- The corresponding four-model full-70 scores are under
  `outputs/eval/external_compare_v1_20260812/scores/casp14_full70/`.
- Reference-based scoring for CASP15, CASP16, CAMEO22, Apo, and CoDNaS is not
  complete. The first external OpenStructure attempt stopped on CASP15 T1104
  with `Target chains must have valid chain names`; therefore no structural
  accuracy is inferred from the runtime/VRAM tables.
- Raw per-target timing and memory records are under
  `outputs/eval/external_compare_v1_20260812/predictions_per_target/`.
