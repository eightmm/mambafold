# Active ESMC-6B external benchmark status

Updated: 2026-08-20 (Asia/Seoul)

> **Provisional boundary.** The active MambaFold model is the ESMC-6B
> conditioned track. Step 170k is a prerelease EMA, and the geometry fine-tune
> initialized from it is still training. No value on this page is a final
> checkpoint-selection result.

This page separates recorded engineering measurements from the next admissible
comparison. The active baseline roster is SimpleFold-360M, SimpleFold-3B when
the identical scale-reference contract is available, ESMFold v1, and DPLM-2
Bit 650M. OmegaFold is excluded because its OOM-limited coverage was
incomplete. The frozen ESM3 project is an archive, not an active baseline.

## What should be compared next

| Priority | Dataset | Required gate | Reporting role |
| ---: | --- | --- | --- |
| 1 | CASP16 strict single-chain (21 inputs) | exclude exact matches `T1227s1`, `T1243`, then MMseqs2 homology filter | first coordinate-training-clean comparison on the admitted subset |
| 2 | CASP15 strict single-chain (22 inputs) | exclude exact matches `T1106s2`, `T1120`, then MMseqs2 homology filter | second coordinate-training-clean comparison on the admitted subset |
| development | CASP14 whole-chain (70) | none changes its prior use during development | preview reproduction only |
| diagnostic | CAMEO22, Apo, CoDNaS | report direct overlap; do not relabel as external | reconstruction and conformational diagnostics only |

The exact-overlap tool and full claim rules are in
[`../../benchmarks/BENCHMARK_POLICY.md`](../../benchmarks/BENCHMARK_POLICY.md).
Passing those gates supports only a coordinate-training-clean claim for the
declared RCSB/AFDB corpus. It cannot establish that a sequence was unseen by
ESMC-6B.

## Current step-170k preview record

The only completed 170k reference-scored set is CASP14: 70 targets, seed 0,
500-step SDE, OpenStructure 2.9.1.

| Checkpoint | N | GDT-TS ↑ | GDT-HA ↑ | TM-score ↑ | all-atom lDDT ↑ | backbone lDDT ↑ | RMSD (Å) ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MambaFold-ESMC-6B, step 170k EMA preview | 70 | 0.682 | 0.544 | 0.761 | 0.646 | 0.769 | 6.146 |

CASP14 is not confirmatory: it was used for checkpoint, sampler, and guidance
development; six sequences exactly match the coordinate-training corpus; and
ESMC-6B sequence pretraining postdates the targets.

## Earlier 119.5k screening snapshot

Before the 170k preview, step 119.5k generated all 462 unique sequences across
the six committed FASTA collections. Those results are retained to document
completed work, but they must not be presented as the current checkpoint or a
leakage-controlled comparison. ESMC timing loaded precomputed ESMC-6B
embeddings and therefore measures the folding head, not the end-to-end PLM
pipeline.

| Dataset | Seed-0 coverage | Mean (median) folding-head time, s | Peak allocated VRAM, GiB |
| --- | ---: | ---: | ---: |
| CASP14 | 70/70 | 31.64 (35.00) | 2.39 |
| CASP15 strict single-chain | 22/22 | 30.53 (24.85) | 2.37 |
| CASP16 strict single-chain | 21/21 | 30.14 (34.62) | 2.39 |
| CAMEO22 | 183/183 | 31.57 (34.93) | 2.39 |
| Apo | 90/90 | 30.02 (24.83) | 2.39 |
| CoDNaS | 77/77 | 31.76 (35.26) | 2.39 |

### CASP16 full-set screening

These are the unfiltered 21-target step-119.5k rows. The table is diagnostic;
rerun the step-170k/final checkpoint on the admitted subset before using it in
the primary comparison.

| Model | N | GDT-TS ↑ | GDT-HA ↑ | TM-score ↑ | all-atom lDDT ↑ | backbone lDDT ↑ | RMSD (Å) ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MambaFold-ESMC-6B, step 119.5k | 21 | 0.527 | 0.377 | 0.652 | 0.577 | 0.708 | 11.143 |
| SimpleFold-360M | 21 | 0.530 | 0.406 | 0.638 | 0.616 | 0.700 | 13.387 |
| ESMFold v1 | 21 | 0.622 | 0.500 | 0.706 | 0.668 | 0.755 | 11.133 |
| DPLM-2 Bit 650M | 21 | 0.393 | 0.274 | 0.524 | 0.201* | 0.554 | 17.340 |

### CASP15 full-set screening

These are the unfiltered 22-target step-119.5k rows and have the same
diagnostic-only status.

| Model | N | GDT-TS ↑ | GDT-HA ↑ | TM-score ↑ | all-atom lDDT ↑ | backbone lDDT ↑ | RMSD (Å) ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MambaFold-ESMC-6B, step 119.5k | 22 | 0.602 | 0.463 | 0.679 | 0.646 | 0.772 | 13.407 |
| SimpleFold-360M | 22 | 0.590 | 0.461 | 0.670 | 0.681 | 0.766 | 13.981 |
| ESMFold v1 | 22 | 0.649 | 0.535 | 0.705 | 0.709 | 0.798 | 12.824 |
| DPLM-2 Bit 650M | 22 | 0.441 | 0.323 | 0.546 | 0.236* | 0.607 | 18.348 |

`*` DPLM-2 emits `N`, `CA`, `C`, `O`, and `CB`, not complete side chains.
Its all-atom lDDT is not comparable; backbone lDDT is the appropriate local
quality column.

## High-overlap diagnostics

- CAMEO22: 157/183 target PDB IDs and 145/183 exact sequences occur in the
  coordinate-training corpus.
- Apo: 88/90 target IDs and 88/90 exact RCSB sequences occur in training.
- CoDNaS: 77/77 target IDs and 76/77 exact sequences occur in training.

The step-119.5k CAMEO22 reconstruction score was GDT-TS 0.862, TM-score
0.891, all-atom lDDT 0.811, backbone lDDT 0.904, and RMSD 2.698 Å over 183
targets. These values demonstrate pipeline completion on a high-overlap set;
they are not evidence of external generalization. Apo/CoDNaS five-sample,
two-reference-state structural evaluation was not completed, so no
state-recovery or diversity claim is made.

## Confirmatory path

Freeze the final checkpoint, sampler, seeds, exact/MMseqs2 filter, and scorer
before collecting a prospective post-freeze RCSB/CAMEO window. CASP17 can be
added after official references and results are public, but because MambaFold
was not a live entrant it must be labeled retrospective.

Historical ESM3/OmegaFold runtime and common-subset tables remain preserved in
[`external_common_results.md`](external_common_results.md) as an explicitly
archived engineering snapshot. They are not part of the active comparison.
