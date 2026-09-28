# MambaFold: Sequence-Conditioned All-Atom Protein Folding with a Mamba Trunk

**Draft status:** research manuscript, 25 September 2026. Authors and affiliations to be supplied by the research team. Numerical results below refer to the fixed Run A final EMA and one seed; they should not be read as a claim of state-of-the-art accuracy or architecture superiority.

## Abstract

Protein structure prediction commonly uses attention or explicit pair representations in its folding module. We study whether a state-space sequence mixer can support single-chain, sequence-conditioned, all-atom structure generation. MambaFold combines frozen ESMC-6B sequence embeddings with a 370.48M-parameter folding model whose residue trunk and cross-residue atom streams use Mamba-3 blocks, without attention or a learned pair tensor in the folding model. We train with rigid-aligned flow matching and all-atom lDDT on experimental structures admitted by a 1 May 2020 release cutoff and AlphaFold DB SwissProt v4 structures. A fixed 500-step SDE sampler yields mean TM-scores of **0.641** on 19 admitted CASP15 targets and **0.611** on 18 admitted CASP16 targets, versus **0.631** and **0.587** for SimpleFold-360M on the identical targets and references. MambaFold's all-atom lDDT is lower (**0.516/0.474** versus **0.659/0.579**), while backbone lDDT is similar. These results demonstrate a workable Mamba folding trunk for global structure prediction, with a clear remaining atom-level accuracy gap. The external comparison is not a controlled mixer ablation: the systems differ in pretrained encoder, training corpus, and optimization.

## 1. Introduction

Recent generative folding systems have shown that a general-purpose transformer can produce protein structures without a learned pair stack. [SimpleFold](https://arxiv.org/abs/2509.18480) makes this case with flow matching and a scaled transformer folding model. [Mamba-3](https://arxiv.org/abs/2603.15569) offers a different sequence mixer based on selective state-space dynamics. This motivates a narrower empirical question: can a folding network based on that mixer produce useful sequence-conditioned, all-atom structures under a generative training objective?

Mamba is already used in protein modeling. [LC-PLM](https://arxiv.org/abs/2411.08909) develops a Mamba-based protein language model, and [PI-Mamba](https://arxiv.org/abs/2603.26705) generates protein backbones from length and geometry constraints. Our task differs: predict all standard heavy atoms from a specified amino-acid sequence. We therefore describe this work as an implementation and evaluation of a Mamba **folding trunk**, without a priority claim for Mamba in protein modeling.

The result is intentionally limited. Frozen ESMC-6B embeddings bring information from a separate attention-based sequence model. The experiments show that the downstream folding module can operate without attention or pair tensors; they do not show that an entirely attention-free sequence-to-structure pipeline is sufficient.

## 2. Methods

### 2.1 Data and coordinate exposure

The experimental stream starts from the Boltz RCSB processed-target snapshot and applies the reference SimpleFold-style size, resolution, and release-date filters. The declared corpus figure is **161,613 admitted RCSB entries**; **158,210** contain a protein chain and enter the local farm. Monomer chains are extracted as individual training examples. The distillation stream contains **269,003** AlphaFold DB SwissProt v4 records from a frozen accession list. AFESM, used by SimpleFold, is absent. Standard amino-acid single chains of at least 20 residues are the modeled units. Detailed source and admission rules are in [`../../docs/data_contract.md`](../../docs/data_contract.md).

We audit benchmark targets against the union of both structure streams. Exact sequence checks are followed by MMseqs2 at ≥30% identity over ≥80% of the full benchmark query. This excludes 8/70 CASP14, 3/22 CASP15, and 3/21 CASP16 targets; the corresponding admitted counts are 62, 19, and 18. CAMEO22 has 68/183 full-chain-admitted targets, without an official domain-level gate. These filters address coordinate-training similarity under this corpus contract. We cannot establish whether the frozen ESMC-6B encoder encountered any benchmark sequence in its own pretraining.

### 2.2 Folding model

The folding model reads amino-acid identity, frozen 2,560-dimensional ESMC-6B residue embeddings, current atom coordinates, and flow time. An atom encoder maps 14 standard heavy-atom slots per residue to residue features. A bidirectional Mamba-3 residue trunk (width 1,024; 15 blocks) mixes information along the chain. Mamba-based cross-residue backbone streams carry information between N, CA, C, O, and CB channels. An atom decoder returns a velocity for each atom slot. The folding network has no attention block, no learned pair representation, and no triangle update. The checked-in config reports **370,480,666 parameters** for the folding model, excluding ESMC-6B.

Training uses rigid-aligned flow matching plus exact all-atom lDDT. The Run A model was trained for 300,000 optimizer steps with a 1,024-residue crop ceiling, one protein per GPU, 16 noise/augmentation copies accumulated over two microsteps, and EMA decay 0.999. The final EMA is the reported checkpoint. There is no validation-based checkpoint selection. These choices and deviations from SimpleFold, including ESMC-6B rather than ESM2-3B and a different crop/optimizer schedule, are declared in [`../../configs/run_a_mamba.yaml`](../../configs/run_a_mamba.yaml) and [`../../docs/data_contract.md`](../../docs/data_contract.md).

### 2.3 Inference and evaluation

We sample one prediction per target with 500-step SDE (tau 0.01, seed 0) from the fixed final EMA. The same sampler settings produced the earlier CASP14/CAMEO22 Run A results and were frozen before the new CASP15/16 scores. OpenStructure 2.9.1 computes TM-score, all-atom lDDT, and backbone lDDT using the same command and staged reference coordinates for both systems. CASP15 official domain/EU scores are averaged within target by mapped-residue count, then targets are averaged equally. CASP16 uses official whole-chain references. SimpleFold-360M is reaggregated from complete seed-0 target-level results on precisely the same predeclared target lists, without selecting for baseline success. Reference-file SHA-256 hashes agree for every admitted comparison pair. Full commands, input hashes, and machine-readable target rows are in [`reproducibility.md`](reproducibility.md) and [`data/matched_results.json`](data/matched_results.json).

## 3. Results

### 3.1 Same-target external comparison

| Benchmark and role | Admitted / input | MambaFold TM ↑ | SimpleFold-360M TM ↑ | MambaFold all-atom lDDT ↑ | SimpleFold-360M all-atom lDDT ↑ | MambaFold backbone lDDT ↑ | SimpleFold-360M backbone lDDT ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CASP15, post-freeze check | 19 / 22 | **0.641** | 0.631 | 0.516 | **0.659** | 0.739 | 0.745 |
| CASP16, post-freeze check | 18 / 21 | **0.611** | 0.587 | 0.474 | **0.579** | **0.676** | 0.668 |
| CASP14, development-exposed | 62 / 70 | 0.615 | **0.657** | 0.452 | **0.619** | 0.647 | **0.698** |
| CAMEO22, full-chain diagnostic | 68 / 183 | 0.706 | **0.709** | 0.577 | **0.693** | 0.760 | **0.773** |

Values are unweighted target means. SimpleFold-360M is a size-near public baseline, not a matched training/control arm. The MambaFold model excludes AFESM, while the SimpleFold training recipe includes it. CASP14 was used during model-development inspection. The older `casp14_common62` file was excluded because six of its target IDs differ from this work's admitted 62; the CASP14 baseline above was recomputed from the complete 70-target source.

The paired CASP15 TM-score difference is +0.010 (MambaFold minus SimpleFold; 10,000-target-bootstrap percentile interval −0.026 to +0.041). The CASP16 difference is +0.024 (−0.024 to +0.080). These intervals include zero, so the observed TM-score advantages are small and uncertain. All-atom lDDT differences are −0.143 on CASP15 (−0.183 to −0.105) and −0.106 on CASP16 (−0.167 to −0.042). Backbone lDDT differences are −0.006 and +0.008, respectively. The combination suggests that global and backbone placement is much more competitive than atom-level local accuracy. The stereochemistry diagnostic below identifies a major source of the atom-level score gap.

### 3.2 Stereochemistry diagnostic

We rescored the identical admitted predictions and references with OpenStructure 2.9.1 using `--lddt-no-stereochecks`. This is a diagnostic score, not the benchmark result or a physically valid correction to the structures.

| Dataset | MambaFold standard / no-check lDDT | SimpleFold standard / no-check lDDT | MambaFold / SimpleFold mean reported clashes per target |
| --- | ---: | ---: | ---: |
| CASP15 (19 targets, 28 reference pairs) | 0.516 / 0.670 | 0.659 / 0.673 | 349.8 / 10.1 |
| CASP16 (18 targets, 18 reference pairs) | 0.474 / 0.608 | 0.579 / 0.595 | 157.7 / 2.7 |

The change in the between-model gap under this scoring switch is 0.140 on CASP15, versus a standard-score gap of 0.143, and 0.119 on CASP16, versus a standard-score gap of 0.106. Clashes are substantially more frequent in the MambaFold structures. The diagnostic supports stereochemical violations as the dominant source of the *measured* all-atom lDDT deficit on these cohorts. It does not show that a geometry repair will recover the same amount: moving atoms can change local distances and the fold. Reference parity, aggregation, local raw-score paths, and compact target-level results are recorded in [`reproducibility.md`](reproducibility.md) and [`data/stereo_diagnosis.json`](data/stereo_diagnosis.json).

### 3.3 Scope of the demonstration

All 22 CASP15 and 21 CASP16 inputs produced a prediction and a valid official-reference score. On the predeclared coordinate-homology-admitted subsets, the 500-step Mamba folding model reaches mean TM-score above 0.60 on both sets. CASP14 and CAMEO22 provide wider context but do not serve as untouched prospective tests. The measured outputs support feasibility of a Mamba sequence mixer in this folding architecture. They do not establish which component causes the score difference, nor do they establish accuracy parity in all-atom detail.

## 4. Limitations and next experiments

The strongest limitation is the lack of a matched transformer trunk trained on the same embeddings, data, and objective. SimpleFold differs in PLM, corpus size/composition, cropping, and optimization, so no mixer-specific causal conclusion follows. A second limit is the frozen ESMC-6B encoder: the pipeline uses attention upstream and its sequence-pretraining exposure is unaudited. Third, CASP14/CAMEO22 were seen during method development, and earlier projects exposed the research team to CASP15/16 target families; the latter are post-freeze checks of this checkpoint rather than fully blind prospective validation. Fourth, only one Run A checkpoint and one sampling seed are reported. Finally, the all-atom lDDT deficit is substantial and a practical obstacle for applications needing accurate side chains, contacts, or steric detail.

The immediate accuracy experiment is reference-free geometry refinement of the predicted structures, tuned on development targets and then frozen before testing on CASP15/16 with the standard OpenStructure lDDT. Its acceptance criteria are fewer clashes, higher standard all-atom lDDT, and no material loss in TM-score or backbone lDDT. A previous geometry fine-tune reduced mean reported clashes from 175.1 to 134.1 on CASP14 and from 123.4 to 68.9 on CAMEO22, yet improved all-atom lDDT by only 0.006 and 0.011 while slightly lowering TM-score; further training with the same settings is not yet justified. The next decisive architecture experiment is a parameter- and compute-matched transformer trunk with all other inputs fixed. A separate, genuinely prospective protein set with a declared structure-release cutoff should test generalization. Multiple seeds, per-residue geometry and clash analysis, and end-to-end cost accounting (including ESMC-6B) should accompany any stronger efficiency or biological-utility claim. Confidence calibration is a separate analysis and is not used to support the present folding conclusion.

## References

1. Wang et al. [SimpleFold: Folding Proteins is Simpler than You Think](https://arxiv.org/abs/2509.18480), 2025; [official code and released benchmark predictions](https://github.com/apple/ml-simplefold).
2. Lahoti et al. [Mamba-3: Improved Sequence Modeling using State Space Principles](https://arxiv.org/abs/2603.15569), 2026.
3. Wu and Zhu. [PI-Mamba: Linear-Time Protein Backbone Generation via Spectrally Initialized Flow Matching](https://arxiv.org/abs/2603.26705), 2026.
4. [LC-PLM: Long-context Protein Language Model](https://arxiv.org/abs/2411.08909), 2024.
5. [ESM Cambrian technical introduction](https://www.evolutionaryscale.ai/blog/esm-cambrian), 2024.
6. [OpenStructure 2.9.1 documentation](https://openstructure.org/docs/2.9.1/).
