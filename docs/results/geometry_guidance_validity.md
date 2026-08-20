# Archived geometry-guidance validity experiment

> **Archive only.** This mixed ESM3/ESMC step-119.5k CASP14 experiment is
> retained as engineering evidence for a guidance prototype. It is not an
> active cross-model comparison, does not validate the step-170k preview or
> the ongoing geometry fine-tune, and must not be used as a final model claim.
> The active track is ESMC-6B only; CASP14 remains a development set.

Snapshot: 2026-08-13 (Asia/Seoul)

## Conclusion

Ground-truth-free late geometry guidance passed the pre-registered validity
gate for the frozen MambaFold-ESM3 artifact and the provisional
MambaFold-ESMC-6B checkpoint snapshot. It substantially corrected local bond
lengths without a material loss of global fold accuracy. It should still be
treated as an opt-in **bond-geometry correction**, not as a complete
stereochemical refinement method: hard-clash counts did not improve, and an
existing C-terminal C-alpha inversion in the ESMC-6B output was not repaired.
CASP14 is used here retrospectively as engineering evidence, not as an
untouched confirmatory evaluation.

## Evaluation contract

- Dataset: 69 CASP14 whole-chain targets. T1061 was excluded because an earlier
  exploratory run on that target selected the guidance scale.
- Checkpoints: frozen MambaFold-ESM3 step 120,000 and provisional
  MambaFold-ESMC-6B step 119,500 EMA weights from the then-active training program.
- Controlled inference: fresh guided-off and guided-on predictions used the
  same runner, H100 device class, precomputed ESM feature directories,
  seed 0, 500-step SDE, logarithmic time grid, tau 0.01, epsilon 0.01, and
  diffusion cutoff 0.99. The two conditions differ only in guidance scale.
- Independent change: geometry-guidance scale 0.1, active from flow time 0.5
  at every remaining step, with bond/CA-angle/CA-clash weights
  1.0/0.25/0.1. No experimental structure is used by the guidance energy.
- Scoring: OpenStructure 2.9.1 plus local bond, hard-clash, and C-alpha
  chirality diagnostics. Reported means are paired target means; confidence
  intervals use 20,000 deterministic paired target-bootstrap resamples.
- Pre-registered PASS gate, required for both models: at least 25% lower mean
  first-shell bond-violation fraction above 0.10 Angstrom, mean GDT-TS change
  at least -0.01, and no more than 10% increases in OpenStructure clash or bad
  angle rates.

## Pre-registered result

| Model | Bond violations, off -> on | Relative change | Bond MAE, off -> on (A) | GDT-TS change [95% CI] | OST clash change | OST bad-angle change | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MambaFold-ESM3, step 120,000 | 0.106% -> 0.041% | -60.8% | 0.01200 -> 0.00490 | +0.00001 [-0.00016, +0.00020] | +0.47% | -3.81% | **PASS** |
| MambaFold-ESMC-6B, step 119,500 | 0.282% -> 0.141% | -50.0% | 0.01195 -> 0.00450 | +0.00017 [-0.00007, +0.00043] | +0.34% | +4.05% | **PASS** |

The bond-MAE reductions were 59.1% for ESM3 and 62.3% for ESMC-6B. Their
paired 95% intervals excluded zero: -0.00709 A
[-0.00743, -0.00677] and -0.00744 A [-0.00766, -0.00724], respectively.
The intervals for GDT-TS included zero and remained well inside the accuracy
guardrail.

## Broader validity diagnostics

Pooled counts below are included to make rare failures visible. They are not
the equal-target means used by the pre-registered decision.

| Model | Condition | Bond violations / 85,310 | OST bad bonds | Local hard clashes | OST clashes | Wrong C-alpha / 16,143 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ESM3 120k | Off | 115 | 22 | 723 | 1,356 | 0 |
| ESM3 120k | On | **36** | **15** | 731 | 1,369 | 0 |
| ESMC-6B 119.5k | Off | 275 | 58 | 1,260 | 2,286 | 1 |
| ESMC-6B 119.5k | On | **107** | **41** | 1,276 | 2,295 | 1 |

- Equal-target local hard-clash rates increased by 0.87% for ESM3 and 2.15%
  for ESMC-6B. The ESM3 paired interval included zero
  [-0.026, +0.096] clashes/1k atoms; the ESMC-6B increase did not
  [+0.050, +0.269]. Pooled hard clashes increased by 8 and 16, respectively.
- The sole ESMC-6B chirality exception is T1027 residue A168, the C-terminal
  residue. It already existed without guidance: normalized signed C-alpha
  volume -0.183 off and -0.132 on. Guidance neither introduced nor repaired
  it. No ESM3 inversion and no degenerate C-alpha centre was observed.
- OpenStructure bad bonds fell from 22 to 15 for ESM3 and from 58 to 41 for
  ESMC-6B. OpenStructure clash counts rose by 13 and 9, respectively, while
  their equal-target paired intervals included zero.
- These issue lists and local thresholds are validity diagnostics, not
  MolProbity clashscores or a force-field energy evaluation.

## Interpretation

The tested guidance differentiates a lightweight energy over the predicted
clean structure and adds its normalized gradient late in SDE sampling. Its
terms constrain first-shell bond lengths, a C-alpha virtual-angle floor, and a
C-alpha-only clash penalty. The experiment shows that this is enough to repair
bond lengths while preserving the learned fold, but it does not directly
enforce all covalent angles, peptide planarity, side-chain chirality, torsions,
or atom-radius-aware nonbonded exclusion. The small hard-clash increase and
failure to repair T1027 are consistent with those missing terms.

The evidence therefore supports using scale 0.1 as an optional bond-cleanup
preset. A default full-validity preset should first add a chirality barrier,
topology-aware all-atom repulsion with bonded exclusions and van der Waals
radii, and explicit angle/peptide-planarity terms, then repeat this same paired
69-target gate.

## Runtime and provenance

- One-target T1058 smoke: ESM3 on A5000 took 38.88 seconds without guidance
  and 42.87 seconds with it (+10.3%); ESMC-6B on H100 took 24.26 and 24.16
  seconds. These are device-specific diagnostics, not a cross-model speed
  comparison.
- Full guided inference on one H100: ESM3 69/69 in 47:02 and ESMC-6B 69/69
  in 1:01:09. Fresh unguided controls later took 27:23 and 27:41. Whole-job
  differences include checkpoint-cache and lazy-kernel state, so they are not
  a controlled estimate of guidance overhead.
- Controlled experiment jobs: fresh unguided inference 53704, paired scoring
  53706, and final aggregation 53707. The earlier 53676/53678/53679 comparison
  against legacy stored baselines was retained as a provisional artifact but
  is not the reported causal comparison.
- Raw summary:
  `outputs/eval/geometry_guidance_validity_v2/summary.json`.
- Generated detailed table:
  `outputs/eval/geometry_guidance_validity_v2/RESULTS.md`.
