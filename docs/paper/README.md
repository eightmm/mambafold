# MambaFold paper package

This is a research draft and its auditable result package. The claim is bounded to a **Mamba folding trunk** that predicts single-chain all-atom structures from frozen ESMC-6B sequence embeddings. ESMC-6B itself uses attention; the full sequence-to-structure pipeline is not attention-free. The SimpleFold-360M comparison is an external baseline, not a controlled architecture ablation.

- `manuscript.md`: submission-oriented draft with methods, results, limitations, and references.
- `reproducibility.md`: fixed checkpoint, sampler, target selection, reference/scorer contract, and commands.
- `data/simplefold_360m_*_admitted.json`: target-level SimpleFold-360M scores recomputed on the exact admitted target IDs, with source-summary and ID-list SHA-256 hashes.
- `data/matched_results.json`: machine-readable summary of all confirmed comparisons, generated after scoring.
- `data/stereo_diagnosis.json`: matched CASP15/16 standard and no-stereocheck lDDT, plus stereochemical violation counts; the no-check scores are diagnostic only.
- `late_geometry_experiment.md` and `data/late_geometry_pilot.json`: selected CASP14 pilot of bounded late-step SDE geometry guidance; full-cohort evaluation is pending.

CASP14 and CAMEO22 were inspected during development and serve as exploratory/diagnostic evidence. CASP15/16 were newly scored after this Run A checkpoint and SDE sampler were fixed, but the research team had seen these target families in an earlier project; this is a post-freeze check, not a fully blind prospective test. The coordinate-training homology screen is stricter than exact matching; it does not audit ESMC-6B pretraining exposure.

Before external submission, verify authorship, affiliations, checkpoint distribution rights, all-atom geometry, and at least one independent rerun of the reported tables. The current result package supports a feasibility/proof-of-concept framing, not an architecture superiority or state-of-the-art claim.
