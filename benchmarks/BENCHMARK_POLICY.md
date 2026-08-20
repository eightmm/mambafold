# Benchmark admission policy

This policy separates stable evaluation inputs from claims about
generalization. It applies to the active MambaFold-ESMC-6B track. ESM3 is an
immutable legacy archive, and OmegaFold is not in the active comparator roster.

## Claim boundary

A target may be called **coordinate-training-clean under this audit** only if:

1. its PDB/chain identity is absent from every coordinate-training source;
2. its canonical amino-acid sequence has no exact match in any source; and
3. it passes the predeclared MMseqs2 homology screen against the union of RCSB
   and AFDB SwissProt training sequences.

For the full-chain gate, exclude a target when MMseqs2 finds at least 30%
sequence identity over at least 80% of the benchmark-query sequence. Do not
require reciprocal coverage of the training sequence: training uses contiguous
random crops, so a target embedded in a longer training chain is relevant.
Also search each official CASP evaluation-unit/domain sequence and exclude its
parent target when a hit reaches 30% identity over 80% of that domain query.
For a future set without official domains, freeze a domain segmentation before
prediction. Record the MMseqs2 version, database hashes, commands, query and
training-sequence coverage, identity, and excluded IDs. Freeze every rule
before looking at model scores; do not tune it to improve a result.

Passing these gates does **not** establish that a target was unseen by
ESMC-6B. ESMC is trained on large sequence collections whose complete
per-sequence membership is not available here. Allowed wording is therefore
"coordinate-training-clean under the declared RCSB/AFDB audit," never
"training-clean," "unseen by ESMC," or "leakage-free."
See the [ESMC-6B model card](https://huggingface.co/biohub/ESMC-6B) for the
published pretraining-data description.

The gate audits the MambaFold folding head only. Comparator training corpora
and cutoffs differ, so an admitted-target score table is a controlled target
and scoring comparison, not proof that every baseline had equal data exposure.

## Required artifacts

For every scored set, preserve:

- the original FASTA and SHA-256;
- canonical training FASTAs and their SHA-256 values;
- the JSON output of `audit_sequence_overlap.py`;
- the exact-match-clean intermediate FASTA and target-ID list;
- the MMseqs2 result table, version, database hashes, and command;
- the final admitted FASTA, excluded target list, reference hashes, checkpoint
  hash, sampler settings, seeds, and scorer version.

First export the exact active coordinate-training sequences. The RCSB command
applies the committed training file list; AFDB is scanned in full because the
active source has `file_list: null`:

```bash
uv run --no-sync python scripts/build_metadata.py \
  --npz_dir data/rcsb_boltz_official_full \
  --file_list data/splits/train_boltz_official_full.txt \
  --out_tsv /path/to/audit/rcsb-training.tsv \
  --out_fasta /path/to/audit/rcsb-training.fasta \
  --fail_on_error

uv run --no-sync python scripts/build_metadata.py \
  --npz_dir data/afdb_swissprot/npz \
  --out_tsv /path/to/audit/afdb-swissprot-training.tsv \
  --out_fasta /path/to/audit/afdb-swissprot-training.fasta \
  --fail_on_error
```

Run the exact gate before MMseqs2:

```bash
PYTHONPATH=. uv run --no-sync python benchmarks/audit_sequence_overlap.py \
  --targets benchmarks/external_testsets/casp16_single_chain_21.fasta \
  --training /path/to/rcsb-training.fasta \
  --training /path/to/afdb-swissprot-training.fasta \
  --out /path/to/casp16-exact-overlap.json \
  --write-exact-clean-fasta /path/to/casp16-exact-clean.fasta \
  --write-exact-clean-ids /path/to/casp16-exact-clean-ids.txt
```

This utility checks exact sequence identity only. Its FASTA and ID list are not
ready for scoring until the independent MMseqs2 gate is complete. After that
gate, score only the final admitted IDs:

```bash
uv run --no-sync python benchmarks/score_external_openstructure.py \
  --dataset casp16 \
  --model mambafold_esmc6b_step170000 \
  --target-ids /path/to/casp16-admitted-ids.txt \
  --out-dir /path/to/scores/casp16/mambafold-esmc6b-step170000
```

Use the same admitted ID file for `simplefold_360m`, `esmfold_v1`, and
`dplm2_bit_650m` so every active model is averaged over exactly the same
targets.

## Dataset roles and known overlap

| Dataset | Current role | Known coordinate-training overlap | Allowed use |
| --- | --- | --- | --- |
| CASP16 strict single-chain (21) | primary scoreable set, first | exact matches `T1227s1`, `T1243`; homology audit pending | score the admitted subset after both gates |
| CASP15 strict single-chain (22) | primary scoreable set, second | exact matches `T1106s2`, `T1120`; homology audit pending | score the admitted subset after both gates |
| CASP14 whole-chain (70) | development/reproduction | exact matches `T1029`, `T1030`, `T1034`, `T1065s2`, `T1082`, `T1092`; used during model/sampler development | retrospective engineering evidence only |
| CAMEO22 (183) | overlap diagnostic | 157/183 target IDs and 145/183 exact sequences occur in coordinate training | do not use for external-generalization claims |
| Apo (90) | conformational diagnostic | 88/90 target IDs and 88/90 exact RCSB sequences occur in training | state-recovery/diversity diagnostics only |
| CoDNaS (77) | conformational diagnostic | 77/77 target IDs and 76/77 exact sequences occur in training | state-recovery/diversity diagnostics only |

Counts describe the current RCSB/AFDB coordinate-training inventory and must
be recomputed if that inventory changes.

## Confirmatory evaluation

The first new candidate is a preregistered RCSB temporal set from initial
releases between 2025-11-01 and 2026-06-30, after the AFDB v6 release used by
the active coordinate source. Admit only A1 biological-assembly protein
monomers of length 40--1,024 with standard residues, at least 90% observed
backbone, and either X-ray resolution at most 3.0 Å or cryo-EM resolution at
most 3.5 Å. Cluster the candidate set internally at 30% identity, then apply
the full-chain and domain-local training-source gates above. Freeze its target
and reference manifest before model scoring.

The strongest prospective path is to freeze the final checkpoint, sampler,
seeds, filter, and scorer before acquiring a new RCSB/CAMEO window. CASP17
strict A1 monomers can provide a second post-freeze benchmark after official
references and assessments become public. MambaFold did not participate as a
live CASP17 entrant, so a later evaluation is retrospective and must not be
called an official CASP submission or officially blind prediction.

External provenance: [CASP17 schedule](https://predictioncenter.org/casp17/index.cgi),
[CAMEO workflow](https://cameo3d.org/help), and
[AFDB release notes](https://www.ebi.ac.uk/pdbe/news/alphafold-database-release-notes).

## Comparator and metric rules

- [SimpleFold-360M](https://github.com/apple/ml-simplefold) is the primary
  size-matched baseline.
- SimpleFold-3B is a scale reference only where the identical target/scoring
  contract is available.
- ESMFold v1 and DPLM-2 Bit 650M are retained as public baselines.
- OmegaFold is excluded because its OOM-limited output is not a full-set
  comparison.
- ESM3 is historical archive material, not an active baseline.
- Report coverage before averages and never define a common subset by a
  baseline's inference success.
- DPLM-2 all-atom lDDT is not comparable because its output omits full
  side-chain atoms; use backbone lDDT for the local-quality comparison.
