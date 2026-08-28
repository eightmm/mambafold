# Data contract

Status: in progress, 2026-08-20 (Asia/Seoul)

This is the contract `mambafold-clean` is built to, and the evidence behind each
rule. It does not govern the previous ESMC-6B track, which keeps its own
contract in the original repository.

## Why the corpus was rebuilt

Two independent defects in the previous corpus, either of which is enough to
sink a comparison against SimpleFold.

**AlphaFold DB version drift.** SimpleFold's published list names v4 ids
(`AF-Q87AD2-F1-model_v4`) and its config points at `swissprot_pdb_v4_boltz`. The
previous download script resolved each accession through the live AlphaFold API,
which serves the *current* version, and the old manifest records what actually
arrived: source `AF-Q189A6-F1-model_v4`, delivered `AF-Q189A6-F1-model_v6`,
created `2025-08-01`. Those are v6 coordinates from a generator whose training
cutoff is not established here, so the distillation stream could not be declared
clean with respect to CASP15/16 or any post-2022 benchmark.

Sequence-level homology screening does not repair that. A generator that saw a
structure transfers fold knowledge below any identity threshold worth applying.
AFDB v4 is AlphaFold 2 output with a 2018-04-30 training cutoff, which precedes
every benchmark in the roster, so switching to v4 removes the channel by
construction instead of by screening.

**Reproducibility.** Because the old path asked for "current", rerunning it
later produced a different corpus. The EBI versioned archive is frozen.

Choosing v4 settles leakage, reproducibility, and SimpleFold parity at once.

## What SimpleFold actually does

Read from the reference implementation at commit `c7a5570`.

`configs/data/pdb.yaml` declares **three** filters for `rcsb_protein`, not two.
Lines 13–18 are the per-dataset list:

```yaml
filters:
  - _target_: ...filter.dynamic.resolution.ResolutionFilter
    resolution: 5.0
  - _target_: ...filter.dynamic.date.DateFilter
    date: "2020-05-01"
    ref: released
```

and lines 20–23 declare a top-level list that
`src/simplefold/datasets/train_datamodule.py:268-273` applies to **every**
dataset *before* the per-dataset filters:

```yaml
filters:
  - _target_: ...filter.dynamic.size.SizeFilter
    min_chains: 1
    max_chains: 300
```

Semantics, from the filter implementations:

- `size.py:44-46` — `num_chains <= max_chains and num_valid >= min_chains`,
  where `num_valid` counts chains with `valid` true, of any molecule type.
- `resolution.py:42` — `structure.resolution <= 5.0`, with no missing-value
  branch.
- `date.py` — uses `released`, falls back to `deposited` when empty, rejects a
  record with neither.

Nothing else gates the RCSB stream. `pdb_sp.yaml:22-31` gives SwissProt no date
filter at all, only the published `record_list`; `pdb_sp_afesm.yaml:33-40` gives
AFESM no filter. The whole `filter/dynamic/` package is `date`, `max_residues`,
`resolution`, `size`, and `subset` — there is no identity or cluster filter
anywhere in the repository, and the test-set configs are plain target
directories with no gate. `train_datamodule.py:300` sets
`val_records = train_records[-16:]` under the comment
`# this is a dummy validation set, as we disable validation in training`.

So SimpleFold's entire leakage control is the date cutoff on experimental PDB.
The policy in [`../benchmarks/BENCHMARK_POLICY.md`](../benchmarks/BENCHMARK_POLICY.md)
is stricter than that, and remains in force here.

## The resolution rule is a no-op on this archive

Every record in the Boltz snapshot carries `resolution: 0.0` — all 216,870 of
them, across X-ray, EM, and NMR alike. `ResolutionFilter` therefore admits the
entire RCSB corpus in the reference pipeline. It expresses an intent the
preprocessed data cannot support.

That leaves a choice, and the pipeline makes it explicit rather than silent:

- `RESOLUTION_SOURCE=boltz` (default) reproduces the reference behaviour, so the
  corpus matches the one SimpleFold trained on.
- `RESOLUTION_SOURCE=rcsb_api` applies the real resolutions fetched in stage 01
  and yields a stricter corpus than SimpleFold trained on — a deliberate
  deviation, not a match.

The report always records how many entries the other mode would have decided,
so the difference is never invisible.

## The contract

| Component | Rule | Deviation from SimpleFold |
| --- | --- | --- |
| RCSB source | Boltz `rcsb_processed_targets`, 2024-12-20 snapshot, 216,870 records | none |
| RCSB size | `num_chains <= 300` and at least one valid chain | none |
| RCSB molecule type | at least one protein chain (`mol_type == 0`) | ours; SimpleFold's dataset is `rcsb_protein` but its filters do not say so |
| RCSB date | released ≤ 2020-05-01, deposit fallback, no-date rejected | none |
| RCSB resolution | ≤ 5.0 Å from the Boltz field, which is a no-op | none in effect |
| RCSB chains | `extract_monomer_chains` at load time; `single_chain_only` off | ours; SimpleFold's example is a whole entry, never split per chain |
| Homomer copies | `dedup_homomer_chains`: one chain per distinct sequence per entry, keeping the best-resolved copy | ours; reduces a deviation rather than adding one — see below |
| Observed-atom gate | none (`min_obs_ratio: 0.0`) | none — SimpleFold has no such filter either |
| AFDB | SwissProt v4 from the frozen EBI archive, restricted to the published list (269,003 ids) | none |
| AFESM | not used | SimpleFold-360M pretrains with it; this corpus is a strict subset |
| PLM | ESMC-6B, 2560-d, pinned revision | SimpleFold uses ESM2-3B |
| Trunk | pure Mamba; attention and the pair stack are deleted from `src/`, not disabled | SimpleFold is all-attention |
| Loss | rigid-aligned flow matching + exact all-atom lDDT; pretraining uses α(t)=1, the 1+8·ReLU(t−0.5) ramp belongs to a separately declared fine-tune | no auxiliary loss surface in the folding trainer |
| Crop size | one fixed 1024, no schedule | SimpleFold's `pdb.yaml` sets `max_tokens: 256` |
| Crop shape | contiguous sequence window | **SimpleFold crops spatially** — `BoltzCropper` anchors on a resolved token and expands by distance, so its crop can be sequence-discontiguous and can span chains |
| Validation | none; one `admitted_all` list, fixed steps, last EMA | none — SimpleFold also disables it |
| Time sampling | `0.90·logit_normal(m=0.8, s=1.7) + 0.10·U(0,1)`, bounded by `t_eps=1e-4` | **the uniform weight is 0.10, SimpleFold's is 0.02** — same functional form and same `m`, `s`, `t_eps` (`simplefold.py:389`), only the mixing constant differs. The intent was to widen low-`t` coverage; measured over 4M draws it does the opposite, because the form is a convex combination and averaging against `U(0,1)` pulls draws toward 0.5: `P(t<0.1)` falls from 3.44% at 0.02 to 1.81% at 0.10. Run A ships with 0.10 and the geometric fine-tune switches to `t_schedule: uniform` instead |
| lDDT time weight | α(t)=1 in pretraining | none — SimpleFold's `lddt_weight_schedule` is `False` in `configs/experiment/train.yaml`, "set to True in finetuning phase" |
| Coordinate scale | divide by 16.0 | none — `processor.scale: 16.0`. `ref_scale: 5.0` applies to reference conformers, which this model has no input for |
| Centering | centroid over every canonical atom slot | none — SimpleFold centres with `atom_pad_mask`, not the resolved mask |
| Augmentation | random SO(3) rotation + N(0,1) translation, per copy | none — `center_random_augmentation(s_trans=1.0)` |
| Interpolant | `x_t = t·x_1 + (1−t)·x_0`, target `x_1 − x_0`, noise `randn_like(coords)` unmasked outside padding | none — `LinearPath` with `alpha_t = t`, `sigma_t = 1−t` |
| Noise multiplicity | 16 independent augment/noise copies per sampled protein | none — matches SimpleFold's `processor.multiplicity: 16` |
| Optimizer | AdamW, lr 1e-4, weight_decay 0.0 | none — `configs/model/simplefold.yaml` |
| LR schedule | linear warmup 1e-6 → 1e-4 over 5,000 steps, then held flat | none — `utils/lr_scheduler.py::LinearWarmup` has no decay phase |
| Grad clip | 2.0 | none — `clip_grad_norm_val: 2.0` |
| EMA | 0.999 | none |
| Steps / warmup | 300,000 / 5,000 | matched to `configs/experiment/train.yaml`; the *token* throughput is not matched — see below |

Because the chain rule differs, admitted counts will not equal SimpleFold's. The
rules are matched; the counts are reported, not forced.

### Implementation details and remaining deviations

**The observed-atom gate was removed.** It had no SimpleFold counterpart:
`configs/data/pdb.yaml` declares exactly `SizeFilter`, `ResolutionFilter` and
`DateFilter`; the available filter classes are date, max_residues, resolution,
size, subset, ligand and polymer, and none is an observation-ratio filter.
`tokenize/boltz_protein.py` records `is_present` as `resolved_mask` metadata and
drops nothing on it, and `crop/boltz.py` picks its anchor among resolved tokens
but never rejects a crop for its resolved fraction. At 0.5 it dropped 18,683 of
496,400 RCSB chains (3.8%) and 687 entries outright — `4cau`, a 300-chain entry,
yielded nothing at all.

The collator follows SimpleFold here: centering and corruption use the canonical
`atom_mask`, while FM alignment, FM MSE, and lDDT supervision use
`valid_mask = atom_mask & observed_mask`. Thus unresolved canonical slots see
the same interpolant/noise input distribution as slots that will be generated at
inference, but never contribute a target loss. A resolved-mask input feature is
deliberately absent because it would expose training-only information.

**Chain-level examples multiply SimpleFold's sampling weight.** One SimpleFold
example is one entry, drawn uniformly, with no clustering. One of ours is one
chain, so a 300-chain homomer is drawn 300 times where SimpleFold draws it once.

**Homomer dedup is the correction for that, not an extra deviation.** 269,053 of
503,669 admitted RCSB chains (53.4%) repeat a sequence already present in their
own entry, and 42.0% of entries contain such a repeat. The FM loss rigid-aligns
its target, so copies related by a rigid transform are the same target. Keeping
one copy per distinct sequence per entry — the copy resolving the most atoms —
restores an entry-level measure while still exposing every distinct chain.
Cross-entry repeats of a protein are untouched, exactly as in SimpleFold.

**Noise multiplicity is matched.** SimpleFold's processor sets
`multiplicity: 16`, so each sampled protein enters the step with sixteen
independent rotations, translations, timesteps, and noise draws. This project
uses `copies_per_protein: 16` for the same per-protein variance reduction. The
global number of distinct proteins per optimizer step is the DDP world size in
both recipes; no fixed GPU count is assumed here. Feasibility at the worst-case
1024-residue length remains a preflight memory gate, not a reason to silently
change the objective or multiplicity.

The crop-shape row matters more than the crop-size row. A spatial crop yields a
sequence-discontiguous token set, which breaks the locality a state-space scan
relies on; a contiguous window is what a Mamba trunk can use. It is also close to
moot in practice here: 98.85% of chains fit inside 1024 whole, so cropping fires
on 1.15% of examples at all.

## Consequence for evaluation

The date rule works on the experimental half. Against the admitted RCSB corpus,
exact sequence overlap with CASP14, CASP15, and CASP16 is zero, where the
previous track's corpus matched six, two, and two targets.

The distillation half is a different matter, and only the union gate sees it.
AFDB SwissProt carries no date filter — SimpleFold does not filter it either —
and it covers UniProt, so a natural protein that later became a CASP target sits
in the corpus verbatim. Against RCSB ∪ AFDB v4, over 731,564 training sequences:

| Benchmark | Targets | Exact overlap | Excluded |
| --- | ---: | ---: | --- |
| CASP14 whole-chain | 70 | 0 | — |
| CASP15 strict single-chain | 22 | 1 | `T1106s2` |
| CASP16 strict single-chain | 21 | 2 | `T1227s1`, `T1243` |

`T1106s2` matches `AF-P0C2N2` and `AF-P61417`; `T1227s1` matches `AF-P0AD04`;
`T1243` matches `AF-P0CH91` and `AF-Q7U1K0`. A gate run against the experimental
half alone would have called every one of them clean.

The MMseqs2 screen then excludes strictly more — 30% identity over 80% of the
query, against all 731,564 training sequences:

| Benchmark | Targets | Exact | Homology | Scoreable |
| --- | ---: | ---: | ---: | ---: |
| CASP14 whole-chain | 70 | 0 | 8 | **62** |
| CASP15 strict single-chain | 22 | 1 | 3 | **19** |
| CASP16 strict single-chain | 21 | 2 | 3 | **18** |

The exact gate called CASP14 completely clean and it is not. `T1045s1` is a
154-residue prefix of a 157-residue training sequence: whole-string comparison
finds no match while every residue of the target is in training verbatim. That
embedded-target case is precisely why coverage is measured on the query only.
`T1100` and `T1133` sit at 92.5% and 92.8% identity — near-identical, and equally
invisible to an exact check.

Four exclusions match RCSB rather than AFDB. The release cutoff removes exact
matches, never homologs: `2b5e` and `5t91` were admitted years before
2020-05-01 and remain 40% and 39% homologs of CASP targets. Reportable coverage
is 62/70, 19/22, and 18/21; the full result with MMseqs2 version, input hashes,
and command is in `data/audit/homology_gate.json`.

The same three targets were flagged under the previous contract; what changed is
the attribution. There they were mixed in with RCSB matches, and the CASP14 six
and CASP15's `T1120` were RCSB-side. The date cut removed those, leaving only
the distillation channel — which the AFDB half of the gate now names precisely.

CASP14 returns as a comparable benchmark on one condition: sampler settings are
frozen against the post-cutoff pool rather than against CASP14. The previous
geometry terms and guidance are not carried in this project.

`data/splits/heldout_post_cutoff.txt` — every entry released after the cutoff —
is the confirmatory selection set.

## Corpus materialisation

Nothing large is copied. Both roots are on `/dev/sda1`, so structure records are
symlinks into the frozen Boltz snapshot and embeddings are hard links into the
previously computed ESMC-6B cache. A spot check on the first build confirmed a
linked embedding and its source sharing inode `65273698012`, with `df` reporting
54T used before and after.

The ESMC-6B cache is addressed by the SHA-256 of the canonical sequence, so the
v6-era embeddings are reusable for the v4 records wherever the sequence is
unchanged.

That premise — an AlphaFold DB version changes the prediction, not the UniProt
sequence — is mostly true and measurably not always true. Comparing all 268,977
accessions found **338 whose sequence changed between v4 and v6**, a rate of
0.126%. UniProt revises canonical sequences between AlphaFold DB releases, and
the accession is not a safe join key on its own.

Stage 07 joins on the accession without opening a structure record, and guards
each link by comparing the cached array's row count against the v4 sequence
length — an embedding carries one row per residue up to the cap. That guard is
cheap and catches every length change, but it is blind in two ways: a
same-length substitution passes it, and at or above the 1024-residue cap every
sequence yields exactly 1024 rows so it says nothing at all. Stage 07 therefore
compares the 2,779 capped-length accessions against their v6 records directly,
and samples 3,000 more at random.

The sample fired: 7 of 3,000 disagreed. Stage 07b then compared every accession,
which is the only way to bound the problem rather than estimate it:

| Outcome | Records |
| --- | ---: |
| sequence match | 268,639 |
| sequence mismatch | 338 |
| — caught by the row-count guard, never linked | 235 |
| — same length, linked, then removed by 07b | **103** |
| absent from the v6 corpus | 26 |
| needing a fresh ESMC-6B pass | 364 |

The 103 are the point. They passed every cheap check, were linked, and each one
paired a v4 record with the embedding of a *different* protein. Nothing
downstream would have complained: the loader resolves embeddings by sequence
hash and would have trained on them silently. Stage 07b removes those links and
routes the accession to recomputation instead.

An earlier approach — re-deriving the whole mapping by reading all 268,977 v6
records serially — was abandoned after 224,311 links when it measured roughly
6 MB/min under concurrent load. The same work runs in six minutes across 16
workers, which is why 07b verifies exhaustively rather than by sample.

## History

The first build of this corpus used the RCSB Data API for dates and resolutions
and omitted the size rule. An adversarial audit of that build raised 24 findings
and confirmed 11. Two changed the corpus:

- The missing `SizeFilter` had admitted 830 entries the reference discards — 545
  with more than 300 chains, the largest at 3,720, and 285 whose chains are all
  invalid. Because the sampler is chain-level, those 830 entries contributed
  about 22% of all admitted chains, heavily oversampling a few hundred viral
  capsid sequences.
- The RCSB API returned no record for 122 obsoleted accessions, and its real
  resolutions excluded 2,490 entries that the reference pipeline admits.

Reading the Boltz manifest directly fixed all of it: it carries the exact fields
the reference filters on, covers all 216,870 records with no lookup failures,
and needs no network. Rebuilding against it newly excluded exactly 830 entries,
all by the size rule — independently matching the audit's count.

The audit also found that `np.savez` appends `.npz` to any path lacking it, so
the v4 converter's `.npz.tmp` temporary was written as `.npz.tmp.npz` and the
subsequent rename raised `FileNotFoundError` into a broad `except`. Every
conversion would have failed while reporting itself as 269,003 individual
failures. Caught before the archive finished downloading.

## Recorded results

Built by the pipeline on 2026-08-20 over the full 216,870-record Boltz snapshot.

| Bucket | Entries |
| --- | ---: |
| admitted | 158,210 |
| admitted, train | 152,466 |
| admitted, val | 5,744 |
| held out, released after cutoff | 52,608 |
| excluded, no protein chain | 4,413 |
| excluded by the size rule | 1,639 |
| — more than 300 chains | 1,167 |
| — no valid chain | 472 |
| excluded by resolution | 0 |
| excluded, no usable date | 0 |
| absent from the Boltz manifest | 0 |

158,210 + 52,608 + 4,413 + 1,639 = 216,870: the four buckets are mutually
disjoint and account for every record. The manifest resolved all of them, unlike
the RCSB Data API which failed on 124 obsoleted accessions. No record needed the
deposit-date fallback. The resolution rule excluded nothing, as expected on an
archive whose resolution field is uniformly 0.0; under
`RESOLUTION_SOURCE=rcsb_api` a further 2,337 entries would have been excluded and
12,823 carry no API resolution at all.

The admitted corpus holds 513,147 protein chains, of which 503,669 have a
non-empty canonical sequence — the remaining 9,478 are chains whose residues are
all non-standard or `UNK`, which the training loader skips for the same reason.
Those 503,669 chains reduce to 110,920 distinct sequences, and every one of them
was already present in the previous track's ESMC-6B cache, so the corpus needed
no new embedding computed. The training split alone holds 471,363 chains and
462,561 sequences.

### The protein requirement earns its place

Widening the universe from the previous track's 211,742-entry symlink farm to
the full snapshot adds 3,879 entries — but 3,878 of them have no protein chain
at all. They are nucleic-acid structures such as `101d`, admitted because the
size, date, and resolution rules say nothing about molecule type. They produce
no training example, contribute nothing to the FASTA, and would only pad the
corpus counts and the link farm.

Exactly one genuinely new protein entry came from the wider universe. The
protein requirement therefore recovers what the previous farm had implicitly
been doing, while keeping the admission rule explicit and auditable.

## Open items

- MMseqs2 30% identity / 80% query coverage screen. Never run. No
  generalization claim stands without it.
- Stage 05b, the GPU embedding pass, turned out not to be needed: every
  admitted sequence was already cached. It stays in the pipeline as a safety
  net for a future corpus change and has never been run.
- Training configs. `configs/` is deliberately empty; nothing may be copied in
  from the previous track without being re-derived against this contract.
- (Closed 2026-08-21.) The transformer control arm's matching basis is moot:
  the arm was cancelled, so nothing needs matching.
- The previous track's `geoft50k` run, still training against the superseded
  contract and holding a GPU this project will need.
