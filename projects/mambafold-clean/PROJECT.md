# PROJECT.md

## Status

- State: Run A trained; fixed-checkpoint external evaluation completed; paper draft available
- Root: repository root
- Last updated: 2026-09-25

## Project

- Name: MambaFold-clean
- Type: ml
- Goal: single-chain all-atom protein structure generation by direct all-atom
  flow matching, on a corpus that reproduces SimpleFold's data admission so the
  architecture claim and the benchmark claims rest on one contract.
- Core claim: a Mamba folding trunk without attention or a learned pair stack
  can produce useful sequence-conditioned, all-atom protein structures from
  frozen ESMC-6B embeddings. The upstream ESMC-6B encoder uses attention.
  The current external baseline is not a matched training/control arm.
- Why a separate root: the previous ESMC-6B track's corpus and this one make
  contradictory statements about the same benchmarks. Under the old contract
  CASP14 carries six exact training matches and is development-only; here it
  carries none by date and is reportable. A shared namespace would let a number
  be cited against the wrong contract. **No number produced under the previous
  contract may be reported here.**
- Scope: single-chain proteins, standard amino acids, monomer chains extracted
  from every admitted entry. MSA-free, conditioned on pinned sequence-only
  ESMC-6B embeddings.
- Non-goals: multimer/interface prediction, ligands, nucleic acids, metals,
  cofactors, water, non-standard residues/PTMs.

## Data

Full contract and evidence: `docs/data_contract.md`. Build: `pipeline/README.md`.
Reporting rules: `benchmarks/BENCHMARK_POLICY.md`.

- **Sources.** RCSB from the Boltz `rcsb_processed_targets` 2024-12-20 snapshot
  (216,870 records) and AFDB SwissProt **v4** from the frozen EBI archive,
  restricted to SimpleFold's published `swissprot_list.csv`.
- **Admission**, reproducing all three SimpleFold filters in their order:
  `SizeFilter(min_chains=1, max_chains=300)` from the top-level `filters:`
  block, then `ResolutionFilter(5.0)`, then
  `DateFilter(date="2020-05-01", ref="released")`.
- **Resolution is a no-op** on this archive: every Boltz record carries
  `resolution: 0.0`, so the reference filter admits everything.
  `RESOLUTION_SOURCE=boltz` reproduces that; `rcsb_api` is a stricter deviation.
- **AFDB v4, not v6.** The previous track resolved accessions through the live
  API and received v6 coordinates created 2025-08-01, from a generator whose
  cutoff is not established. v4 is AlphaFold 2 output with a 2018-04-30 cutoff,
  which precedes every benchmark, so the distillation channel is clean by
  construction rather than by screening.
- **AFESM is not used.** This corpus is therefore a strict subset of
  SimpleFold's, and any parity result is conservative.
- **PLM.** ESMC-6B, 2560-d, addressed by SHA-256 of the canonical sequence.
  268,639 of 268,977 AFDB accessions share a sequence between v4 and v6 and
  reuse the cached embedding; the 338 that changed were recomputed.
- Nothing is copied: structures are symlinks into the Boltz snapshot,
  embeddings are hard links. External inputs are declared only in
  `config/paths.env`.

### Current counts

| Bucket | Records |
| --- | ---: |
| **admitted (SimpleFold-exact) — the corpus figure** | **161,613** |
| admitted, protein-chain filter on (what the farm holds) | 158,210 |
| held out, released after cutoff (SimpleFold-exact) | 53,618 |
| held out, protein-chain filter on | 52,608 |
| excluded, no protein chain | 4,413 |
| excluded by the size rule | 1,639 |
| AFDB SwissProt v4 | 269,003 |

The two admitted figures differ only by entries with no protein chain, which
produce no training example either way, so the two corpora train identically.
SimpleFold's filters have no molecule-type rule, so **161,613 is the corpus
figure** — it is what its published "~160K" corresponds to. The protein
requirement is this project's addition; the 158,210 farm is what that rule
leaves on disk, and both numbers are always reported together.

Both are pipeline outputs, not arithmetic: stage 02 writes
`protein_filter_cross_report` into `data/splits/report.json`, carrying
`simplefold_exact_admitted: 161613` alongside `non_protein_pre_cutoff: 3403`
and `non_protein_post_cutoff: 1010`. Re-derived independently from the Boltz
manifest, both figures reproduce exactly.

### Chain-level totals

Every entry-level count above is an *entry*. The training unit is a *chain*,
and `extract_monomer_chains` indexes every protein chain of every admitted
entry as its own monomer example. `data/audit/chain_length_distribution.json`:

`11_verify_training_readiness.py` read every record of both sources and found
**nothing blocking**: no missing embedding, no row-count mismatch, no width
mismatch, no unreadable record. Every one of the 765,241 chains resolves a
2560-d embedding with the right number of rows.

| | RCSB | AFDB v4 | Combined |
| --- | ---: | ---: | ---: |
| records | 158,210 | 269,003 | 427,213 |
| trainable records | 156,949 | 268,841 | **425,790** |
| chains (≥20 aa) | 496,400 | 268,841 | **765,241** |
| residues | 139,310,232 | 85,078,216 | **224,388,448** |
| median / p90 / p99 | 238 / 509 / 1071 | 282 / 554 / 1027 | 248 / 521 / 1057 |
| max | 5,037 | 2,551 | 5,037 |

Records yielding nothing are all short, not broken: 1,250 RCSB and 162 AFDB
fall under the 20-residue floor, and 11 RCSB entries carry no protein chain.

RCSB's 496,400 is confirmed twice, by `04_build_fasta.py`'s per-chain TSV and
independently by the readiness stage. The length quantiles come from
`data/audit/chain_length_distribution.json`, whose AFDB side is derived from
`manifest.tsv` and so counts one chain (77 residues) that the readiness stage,
reading the file itself, canonicalises to under the floor. The readiness figures
are the authoritative ones.

A stricter population still — chains that also admit a valid observed-atom crop
— is what the loader actually indexes;
`pipeline/13_prebuild_loader_caches.py` measures it and writes
`data/audit/loader_caches.json`.

## Architecture

- Pure Mamba trunk. **There is no attention and no pair stack anywhere in the
  model, and neither is reachable by config.** `bimamba3.py`'s hybrid attention
  layer, `model/fold/pair_blocks.py` and `model/fold/multiplicative_update.py`
  were deleted, along with `use_pair_stack`, `trunk_attn_layers`,
  `trunk_attn_every`, `n_attn_heads`, `d_pair`, `n_pair_blocks`, `n_pair_heads`,
  `pair_mult_c` and `pair_use_cueq`.

  Deleted rather than switched off, for two reasons. A trunk that *could* be
  given attention by flipping a flag cannot support the claim that a pure SSM
  trunk suffices — the claim has to be a property of the code, not of a default.
  And a disabled O(L²) triangle path still has to be carried, sized, and
  reasoned about at every crop decision. That reasoning stands on its own — it
  is why attention is absent rather than disabled — and does not depend on the
  control arm, which was cancelled (see below).
- Three SSM levels, atom → token → atom: AtomEncoder over each residue's atom
  slots, a pair-free BiMamba residue trunk, an AtomDecoder reading out per-atom
  velocity.
- There is no learned O(L²) pair tensor, including in the output path.
- Loss: rigid-aligned flow matching + exact all-atom lDDT only. Pretraining uses
  α(t)=1; a separately declared fine-tune may use 1+8·ReLU(t−0.5).
  Confidence is trained later against a frozen folding model.

### Shared-memory limits (measured)

The Mamba-3 MIMO backward kernel's dynamic shared memory, at headdim=64,
d_model=1024, bf16, fits exactly:

```
smem = 27,008 + 736 × d_state + 250 × chunk_size   [bytes]
```

Four measurements, zero residual. `chunk_size = base // mimo_rank`, base 32 on
pre-Hopper and 64 on Hopper+, with a kernel minimum of 8.

The `base` above is what `_default_chunk_size` derives from
`get_device_capability() >= (9, 0)`, so **sm_120 counts as Hopper+**. Rank 1 is
a separate case: it returns 64 on every device and takes the SISO kernel, whose
budget is not this formula (the one measurement is `d_state=256` → 114,944 B).

- RTX 6000 Ada, limit 101,376 B: `d_state=64` works at rank 1/2/4; `d_state=128`
  works **only at rank 1**, which takes the lighter SISO kernel; `rank=8` is
  impossible because chunk falls to 4.
- `mimo_rank` matters through the kernel path, not the budget: rank 2→4 halves
  chunk and saves 2,000 B, while `d_state=256` on SISO (114,944 B) is *lighter*
  than `d_state=128` on MIMO (123,216 B).

Applying the formula to the rest of the fleet:

| GPU | optin limit | `d_state=64` | `d_state=128` | `d_state=256` |
| --- | ---: | --- | --- | --- |
| RTX 6000 Ada, sm_89 | 101,376 | r2/r4 ok | **rank 1 only** | none |
| H100, sm_90 | 232,448 | r2/r4/r8 ok | r2/r4/r8 ok | r2 ok, 223,424 B |
| RTX PRO 6000, sm_120 | 101,376 † | r2/r4/r8 ok | none | none |

† predicted from the CUDA per-block maximum for sm_120, not measured; jobs
56821/56822 measure it. It does not change the choice, because the only
partition with more than one GPU is Ada: `6000ada` is two nodes of eight,
`heavy` is one H100 and one RTX PRO 6000. A multi-GPU run is an Ada run.

## Training plan

Deliberately the same shape as SimpleFold, so the comparison means something.

- **No validation.** SimpleFold disables it outright (`limit_val_batches: 0.0`,
  `num_sanity_val_steps: 0`) and its checkpoint callback monitors
  `trainer/global_step` with `mode: max`, which makes "best k" mean "most recent
  k". It trains a fixed number of steps and takes the last EMA. This project
  does the same, and every admitted record goes to training.
- **One fixed crop of 1024. No schedule, no stages.** At 1024, 98.85% of chains
  fit whole and 98.4% of all residues fall inside the crop, so a curriculum
  would be buying the last 1.6%. Run A samples one distinct protein per GPU and
  repeats it for noise multiplicity, so all rows share a length; no length
  bucketing is needed. `length_bin: 64` matches the SSM kernel's own padding.
- **One arm.** The parameter-matched transformer control trunk was cancelled
  on 2026-08-21 as too expensive: at
  `copies 8 x grad_accum 2` a step is two micro-steps, so even a short 20k-step
  paired probe runs past a day of wall clock for the two arms, before queue time.
  What this costs is stated plainly rather than quietly dropped — no controlled
  comparison remains, so a difference against SimpleFold cannot be attributed to
  the trunk mixer as against any of this project's declared deviations from it.
  The released SimpleFold checkpoint is now the only comparator: a public
  baseline, not a controlled ablation, differing in PLM, corpus size and
  training infrastructure.

### Run A — `configs/run_a_mamba.yaml`

| | | why |
| --- | --- | --- |
| `d_res` / `n_trunk` | 1024 / 15 | 370.48M folding-model parameters in the final Run A config |
| `d_atom` / `n_atom_layers` | 128 / 4 | atom encoding and decoding remain outside the residue trunk |
| `d_state` / `mimo_rank` | 64 / 4 | final Run A uses the MIMO state-space kernel |
| `max_length` | 1024 | one fixed upper bound; no crop-schedule interface |
| loss | `w_fm` + `w_lddt_atom` only | no auxiliary loss knobs in the folding trainer |
| validation | none | no validation loader or CLI knobs; one `admitted_all` list |

The step count, optimizer, warmup, clipping, timestep distribution, and
16-way noise multiplicity match the declared SimpleFold reference recipe. The
global number of distinct proteins per step remains equal to the DDP world size.

Measured on one RTX 6000 Ada (47.4 GiB), worst case — crop 1024, every residue
and atom slot valid, real forward, objective, backward and AdamW step
(`benchmarks/probe_train_memory.py`, `data/audit/train_memory/`):

| model | crop | batch | peak | ms/step |
| --- | ---: | ---: | ---: | ---: |
| 1024/16 | 1024 | 4 | 18.80 GiB | 354 |
| 1024/18 | 1024 | 8 | 34.70 GiB | 753 |
| 1024/20, `d_atom` 160 | 1024 | 4 | 22.89 GiB | 444 |

The exact all-atom lDDT term was measured separately on the same GPU with
batch 8 and 512-row cutoff-neighbor chunks. At 2,048 / 4,096 / 8,192 resolved
atoms per protein, forward+backward took 40 / 78 / 154 ms and peaked at
115 / 253 / 528 MiB. The implementation discovers every ground-truth neighbor
within 15 Å but evaluates predicted distances only for those pairs, so it does
not materialize a dense differentiable pair matrix.

## Evaluation

Two gates, both run. Coverage after exclusions: **CASP14 62/70, CASP15 19/22,
CASP16 18/21**. CAMEO22 has 68/183 targets admitted by the same full-chain
gate, but lacks a frozen domain-local gate and is reported as a secondary
diagnostic rather than a clean-generalization set.

The exact gate alone called CASP14 completely clean and it is not: `T1045s1` is
a 154-residue prefix of a 157-residue training sequence, invisible to
whole-string comparison. The MMseqs2 screen at 30% identity over 80% of the
query excludes 14 targets in total, four of them matching RCSB entries admitted
long before the cutoff — the date rule removes exact matches, never homologs.

Sampler settings must be frozen against the post-cutoff pool, never against
CASP14.

## Commands

```bash
./pipeline/run_all.sh                        # build the corpus, stages 00–10
sbatch scripts/slurm_build_data.sh           # the same on a compute node
sbatch scripts/slurm_compute_missing_embeddings.sh   # GPU: fill embedding gaps
pytest -q tests/test_corpus_invariants.py    # check the corpus against the contract
PYTHONPATH=src python scripts/train.py --config configs/run_a_mamba.yaml
```

## Verification

- `tests/test_corpus_invariants.py` re-derives every admission rule from the
  Boltz manifest and the files on disk, and deliberately does not import the
  pipeline: a stage that mis-applies a filter still writes a self-consistent
  `report.json`, so the report is not evidence for itself.
- After any corpus change, rerun that suite and `pipeline/11_verify_training_readiness.py`.
- Before a long run: config diff reviewed, W&B name/tags set, no run writes into
  an existing output directory.

## Decisions taken

1. **Admitted set**: 161,613, the SimpleFold-exact figure, always reported with
   the 158,210 the farm holds.
2. **Final Run A model size**: `d_res=1024`, `n_trunk=15`, `d_atom=128`, `n_atom_layers=4`
   — 370,480,666 folding-model parameters (370.48M) in the config.
3. **Final Run A Mamba settings**: `d_state=64`, `mimo_rank=4` (MIMO), `headdim=64`,
   `expand=2`.
4. **No transformer control arm** — cancelled 2026-08-21 on cost.
5. **Crop**: one fixed 1024. No schedule.

## The geometric fine-tune is load-bearing

It was recorded as a phase that "may or may not happen". The measurements say it
has to. lDDT is the only term in the main run that sees bond-scale geometry — a
1.5 A bond is 0.19-0.39% of the FM target's variance — and its gradient scales
with `(1-t)`, so under `alpha_mode: const` the 31.5% of draws above t = 0.8
carry 9.8% of it. The main run therefore learns global structure and leaves
stereochemistry to the fine-tune, which is SimpleFold's split and is why `const`
stays. Skipping the fine-tune means shipping a model whose bonds were never
directly supervised.

### Its time schedule is `uniform`, paired with `ramp`

Decided 2026-08-28. The main run samples `t` from SimpleFold's convex form with
`t_uniform_weight: 0.10`, which — measured, not assumed — puts only 1.8% of
draws below t = 0.1 and 4.4% of lDDT's gradient mass there. Raising that
constant from SimpleFold's 0.02 was meant to widen the low-t floor and narrows
it instead, because averaging against `U(0,1)` shrinks both tails rather than
fattening the left one.

The fine-tune uses `t_schedule: uniform` rather than a repaired mixture: it is
already implemented, it removes the schedule as a confound, and it needs no new
sampler. Over 4M draws, against the main run's `logit_normal`/`const`:

| | t<0.1 draws | t<0.1 grad mass | t>0.8 draws | t>0.8 grad mass |
|---|---|---|---|---|
| main run, `logit_normal` w=0.10 / `const` | 1.8% | 4.4% | 31.5% | 9.8% |
| fine-tune, `uniform` / `ramp` | 10.0% | 14.3% | 20.0% | 11.8% |

Low-t exposure rises 5.5x and its gradient mass 3.3x. High t loses draws yet
*gains* gradient mass, because `ramp` offsets the `(1-t)` decay — which is why
the two go together. `uniform` with `const` would leave t > 0.8 holding 4.0% of
the gradient, and that band is where bond-scale error is the residual, so
`uniform` alone would trade one starved end for the other.

The first separately declared phase (`scripts/slurm_geo_finetune.sh`) completed
10k steps as job 61311 on 2026-09-02 from the final 300k EMA with a fresh
optimizer: uniform time, ramp weighting, and bond/angle/non-bonded-clash
weights of 1.0. Its final 2k-step means improved bond MAE from 0.0553 to 0.0476
A and angle MAE from 3.388 to 3.076 degrees, but the old clash diagnostic only
moved from 74.3 to 67.7/1k. That v1 clash was not benchmark-aligned: it used a
0.4-A overlap tolerance, excluded graph distance <=3 (therefore removing 1-3
and 1-4 pairs), and used the resolved-reference mask even though inference
emits every canonical atom.

The corrected v2 definition uses OpenStructure's heavy-atom floors: ordinary
pairs use summed element VDW radii minus 1.5 A, while S-S uses its separate
2.03-1.00 = 1.03-A floor. Only direct intra-residue bonds and the consecutive
peptide C(i)-N(i+1) bond are excluded; 1-3 and 1-4 pairs remain active. Clash
masking uses every canonical output atom, not only atoms resolved in the
training reference. A Huber barrier begins 0.1 A outside the hard threshold,
so atoms near the boundary still receive a gradient, and exact coincidences
receive a deterministic finite descent direction. Detached CA-centred bounding
spheres find a conservative residue neighbour list; gradient-checkpointed
256-residue-pair chunks bound the differentiable workspace without a learned
pair matrix. Hard clashes/1k atoms, smooth clashes/1k atoms, and mean overlap
are logged separately from the optimization surrogate. The continuation is
declared by `scripts/slurm_geo_clash_v2.sh` and starts from the v1 EMA.

Baseline defaults for all three geometry terms remain zero, so the 300k Run A
objective and its comparison contract are unchanged. The fine-tune writes to a
new output directory, initializes both train weights and EMA from the Run A EMA,
and refuses to overwrite an existing run.

## Validated on GPU (2026-08-22, jobs 57848 / 57856 / 57859)

Everything here is measured. Nothing in this section is an estimate.

- **Masking.** A padded batch matches each sequence run alone, and padded
  outputs are exactly zero. This rules out the depthwise-conv boundary, the
  `_flip_by_mask` contiguous-prefix assumption, the pooling softmax, the
  `BackboneStreamMixer` axis order and the AdaLN `repeat` broadcast — all of
  which fail silently rather than raising.
- **Memory.** `peak = 4.89 GiB + 2.538 MiB/residue`, eight points at or above
  1024 residues, max residual 0.223 GiB. Depends on the total residue count and
  not on how crop and batch divide it, checked directly. Add ~3 GiB for the EMA
  copy and DDP buckets: `copies 8 x 1024` projects to 28.2 GiB, 59% of a 47.4
  GiB card; 16 copies does not fit.
- **The model learns.** Four real chains, 400 steps: fm -76%, lDDT loss -21%.
- **Distributed.** One rank, two ranks, checkpoint and resume all pass. The
  per-interval packed metric all-reduce does not hang; the step counter
  continues correctly across a resume.
- **The loader is not the bottleneck.** `data_wait` settles at **0.6%** of step
  time after the first batch. This replaces an estimate of ~26x headroom with a
  reading from the training loop itself.
- **Step time, split (A5000, 768 residues, warm kernels, 422 ms/step).**
  backward 203.3 ms (48.2%), forward with its metric syncs 155.3 ms (36.8%),
  grad clip and AdamW 59.5 ms (14.1%), collate 2.9 ms, host-to-device 0.7 ms,
  unaccounted 0.0 ms. The clip-and-optimizer share is worth naming: 59.5 ms on
  370M parameters is about five hours of a 300k-step run spent on nothing but
  the optimizer.
- **A 3.7x step-time discrepancy is closed, and it was measurement.** The same
  workload read 1091 ms/step in the first overfit probe and 422 ms here. The
  difference is TileLang compilation landing inside the timed region: that probe
  has no prewarm stage and the kernel cache was cold. Neither `use_rigid_align`
  nor the metric `.item()` syncs — the two candidates named at the time —
  accounts for it. Time any step only with warm kernels.
- **The rollout path runs.** `scripts/rollout.py` sampled 6 CASP16 targets under
  both ODE and SDE from a smoke checkpoint: EMA weights loaded, sequence-only
  examples built, structures written. `t1208s1.pdb` carries 2,619 ATOM records,
  which is exactly the atom_mask count computed independently for that sequence.
  The geometry report correctly calls the output garbage — bond RMSD 37-56 A and
  0.0% of CA-CA steps within 0.5 A of ideal, from a model trained 40 steps whose
  zero-initialised head leaves the sampler integrating noise. A metric that says
  "bad" for a model that is bad is the property being checked here.
- **The chain index is built and findable.** 498,546 chains (RCSB 229,705 +
  AFDB 268,841) in 2,424 s. The loader resolves both caches to the files
  `pipeline/13` wrote, which is the point that was broken: the stage builds from
  absolute paths and the loader looks up from config-relative ones, and before
  the key was made spelling-independent the two never met. `slurm_train.sh`'s
  guard passes, so training starts without paying the rebuild.
- **CASP embeddings exist**: 113/113 across the three sets (lengths 35-881), in
  `data/casp_esmc6b`, kept apart from the training caches.
- **Prewarm is a first-run cost only.** Four shapes compiled in 900.0s cold and
  22-37s warm: TileLang caches kernels on disk. The 16 shapes of crop 1024 at
  bin 64 should cost about an hour once, and near nothing on every restart.
- **NCCL is broken on gpu2.** `benchmarks/probe_nccl.py`, which imports nothing
  from this project, segfaults for every NCCL variant tried (with `device_id`,
  without, and with P2P and SHM disabled) while gloo passes init, barrier and a
  value-checked all_reduce. Re-run that probe on any node before trusting it.
  Whether the 6000ada nodes are affected decides whether Run A can use NCCL.

## Open decisions

1. `total_steps`, `lr`, `warmup_steps`, and how many GPUs Run A takes.
2. Whether to embed the post-cutoff pool. Not needed under the SimpleFold-style
   plan, which selects nothing; needed if a selection set is ever wanted.

## Not yet done

- The rollout script. This is now the only path to any number at all: with the
  control arm cancelled, an external-benchmark rollout is the sole evidence the
  project produces. `src/mambafold/sampling/` and `src/mambafold/structure_io.py`
  exist and have zero callers because nothing invokes them yet.
- CASP domain-level homology queries: the policy requires them, stage 10 runs
  whole-chain queries only.
- The loader chain index. `pipeline/13_prebuild_loader_caches.py` exists but has
  not completed a run, so no training job has started from a warm cache and
  `data/audit/loader_caches.json` does not exist yet.
- H100 and RTX PRO 6000 shared-memory limits are predicted, not measured.
