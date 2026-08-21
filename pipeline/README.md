# Data pipeline

Nine ordered stages take this project from an empty `data/` to a frozen,
leakage-gated corpus. The numbering is the dependency order.

```bash
./pipeline/run_all.sh          # everything, resumable
./pipeline/run_all.sh 06 07    # only these stages
sbatch scripts/slurm_build_data.sh
```

Every stage is idempotent. An interrupted run costs the time already spent and
nothing else — the archive download resumes, conversions skip finished records,
and the link farms reconcile to their id lists.

All external inputs are declared in [`../config/paths.env`](../config/paths.env)
rather than hidden in the stages, so the corpus provenance is auditable in one
place.

| Stage | Reads | Writes |
| --- | --- | --- |
| `00_fetch_external.sh` | SimpleFold id list URL, AlphaFold DB v4 archive URL | `data/external/` |
| `01_fetch_rcsb_metadata.py` | RCSB Data API | `data/splits/rcsb_entry_metadata.tsv` |
| `02_build_admission_splits.py` | Boltz manifest | `data/splits/{train,val,admitted_all,heldout_post_cutoff,…}.txt`, `report.json` |
| `03_link_structures.py` | Boltz snapshot, admission lists | `data/rcsb_train/`, `data/rcsb_heldout_post_cutoff/` |
| `04_build_fasta.py` | structure farm | `data/audit/rcsb-{admitted,training}.{fasta,tsv}` |
| `05_link_rcsb_embeddings.py` | admitted FASTA, legacy ESMC-6B cache | `data/rcsb_esmc6b/by_sequence/`, missing-embedding list |
| `06_convert_afdb_v4.py` | v4 archive, SimpleFold id list | `data/afdb_swissprot_v4/{npz,manifest.tsv,sequences.tsv}` |
| `07_link_afdb_embeddings.py` | v4 sequences, legacy ESMC-6B cache | `data/afdb_esmc6b/by_sequence/`, sequence audit |
| `07b_verify_afdb_sequences.py` | every v4 and v6 record | verified cache, mismatch list, links removed |
| `08_leakage_gates.py` | both training FASTAs, benchmark FASTAs | `data/audit/*-exact-overlap.json`, gate summary |
| `09_freeze_manifests.py` | the finished corpus | SHA-256 manifests, link-target manifest |
| `10_homology_gate.py` | both training FASTAs, benchmark FASTAs | MMseqs2 screen, per-hit identity and coverage, admitted counts |
| `11_verify_training_readiness.py` | both corpora | per-record verdict: loadable, chains canonicalise, embeddings present and correctly shaped |
| `12_compute_missing_embeddings.py` | the caches themselves | the embeddings nothing could supply — **GPU, run separately** |

## Checking the result

```bash
pytest -q tests/test_corpus_invariants.py
```

The checks re-derive every admission rule from the Boltz manifest and the files
on disk, and deliberately do not import the pipeline. A stage that mis-applies a
filter still writes a self-consistent `report.json`, so the report is not
evidence for itself; the manifest is. They skip themselves when the corpus has
not been built.

## Notes that matter

**Stage 02 also requires a protein chain.** The reference filters say nothing
about molecule type, so a nucleic-acid-only entry passes all three. The full
Boltz snapshot holds thousands of them; they yield no training example and would
only pad the corpus. `--allow_non_protein` turns the requirement off.

**Stage 02 reproduces three reference filters, not two.** SimpleFold declares
`ResolutionFilter` and `DateFilter` per-dataset in `configs/data/pdb.yaml`, and
`SizeFilter(min_chains=1, max_chains=300)` in the same file's top-level
`filters:` block, which `train_datamodule.py` applies *first* to every dataset.
Omitting the size rule admits several hundred entries the reference discards,
including assemblies with thousands of chains that then dominate a chain-level
sampler. See [`../docs/data_contract.md`](../docs/data_contract.md).

**Stage 02's resolution rule has two modes.** Every record in this Boltz archive
carries `resolution: 0.0`, so the reference `ResolutionFilter` is a no-op on it.
`RESOLUTION_SOURCE=boltz` reproduces that exactly; `rcsb_api` applies the real
resolutions and produces a stricter corpus than SimpleFold trained on. The
report always records what the other mode would have decided.

**Stage 06 runs in parallel.** A completed archive is seekable, so one
sequential pass records where each wanted member's data begins and the workers
seek to it afterwards — rather than each re-reading 39 GB. `--workers` defaults
to the Slurm allocation so a batch job never oversubscribes its cgroup. The
index is cached beside the corpus; the stage skips records it already converted.

**Stages 03, 05 and 07 prune.** The farms and the embedding cache are
reconciled to their inputs, not merely topped up. When the admission rule
changes, a record the new rule rejects leaves the corpus; otherwise the
directory still offers it to a training run.

**Stage 08 is the first of two gates and the weaker one.** It compares whole
strings, so it cannot see a target embedded in a longer training sequence.
Stage 10 runs the MMseqs2 screen at 30% identity over 80% of the query and
excludes strictly more: 8 further CASP14 targets, 3 in CASP15, 3 in CASP16.
One of them, `T1045s1`, is a 154-residue prefix of a 157-residue training
sequence — no exact match, every residue in training. Coverage is measured on
the query only for exactly that reason. No generalization claim stands on
stage 08 alone.

**Stage 07b is not optional.** Stage 07's guards are cheap and blind in one
direction: an embedding stores one row per residue, so the row count catches
every length change and nothing else. A same-length sequence change passes it,
and above the 1024-residue cap the count says nothing at all. On this corpus
338 of 268,977 accessions changed sequence between AFDB v4 and v6, and 103 of
them passed every cheap check and were linked to another protein's embedding.
07b compares all of them and removes those links.

**Stage 12 derives its work from the cache, not from a flag.** Asking which
sequences have no embedding cannot miss a record that one stage rejected and
another cleared — a real case here, which an earlier flag-driven version missed
by exactly one entry. It needs a GPU:

```bash
sbatch scripts/slurm_compute_missing_embeddings.sh
```

`run_all.sh` does not call them, so the CPU pipeline never blocks on a GPU
allocation.

**Stage 11 counts chains, not entries.** Every other report counts entries, and
the training unit is the chain. It also applies the loader's `min_length=20`
before demanding an embedding: several RCSB entries carry single-residue protein
chains that the loader discards, and asking for their embeddings reports a
healthy corpus as broken.

```bash
sbatch scripts/slurm_compute_missing_embeddings.sh
```

`run_all.sh` does not call it, so the CPU pipeline never blocks on a GPU
allocation.
