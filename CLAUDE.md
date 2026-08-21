# oh-my-setting Loader

Read `PROJECT.md` first. Project work starts only after it is filled and confirmed.
Then follow `/home/snu_sjm0/.oh-my-setting/templates/project-ml-AGENTS.md` for shared `ml` rules.
Project rules override global defaults.

## Local Agent Rules

- New/broad work or draft `PROJECT.md`: interview -> update `PROJECT.md` -> confirm -> code.
- New/major docs: interview -> concrete outline -> confirm -> write.
- Keep edits task-scoped; do not rewrite unrelated files.
- End with: changed, verified, not verified, next command.

## This root specifically

- The data contract is `docs/data_contract.md`. It is confirmed. Changing an
  admission rule changes the corpus and every number derived from it, so treat
  it as a contract change: update the document, rerun the pipeline, rerun
  `tests/test_corpus_invariants.py`.
- Reporting rules are `benchmarks/BENCHMARK_POLICY.md`. Coverage after both
  leakage gates is CASP14 62/70, CASP15 19/22, CASP16 18/21. Never report a
  number from the previous ESMC-6B track here, and never mix the two corpora.
- External inputs live only in `config/paths.env`. Do not hardcode a path
  anywhere else, and never commit that file — `config/paths.env.example` is the
  tracked template.
- `run_all.sh` is read incrementally by bash while it runs. Edit it with a
  temp-file-and-rename, never in place, or a running job resumes at a stale byte
  offset. Python stages are launched as fresh processes and are safe to edit.
