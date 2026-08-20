# Tests

Focused checks cover the current single-chain all-atom code path.

| Test | Coverage |
| --- | --- |
| `test_mamba3.py` | Bi-Mamba and direct all-atom model shapes |
| `test_geometry_loss.py` | geometry auxiliary losses |
| `test_pair_blocks.py` | pair-block masks and gradients (ablation code) |
| `test_distributed.py` | distributed length synchronization |
| `test_esm.py` | ESM embedding/cache behavior |
| `test_length_sampler.py` | length-balanced sampling |
| `test_sequence_cache.py` | canonical-sequence cache identity |
| `test_build_metadata.py` | deterministic active-split training FASTA path resolution |
| `test_validate_boltz_rcsb.py` | Boltz RCSB validation helpers |
| `test_audit_sequence_overlap.py` | exact benchmark/training overlap report and filtered FASTA/ID outputs |
| `test_score_external_openstructure.py` | active comparator roster, target filters, and model-independent CASP14 references |
| `test_summarize_external_openstructure.py` | complete four-model comparison tables and fail-closed target identity |

Run the focused suite with:

```bash
uv run pytest -q
```

CUDA extension initialization can dominate the first run. The active ESMC-6B
FASTA CLI has parser, feature, and artifact checks; full structure generation
requires a CUDA-capable environment plus separately distributed MambaFold and
ESMC-6B artifacts. The overlap unit test covers the exact gate only; MMseqs2
homology screening remains an external benchmark-admission step.
