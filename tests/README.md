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
| `test_validate_boltz_rcsb.py` | Boltz RCSB validation helpers |

Run the focused suite with:

```bash
uv run pytest -q
```

CUDA extension initialization can dominate the first run. The frozen ESM3
FASTA CLI also has a lightweight parser/feature smoke check in its release
verification workflow; full structure generation requires a CUDA-capable ESM3
environment and the separately distributed checkpoint.
