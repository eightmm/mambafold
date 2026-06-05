# Tests

Focused checks for the active single-chain model path:

- `test_mamba3.py` — Mamba block and Stage 1 smoke shape checks
- `test_pair_blocks.py` — pair block shape/mask/gradient checks
- `test_geometry_loss.py` — geometry auxiliary losses

Run:

```bash
uv run pytest -q
```
