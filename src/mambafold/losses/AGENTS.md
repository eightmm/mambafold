# losses/

Active loss modules:

- `ca_only.py` — Stage 1 C-alpha FM/lDDT/bond/distogram helpers.
- `lddt.py` — soft C-alpha lDDT and per-residue lDDT helpers.
- `geometry.py` — Stage 2 bond and clash losses.

`train/engine.py` owns the composite Stage 1, Stage 2, and joint losses.
Do not add standalone loss modules unless they are used by the active engine or tests.
