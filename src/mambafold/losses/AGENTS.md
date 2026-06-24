# losses/

Active loss modules:

- `ca_only.py` — C-alpha topology auxiliary helpers.
- `lddt.py` — soft C-alpha and sampled all-atom LDDT helpers.
- `geometry.py` — bond and clash losses.

`train/engine.py` owns the active direct all-atom composite loss.
