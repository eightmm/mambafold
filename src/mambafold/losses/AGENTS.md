# losses/

Active loss modules:

- `ca_only.py` — C-alpha topology auxiliary helpers.
- `lddt.py` — soft C-alpha and chunked exact all-atom LDDT helpers.
- `geometry.py` — bond and clash losses.

- `stereochemistry.py` — chirality and local stereochemistry terms.

`train/engine.py` owns the composite loss. In this project the baseline run
weights only flow matching and all-atom lDDT; `geometry.py` and
`stereochemistry.py` are zero-weight there and belong to a separate geometry
fine-tune. See [`docs/data_contract.md`](../../../docs/data_contract.md).
