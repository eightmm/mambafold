# losses/

`lddt.py` implements the chunked exact all-atom soft lDDT term. It uses every
resolved atom; `pair_chunk_size` bounds workspace without approximating the atom
set.

`train/engine.py` composes this with rigid-aligned all-atom flow matching. No
auxiliary or confidence loss belongs to the folding baseline. See
[`docs/data_contract.md`](../../../docs/data_contract.md).
