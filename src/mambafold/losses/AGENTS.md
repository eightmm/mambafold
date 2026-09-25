# losses/

`lddt.py` implements the chunked exact all-atom soft lDDT term. It uses every
resolved atom; `pair_chunk_size` bounds workspace without approximating the atom
set.

`geometry.py` implements the separately configured geometric fine-tune terms:
ground-truth bond lengths, ground-truth bond angles, and a memory-bounded
OpenStructure-aligned heavy-atom clash penalty. Clash topology excludes only
direct covalent bonds (1-2), so 1-3 and 1-4 pairs remain active, and its mask is
the canonical emitted atom set rather than the resolved-reference subset. These
weights remain zero in baseline pretraining.

`train/engine.py` composes these with rigid-aligned all-atom flow matching. No
confidence loss belongs to the folding objective. See
[`docs/data_contract.md`](../../../docs/data_contract.md).
