import numpy as np
import torch

from mambafold.data.collate import ProteinCollator
from mambafold.data.constants import MAX_ATOMS_PER_RES
from mambafold.data.dataset import _gather_esm_rows
from mambafold.data.esm import _map_hf_esmc_key
from mambafold.data.types import ProteinExample


def test_map_hf_esmc_transformer_keys():
    assert _map_hf_esmc_key("esmc.embed.weight") == "embed.weight"
    assert (
        _map_hf_esmc_key("esmc.transformer.blocks.7.attn.layernorm_qkv.layer_norm_weight")
        == "transformer.blocks.7.attn.layernorm_qkv.0.weight"
    )
    assert (
        _map_hf_esmc_key("esmc.transformer.blocks.7.attn.layernorm_qkv.weight")
        == "transformer.blocks.7.attn.layernorm_qkv.1.weight"
    )
    assert (
        _map_hf_esmc_key("esmc.transformer.blocks.7.ffn.fc1_weight")
        == "transformer.blocks.7.ffn.1.weight"
    )
    assert (
        _map_hf_esmc_key("esmc.transformer.blocks.7.ffn.fc2_weight")
        == "transformer.blocks.7.ffn.3.weight"
    )


def test_map_hf_esmc_head_and_extra_state():
    assert _map_hf_esmc_key("lm_head.3.weight") == "sequence_head.3.weight"
    assert _map_hf_esmc_key("esmc.transformer.norm.weight") == "transformer.norm.weight"
    assert _map_hf_esmc_key("esmc.transformer.blocks.0.ffn._extra_state") is None


def test_gather_esm_rows_vectorized_across_chains():
    chain0 = np.arange(5 * 3, dtype=np.float32).reshape(5, 3)
    chain1 = (100 + np.arange(4 * 3, dtype=np.float32)).reshape(4, 3)
    entries = [
        (0, 3, 0, 0, 0),
        (1, 1, 0, 1, 1),
        (2, 4, 0, 0, 0),
        (3, 0, 0, 1, 1),
    ]

    gathered = _gather_esm_rows(entries, {0: chain0, 1: chain1})

    expected = torch.from_numpy(np.stack([chain0[3], chain1[1], chain0[4], chain1[0]]))
    assert gathered is not None
    assert torch.equal(gathered, expected)


def test_gather_esm_rows_contiguous_single_chain_is_zero_copy():
    chain = np.arange(8 * 4, dtype=np.float16).reshape(8, 4)
    entries = [(i, i + 2, i, 0, 0) for i in range(4)]

    gathered = _gather_esm_rows(entries, {0: chain})

    assert gathered is not None
    assert torch.equal(gathered, torch.from_numpy(chain[2:6]))
    assert gathered.data_ptr() == torch.from_numpy(chain[2:6]).data_ptr()


def test_collator_preserves_float16_esm_cache_dtype():
    length = 3
    atoms = MAX_ATOMS_PER_RES
    example = ProteinExample(
        res_type=torch.zeros(length, dtype=torch.long),
        atom_type=torch.zeros(length, atoms, dtype=torch.long),
        pair_type=torch.zeros(length, atoms, dtype=torch.long),
        coords=torch.zeros(length, atoms, 3),
        atom_mask=torch.ones(length, atoms, dtype=torch.bool),
        observed_mask=torch.ones(length, atoms, dtype=torch.bool),
        res_seq_nums=torch.arange(length),
        seq_len=length,
        esm=torch.arange(length * 8, dtype=torch.float16).reshape(length, 8),
    )

    batch = ProteinCollator(augment=False)([example])

    assert batch is not None
    assert batch.esm is not None
    assert batch.esm.dtype == torch.float16
    assert torch.equal(batch.esm[0, :length], example.esm)
