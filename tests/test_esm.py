import numpy as np
import pytest
import torch

from mambafold.data.collate import ProteinCollator
from mambafold.data.constants import MAX_ATOMS_PER_RES
from mambafold.data.dataset import _gather_esm_rows
from mambafold.data.esm import (
    ESMC_6B_MODEL_NAME,
    ESMC_6B_REPO_ID,
    ESMEmbedder,
    _map_hf_esmc_key,
)
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


def test_esm_embedder_defaults_to_esmc_6b():
    embedder = ESMEmbedder(device="cpu")

    assert embedder.model_name == ESMC_6B_MODEL_NAME


@pytest.mark.parametrize(
    "model_name",
    [ESMC_6B_MODEL_NAME, ESMC_6B_REPO_ID, "esmc-600m", "esm3-open"],
)
def test_esm_embedder_accepts_explicit_supported_model_families(model_name):
    embedder = ESMEmbedder(model_name, device="cpu")

    assert embedder.model_name == model_name


@pytest.mark.parametrize("model_name", ["", "esm", "some-hf-model"])
def test_esm_embedder_rejects_ambiguous_model_names(model_name):
    with pytest.raises(ValueError, match="explicit 'esmc\\*' or 'esm3\\*'"):
        ESMEmbedder(model_name, device="cpu")


def test_esm_embedder_dispatches_esmc_and_explicit_esm3_legacy_branches():
    calls = []

    class FakeModel:
        family = ""

        @classmethod
        def from_pretrained(cls, name):
            calls.append((cls.family, name))
            return cls()

        def to(self, _device):
            return self

        def eval(self):
            return self

        def parameters(self):
            return []

    class FakeESM3(FakeModel):
        family = "esm3"

    class FakeESMC(FakeModel):
        family = "esmc"

    api = (FakeESM3, FakeESMC, object, object)
    esmc = ESMEmbedder("esmc-test", device="cpu")
    esmc._api = api
    esm3 = ESMEmbedder("esm3-open", device="cpu")
    esm3._api = api

    assert isinstance(esmc._get_client(), FakeESMC)
    assert isinstance(esm3._get_client(), FakeESM3)
    assert calls == [("esmc", "esmc-test"), ("esm3", "esm3-open")]


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
