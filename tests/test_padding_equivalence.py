"""A padded batch must give the same answer as each sequence run alone.

Every masking bug in this model shares one signature: a padded position leaks
into a valid one and the result depends on what the batch happened to contain.
The candidates are real and each is easy to get wrong —

* the Mamba depthwise causal conv (k=4) reads across the padding boundary
  unless the input is zeroed at masked positions before it,
* `_flip_by_mask` needs a contiguous valid prefix or it reverses the wrong
  window,
* the slot-pooling softmax must be -inf-masked, and its masked mean must divide
  by the real occupancy rather than the slot count,
* `BackboneStreamMixer` reshapes [B, L, 5, d] to [B*5, L, d] and the mask has to
  be tiled in the matching axis order,
* the trunk's AdaLN modulation is broadcast, so a wrong `repeat` silently pairs
  a residue with another example's time.

None of them raises. Each produces a loss curve that looks plausible and
plateaus for no visible reason. This test catches all of them at once, and it
needs a GPU because the SSM kernels are CUDA-only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mambafold.data.constants import MAX_ATOMS_PER_RES  # noqa: E402
from mambafold.data.types import ProteinBatch  # noqa: E402
from mambafold.model.fold.all_atom import MambaFoldAllAtom  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Mamba-3 SSM kernels are CUDA-only"
)


def _batch(lengths: list[int], max_len: int, d_plm: int, device) -> ProteinBatch:
    """A batch whose rows have different real lengths, padded to `max_len`."""
    batch_size = len(lengths)
    atoms = MAX_ATOMS_PER_RES
    generator = torch.Generator(device="cpu").manual_seed(0)
    res_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    atom_mask = torch.zeros(batch_size, max_len, atoms, dtype=torch.bool)
    for row, length in enumerate(lengths):
        res_mask[row, :length] = True
        # Ragged occupancy inside each residue, like real side chains.
        for i in range(length):
            atom_mask[row, i, : 4 + (i % 11)] = True
    coords = torch.randn(batch_size, max_len, atoms, 3, generator=generator)
    coords = coords * atom_mask.unsqueeze(-1)
    zeros = torch.zeros(batch_size, max_len, dtype=torch.long)
    return ProteinBatch(
        res_type=torch.randint(0, 20, (batch_size, max_len), generator=generator),
        res_seq_nums=torch.arange(max_len).expand(batch_size, max_len).contiguous(),
        atom_type=torch.randint(0, 30, (batch_size, max_len, atoms), generator=generator),
        pair_type=torch.randint(0, 30, (batch_size, max_len, atoms), generator=generator),
        res_mask=res_mask,
        atom_mask=atom_mask,
        valid_mask=atom_mask,
        ca_mask=res_mask,
        chain_id=zeros,
        entity_id=zeros.clone(),
        sym_id=zeros.clone(),
        is_nterm=torch.zeros(batch_size, max_len, dtype=torch.bool),
        is_cterm=torch.zeros(batch_size, max_len, dtype=torch.bool),
        x_clean=coords,
        x_t=coords.clone(),
        eps=torch.zeros_like(coords),
        t=torch.full((batch_size, 1, 1, 1), 0.5),
        esm=torch.randn(batch_size, max_len, d_plm, generator=generator),
    ).to(device)


def _row(batch: ProteinBatch, index: int, length: int) -> ProteinBatch:
    """Extract one row, cropped to its own length — no padding at all."""
    from dataclasses import fields, replace

    cropped = {}
    for field in fields(batch):
        value = getattr(batch, field.name)
        if not isinstance(value, torch.Tensor):
            continue
        cropped[field.name] = (
            value[index : index + 1]
            if field.name == "t"
            else value[index : index + 1, :length]
        )
    return replace(batch, **cropped)


_D_PLM = 32


def _model() -> MambaFoldAllAtom:
    """A small model that keeps every value the SSM kernels actually see.

    Only `d_res`, `n_trunk` and the PLM widths are shrunk. Everything the
    TileLang kernels specialise on — `expand`, `headdim`, `d_state`,
    `mimo_rank`, `atom_mixer` and the atom-side ranks — is exactly what
    `configs/run_a_mamba.yaml` trains, because a padding-equivalence test that
    exercises a different kernel configuration proves nothing about the model
    that runs.

    The previous fixture got both wrong. It used `headdim=8`, which the Mamba-3
    kernel itself warns is untested ("consider one of the tested headdim_v: 32,
    64, 128") and which makes the GEMM tiling degenerate — m_warp 1 x n_warp 1
    against num_warps 4 — so the test could not run at all on a GPU. And it left
    `atom_mixer` at the constructor default `"mamba"`, so it would have tested an
    atom-axis scan that the config does not train.

    d_atom stays at the config's 128 so the cross-residue mixer lands on the same
    d_inner 256 / nheads 4 as the real model.
    """
    return (
        MambaFoldAllAtom(
            d_res=256,
            n_trunk=2,
            d_res_type=8,
            d_res_pos=8,
            d_plm=_D_PLM,
            d_plm_proj=16,
            d_ca_emb=16,
            d_temb=32,
            use_plm=True,
            d_state=64,
            mimo_rank=4,
            expand=2,
            headdim=64,
            d_atom=128,
            n_atom_layers=2,
            atom_mixer="mlp",
            atom_d_state=64,
            atom_mimo_rank=2,
            n_atom_cross_layers=1,
        )
        .to(torch.device("cuda"))
        .eval()
    )


def test_padding_does_not_change_any_valid_position() -> None:
    device = torch.device("cuda")
    d_plm = _D_PLM
    torch.manual_seed(0)
    model = _model()

    lengths = [37, 61, 12]
    padded = _batch(lengths, max_len=96, d_plm=d_plm, device=device)
    with torch.no_grad():
        together = model(padded)["v_atom"]

    for index, length in enumerate(lengths):
        with torch.no_grad():
            alone = model(_row(padded, index, length))["v_atom"]
        mask = padded.atom_mask[index, :length].unsqueeze(-1)
        a = (together[index, :length] * mask).float()
        b = (alone[0] * mask).float()
        scale = max(1e-6, float(b.abs().max()))
        assert torch.allclose(a, b, atol=2e-3 * scale, rtol=2e-3), (
            f"row {index} (length {length}) changed when batched with padding: "
            f"max |delta| = {float((a - b).abs().max()):.3e}, scale {scale:.3e}"
        )


def test_padded_positions_stay_exactly_zero() -> None:
    device = torch.device("cuda")
    d_plm = _D_PLM
    torch.manual_seed(0)
    model = _model()
    batch = _batch([37, 61, 12], max_len=96, d_plm=d_plm, device=device)
    with torch.no_grad():
        out = model(batch)["v_atom"]
    assert float(out[~batch.atom_mask].abs().max()) == 0.0
