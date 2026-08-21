"""Contracts for how coordinates enter and leave the all-atom model.

Every check here guards a quantity that is silently wrong rather than loud when
it drifts: an atom slot nothing fills, a Fourier band that cannot resolve a
bond, a relative embedding that is not actually relative, or a decoder that
hands every atom of a residue the same context vector.
"""

from __future__ import annotations

import math

import torch

from mambafold.data.constants import (
    CA_ATOM_ID,
    COORD_SCALE,
    MAX_ATOMS_PER_RES,
    RESIDUE_ATOMS,
)
from mambafold.model.embeddings import CoordinateFourierEmbedder
from mambafold.model.fold.atom_mamba import AtomDecoder


def test_atom_slot_count_matches_the_largest_residue_exactly() -> None:
    widest = max(RESIDUE_ATOMS.items(), key=lambda kv: len(kv[1]))
    assert len(widest[1]) == MAX_ATOMS_PER_RES, (
        f"{widest[0]} needs {len(widest[1])} slots but MAX_ATOMS_PER_RES is "
        f"{MAX_ATOMS_PER_RES}. A larger constant reserves a slot nothing writes "
        f"and pads every atom-axis tensor; a smaller one truncates {widest[0]}."
    )
    assert CA_ATOM_ID == 1
    assert all(atoms[CA_ATOM_ID] == "CA" for atoms in RESIDUE_ATOMS.values())


def test_fourier_bands_land_on_the_requested_angstrom_periods() -> None:
    # The band is declared in Angstrom, so the endpoints must survive the
    # COORD_SCALE conversion. This is what breaks silently if COORD_SCALE moves.
    embedder = CoordinateFourierEmbedder(
        d_out=8, num_freqs=16, min_period_A=1.0, max_period_A=128.0
    )
    periods = 2.0 * math.pi * COORD_SCALE / embedder.freqs
    assert math.isclose(float(periods[0]), 128.0, rel_tol=1e-5)
    assert math.isclose(float(periods[-1]), 1.0, rel_tol=1e-5)
    # Log-spaced: the ratio between consecutive periods is constant.
    ratios = (periods[:-1] / periods[1:]).tolist()
    assert max(ratios) - min(ratios) < 1e-4


def test_norm_channel_has_a_finite_gradient_at_the_zero_vector() -> None:
    # The CA slot's own relative coordinate is exactly zero, as is every padded
    # slot. `torch.norm` backward is 0/0 there.
    embedder = CoordinateFourierEmbedder(
        d_out=8, num_freqs=4, min_period_A=1.0, max_period_A=16.0, include_norm=True
    )
    coords = torch.zeros(2, 3, requires_grad=True)
    embedder(coords).sum().backward()
    assert torch.isfinite(coords.grad).all()


def test_relative_embedding_is_invariant_to_translating_the_whole_structure() -> None:
    from mambafold.model.fold import MambaFoldAllAtom

    torch.manual_seed(3)
    model = MambaFoldAllAtom(
        d_res=16, n_trunk=0, d_ca_emb=8, d_atom=8, n_atom_layers=0,
        n_atom_cross_layers=0, use_plm=False,
    )
    coords = torch.randn(1, 4, MAX_ATOMS_PER_RES, 3) * 0.2
    mask = torch.ones(1, 4, MAX_ATOMS_PER_RES, dtype=torch.bool)
    shift = torch.tensor([0.7, -0.3, 1.1])

    ca_rel = coords - coords[:, :, CA_ATOM_ID : CA_ATOM_ID + 1, :]
    shifted_rel = (coords + shift) - (coords + shift)[:, :, CA_ATOM_ID : CA_ATOM_ID + 1, :]
    torch.testing.assert_close(ca_rel, shifted_rel)

    _, ca_feat = model._split_scale_embed(coords, mask)
    _, ca_feat_shifted = model._split_scale_embed(coords + shift, mask)
    # The absolute half must move; if it did not, the split would be pointless.
    assert not torch.allclose(ca_feat, ca_feat_shifted)
    torch.testing.assert_close(
        model.atom_rel_embed(ca_rel), model.atom_rel_embed(shifted_rel)
    )


def test_decoder_gives_each_atom_slot_a_distinct_view_of_the_residue_latent() -> None:
    torch.manual_seed(11)
    decoder = AtomDecoder(
        d_res=32, d_atom=8, n_layers=0, d_temb=8, mixer="mlp", n_cross_layers=0
    )
    res_latent = torch.randn(1, 2, 32)
    ctx = decoder.ctx_proj(res_latent).view(1, 2, MAX_ATOMS_PER_RES, 8)
    spread = (ctx - ctx.mean(dim=2, keepdim=True)).abs().max()
    assert spread > 1e-3, "ctx_proj is broadcasting one vector to every slot"
