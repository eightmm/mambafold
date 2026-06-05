"""Stage 2 — All-atom flow matching conditioned on Stage 1 CA + trunk latent.

Inputs:
    pair_type [B, L, A]          — residue-atom identity (dataset)
    x_t_atom  [B, L, A, 3]       — FM-corrupted atom coords (CA slot pre-filled with s1_ca_cond)
    s1_ca     [B, L, 3]          — Stage 1 final/noisy CA condition
    s1_latent [B, L, d_res]      — Stage 1 trunk latent (rich pair-aware context)
    t         [B, 1, 1, 1]       — independent FM time
    res_type, esm, chain_id, ... — same conditioning as Stage 1

Output:
    v_atom    [B, L, A, 3]       — FM velocity; CA residual is regularized by anchor loss

Param target ≈ 75M.

See `docs/architecture.md` §4 for the design rationale.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from mambafold.data.constants import AA_TO_ID, CA_ATOM_ID, MAX_ATOMS_PER_RES
from mambafold.data.types import ProteinBatch
from mambafold.model.bimamba3 import MambaStack
from mambafold.model.embeddings import AtomFeatureEmbedder
from mambafold.model.fold.conditioning import CAAnchoredFourier, Stage1LatentBroadcast

NUM_RES_TYPES = len(AA_TO_ID)


def _group_atoms_to_residues(atom_tok: Tensor, atom_mask: Tensor) -> Tensor:
    """Masked mean pool atoms → residues. [B, L, A, D] → [B, L, D]."""
    mask_f = atom_mask.unsqueeze(-1).to(atom_tok.dtype)
    return (atom_tok * mask_f).sum(dim=2) / mask_f.sum(dim=2).clamp(min=1)


class _ResidueToAtomBroadcast(nn.Module):
    """Project residue tokens back to atom slots with per-slot learnable gate."""

    def __init__(self, d_res: int, d_atom: int):
        super().__init__()
        self.proj = nn.Linear(d_res, d_atom)
        self.slot_gate = nn.Embedding(MAX_ATOMS_PER_RES, d_atom)
        nn.init.zeros_(self.slot_gate.weight)

    def forward(self, res_tok: Tensor, atom_mask: Tensor) -> Tensor:
        A = atom_mask.shape[2]
        gate = torch.sigmoid(self.slot_gate(torch.arange(A, device=res_tok.device)))
        broadcast = self.proj(res_tok).unsqueeze(2) * gate.unsqueeze(0).unsqueeze(0)
        return broadcast * atom_mask.unsqueeze(-1).to(res_tok.dtype)


class MambaFoldStage2(nn.Module):
    """All-atom FM refinement conditioned on Stage 1 C-alpha.

    The CA slot is initialized from Stage 1, but the velocity head may refine it.
    Training keeps this residual motion bounded with an explicit anchor loss.

    Args:
        d_atom: Atom token dim throughout encoder/decoder. Default 384.
        d_res_polish: Residue dim of the optional mid-trunk polish. Default 512.
        n_atom_enc: Atom encoder BiMamba depth. Default 4.
        n_polish:   Residue polish BiMamba depth. Default 4.
        n_atom_dec: Atom decoder BiMamba depth. Default 4.
        d_s1_res: Stage 1 trunk latent dim (must match Stage 1's d_res).
        d_ca_anchor: CAAnchoredFourier output dim. Default 64.
        d_res_type_atom, d_res_pos, d_atom_slot, d_fourier — passed to
            AtomFeatureEmbedder-style atom features.
        bidirectional: BiMamba (default True) vs causal.
        SSM hyperparams (d_state, mimo_rank, expand, headdim) — defaults
            match the atom encoder/decoder conventions.
    """

    def __init__(
        self,
        d_atom: int = 384,
        d_res_polish: int = 512,
        n_atom_enc: int = 4,
        n_polish: int = 4,
        n_atom_dec: int = 4,
        d_s1_res: int = 1024,
        d_ca_anchor: int = 64,
        d_res_type_atom: int = 32,
        d_res_pos: int = 0,
        d_atom_slot: int = 32,
        d_fourier: int = 128,
        bidirectional: bool = True,
        d_state: int = 64,
        mimo_rank: int = 4,
        expand: int = 2,
        headdim: int = 64,
    ):
        super().__init__()
        self.d_atom = d_atom
        self.d_res_polish = d_res_polish

        # ── Atom embedding ──────────────────────────────────────────────
        # Atom feature stack: pair_type + Fourier(coords) + slot + res_type
        self.atom_embed = AtomFeatureEmbedder(
            d_atom=d_atom,
            d_fourier=d_fourier,
            d_res_pos=d_res_pos,
            d_atom_slot=d_atom_slot,
            d_res_type_atom=d_res_type_atom,
        )

        # ── Conditioning from Stage 1 ──────────────────────────────────
        self.ca_anchor = CAAnchoredFourier(d_out=d_ca_anchor, num_freqs=8)
        self.ca_anchor_proj = nn.Linear(d_ca_anchor, d_atom)
        self.s1_latent_broadcast = Stage1LatentBroadcast(d_res=d_s1_res, d_atom=d_atom)
        self.t_embed = nn.Linear(1, d_atom)
        # Stage 1 scaffold cues: pseudo-Cβ direction + per-residue confidence,
        # broadcast over the atom axis. Zero-init so they start as no-ops.
        self.pcb_proj = nn.Linear(3, d_atom)
        self.conf_proj = nn.Linear(1, d_atom)
        nn.init.zeros_(self.pcb_proj.weight); nn.init.zeros_(self.pcb_proj.bias)
        nn.init.zeros_(self.conf_proj.weight); nn.init.zeros_(self.conf_proj.bias)

        # ── Atom encoder (per-residue BiMamba over the A atom slots) ────
        self.atom_encoder = MambaStack(
            d_atom, n_atom_enc,
            d_state=d_state, mimo_rank=mimo_rank, expand=expand, headdim=headdim,
            bidirectional=bidirectional,
        )

        # ── Residue polish (light Mamba pass at d_res_polish) ───────────
        self.atom_to_res_proj = nn.Linear(d_atom, d_res_polish)
        self.residue_polish = MambaStack(
            d_res_polish, n_polish,
            d_state=d_state, mimo_rank=mimo_rank, expand=expand, headdim=headdim,
            bidirectional=bidirectional,
        )
        self.res_to_atom = _ResidueToAtomBroadcast(d_res_polish, d_atom)

        # ── Atom decoder + velocity head ────────────────────────────────
        self.atom_decoder = MambaStack(
            d_atom, n_atom_dec,
            d_state=d_state, mimo_rank=mimo_rank, expand=expand, headdim=headdim,
            bidirectional=bidirectional,
        )
        self.v_head = nn.Sequential(
            nn.LayerNorm(d_atom),
            nn.Linear(d_atom, 3),
        )

    # ── helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def inject_ca_slot(x_t_atom: Tensor, s1_ca: Tensor) -> Tensor:
        """Initialize the CA slot of x_t_atom from Stage 1's predicted CA.

        Args:
            x_t_atom: [B, L, A, 3]
            s1_ca:    [B, L, 3]
        Returns: new tensor [B, L, A, 3] with CA slot overwritten.
        """
        out = x_t_atom.clone()
        out[..., CA_ATOM_ID, :] = s1_ca
        return out

    # ── forward ─────────────────────────────────────────────────────────

    def forward(
        self,
        batch: ProteinBatch,
        s1_ca: Tensor,
        s1_latent: Tensor,
        s1_pcb: Tensor | None = None,
        s1_conf: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            batch:     ProteinBatch with x_t (CA slot already pre-injected by
                       caller), pair_type, atom_mask, res_type, t, etc.
            s1_ca:     [B, L, 3]      Stage 1's final CA estimate
            s1_latent: [B, L, d_s1_res] Stage 1's trunk latent
            s1_pcb:    [B, L, 3]      Stage 1 pseudo-Cβ direction (optional)
            s1_conf:   [B, L]         Stage 1 per-residue confidence (optional)
        Returns:
            v_atom:    [B, L, A, 3]   FM velocity for atoms
        """
        B, L = batch.res_type.shape
        A = MAX_ATOMS_PER_RES

        # 1. Atom feature embedding (pair_type + Fourier + slot + res_type)
        atom = self.atom_embed(
            batch.pair_type, batch.x_t, batch.atom_mask,
            res_pos_feat=None, res_type=batch.res_type,
        )                                                                  # [B,L,A,d_atom]

        # 2. CA anchor + Stage 1 latent + time features
        anchor = self.ca_anchor(batch.x_t, s1_ca)                          # [B,L,A,d_ca_anchor]
        atom = atom + self.ca_anchor_proj(anchor)

        atom = atom + self.s1_latent_broadcast(s1_latent)                  # broadcast over A

        # Stage 1 scaffold cues (broadcast over the atom axis).
        if s1_pcb is not None:
            atom = atom + self.pcb_proj(s1_pcb).unsqueeze(2)               # [B,L,1,d_atom]
        if s1_conf is not None:
            atom = atom + self.conf_proj(s1_conf.unsqueeze(-1)).unsqueeze(2)

        # batch.t shape [B, 1, 1, 1]. Reduce to [B, 1] for the Linear, then
        # add two singleton axes so the result broadcasts to [B, L, A, d_atom].
        t_scalar = batch.t.view(B, 1)                                       # [B, 1]
        t_feat = self.t_embed(t_scalar)                                     # [B, d_atom]
        atom = atom + t_feat[:, None, None, :]                              # [B, 1, 1, d_atom]

        # 3. Atom encoder per-residue (flatten over BL; the A atom slots are the seq axis)
        atom_flat = atom.reshape(B * L, A, -1)
        am_flat = batch.atom_mask.reshape(B * L, A)
        atom_flat = self.atom_encoder(atom_flat, am_flat)
        atom = atom_flat.reshape(B, L, A, -1)

        # 4. Group → residue polish → ungroup (+residual on atom dim)
        res = _group_atoms_to_residues(atom, batch.atom_mask)              # [B,L,d_atom]
        res = self.atom_to_res_proj(res)                                   # [B,L,d_res_polish]
        res = self.residue_polish(res, batch.res_mask)
        atom = atom + self.res_to_atom(res, batch.atom_mask)

        # 5. Atom decoder
        atom_flat = atom.reshape(B * L, A, -1)
        atom_flat = self.atom_decoder(atom_flat, am_flat)
        atom = atom_flat.reshape(B, L, A, -1)

        # 6. Velocity head, mask padding atoms.
        v_atom = self.v_head(atom)
        return v_atom * batch.atom_mask.unsqueeze(-1).to(v_atom.dtype)
