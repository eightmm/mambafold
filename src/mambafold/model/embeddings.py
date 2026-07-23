"""Feature embeddings for atom and residue tokens."""

import torch
import torch.nn as nn
from torch import Tensor

from mambafold.data.constants import AA_TO_ID, MAX_ATOMS_PER_RES, NUM_PAIR_TYPES

NUM_RES_TYPES = len(AA_TO_ID)  # 21 (20 AAs + UNK)


class CoordinateFourierEmbedder(nn.Module):
    """Fourier positional embedding for 3D coordinates."""

    def __init__(self, d_out: int = 128, num_freqs: int = 16):
        """
        Args:
            d_out (int): Output embedding dimension. Default: 128.
            num_freqs (int): Number of Fourier frequency bands. The raw input
                dimension is 3 + 3 * 2 * num_freqs (3 raw coords + sin/cos per
                coord per frequency). Default: 16.
        """
        super().__init__()
        self.num_freqs = num_freqs
        # 3 coords × (sin + cos) × num_freqs + 3 raw
        raw_dim = 3 + 3 * 2 * num_freqs
        self.proj = nn.Linear(raw_dim, d_out)

        # Fixed frequency bands: 2^-3 to 2^4 = [0.125, 16]
        # Coords are centered ~[-5, 5] in normalized units (COORD_SCALE=10).
        # Max phase = 5 * 16 = 80, safe for bfloat16.
        freqs = 2.0 ** torch.linspace(-3, 4, num_freqs)
        self.register_buffer("freqs", freqs)  # [num_freqs]

    def forward(self, coords: Tensor) -> Tensor:
        """
        Args:
            coords (Tensor): 3D coordinates of shape [*, 3].

        Returns:
            Tensor: Fourier-encoded coordinate embedding of shape [*, d_out].
                Internally expands coords to [*, 3, num_freqs], computes sin/cos,
                concatenates with raw coords, then projects to d_out.
        """
        # Expand: [*, 3, 1] * [num_freqs] -> [*, 3, num_freqs]
        scaled = coords.unsqueeze(-1) * self.freqs  # [*, 3, F]
        fourier = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1).flatten(-2)
        return self.proj(torch.cat([coords, fourier], dim=-1))


class SequenceFourierEmbedder(nn.Module):
    """AF3 / Boltz / Protenix-style residue-position embedding (1D variant).

    Per-residue features:
      * per-chain absolute residue index (`rel`, chain-reset; comes pre-computed
        in `res_seq_nums`) — raw + Fourier sin/cos
      * categorical chain_id Embedding   — distinguishes chains
      * categorical entity_id Embedding  — homomer grouping (same sequence)
      * categorical sym_id Embedding     — copy number within an entity

    Chain-length normalization (`rel/span`, formerly `rel_norm`) is REMOVED —
    no current SOTA folding model (AF3, Boltz-1/2, Protenix) uses it; it
    introduces implicit dependency on chain length and crop window.
    """

    MAX_CHAINS = 64
    MAX_ENTITIES = 64
    MAX_SYM = 32

    def __init__(
        self,
        d_out: int = 64,
        num_freqs: int = 8,
        d_chain: int = 16,
        d_entity: int = 16,
        d_sym: int = 8,
    ):
        super().__init__()
        self.chain_embed = nn.Embedding(self.MAX_CHAINS, d_chain)
        self.entity_embed = nn.Embedding(self.MAX_ENTITIES, d_entity)
        self.sym_embed = nn.Embedding(self.MAX_SYM, d_sym)
        # raw rel (1) + sin/cos rel × num_freqs (2F) + categorical embeds
        in_dim = 1 + 2 * num_freqs + d_chain + d_entity + d_sym
        self.proj = nn.Linear(in_dim, d_out)
        freqs = 2.0 ** torch.linspace(0, 4, num_freqs)
        self.register_buffer("freqs", freqs)

    def forward(
        self,
        seq_nums: Tensor,
        mask: Tensor,
        chain_id: Tensor | None = None,
        entity_id: Tensor | None = None,
        sym_id: Tensor | None = None,
    ) -> Tensor:
        valid = mask.to(torch.bool)
        rel = seq_nums.to(self.freqs.dtype)  # already per-chain 0-base from dataset

        if chain_id is None:
            chain_id = torch.zeros_like(seq_nums, dtype=torch.long)
        if entity_id is None:
            entity_id = torch.zeros_like(seq_nums, dtype=torch.long)
        if sym_id is None:
            sym_id = torch.zeros_like(seq_nums, dtype=torch.long)

        rel_scaled = rel.unsqueeze(-1) * self.freqs
        fourier = torch.cat([torch.sin(rel_scaled), torch.cos(rel_scaled)], dim=-1)

        cid = chain_id.clamp(max=self.MAX_CHAINS - 1)
        eid = entity_id.clamp(max=self.MAX_ENTITIES - 1)
        sid = sym_id.clamp(max=self.MAX_SYM - 1)
        ch_e = self.chain_embed(cid)
        en_e = self.entity_embed(eid)
        sy_e = self.sym_embed(sid)

        feat = torch.cat([rel.unsqueeze(-1), fourier, ch_e, en_e, sy_e], dim=-1)
        out = self.proj(feat)
        return out * valid.unsqueeze(-1).to(out.dtype)


class AtomFeatureEmbedder(nn.Module):
    """Embed atom features into atom tokens.

    Chemical identity: pair_embed (residue×atom joint, 167 unique pairs)
    Geometry: coord_embed (Fourier)
    Auxiliary: optional res_pos, optional atom_slot
    """

    def __init__(
        self,
        d_atom: int = 256,
        d_fourier: int = 128,
        d_res_pos: int = 0,
        d_atom_slot: int = 0,
        d_res_type_atom: int = 32,
    ):
        """
        Args:
            d_atom (int): Atom token embedding dimension. Used for pair_embed
                output and the final projection output [B, L, A, d_atom].
                Default: 256.
            d_fourier (int): Output dimension of the coordinate Fourier embedder
                [B, L, A, d_fourier]. Default: 128.
            d_res_pos (int): Dimension of an optional residue-level position
                feature broadcast to each atom [B, L, d_res_pos → B, L, A, d_res_pos].
                Set to 0 to disable. Default: 0.
            d_atom_slot (int): Dimension of per-slot (intra-residue atom index)
                learnable embedding [A, d_atom_slot]. Set to 0 to disable.
                Default: 0.
            d_res_type_atom (int): Dimension of a residue-type embedding
                broadcast to every atom slot [B, L, d_rt → B, L, A, d_rt].
                Gives the atom encoder explicit context about which AA the
                residue is. Set to 0 to disable. Default: 32.
        """
        super().__init__()
        self.pair_embed = nn.Embedding(NUM_PAIR_TYPES, d_atom)
        self.coord_embed = CoordinateFourierEmbedder(d_out=d_fourier)
        self.atom_slot_embed = (
            nn.Embedding(MAX_ATOMS_PER_RES, d_atom_slot) if d_atom_slot > 0 else None
        )
        self.res_type_embed = (
            nn.Embedding(NUM_RES_TYPES, d_res_type_atom) if d_res_type_atom > 0 else None
        )
        in_dim = d_atom + d_fourier + d_res_pos + d_atom_slot + d_res_type_atom
        self.proj = nn.Linear(in_dim, d_atom)

    def forward(
        self,
        pair_type: Tensor,  # [B, L, A]
        coords: Tensor,  # [B, L, A, 3]
        atom_mask: Tensor,  # [B, L, A]
        res_pos_feat: Tensor | None = None,  # [B, L, d_res_pos]
        res_type: Tensor | None = None,  # [B, L]
    ) -> Tensor:
        """
        Args:
            pair_type (Tensor): Residue-atom pair type indices of shape [B, L, A].
                Integer values in [0, NUM_PAIR_TYPES).
            coords (Tensor): Atom 3D coordinates of shape [B, L, A, 3].
            atom_mask (Tensor): Boolean or float atom validity mask of shape [B, L, A].
                Padding atoms (mask == 0) are zeroed in the output.
            res_pos_feat (Tensor | None): Optional residue-level position features
                of shape [B, L, d_res_pos], broadcast to each atom slot.
                Pass None when d_res_pos == 0.

        Returns:
            Tensor: Atom token embeddings of shape [B, L, A, d_atom].
                Padding atom positions are zeroed via atom_mask.
        """
        B, L, A = pair_type.shape

        parts = [self.pair_embed(pair_type), self.coord_embed(coords)]

        if res_pos_feat is not None:
            parts.append(res_pos_feat.unsqueeze(2).expand(-1, -1, A, -1))

        if self.atom_slot_embed is not None:
            slot_ids = torch.arange(A, device=coords.device)
            parts.append(self.atom_slot_embed(slot_ids).view(1, 1, A, -1).expand(B, L, -1, -1))

        if self.res_type_embed is not None and res_type is not None:
            rt = self.res_type_embed(res_type)  # [B, L, d_res_type_atom]
            parts.append(rt.unsqueeze(2).expand(-1, -1, A, -1))

        out = self.proj(torch.cat(parts, dim=-1))
        return out * atom_mask.unsqueeze(-1).to(out.dtype)
