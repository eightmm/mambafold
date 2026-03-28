"""Feature embeddings for atom and residue tokens."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mambafold.data.constants import BACKBONE_ATOMS, CA_ATOM_ID, MAX_ATOMS_PER_RES, NUM_AA_TYPES, NUM_ATOM_TYPES, NUM_ELEMENT_TYPES, NUM_PAIR_TYPES, ATOM_TYPE_TO_ELEMENT


class CoordinateFourierEmbedder(nn.Module):
    """Fourier positional embedding for 3D coordinates."""

    def __init__(self, d_out: int = 128, num_freqs: int = 16):
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
            coords: [*, 3]

        Returns: [*, d_out]
        """
        # Expand: [*, 3, 1] * [num_freqs] -> [*, 3, num_freqs]
        scaled = coords.unsqueeze(-1) * self.freqs  # [*, 3, F]
        fourier = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1).flatten(-2)
        return self.proj(torch.cat([coords, fourier], dim=-1))


class SequenceFourierEmbedder(nn.Module):
    """Fourier embedding for residue indices with chain-relative normalization."""

    def __init__(self, d_out: int = 64, num_freqs: int = 8):
        super().__init__()
        raw_dim = 2 + 4 * num_freqs
        self.proj = nn.Linear(raw_dim, d_out)
        freqs = 2.0 ** torch.linspace(0, 4, num_freqs)
        self.register_buffer("freqs", freqs)

    def forward(self, seq_nums: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            seq_nums: [B, L] integer residue sequence numbers
            mask: [B, L] bool valid residue mask

        Returns: [B, L, d_out]
        """
        seq = seq_nums.to(self.freqs.dtype)
        valid = mask.to(torch.bool)

        fill = torch.full_like(seq, torch.finfo(seq.dtype).max)
        first = torch.where(valid, seq, fill).amin(dim=1, keepdim=True)
        first = torch.where(valid.any(dim=1, keepdim=True), first, torch.zeros_like(first))

        rel = torch.where(valid, seq - first, torch.zeros_like(seq))
        span = rel.amax(dim=1, keepdim=True).clamp(min=1.0)
        rel_norm = rel / span

        rel_scaled = rel.unsqueeze(-1) * self.freqs
        norm_scaled = rel_norm.unsqueeze(-1) * self.freqs
        fourier = torch.cat(
            [
                torch.sin(rel_scaled), torch.cos(rel_scaled),
                torch.sin(norm_scaled), torch.cos(norm_scaled),
            ],
            dim=-1,
        )
        feat = torch.cat([rel.unsqueeze(-1), rel_norm.unsqueeze(-1), fourier], dim=-1)
        out = self.proj(feat)
        return out * valid.unsqueeze(-1).to(out.dtype)


class ResidueLocalFrameEmbedder(nn.Module):
    """Embed per-residue atom coordinates in a backbone-local frame."""

    def __init__(self, d_out: int = 64, num_freqs: int = 8, eps: float = 1e-6):
        super().__init__()
        self.coord_embed = CoordinateFourierEmbedder(d_out=d_out, num_freqs=num_freqs)
        self.eps = eps

    def forward(self, coords: Tensor, atom_mask: Tensor) -> Tensor:
        local_coords = self._to_local_frame(coords, atom_mask)
        out = self.coord_embed(local_coords)
        return out * atom_mask.unsqueeze(-1).to(out.dtype)

    def _to_local_frame(self, coords: Tensor, atom_mask: Tensor) -> Tensor:
        n = coords[:, :, 0]
        ca = coords[:, :, CA_ATOM_ID]
        c = coords[:, :, 2]

        ca_mask = atom_mask[:, :, CA_ATOM_ID]
        origin = torch.where(ca_mask.unsqueeze(-1), ca, torch.zeros_like(ca))

        x_vec = c - origin
        x_norm = torch.linalg.norm(x_vec, dim=-1, keepdim=True)
        x_axis = x_vec / x_norm.clamp(min=self.eps)

        n_vec = n - origin
        n_proj = (n_vec * x_axis).sum(dim=-1, keepdim=True) * x_axis
        y_vec = n_vec - n_proj
        y_norm = torch.linalg.norm(y_vec, dim=-1, keepdim=True)
        y_axis = y_vec / y_norm.clamp(min=self.eps)

        z_vec = torch.cross(x_axis, y_axis, dim=-1)
        z_norm = torch.linalg.norm(z_vec, dim=-1, keepdim=True)
        z_axis = z_vec / z_norm.clamp(min=self.eps)
        y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=-1), dim=-1, eps=self.eps)

        frame_ok = (
            atom_mask[:, :, 0]
            & ca_mask
            & atom_mask[:, :, 2]
            & (x_norm.squeeze(-1) > self.eps)
            & (y_norm.squeeze(-1) > self.eps)
            & (z_norm.squeeze(-1) > self.eps)
        )
        basis = torch.stack([x_axis, y_axis, z_axis], dim=-1)
        eye = torch.eye(3, device=coords.device, dtype=coords.dtype).view(1, 1, 3, 3)
        basis = torch.where(frame_ok.unsqueeze(-1).unsqueeze(-1), basis, eye)

        rel = coords - origin.unsqueeze(2)
        return torch.einsum("blaj,bljk->blak", rel, basis)


class AtomFeatureEmbedder(nn.Module):
    """Embed atom/residue/pair type and coordinate features into atom tokens.

    Three complementary embeddings summed before projection:
      - res_type_embed:  amino acid identity (broadcast over atoms)
      - atom_type_embed: atomic element/type properties
      - pair_embed:      exact (residue, atom) chemical identity (167 unique pairs)
    """

    def __init__(
        self,
        d_atom: int = 256,
        d_fourier: int = 128,
        d_res_pos: int = 0,
        d_atom_slot: int = 0,
        d_local_frame: int = 0,
    ):
        super().__init__()
        self.res_type_embed = nn.Embedding(NUM_AA_TYPES, d_atom)
        self.atom_type_embed = nn.Embedding(NUM_ATOM_TYPES, d_atom)
        self.pair_embed = nn.Embedding(NUM_PAIR_TYPES, d_atom)
        self.coord_embed = CoordinateFourierEmbedder(d_out=d_fourier)
        self.atom_slot_embed = nn.Embedding(MAX_ATOMS_PER_RES, d_atom_slot) if d_atom_slot > 0 else None
        self.local_frame_embed = ResidueLocalFrameEmbedder(d_out=d_local_frame) if d_local_frame > 0 else None
        # Element class embedding (C/N/O/S) — derived from atom_type_id via buffer
        atom_to_elem = torch.tensor(ATOM_TYPE_TO_ELEMENT, dtype=torch.long)
        self.register_buffer('atom_to_elem', atom_to_elem)
        self.element_embed = nn.Embedding(NUM_ELEMENT_TYPES, d_atom)
        # +1 for observed_mask binary feature (1=observed, 0=missing/padding)
        # +1 for backbone/sidechain flag
        in_dim = d_atom + d_fourier + d_res_pos + d_atom_slot + d_local_frame + 2
        self.proj = nn.Linear(in_dim, d_atom)

    def forward(
        self,
        res_type: Tensor,              # [B, L]
        res_pos_feat: Tensor | None,   # [B, L, d_res_pos]
        atom_type: Tensor,             # [B, L, A]
        pair_type: Tensor,             # [B, L, A]
        coords: Tensor,                # [B, L, A, 3]
        atom_mask: Tensor,             # [B, L, A]
        observed_mask: Tensor | None = None,  # [B, L, A] — 1=observed, 0=missing
    ) -> Tensor:
        """Returns: [B, L, A, d_atom] atom token embeddings"""
        B, L, A = atom_type.shape

        feat = (
            self.res_type_embed(res_type).unsqueeze(2).expand(-1, -1, A, -1)  # [B, L, A, d_atom]
            + self.atom_type_embed(atom_type)                                  # [B, L, A, d_atom]
            + self.pair_embed(pair_type)                                       # [B, L, A, d_atom]
            + self.element_embed(self.atom_to_elem[atom_type])                 # [B, L, A, d_atom]
        )
        parts = [feat, self.coord_embed(coords)]

        if res_pos_feat is not None:
            parts.append(res_pos_feat.unsqueeze(2).expand(-1, -1, A, -1))

        if self.atom_slot_embed is not None:
            slot_ids = torch.arange(A, device=coords.device)
            slot_feat = self.atom_slot_embed(slot_ids).view(1, 1, A, -1).expand(B, L, -1, -1)
            parts.append(slot_feat)

        if self.local_frame_embed is not None:
            parts.append(self.local_frame_embed(coords, atom_mask))

        # observed_mask binary flag: tells model which atoms are experimentally observed
        obs = observed_mask if observed_mask is not None else atom_mask
        parts.append(obs.unsqueeze(-1).to(coords.dtype))  # [B, L, A, 1]

        # backbone/sidechain flag: slots 0-3 are N,CA,C,O (backbone)
        is_backbone = (torch.arange(A, device=coords.device) < len(BACKBONE_ATOMS)).float()
        parts.append(is_backbone.view(1, 1, A, 1).expand(B, L, -1, -1))

        out = self.proj(torch.cat(parts, dim=-1))
        return out * atom_mask.unsqueeze(-1).to(out.dtype)
