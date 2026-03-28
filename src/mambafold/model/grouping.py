"""Atom-residue grouping and ungrouping operations."""

import torch
import torch.nn as nn
from torch import Tensor

from mambafold.data.constants import MAX_ATOMS_PER_RES


def group_atoms_to_residues(atom_tok: Tensor, atom_mask: Tensor) -> Tensor:
    """Masked average pooling: atoms → residue tokens.

    Args:
        atom_tok: [B, L, A, D] atom embeddings
        atom_mask: [B, L, A] bool

    Returns: [B, L, D] residue tokens
    """
    mask_f = atom_mask.unsqueeze(-1).to(atom_tok.dtype)  # [B, L, A, 1]
    pooled = (atom_tok * mask_f).sum(dim=2)              # [B, L, D]
    count = mask_f.sum(dim=2).clamp(min=1)               # [B, L, 1]
    return pooled / count


class ResidueToAtomBroadcast(nn.Module):
    """Project residue tokens and broadcast to atom positions."""

    def __init__(self, d_res: int, d_atom: int):
        super().__init__()
        self.proj = nn.Linear(d_res, d_atom)
        self.slot_gate = nn.Embedding(MAX_ATOMS_PER_RES, d_atom)  # per-slot multiplicative gate
        nn.init.zeros_(self.slot_gate.weight)

    def forward(self, res_tok: Tensor, atom_mask: Tensor) -> Tensor:
        """
        Args:
            res_tok: [B, L, d_res]
            atom_mask: [B, L, A] bool

        Returns: [B, L, A, d_atom]
        """
        projected = self.proj(res_tok)                              # [B, L, d_atom]
        A = atom_mask.shape[2]
        gate = torch.sigmoid(self.slot_gate(torch.arange(A, device=res_tok.device)))  # [A, d_atom]
        broadcast = projected.unsqueeze(2) * gate.unsqueeze(0).unsqueeze(0)           # [B, L, A, d_atom]
        return broadcast * atom_mask.unsqueeze(-1).to(projected.dtype)
