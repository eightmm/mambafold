"""Atom-residue grouping and ungrouping operations."""

import torch.nn as nn
from torch import Tensor


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

    def forward(self, res_tok: Tensor, atom_mask: Tensor) -> Tensor:
        """
        Args:
            res_tok: [B, L, d_res]
            atom_mask: [B, L, A] bool

        Returns: [B, L, A, d_atom]
        """
        projected = self.proj(res_tok)                              # [B, L, d_atom]
        A = atom_mask.shape[2]
        broadcast = projected.unsqueeze(2).expand(-1, -1, A, -1)   # [B, L, A, d_atom]
        return broadcast * atom_mask.unsqueeze(-1).to(projected.dtype)
