"""Atom decoder: Mamba-3 stack + gradient prediction head."""

import torch.nn as nn
from torch import Tensor

from mambafold.model.ssm.bimamba3 import MambaStack


class AtomDecoder(nn.Module):
    """Per-residue Mamba-3 decoder + gradient head."""

    def __init__(
        self,
        d_atom: int = 256,
        n_layers: int = 4,
        d_state: int = 32,
        mimo_rank: int = 2,
        expand: int = 2,
        headdim: int = 64,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.stack = MambaStack(d_atom, n_layers, d_state=d_state, mimo_rank=mimo_rank,
                                expand=expand, headdim=headdim, bidirectional=bidirectional)
        self.grad_head = nn.Sequential(
            nn.LayerNorm(d_atom),
            nn.Linear(d_atom, d_atom // 2),
            nn.GELU(),
            nn.Linear(d_atom // 2, 3),
        )

    def forward(self, atom_tok: Tensor, atom_mask: Tensor) -> Tensor:
        """
        Args:
            atom_tok: [B*L, A, d_atom]
            atom_mask: [B*L, A] bool

        Returns: [B*L, A, 3] predicted EqM gradient
        """
        atom_tok = self.stack(atom_tok, atom_mask)
        return self.grad_head(atom_tok) * atom_mask.unsqueeze(-1).to(atom_tok.dtype)
