"""Typed data containers for protein batches."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ProteinExample:
    """Single protein structure (pre-batching)."""
    res_type: torch.Tensor        # [L] int — AA type IDs
    atom_type: torch.Tensor       # [L, A] int — atom type IDs per slot
    pair_type: torch.Tensor       # [L, A] int — (residue, atom) pair IDs
    coords: torch.Tensor          # [L, A, 3] float — ground truth coordinates
    atom_mask: torch.Tensor       # [L, A] bool — valid atom slots
    observed_mask: torch.Tensor   # [L, A] bool — experimentally observed atoms
    res_seq_nums: torch.Tensor    # [L] int — residue sequence numbers
    seq_len: int                  # number of residues
    esm: Optional[torch.Tensor] = None  # [L, d_esm] float — pre-computed ESM embeddings


@dataclass
class ProteinBatch:
    """Batched protein data for training/inference."""
    # Sequence info
    res_type: torch.Tensor        # [B, L] int
    res_seq_nums: torch.Tensor    # [B, L] int — residue sequence numbers / indices
    atom_type: torch.Tensor       # [B, L, A] int
    pair_type: torch.Tensor       # [B, L, A] int — (residue, atom) pair IDs
    res_mask: torch.Tensor        # [B, L] bool — valid residues (padding mask)
    atom_mask: torch.Tensor       # [B, L, A] bool — valid atom slots
    valid_mask: torch.Tensor      # [B, L, A] bool — atom_mask & observed_mask
    ca_mask: torch.Tensor         # [B, L] bool — has C-alpha

    # Coordinates
    x_clean: torch.Tensor         # [B, L, A, 3] float — normalized ground truth
    x_gamma: torch.Tensor         # [B, L, A, 3] float — corrupted coordinates
    eps: torch.Tensor             # [B, L, A, 3] float — noise
    gamma: torch.Tensor           # [B, 1, 1, 1] float — interpolation factor

    # Conditioning
    esm: Optional[torch.Tensor]   # [B, L, d_plm] float — optional external PLM embeddings

    @property
    def device(self) -> torch.device:
        return self.res_type.device

    @property
    def batch_size(self) -> int:
        return self.res_type.shape[0]

    @property
    def max_len(self) -> int:
        return self.res_type.shape[1]

    def with_coords(self, new_coords: torch.Tensor) -> "ProteinBatch":
        """Return a copy with x_gamma replaced (for sampling)."""
        return ProteinBatch(
            res_type=self.res_type,
            res_seq_nums=self.res_seq_nums,
            atom_type=self.atom_type,
            pair_type=self.pair_type,
            res_mask=self.res_mask,
            atom_mask=self.atom_mask,
            valid_mask=self.valid_mask,
            ca_mask=self.ca_mask,
            x_clean=self.x_clean,
            x_gamma=new_coords,
            eps=self.eps,
            gamma=self.gamma,
            esm=self.esm,
        )

    def to(self, device: torch.device) -> "ProteinBatch":
        """Move all tensors to device."""
        fields = {}
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                fields[k] = v.to(device)
            else:
                fields[k] = v
        return ProteinBatch(**fields)
