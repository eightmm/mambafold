"""Typed data containers for protein batches."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ProteinExample:
    """Single protein structure (pre-batching).

    All feature tensors here must be derivable from information available at
    inference time (sequence + user-supplied chain boundaries). No observation-
    quality signals (e.g. B-factor, missing-atom fraction) are stored.
    """
    res_type: torch.Tensor        # [L] int — AA type IDs (21 classes: 20 AA + UNK)
    atom_type: torch.Tensor       # [L, A] int — atom type IDs per slot
    pair_type: torch.Tensor       # [L, A] int — (residue, atom) pair IDs
    coords: torch.Tensor          # [L, A, 3] float — ground truth coordinates
    atom_mask: torch.Tensor       # [L, A] bool — valid atom slots (derivable from res_type)
    observed_mask: torch.Tensor   # [L, A] bool — experimentally observed atoms (TRAIN ONLY: loss masking)
    res_seq_nums: torch.Tensor    # [L] int — residue sequence numbers (within chain)
    seq_len: int                  # number of residues
    chain_id: torch.Tensor = None         # [L] int — 0-based chain index within this example
    entity_id: torch.Tensor = None        # [L] int — shared across chains with identical sequence (homomer grouping)
    sym_id: torch.Tensor = None           # [L] int — copy number within an entity (AF3-style; 0..n_copies-1)
    is_nterm: torch.Tensor = None         # [L] bool — first residue of its ORIGINAL chain
    is_cterm: torch.Tensor = None         # [L] bool — last residue of its ORIGINAL chain
    esm: Optional[torch.Tensor] = None    # [L, d_esm] float — pre-computed ESM embeddings

    def __post_init__(self):
        # Auto-fill single-chain defaults for backward compatibility.
        if self.chain_id is None:
            self.chain_id = torch.zeros(self.seq_len, dtype=torch.long)
        if self.entity_id is None:
            self.entity_id = self.chain_id.clone()
        if self.sym_id is None:
            self.sym_id = torch.zeros(self.seq_len, dtype=torch.long)
        if self.is_nterm is None:
            self.is_nterm = torch.zeros(self.seq_len, dtype=torch.bool)
        if self.is_cterm is None:
            self.is_cterm = torch.zeros(self.seq_len, dtype=torch.bool)


@dataclass
class ProteinBatch:
    """Batched protein data for training/inference.

    Only inference-available features are fed to the model; `observed_mask` /
    `valid_mask` are carried for loss masking but never used as input features.
    """
    # Sequence info
    res_type: torch.Tensor        # [B, L] int
    res_seq_nums: torch.Tensor    # [B, L] int — residue sequence numbers within chain
    atom_type: torch.Tensor       # [B, L, A] int
    pair_type: torch.Tensor       # [B, L, A] int — (residue, atom) pair IDs
    res_mask: torch.Tensor        # [B, L] bool — valid residues (padding mask)
    atom_mask: torch.Tensor       # [B, L, A] bool — valid atom slots
    valid_mask: torch.Tensor      # [B, L, A] bool — atom_mask & observed_mask (LOSS ONLY — do not feed to model)
    ca_mask: torch.Tensor         # [B, L] bool — has C-alpha

    # Chain / entity indexing (0 for single-chain fallback)
    chain_id: torch.Tensor        # [B, L] int — per-chain unique index
    entity_id: torch.Tensor       # [B, L] int — shared across identical sequences (homomer signal)
    sym_id: torch.Tensor          # [B, L] int — copy number within an entity (AF3 style)
    is_nterm: torch.Tensor        # [B, L] bool — first residue of its original chain
    is_cterm: torch.Tensor        # [B, L] bool — last residue of its original chain

    # Coordinates
    x_clean: torch.Tensor         # [B, L, A, 3] float — normalized ground truth
    x_t: torch.Tensor         # [B, L, A, 3] float — corrupted coordinates
    eps: torch.Tensor             # [B, L, A, 3] float — noise
    t: torch.Tensor               # [B, 1, 1, 1] float — interpolation time ∈ [0, 1]

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
        """Return a copy with x_t replaced (for sampling)."""
        return ProteinBatch(
            res_type=self.res_type,
            res_seq_nums=self.res_seq_nums,
            atom_type=self.atom_type,
            pair_type=self.pair_type,
            res_mask=self.res_mask,
            atom_mask=self.atom_mask,
            valid_mask=self.valid_mask,
            ca_mask=self.ca_mask,
            chain_id=self.chain_id,
            entity_id=self.entity_id,
            sym_id=self.sym_id,
            is_nterm=self.is_nterm,
            is_cterm=self.is_cterm,
            x_clean=self.x_clean,
            x_t=new_coords,
            eps=self.eps,
            t=self.t,
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

    def truncate_length(self, max_L: int) -> "ProteinBatch":
        """Return a copy with the L-axis cropped to `max_L`. No-op if already shorter.

        `t` is shape-invariant so it's passed through.
        """
        if self.res_type.shape[1] <= max_L:
            return self
        sl = (slice(None), slice(0, max_L))
        return ProteinBatch(
            res_type=self.res_type[sl],
            res_seq_nums=self.res_seq_nums[sl],
            atom_type=self.atom_type[sl],
            pair_type=self.pair_type[sl],
            res_mask=self.res_mask[sl],
            atom_mask=self.atom_mask[sl],
            valid_mask=self.valid_mask[sl],
            ca_mask=self.ca_mask[sl],
            chain_id=self.chain_id[sl],
            entity_id=self.entity_id[sl],
            sym_id=self.sym_id[sl],
            is_nterm=self.is_nterm[sl],
            is_cterm=self.is_cterm[sl],
            x_clean=self.x_clean[sl],
            x_t=self.x_t[sl],
            eps=self.eps[sl],
            t=self.t,
            esm=self.esm[sl] if self.esm is not None else None,
        )
