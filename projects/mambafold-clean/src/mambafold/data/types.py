"""Typed data containers for protein batches."""

from dataclasses import dataclass, fields, replace
from typing import Optional

import torch

from mambafold.data.constants import PAIR_PAD_ID


@dataclass
class ProteinExample:
    """Single protein structure (pre-batching).

    All feature tensors here must be derivable from information available at
    inference time (sequence + user-supplied chain boundaries). No observation-
    quality signals (e.g. B-factor, missing-atom fraction) are stored.
    """

    res_type: torch.Tensor  # [L] int — AA type IDs (21 classes: 20 AA + UNK)
    atom_type: torch.Tensor  # [L, A] int — atom type IDs per slot
    pair_type: torch.Tensor  # [L, A] int — (residue, atom) pair IDs
    coords: torch.Tensor  # [L, A, 3] float — ground truth coordinates
    atom_mask: torch.Tensor  # [L, A] bool — valid atom slots (derivable from res_type)
    observed_mask: (
        torch.Tensor
    )  # [L, A] bool — experimentally observed atoms (TRAIN ONLY: loss masking)
    res_seq_nums: torch.Tensor  # [L] int — residue sequence numbers (within chain)
    seq_len: int  # number of residues
    chain_id: torch.Tensor = None  # [L] int — 0-based chain index within this example
    entity_id: torch.Tensor = (
        None  # [L] int — shared across chains with identical sequence (homomer grouping)
    )
    sym_id: torch.Tensor = None  # [L] int — copy number within an entity (AF3-style; 0..n_copies-1)
    is_nterm: torch.Tensor = None  # [L] bool — first residue of its ORIGINAL chain
    is_cterm: torch.Tensor = None  # [L] bool — last residue of its ORIGINAL chain
    esm: Optional[torch.Tensor] = None  # [L, d_esm] float — pre-computed ESM embeddings

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
    res_type: torch.Tensor  # [B, L] int
    res_seq_nums: torch.Tensor  # [B, L] int — residue sequence numbers within chain
    atom_type: torch.Tensor  # [B, L, A] int
    pair_type: torch.Tensor  # [B, L, A] int — (residue, atom) pair IDs
    res_mask: torch.Tensor  # [B, L] bool — valid residues (padding mask)
    atom_mask: torch.Tensor  # [B, L, A] bool — valid atom slots
    valid_mask: (
        torch.Tensor
    )  # [B, L, A] bool — atom_mask & observed_mask (LOSS ONLY — do not feed to model)
    ca_mask: torch.Tensor  # [B, L] bool — has C-alpha

    # Chain / entity indexing (0 for single-chain fallback)
    chain_id: torch.Tensor  # [B, L] int — per-chain unique index
    entity_id: torch.Tensor  # [B, L] int — shared across identical sequences (homomer signal)
    sym_id: torch.Tensor  # [B, L] int — copy number within an entity (AF3 style)
    is_nterm: torch.Tensor  # [B, L] bool — first residue of its original chain
    is_cterm: torch.Tensor  # [B, L] bool — last residue of its original chain

    # Coordinates
    x_clean: torch.Tensor  # [B, L, A, 3] float — normalized ground truth
    x_t: torch.Tensor  # [B, L, A, 3] float — corrupted coordinates
    eps: torch.Tensor  # [B, L, A, 3] float — noise
    t: torch.Tensor  # [B, 1, 1, 1] float — interpolation time ∈ [0, 1]

    # Conditioning
    # [B, L, d_plm] float — optional external PLM embeddings. May carry a
    # single row when every example in the batch is the same protein
    # (`copies_per_protein` > 1); the model broadcasts it.
    esm: Optional[torch.Tensor]
    x_self_cond: Optional[torch.Tensor] = None  # [B, L, A, 3] detached x0 estimate

    @property
    def device(self) -> torch.device:
        return self.res_type.device

    @property
    def batch_size(self) -> int:
        return self.res_type.shape[0]

    @property
    def max_len(self) -> int:
        return self.res_type.shape[1]

    def to(self, device: torch.device) -> "ProteinBatch":
        """Move all tensor fields to `device`; non-tensors pass through.

        `non_blocking=True` is safe and useful here because the loader sets
        `pin_memory=True`: the copies are queued on the current stream and every
        consumer is on that same stream, so ordering holds while the transfer
        overlaps whatever the GPU is still finishing. It is a small win — about
        15 MB per micro-step at crop 1024 with 8 copies — but it is free.
        """
        moved = {
            f.name: v.to(device, non_blocking=True)
            for f in fields(self)
            if isinstance(v := getattr(self, f.name), torch.Tensor)
        }
        return replace(self, **moved)

    def pad_to_length(self, max_L: int) -> "ProteinBatch":
        """Right-pad every residue-axis tensor to ``max_L``.

        This is used after DDP ranks agree on one global padded length. It is
        intentionally safe on CPU batches so the expanded tensors are moved to
        CUDA only once, avoiding a transient double allocation near the VRAM
        limit.
        """
        current = self.max_len
        if current >= max_L:
            return self
        pad_rows = max_L - current
        padded = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == "t" or not isinstance(value, torch.Tensor):
                continue
            shape = list(value.shape)
            shape[1] = pad_rows
            fill_value = PAIR_PAD_ID if f.name == "pair_type" else 0
            extra = torch.full(
                shape,
                fill_value,
                dtype=value.dtype,
                device=value.device,
            )
            padded[f.name] = torch.cat((value, extra), dim=1)
        return replace(self, **padded)
