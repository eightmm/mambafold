"""Collation and batching for protein data."""

from typing import Optional

import torch

from mambafold.data.constants import CA_ATOM_ID, MAX_ATOMS_PER_RES, PAIR_PAD_ID
from mambafold.data.transforms import center_and_scale, eqm_corrupt, random_so3_augment
from mambafold.data.types import ProteinBatch, ProteinExample


class ProteinCollator:
    """Collate ProteinExamples into a padded ProteinBatch with EqM corruption."""

    def __init__(
        self,
        augment: bool = True,
        copies_per_protein: int = 1,
        gamma_schedule: str = "logit_normal",
        max_length: Optional[int] = None,
    ):
        self.augment = augment
        self.copies_per_protein = copies_per_protein
        self.gamma_schedule = gamma_schedule
        self.max_length = max_length

    def __call__(self, examples: list[ProteinExample]) -> ProteinBatch | None:
        # Filter None examples
        examples = [e for e in examples if e is not None]
        if len(examples) == 0:
            return None

        # Apply per-example transforms and duplicate for multiple corruptions
        processed = []
        for ex in examples:
            ex = center_and_scale(ex)
            if self.augment:
                ex = random_so3_augment(ex)
            for _ in range(self.copies_per_protein):
                processed.append(ex)

        B = len(processed)
        # Use fixed max_length if provided (prevents TileLang recompilation for varying lengths)
        max_L = self.max_length if self.max_length is not None else max(ex.seq_len for ex in processed)
        A = MAX_ATOMS_PER_RES

        # Initialize batch tensors
        res_type = torch.zeros(B, max_L, dtype=torch.long)
        res_seq_nums = torch.zeros(B, max_L, dtype=torch.long)
        atom_type = torch.zeros(B, max_L, A, dtype=torch.long)
        pair_type = torch.full((B, max_L, A), PAIR_PAD_ID, dtype=torch.long)
        res_mask = torch.zeros(B, max_L, dtype=torch.bool)
        atom_mask = torch.zeros(B, max_L, A, dtype=torch.bool)
        valid_mask = torch.zeros(B, max_L, A, dtype=torch.bool)
        ca_mask = torch.zeros(B, max_L, dtype=torch.bool)
        x_clean = torch.zeros(B, max_L, A, 3)
        x_gamma = torch.zeros(B, max_L, A, 3)
        eps = torch.zeros(B, max_L, A, 3)
        gamma = torch.zeros(B, 1, 1, 1)

        for i, ex in enumerate(processed):
            L = ex.seq_len
            res_type[i, :L] = ex.res_type
            res_seq_nums[i, :L] = ex.res_seq_nums
            atom_type[i, :L] = ex.atom_type
            pair_type[i, :L] = ex.pair_type
            res_mask[i, :L] = True
            atom_mask[i, :L] = ex.atom_mask
            valid_mask[i, :L] = ex.atom_mask & ex.observed_mask
            ca_mask[i, :L] = ex.atom_mask[:, CA_ATOM_ID] & ex.observed_mask[:, CA_ATOM_ID]
            x_clean[i, :L] = ex.coords

            # EqM corruption
            xg, ep, gm = eqm_corrupt(ex.coords, ex.atom_mask, self.gamma_schedule)
            x_gamma[i, :L] = xg
            eps[i, :L] = ep
            gamma[i, 0, 0, 0] = gm

        # ESM embeddings from pre-computed dataset
        esm = None
        esm_list = [ex.esm for ex in processed]
        if all(e is not None and e.shape[0] > 0 for e in esm_list):
            d_esm = esm_list[0].shape[-1]
            esm = torch.zeros(B, max_L, d_esm, dtype=torch.float32)
            for i, ex in enumerate(processed):
                n = min(ex.seq_len, ex.esm.shape[0], max_L)
                esm[i, :n] = ex.esm[:n]

        return ProteinBatch(
            res_type=res_type,
            res_seq_nums=res_seq_nums,
            atom_type=atom_type,
            pair_type=pair_type,
            res_mask=res_mask,
            atom_mask=atom_mask,
            valid_mask=valid_mask,
            ca_mask=ca_mask,
            x_clean=x_clean,
            x_gamma=x_gamma,
            eps=eps,
            gamma=gamma,
            esm=esm,
        )
