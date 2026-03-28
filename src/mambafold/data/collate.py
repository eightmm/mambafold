"""Collation and batching for protein data."""

from typing import Optional

import torch
from torch.utils.data import Sampler

from mambafold.data.constants import CA_ATOM_ID, MAX_ATOMS_PER_RES, PAIR_PAD_ID
from mambafold.data.transforms import center_and_scale, eqm_corrupt, random_so3_augment
from mambafold.data.types import ProteinBatch, ProteinExample


class ProteinCollator:
    """Collate ProteinExamples into a padded ProteinBatch with EqM corruption."""

    def __init__(
        self,
        augment: bool = True,
        copies_per_protein: int = 1,
        esm_cache_dir: Optional[str] = None,
        gamma_schedule: str = "logit_normal",
    ):
        self.augment = augment
        self.copies_per_protein = copies_per_protein
        self.esm_cache_dir = esm_cache_dir
        self.gamma_schedule = gamma_schedule

    def __call__(self, examples: list[ProteinExample]) -> ProteinBatch:
        # Filter None examples
        examples = [e for e in examples if e is not None]
        if len(examples) == 0:
            raise ValueError("Empty batch after filtering")

        # Apply per-example transforms and duplicate for multiple corruptions
        processed = []
        for ex in examples:
            ex = center_and_scale(ex)
            if self.augment:
                ex = random_so3_augment(ex)
            for _ in range(self.copies_per_protein):
                processed.append(ex)

        B = len(processed)
        max_L = max(ex.seq_len for ex in processed)
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
            esm=None,  # loaded separately
        )


class LengthBucketSampler(Sampler):
    """Batch proteins by similar length to minimize padding waste."""

    def __init__(self, lengths: list[int], max_atoms: int = 32768, shuffle: bool = True):
        self.lengths = lengths
        self.max_atoms = max_atoms
        self.shuffle = shuffle

    def __iter__(self):
        # Sort indices by length
        indices = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])
        if self.shuffle:
            # Add small random perturbation to avoid identical batches
            import random
            random.shuffle(indices)
            indices = sorted(indices, key=lambda i: self.lengths[i] + random.randint(-10, 10))

        # Greedily fill batches by atom count
        batches = []
        current_batch = []
        current_atoms = 0

        for idx in indices:
            atom_count = self.lengths[idx] * MAX_ATOMS_PER_RES
            if current_atoms + atom_count > self.max_atoms and len(current_batch) > 0:
                batches.append(current_batch)
                current_batch = []
                current_atoms = 0
            current_batch.append(idx)
            current_atoms += atom_count

        if current_batch:
            batches.append(current_batch)

        if self.shuffle:
            import random
            random.shuffle(batches)

        yield from batches

    def __len__(self):
        # Approximate number of batches based on average atom count per protein
        return max(1, len(self.lengths) * MAX_ATOMS_PER_RES // self.max_atoms)
