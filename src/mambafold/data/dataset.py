"""AFDB dataset loading and canonicalization."""

import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

from mambafold.data.constants import (
    AA_TO_ID,
    ATOM_NAME_TO_ID,
    CA_ATOM_ID,
    MAX_ATOMS_PER_RES,
    PAIR_PAD_ID,
    PAIR_TO_ID,
    RESIDUE_ATOM_TO_SLOT,
    RESIDUE_ATOMS,
)
from mambafold.data.types import ProteinExample


class AFDBDataset(Dataset):
    """Dataset for AFDB .pt files with canonical atom slot mapping."""

    def __init__(
        self,
        data_dir: str,
        max_length: int = 256,
        filter_std_aa: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.max_length = max_length
        self.filter_std_aa = filter_std_aa

        # Collect all .pt files
        self.files = sorted(self.data_dir.glob("*.pt"))
        if len(self.files) == 0:
            raise ValueError(f"No .pt files found in {data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> ProteinExample | None:
        raw = torch.load(self.files[idx], weights_only=False, map_location="cpu")
        return self._canonicalize(raw)

    def _canonicalize(self, raw: dict) -> ProteinExample | None:
        """Convert raw .pt dict to canonical ProteinExample."""
        res_names = raw["res_names"]
        atom_names_per_res = raw["atom_names"]
        coords_per_res = raw["coords"]
        is_observed_per_res = raw["is_observed"]
        L = len(res_names)

        if L == 0:
            return None

        # Filter to standard amino acids
        if self.filter_std_aa:
            valid_idx = [i for i, r in enumerate(res_names) if r in AA_TO_ID and r != "UNK"]
            if len(valid_idx) == 0:
                return None
            res_names = [res_names[i] for i in valid_idx]
            atom_names_per_res = [atom_names_per_res[i] for i in valid_idx]
            coords_per_res = [coords_per_res[i] for i in valid_idx]
            is_observed_per_res = [is_observed_per_res[i] for i in valid_idx]
            L = len(res_names)

        # Random crop if needed
        if L > self.max_length:
            start = torch.randint(0, L - self.max_length, (1,)).item()
            end = start + self.max_length
            res_names = res_names[start:end]
            atom_names_per_res = atom_names_per_res[start:end]
            coords_per_res = coords_per_res[start:end]
            is_observed_per_res = is_observed_per_res[start:end]
            L = self.max_length

        A = MAX_ATOMS_PER_RES

        # Initialize tensors
        res_type = torch.zeros(L, dtype=torch.long)
        atom_type = torch.full((L, A), ATOM_NAME_TO_ID["PAD"], dtype=torch.long)
        pair_type = torch.full((L, A), PAIR_PAD_ID, dtype=torch.long)
        coords = torch.zeros(L, A, 3, dtype=torch.float32)
        atom_mask = torch.zeros(L, A, dtype=torch.bool)
        observed_mask = torch.zeros(L, A, dtype=torch.bool)
        res_seq_nums = torch.arange(L, dtype=torch.long)

        for i in range(L):
            res_name = res_names[i]
            res_type[i] = AA_TO_ID.get(res_name, AA_TO_ID["UNK"])

            slot_map = RESIDUE_ATOM_TO_SLOT.get(res_name, RESIDUE_ATOM_TO_SLOT["UNK"])
            raw_atoms = atom_names_per_res[i]
            raw_coords = coords_per_res[i]
            raw_obs = is_observed_per_res[i]

            for j, atom_name in enumerate(raw_atoms):
                if atom_name in slot_map:
                    slot = slot_map[atom_name]
                    if slot < A:
                        atom_type[i, slot] = ATOM_NAME_TO_ID.get(atom_name, ATOM_NAME_TO_ID["PAD"])
                        pair_type[i, slot] = PAIR_TO_ID.get((res_name, atom_name), PAIR_PAD_ID)
                        coords[i, slot] = torch.tensor(raw_coords[j], dtype=torch.float32)
                        atom_mask[i, slot] = True
                        observed_mask[i, slot] = raw_obs[j] if j < len(raw_obs) else False

        return ProteinExample(
            res_type=res_type,
            atom_type=atom_type,
            pair_type=pair_type,
            coords=coords,
            atom_mask=atom_mask,
            observed_mask=observed_mask,
            res_seq_nums=res_seq_nums,
            seq_len=L,
        )
