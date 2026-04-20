"""AFDB / RCSB dataset loading and canonicalization."""

from pathlib import Path

import numpy as np
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

        # Collect struct .pt files, excluding ESM cache files
        self.files = sorted(
            f for f in self.data_dir.rglob("*.pt")
            if not (f.name.endswith(".esm3.pt") or f.name.endswith(".esmc.pt"))
        )
        if len(self.files) == 0:
            raise ValueError(f"No .pt files found in {data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> ProteinExample | None:
        path = self.files[idx]
        raw = torch.load(path, weights_only=False, map_location="cpu")
        return self._canonicalize(raw, path)

    def _canonicalize(self, raw: dict, path: Path | None = None) -> ProteinExample | None:
        """Convert raw .pt dict to canonical ProteinExample."""
        res_names = raw["res_names"]
        atom_names_per_res = raw["atom_names"]
        coords_per_res = raw["coords"]
        is_observed_per_res = raw["is_observed"]
        L = len(res_names)

        if L == 0:
            return None

        # Load pre-cached ESM3 embeddings [L_raw, 1536] if available
        esm_raw = None
        if path is not None:
            esm_path = path.parent / (path.stem + ".esm3.pt")
            if esm_path.exists():
                esm_raw = torch.load(esm_path, weights_only=True, map_location="cpu")

        # Filter to standard amino acids
        if self.filter_std_aa:
            valid_idx = [i for i, r in enumerate(res_names) if r in AA_TO_ID and r != "UNK"]
            if len(valid_idx) == 0:
                return None
            res_names = [res_names[i] for i in valid_idx]
            atom_names_per_res = [atom_names_per_res[i] for i in valid_idx]
            coords_per_res = [coords_per_res[i] for i in valid_idx]
            is_observed_per_res = [is_observed_per_res[i] for i in valid_idx]
            if esm_raw is not None:
                esm_raw = esm_raw[valid_idx]
            L = len(res_names)

        # Random crop if needed
        if L > self.max_length:
            start = torch.randint(0, L - self.max_length, (1,)).item()
            end = start + self.max_length
            res_names = res_names[start:end]
            atom_names_per_res = atom_names_per_res[start:end]
            coords_per_res = coords_per_res[start:end]
            is_observed_per_res = is_observed_per_res[start:end]
            if esm_raw is not None:
                esm_raw = esm_raw[start:end]
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
            esm=esm_raw,
        )


class RCSBDataset(Dataset):
    """Dataset for Boltz-style .npz files from rcsb_processed_targets.

    Each .npz contains structured arrays: residues, atoms, chains, coords, etc.
    Atoms are stored in canonical ref_atoms[res_name] order, so atom names are
    recovered positionally without decoding the byte-encoded name field.
    Only protein chains (mol_type == 0) and standard residues are used.
    """

    MOL_TYPE_PROTEIN = 0

    def __init__(self, data_dir: str, max_length: int = 512,
                 min_length: int = 20, min_obs_ratio: float = 0.5,
                 file_list: str | None = None, esm_dir: str | None = None):
        self.data_dir = Path(data_dir)
        self.max_length = max_length
        self.min_length = min_length
        self.min_obs_ratio = min_obs_ratio
        self.esm_dir = Path(esm_dir) if esm_dir else None
        if file_list is not None:
            self.files = sorted(
                self.data_dir / line.strip()
                for line in Path(file_list).read_text().splitlines()
                if line.strip()
            )
        else:
            self.files = sorted(self.data_dir.rglob("*.npz"))
        if len(self.files) == 0:
            raise ValueError(f"No .npz files found in {data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> ProteinExample:
        n = len(self.files)
        for attempt in range(n):
            i = (idx + attempt) % n
            try:
                path = self.files[i]
                data = np.load(path)
                ex = self._canonicalize(data, path)
            except Exception:
                ex = None
            if ex is not None:
                return ex
        raise RuntimeError("RCSBDataset: no valid sample in entire dataset")

    def _canonicalize(self, data, path: Path | None = None) -> ProteinExample | None:
        residues = data["residues"]
        atoms = data["atoms"]
        chains = data["chains"]

        # Collect residue indices per protein chain (track original index for ESM lookup)
        protein_chains = []       # filtered residue lists
        protein_chain_origins = [] # original protein-chain index (matches precompute _ch{j}.npy)
        prot_chain_idx = 0
        for ch in chains:
            if ch["mol_type"] != self.MOL_TYPE_PROTEIN:
                continue
            r_start = int(ch["res_idx"])
            r_end = r_start + int(ch["res_num"])
            chain_valid = [
                i for i in range(r_start, r_end)
                if residues[i]["is_standard"] and residues[i]["name"] in AA_TO_ID
                   and residues[i]["name"] != "UNK"
            ]
            if len(chain_valid) >= self.min_length:
                protein_chains.append(chain_valid)
                protein_chain_origins.append(prot_chain_idx)
            prot_chain_idx += 1

        if not protein_chains:
            return None

        # Pick one chain randomly
        pick = int(torch.randint(0, len(protein_chains), (1,)).item())
        valid = protein_chains[pick]
        esm_chain_idx = protein_chain_origins[pick]  # for ESM file lookup

        # Random crop
        esm_start = 0
        if len(valid) > self.max_length:
            esm_start = int(torch.randint(0, len(valid) - self.max_length, (1,)).item())
            valid = valid[esm_start: esm_start + self.max_length]

        L = len(valid)
        A = MAX_ATOMS_PER_RES

        res_type     = torch.zeros(L, dtype=torch.long)
        atom_type    = torch.full((L, A), ATOM_NAME_TO_ID["PAD"], dtype=torch.long)
        pair_type    = torch.full((L, A), PAIR_PAD_ID, dtype=torch.long)
        coords       = torch.zeros(L, A, 3, dtype=torch.float32)
        atom_mask    = torch.zeros(L, A, dtype=torch.bool)
        observed_mask = torch.zeros(L, A, dtype=torch.bool)
        res_seq_nums = torch.arange(L, dtype=torch.long)

        for i, ri in enumerate(valid):
            res = residues[ri]
            res_name = str(res["name"])
            res_type[i] = AA_TO_ID.get(res_name, AA_TO_ID["UNK"])

            slot_map    = RESIDUE_ATOM_TO_SLOT.get(res_name, RESIDUE_ATOM_TO_SLOT["UNK"])
            canon_names = RESIDUE_ATOMS.get(res_name, [])
            a_start     = int(res["atom_idx"])
            a_num       = int(res["atom_num"])

            for j in range(min(a_num, len(canon_names))):
                atom_name = canon_names[j]
                if atom_name not in slot_map:
                    continue
                slot = slot_map[atom_name]
                if slot >= A:
                    continue
                a = atoms[a_start + j]
                atom_type[i, slot]    = ATOM_NAME_TO_ID.get(atom_name, ATOM_NAME_TO_ID["PAD"])
                pair_type[i, slot]    = PAIR_TO_ID.get((res_name, atom_name), PAIR_PAD_ID)
                coords[i, slot]       = torch.tensor(a["coords"], dtype=torch.float32)
                atom_mask[i, slot]    = True
                observed_mask[i, slot] = bool(a["is_present"])

        # Filter low-observation structures
        n_obs = observed_mask.sum().item()
        n_atoms = atom_mask.sum().item()
        if n_atoms > 0 and n_obs / n_atoms < self.min_obs_ratio:
            return None

        # ESM embedding (pre-computed per chain)
        esm = None
        if self.esm_dir is not None and path is not None:
            esm_path = self.esm_dir / f"{path.stem}_ch{esm_chain_idx}.npy"
            if esm_path.exists():
                try:
                    arr = np.load(esm_path)          # [n_chain_residues, d_esm]
                    esm = torch.from_numpy(arr[esm_start:esm_start + L].copy())
                except Exception:
                    pass

        return ProteinExample(
            res_type=res_type,
            atom_type=atom_type,
            pair_type=pair_type,
            coords=coords,
            atom_mask=atom_mask,
            observed_mask=observed_mask,
            res_seq_nums=res_seq_nums,
            seq_len=L,
            esm=esm,
        )
