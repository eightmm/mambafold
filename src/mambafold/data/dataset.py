"""AFDB / RCSB dataset loading and canonicalization."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from mambafold.data.constants import (
    AA_3TO1,
    AA_TO_ID,
    ATOM_NAME_TO_ID,
    MAX_ATOMS_PER_RES,
    PAIR_PAD_ID,
    PAIR_TO_ID,
    RESIDUE_ATOM_TO_SLOT,
    RESIDUE_ATOMS,
)
from mambafold.data.sequence_cache import sequence_embedding_path
from mambafold.data.types import ProteinExample


def _residue_seq_num(residue, fallback: int) -> int:
    """Best-effort original residue number; fallback to contiguous local index."""
    names = getattr(residue.dtype, "names", None) or ()
    for key in ("auth_seq_id", "res_seq_num", "seq_id", "residue_index", "pdb_res_num"):
        if key in names:
            return int(residue[key])
    return int(fallback)


def _gather_esm_rows(
    entries: list[tuple[int, int, int, int, int]],
    per_chain_cache: dict[int, np.ndarray],
) -> torch.Tensor | None:
    """Gather crop-aligned PLM rows with one vectorized copy per chain.

    The previous implementation copied every 2560-dimensional ESMC row into a
    separate NumPy array and Torch tensor.  At L=1024 that creates thousands of
    small allocations per batch item and can exceed the DataLoader timeout.
    """
    if not entries:
        return None

    positions_by_origin: dict[int, list[int]] = {}
    rows_by_origin: dict[int, list[int]] = {}
    embedding_dim: int | None = None
    embedding_dtype = None
    for position, (_ri, loc, _seq_num, _cid, origin) in enumerate(entries):
        arr = per_chain_cache[origin]
        if arr.ndim != 2 or not (0 <= loc < arr.shape[0]):
            return None
        if embedding_dim is None:
            embedding_dim = int(arr.shape[1])
            embedding_dtype = arr.dtype
        elif arr.shape[1] != embedding_dim or arr.dtype != embedding_dtype:
            return None
        positions_by_origin.setdefault(origin, []).append(position)
        rows_by_origin.setdefault(origin, []).append(loc)

    assert embedding_dim is not None and embedding_dtype is not None
    gathered = np.empty((len(entries), embedding_dim), dtype=embedding_dtype)
    for origin, positions in positions_by_origin.items():
        gathered[np.asarray(positions)] = per_chain_cache[origin][
            np.asarray(rows_by_origin[origin])
        ]
    return torch.from_numpy(gathered)


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
            f
            for f in self.data_dir.rglob("*.pt")
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

        # Load legacy pre-cached PLM embeddings if available.
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

        # Legacy path: single chain, single entity; mark true N/C terminus.
        is_nterm_t = torch.zeros(L, dtype=torch.bool)
        is_cterm_t = torch.zeros(L, dtype=torch.bool)
        if L > 0:
            is_nterm_t[0] = True
            is_cterm_t[L - 1] = True
        return ProteinExample(
            res_type=res_type,
            atom_type=atom_type,
            pair_type=pair_type,
            coords=coords,
            atom_mask=atom_mask,
            observed_mask=observed_mask,
            res_seq_nums=res_seq_nums,
            seq_len=L,
            is_nterm=is_nterm_t,
            is_cterm=is_cterm_t,
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

    def _esm_embedding_path(self, stem: str, origin: int, sequence: str) -> Path | None:
        """Resolve sequence-addressed PLM cache with occurrence fallback."""
        if self.esm_dir is None:
            return None
        sequence_path = sequence_embedding_path(self.esm_dir, sequence)
        if sequence_path.exists():
            return sequence_path
        occurrence_path = self.esm_dir / f"{stem}_ch{origin}.npy"
        return occurrence_path if occurrence_path.exists() else None

    def __init__(
        self,
        data_dir: str,
        max_length: int = 512,
        min_length: int = 20,
        min_obs_ratio: float = 0.5,
        file_list: str | None = None,
        esm_dir: str | None = None,
        single_chain_only: bool = False,
        extract_monomer_chains: bool = False,
        chain_index_workers: int = 8,
    ):
        self.data_dir = Path(data_dir)
        self.max_length = max_length
        self.min_length = min_length
        self.min_obs_ratio = min_obs_ratio
        self.esm_dir = Path(esm_dir) if esm_dir else None
        self.single_chain_only = single_chain_only
        # Monomer-extraction mode: index every protein chain of every entry as
        # its own single-chain example (a 4-chain entry → 4 monomers). Supersedes
        # single_chain_only. Builds a cached chain index [(file_idx, origin, len)].
        self.extract_monomer_chains = extract_monomer_chains
        self.chain_index: list[tuple[int, int, int]] | None = None
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
        if extract_monomer_chains:
            from mambafold.data.length_cache import build_chain_index

            self.chain_index = build_chain_index(self, num_workers=chain_index_workers)

    def __len__(self) -> int:
        if self.extract_monomer_chains and self.chain_index is not None:
            return len(self.chain_index)
        return len(self.files)

    def probe_chains(self, path) -> list[tuple[int, int]]:
        """For monomer extraction: list of (protein_chain_origin, seq_len) for
        every valid monomer chain in `path`.

        Fast path — replicates only the residue-keeping + ESM-length clamp that
        set `seq_len` in `_canonicalize` (no coordinate tensors), so the cached
        length matches training exactly. The min_obs_ratio gate is *not* applied
        here (it needs the atom arrays); the rare low-observation chain that
        survives indexing is dropped at load time and skipped to the next, which
        only marginally loosens a bucket. Validity otherwise matches: protein
        chain, >= min_length standard residues, and ESM present when required.
        """
        out: list[tuple[int, int]] = []
        try:
            data = np.load(path)
            chains = data["chains"]
            residues = data["residues"]
        except Exception:
            return out
        stem = Path(path).stem
        origin = -1
        for ch in chains:
            if int(ch["mol_type"]) != self.MOL_TYPE_PROTEIN:
                continue
            origin += 1
            r_start = int(ch["res_idx"])
            r_end = r_start + int(ch["res_num"])
            sequence: list[str] = []
            for i in range(r_start, r_end):
                r = residues[i]
                if r["is_standard"] and str(r["name"]) in AA_TO_ID and str(r["name"]) != "UNK":
                    sequence.append(AA_3TO1[str(r["name"])])
            kept = len(sequence)
            if kept < self.min_length:
                continue
            L = kept
            if self.esm_dir is not None:
                # ESM-loc filter keeps chain-local indices < esm rows → L=min(kept, esm_len).
                p = self._esm_embedding_path(stem, origin, "".join(sequence))
                if p is None:
                    continue  # _canonicalize would return None
                try:
                    L = min(L, int(np.load(p, mmap_mode="r").shape[0]))
                except Exception:
                    continue
            out.append((origin, min(L, self.max_length)))
        return out

    def __getitem__(self, idx: int) -> ProteinExample:
        if self.extract_monomer_chains and self.chain_index is not None:
            n = len(self.chain_index)
            for attempt in range(n):
                fi, origin, _ = self.chain_index[(idx + attempt) % n]
                try:
                    ex = self._canonicalize(
                        np.load(self.files[fi]), self.files[fi], only_chain_origin=origin
                    )
                except Exception:
                    ex = None
                if ex is not None:
                    return ex
            raise RuntimeError("RCSBDataset: no valid chain in entire index")

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

    def probe_length(self, path) -> int | None:
        """Return the seq_len `__getitem__` would yield for `path` (the same value
        the collator pads to), or None if `path` is not a valid sample under the
        current filters (multi-chain when single_chain_only, missing ESM, or
        below min_obs_ratio). Reuses `_canonicalize` so the result exactly matches
        training; used to build the length cache for length-bucketed batching.

        Length is deterministic across epochs: the random crop changes which
        window is taken, not its length (min(kept_residues, max_length)).
        """
        try:
            ex = self._canonicalize(np.load(path), Path(path))
        except Exception:
            return None
        return None if ex is None else int(ex.seq_len)

    def _canonicalize(
        self, data, path: Path | None = None, only_chain_origin: int | None = None
    ) -> ProteinExample | None:
        """Multi-chain canonicalization.

        Walks every protein chain in the entry, stitches them into a single
        flat residue sequence, and tags each residue with `chain_id` (0-based,
        order-preserving) and `res_seq_nums` (original res_idx within that
        chain so the position embedder can see per-chain gaps).

        `only_chain_origin` (monomer extraction): keep ONLY the protein chain
        at that 0-based protein-chain index, yielding a single-chain example for
        it (used by `extract_monomer_chains` to turn each chain of a multimer
        into its own monomer). When set, the `single_chain_only` whole-entry
        filter is bypassed.

        Random crop is taken as a contiguous window over the concatenation;
        chain boundaries inside the crop are preserved in chain_id.
        """
        residues = data["residues"]
        atoms = data["atoms"]
        chains = data["chains"]

        # Gather (residue_idx, chain_local_idx, residue_seq_num, chain_id, chain_origin_idx) entries
        entries: list[tuple[int, int, int, int, int]] = []
        per_chain_counts: list[int] = []  # chain_id → total standard residues (before crop)
        per_chain_sequences: list[str] = []  # chain_id → canonical one-letter sequence
        per_chain_entity: list[
            int
        ] = []  # chain_id → entity_id (same value for chains with identical sequence)
        per_chain_sym: list[int] = []  # chain_id → sym_id (copy number within entity, AF3 style)
        chain_origin_idx = 0  # original protein-chain index (for ESM lookup)
        chain_id_next = 0  # 0-based id for kept chains
        chain_origin_map: list[int] = []  # chain_id → origin_idx
        seq_to_entity: dict[tuple, int] = {}  # residue-name tuple → entity_id
        seq_to_sym_count: dict[tuple, int] = {}  # residue-name tuple → next sym_id (copy 0, 1, ...)
        for ch in chains:
            if ch["mol_type"] != self.MOL_TYPE_PROTEIN:
                continue
            r_start = int(ch["res_idx"])
            r_end = r_start + int(ch["res_num"])
            local = 0
            kept = []
            for i in range(r_start, r_end):
                r = residues[i]
                if r["is_standard"] and r["name"] in AA_TO_ID and r["name"] != "UNK":
                    kept.append((i, local, _residue_seq_num(r, local)))
                    local += 1
            if len(kept) >= self.min_length and (
                only_chain_origin is None or chain_origin_idx == only_chain_origin
            ):
                for ri, loc, seq_num in kept:
                    entries.append((ri, loc, seq_num, chain_id_next, chain_origin_idx))
                per_chain_counts.append(len(kept))
                # Entity id: chains with identical residue-name sequence share an id.
                seq_tuple = tuple(str(residues[ri]["name"]) for ri, _, _ in kept)
                per_chain_sequences.append("".join(AA_3TO1[name] for name in seq_tuple))
                if seq_tuple not in seq_to_entity:
                    seq_to_entity[seq_tuple] = len(seq_to_entity)
                per_chain_entity.append(seq_to_entity[seq_tuple])
                # Sym id: 0-based copy number within an entity (homomer copies)
                sym_id_val = seq_to_sym_count.get(seq_tuple, 0)
                seq_to_sym_count[seq_tuple] = sym_id_val + 1
                per_chain_sym.append(sym_id_val)
                chain_origin_map.append(chain_origin_idx)
                chain_id_next += 1
            chain_origin_idx += 1

        if not entries:
            return None
        if only_chain_origin is None and self.single_chain_only and chain_id_next != 1:
            return None

        # ESM precompute may intentionally cap very long chains (default 2048).
        # When PLM features are requested, only sample residues whose chain-local
        # index has a corresponding ESM row. Otherwise a random tail crop from a
        # very long chain would make the whole batch lose PLM conditioning.
        if self.esm_dir is not None and path is not None:
            esm_lengths: dict[int, int] = {}
            filtered_entries = []
            for ri, loc, seq_num, cid, origin in entries:
                if origin not in esm_lengths:
                    p = self._esm_embedding_path(path.stem, origin, per_chain_sequences[cid])
                    if p is None:
                        return None
                    try:
                        esm_lengths[origin] = int(np.load(p, mmap_mode="r").shape[0])
                    except Exception:
                        return None
                if loc < esm_lengths[origin]:
                    filtered_entries.append((ri, loc, seq_num, cid, origin))
            entries = filtered_entries
            if not entries:
                return None

        # Random contiguous crop over the flat concatenation
        start = 0
        if len(entries) > self.max_length:
            start = int(torch.randint(0, len(entries) - self.max_length, (1,)).item())
            entries = entries[start : start + self.max_length]
        L = len(entries)
        A = MAX_ATOMS_PER_RES

        res_type = torch.zeros(L, dtype=torch.long)
        atom_type = torch.full((L, A), ATOM_NAME_TO_ID["PAD"], dtype=torch.long)
        pair_type = torch.full((L, A), PAIR_PAD_ID, dtype=torch.long)
        coords = torch.zeros(L, A, 3, dtype=torch.float32)
        atom_mask = torch.zeros(L, A, dtype=torch.bool)
        observed_mask = torch.zeros(L, A, dtype=torch.bool)
        res_seq_nums = torch.zeros(L, dtype=torch.long)
        chain_id_t = torch.zeros(L, dtype=torch.long)
        entity_id_t = torch.zeros(L, dtype=torch.long)
        sym_id_t = torch.zeros(L, dtype=torch.long)
        is_nterm_t = torch.zeros(L, dtype=torch.bool)
        is_cterm_t = torch.zeros(L, dtype=torch.bool)

        for i, (ri, loc, seq_num, cid, _origin) in enumerate(entries):
            res = residues[ri]
            res_name = str(res["name"])
            res_type[i] = AA_TO_ID.get(res_name, AA_TO_ID["UNK"])
            res_seq_nums[i] = (
                seq_num  # original residue number if available, else chain-local index
            )
            chain_id_t[i] = cid
            entity_id_t[i] = per_chain_entity[cid]
            sym_id_t[i] = per_chain_sym[cid]
            # Terminus refers to the ORIGINAL chain (not the crop):
            # loc == 0 → N-terminus; loc == len(kept)-1 → C-terminus.
            is_nterm_t[i] = loc == 0
            is_cterm_t[i] = loc == per_chain_counts[cid] - 1

            slot_map = RESIDUE_ATOM_TO_SLOT.get(res_name, RESIDUE_ATOM_TO_SLOT["UNK"])
            canon_names = RESIDUE_ATOMS.get(res_name, [])
            a_start = int(res["atom_idx"])
            a_num = int(res["atom_num"])
            for j in range(min(a_num, len(canon_names))):
                atom_name = canon_names[j]
                if atom_name not in slot_map:
                    continue
                slot = slot_map[atom_name]
                if slot >= A:
                    continue
                a = atoms[a_start + j]
                atom_type[i, slot] = ATOM_NAME_TO_ID.get(atom_name, ATOM_NAME_TO_ID["PAD"])
                pair_type[i, slot] = PAIR_TO_ID.get((res_name, atom_name), PAIR_PAD_ID)
                coords[i, slot] = torch.tensor(a["coords"], dtype=torch.float32)
                atom_mask[i, slot] = True
                observed_mask[i, slot] = bool(a["is_present"])

        # Filter low-observation structures
        n_obs = observed_mask.sum().item()
        n_atoms = atom_mask.sum().item()
        if n_atoms > 0 and n_obs / n_atoms < self.min_obs_ratio:
            return None

        # PLM embeddings: prefer one sequence-addressed file, with legacy
        # `{stem}_ch{origin}.npy` fallback, then reassemble the selected crop.
        esm = None
        if self.esm_dir is not None and path is not None:
            per_chain_cache: dict[int, np.ndarray] = {}
            ok = True
            for _ri, loc, _seq_num, cid, origin in entries:
                if origin not in per_chain_cache:
                    p = self._esm_embedding_path(path.stem, origin, per_chain_sequences[cid])
                    if p is None:
                        ok = False
                        break
                    try:
                        per_chain_cache[origin] = np.load(p, mmap_mode="r")
                    except Exception:
                        ok = False
                        break
                arr = per_chain_cache[origin]
                if loc >= arr.shape[0]:
                    ok = False
                    break
            if ok:
                esm = _gather_esm_rows(entries, per_chain_cache)

        return ProteinExample(
            res_type=res_type,
            atom_type=atom_type,
            pair_type=pair_type,
            coords=coords,
            atom_mask=atom_mask,
            observed_mask=observed_mask,
            res_seq_nums=res_seq_nums,
            seq_len=L,
            chain_id=chain_id_t,
            entity_id=entity_id_t,
            sym_id=sym_id_t,
            is_nterm=is_nterm_t,
            is_cterm=is_cterm_t,
            esm=esm,
        )
