"""Collation and batching for protein data (flow-matching corruption)."""

import math
from typing import Optional

import torch

from mambafold.data.constants import CA_ATOM_ID, MAX_ATOMS_PER_RES, PAIR_PAD_ID
from mambafold.data.transforms import center_and_scale, flow_corrupt, random_se3_augment
from mambafold.data.types import ProteinBatch, ProteinExample


class ProteinCollator:
    """Collate ProteinExamples into a padded ProteinBatch with FM corruption.

    Each batch element gets:
        x_t = t · x_clean + (1-t) · ε,   t ~ the configured schedule
    """

    def __init__(
        self,
        augment: bool = True,
        copies_per_protein: int = 1,
        t_schedule: str = "uniform",
        t_uniform_weight: float = 0.02,
        max_length: Optional[int] = None,
        length_bin: int = 0,
    ):
        if copies_per_protein < 1:
            raise ValueError("copies_per_protein must be positive")
        self.augment = augment
        self.copies_per_protein = int(copies_per_protein)
        self.t_schedule = t_schedule
        self.t_uniform_weight = t_uniform_weight
        self.max_length = max_length
        # length_bin > 0: pad to the next multiple of `length_bin` above the
        # batch's longest sequence (dynamic padding), instead of always to
        # max_length. Bounds the set of distinct shapes (e.g. 128 → 128/256/
        # 384/512) so kernels don't re-JIT per batch, while cutting padding
        # waste from ~67% (fixed 512, median L≈208) to ~12%. Pairs with the
        # length-bucketed batch sampler so batch_max ≈ each sequence length.
        self.length_bin = length_bin

    def __call__(self, examples: list[ProteinExample]) -> ProteinBatch | None:
        examples = [e for e in examples if e is not None]
        if len(examples) == 0:
            return None

        # `copies_per_protein` is SimpleFold's `multiplicity`: the same protein
        # enters the step several times, each copy with its own rotation,
        # translation, time and noise. Everything else about the copies —
        # sequence, atom typing, masks, chain indexing, the PLM embedding — is
        # identical, so it is written once per protein rather than once per row.
        copies = self.copies_per_protein
        base = [center_and_scale(ex) for ex in examples]
        processed = [
            random_se3_augment(ex) if self.augment else ex
            for ex in base
            for _ in range(copies)
        ]

        B = len(processed)
        batch_max = max(ex.seq_len for ex in processed)
        if self.length_bin and self.length_bin > 0:
            # Dynamic padding rounded up to a `length_bin` multiple.
            max_L = math.ceil(batch_max / self.length_bin) * self.length_bin
            if self.max_length is not None:
                max_L = min(max_L, self.max_length)
        else:
            max_L = self.max_length if self.max_length is not None else batch_max
        A = MAX_ATOMS_PER_RES

        res_type = torch.zeros(B, max_L, dtype=torch.long)
        res_seq_nums = torch.zeros(B, max_L, dtype=torch.long)
        atom_type = torch.zeros(B, max_L, A, dtype=torch.long)
        pair_type = torch.full((B, max_L, A), PAIR_PAD_ID, dtype=torch.long)
        res_mask = torch.zeros(B, max_L, dtype=torch.bool)
        atom_mask = torch.zeros(B, max_L, A, dtype=torch.bool)
        valid_mask = torch.zeros(B, max_L, A, dtype=torch.bool)
        ca_mask = torch.zeros(B, max_L, dtype=torch.bool)
        chain_id = torch.zeros(B, max_L, dtype=torch.long)
        entity_id = torch.zeros(B, max_L, dtype=torch.long)
        sym_id = torch.zeros(B, max_L, dtype=torch.long)
        is_nterm = torch.zeros(B, max_L, dtype=torch.bool)
        is_cterm = torch.zeros(B, max_L, dtype=torch.bool)
        x_clean = torch.zeros(B, max_L, A, 3)
        x_t = torch.zeros(B, max_L, A, 3)
        eps = torch.zeros(B, max_L, A, 3)
        t = torch.zeros(B, 1, 1, 1)

        # Copy-invariant fields: one slice assignment per protein, broadcast
        # across that protein's block of rows.
        for j, ex in enumerate(base):
            lo, hi = j * copies, (j + 1) * copies
            L = ex.seq_len
            res_type[lo:hi, :L] = ex.res_type
            res_seq_nums[lo:hi, :L] = ex.res_seq_nums
            atom_type[lo:hi, :L] = ex.atom_type
            pair_type[lo:hi, :L] = ex.pair_type
            res_mask[lo:hi, :L] = True
            atom_mask[lo:hi, :L] = ex.atom_mask
            valid_mask[lo:hi, :L] = ex.atom_mask & ex.observed_mask
            ca_mask[lo:hi, :L] = ex.atom_mask[:, CA_ATOM_ID] & ex.observed_mask[:, CA_ATOM_ID]
            chain_id[lo:hi, :L] = ex.chain_id
            entity_id[lo:hi, :L] = ex.entity_id
            sym_id[lo:hi, :L] = ex.sym_id
            is_nterm[lo:hi, :L] = ex.is_nterm
            is_cterm[lo:hi, :L] = ex.is_cterm

        # Per-copy fields: the augmented coordinates and an independent draw of
        # (t, noise) for each.
        for i, ex in enumerate(processed):
            L = ex.seq_len
            x_clean[i, :L] = ex.coords

            # `atom_mask`, not `atom_mask & observed_mask`. SimpleFold draws
            # `torch.randn_like(coords)` and interpolates with no resolved-atom
            # masking at all; only the loss takes `atom_resolved_mask`. Masking
            # here instead left every unresolved slot at exactly zero — a
            # constant that no real atom takes and that never occurs at
            # inference, where each atom starts from noise. Padding slots stay
            # zero either way, which is what `atom_mask` is for.
            xt, ep, ti = flow_corrupt(
                ex.coords,
                ex.atom_mask,
                self.t_schedule,
                self.t_uniform_weight,
            )
            x_t[i, :L] = xt
            eps[i, :L] = ep
            t[i, 0, 0, 0] = ti

        # ESM embeddings from pre-computed dataset
        esm = None
        esm_list = [ex.esm for ex in base]
        if all(e is not None and e.shape[0] > 0 for e in esm_list):
            d_esm = esm_list[0].shape[-1]
            # Preserve the cache dtype through collation and host-to-device
            # transfer.  ESMC caches are float16 and the model explicitly casts
            # them to its compute dtype before LayerNorm/projection, so an
            # intermediate float32 expansion only doubles pinned memory and
            # PCIe traffic.
            esm_dtype = esm_list[0].dtype
            for embedding in esm_list[1:]:
                esm_dtype = torch.promote_types(esm_dtype, embedding.dtype)
            # Every copy of a protein carries the same embedding, so when the
            # whole batch is one protein the tensor is emitted with a single
            # row and the model broadcasts it. At d_plm 2560 and L 1024 that is
            # the largest tensor in the batch; sending 16 identical copies over
            # PCIe, and projecting all 16 to d_plm_proj, is pure waste.
            rows = 1 if len(base) == 1 and B > 1 else B
            esm = torch.zeros(rows, max_L, d_esm, dtype=esm_dtype)
            if rows == 1:
                ex = base[0]
                n = min(ex.seq_len, ex.esm.shape[0], max_L)
                esm[0, :n] = ex.esm[:n]
            else:
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
            chain_id=chain_id,
            entity_id=entity_id,
            sym_id=sym_id,
            is_nterm=is_nterm,
            is_cterm=is_cterm,
            x_clean=x_clean,
            x_t=x_t,
            eps=eps,
            t=t,
            esm=esm,
        )
