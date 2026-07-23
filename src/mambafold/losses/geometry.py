"""Geometric structure-aware losses: bond length + steric clash.

Coordinates are in normalized units (Å ÷ COORD_SCALE=10). Ideal bond
lengths are pre-divided by the scale so we compare in the same space.

Bond coverage is deliberately limited to the backbone plus Cβ first-shell:
N-CA, CA-C, C-O, CA-CB (within residue) and C(i)-N(i+1) (peptide bond).
These exist in every standard residue (except GLY which has no CB) and
catch the most structure-breaking errors without a full atom14 bond table.
"""

import torch
from torch import Tensor

from mambafold.data.constants import COORD_SCALE

# Slot indices follow the canonical atom14 ordering used in
# mambafold.data.constants.RESIDUE_ATOMS (N, CA, C, O, CB, ...).
_N, _CA, _C, _O, _CB = 0, 1, 2, 3, 4

# Ideal bond lengths in Å (Engh & Huber 2001). Pre-scaled to normalized units.
_IDEAL_A = {
    "N_CA": 1.458,
    "CA_C": 1.525,
    "C_O": 1.229,
    "CA_CB": 1.530,
    "C_N": 1.329,  # peptide bond
}
IDEAL = {k: v / COORD_SCALE for k, v in _IDEAL_A.items()}

# GLY residue type id (no CB). Imported lazily to avoid circular init.
_GLY_ID = None


def _gly_id() -> int:
    global _GLY_ID
    if _GLY_ID is None:
        from mambafold.data.constants import AA_TO_ID

        _GLY_ID = AA_TO_ID["GLY"]
    return _GLY_ID


def bond_length_loss(
    pred_coords: Tensor,  # [B, L, A, 3] predicted (or reconstructed) coords
    res_type: Tensor,  # [B, L]    residue type IDs
    atom_mask: Tensor,  # [B, L, A] valid-atom mask
    res_mask: Tensor,  # [B, L]    valid-residue mask
    chain_id: Tensor | None = None,  # [B, L] optional chain IDs
    res_seq_nums: Tensor | None = None,  # [B, L] optional chain-local residue indices
    true_coords: Tensor | None = None,  # [B, L, A, 3] optional GT coordinates
) -> Tensor:
    """Huber-style bond length deviation in normalized units.

    Sums backbone bonds (N-CA, CA-C, C-O), the CA-CB bond for non-GLY,
    and the peptide C(i)-N(i+1) bond for adjacent valid residues. When
    `chain_id`/`res_seq_nums` are provided, peptide bonds are only applied
    within the same chain and across consecutive residue numbers. When
    `true_coords` is provided, peptide bonds are also gated by the GT C-N
    distance, which catches missing-residue gaps even if residue indices were
    re-numbered during preprocessing.
    Normalization: total deviation divided by number of valid bonds.
    """
    B, L, A, _ = pred_coords.shape
    losses = []
    counts = []

    def _bond(slot_i: int, slot_j: int, ideal: float, extra_mask: Tensor | None = None):
        vi = atom_mask[..., slot_i]
        vj = atom_mask[..., slot_j]
        m = (vi & vj & res_mask).to(pred_coords.dtype)
        if extra_mask is not None:
            m = m * extra_mask.to(pred_coords.dtype)
        d = torch.linalg.norm(pred_coords[..., slot_i, :] - pred_coords[..., slot_j, :], dim=-1)
        err = (d - ideal).abs()
        losses.append((err * m).sum())
        counts.append(m.sum())

    # Within-residue backbone bonds (all residues)
    _bond(_N, _CA, IDEAL["N_CA"])
    _bond(_CA, _C, IDEAL["CA_C"])
    _bond(_C, _O, IDEAL["C_O"])

    # CA-CB (skip GLY)
    non_gly = (res_type != _gly_id()) & res_mask
    _bond(_CA, _CB, IDEAL["CA_CB"], extra_mask=non_gly)

    # Peptide bond C(i) - N(i+1)
    if L >= 2:
        c_i = pred_coords[:, :-1, _C, :]
        n_j = pred_coords[:, 1:, _N, :]
        mi = atom_mask[:, :-1, _C] & res_mask[:, :-1]
        mj = atom_mask[:, 1:, _N] & res_mask[:, 1:]
        peptide = mi & mj
        if chain_id is not None:
            peptide = peptide & (chain_id[:, :-1] == chain_id[:, 1:])
        if res_seq_nums is not None:
            peptide = peptide & ((res_seq_nums[:, 1:] - res_seq_nums[:, :-1]) == 1)
        if true_coords is not None:
            true_cn = torch.linalg.norm(
                true_coords[:, :-1, _C, :] - true_coords[:, 1:, _N, :], dim=-1
            )
            peptide = peptide & (true_cn < (IDEAL["C_N"] + 0.06))
        m = peptide.to(pred_coords.dtype)
        d = torch.linalg.norm(c_i - n_j, dim=-1)
        err = (d - IDEAL["C_N"]).abs()
        losses.append((err * m).sum())
        counts.append(m.sum())

    total_err = torch.stack(losses).sum()
    total_cnt = torch.stack(counts).sum().clamp(min=1)
    return total_err / total_cnt


def ca_clash_loss(
    pred_coords: Tensor,  # [B, L, A, 3]
    res_mask: Tensor,  # [B, L]
    chain_id: Tensor | None = None,  # [B, L] int; None → treat as single chain
    min_dist_A: float = 3.8,
    seq_sep: int = 2,
) -> Tensor:
    """Cα-Cα steric clash penalty (ReLU(min_dist − d))².

    Intra-chain: exclude sequence-adjacent pairs via `seq_sep`.
    Inter-chain: every pair is a valid clash candidate (no adjacency exception),
    because different chains have no sequence neighbour notion.
    """
    min_d = min_dist_A / COORD_SCALE
    ca = pred_coords[:, :, _CA, :]  # [B, L, 3]
    B, L, _ = ca.shape
    d = torch.linalg.norm(ca.unsqueeze(2) - ca.unsqueeze(1), dim=-1)  # [B, L, L]

    idx = torch.arange(L, device=ca.device)
    seq_far = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() > seq_sep  # [L, L]

    if chain_id is None:
        pair = res_mask.unsqueeze(2) & res_mask.unsqueeze(1) & seq_far.unsqueeze(0)
    else:
        same_chain = chain_id.unsqueeze(2) == chain_id.unsqueeze(1)  # [B, L, L]
        intra = same_chain & seq_far.unsqueeze(0)
        inter = ~same_chain  # all cross-chain pairs
        pair = res_mask.unsqueeze(2) & res_mask.unsqueeze(1) & (intra | inter)

    m = pair.to(d.dtype)
    violation = torch.relu(min_d - d).pow(2)
    return (violation * m).sum() / m.sum().clamp(min=1)


def all_atom_clash_loss(
    pred_coords: Tensor,
    valid_mask: Tensor,
    res_mask: Tensor,
    chain_id: Tensor | None = None,
    min_dist_A: float = 1.5,
    max_atoms: int = 2048,
) -> Tensor:
    """Sampled non-bonded heavy-atom clash penalty.

    This is a lightweight AF-style steric term: flatten valid atom slots,
    exclude pairs from the same residue and adjacent residues in the same
    chain, then penalize distances below `min_dist_A`.
    """
    min_d = min_dist_A / COORD_SCALE
    losses = []
    B, L, A, _ = pred_coords.shape
    base_res = torch.arange(L, device=pred_coords.device).view(L, 1).expand(L, A).reshape(-1)
    for b in range(B):
        flat_mask = (valid_mask[b] & res_mask[b].unsqueeze(-1)).reshape(-1)
        idx = flat_mask.nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() < 2:
            continue
        if idx.numel() > max_atoms:
            take = torch.linspace(0, idx.numel() - 1, max_atoms, device=idx.device).long()
            idx = idx[take]

        xyz = pred_coords[b].reshape(-1, 3).index_select(0, idx)
        res_i = base_res.index_select(0, idx)
        d = torch.cdist(xyz, xyz)

        pair = ~torch.eye(idx.numel(), dtype=torch.bool, device=idx.device)
        same_res = res_i[:, None] == res_i[None, :]
        pair = pair & ~same_res
        if chain_id is not None:
            ch_flat = chain_id[b].view(L, 1).expand(L, A).reshape(-1).index_select(0, idx)
            same_chain = ch_flat[:, None] == ch_flat[None, :]
            adjacent = (res_i[:, None] - res_i[None, :]).abs() <= 1
            pair = pair & ~(same_chain & adjacent)

        m = pair.to(d.dtype)
        if m.sum() < 1:
            continue
        violation = torch.relu(min_d - d).pow(2)
        losses.append((violation * m).sum() / m.sum().clamp(min=1))

    if not losses:
        return pred_coords.new_zeros(())
    return torch.stack(losses).mean()
