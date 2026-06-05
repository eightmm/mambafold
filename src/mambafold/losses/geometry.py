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
    "N_CA":  1.458,
    "CA_C":  1.525,
    "C_O":   1.229,
    "CA_CB": 1.530,
    "C_N":   1.329,  # peptide bond
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
    pred_coords: Tensor,      # [B, L, A, 3] predicted (or reconstructed) coords
    res_type: Tensor,         # [B, L]    residue type IDs
    atom_mask: Tensor,        # [B, L, A] valid-atom mask
    res_mask: Tensor,         # [B, L]    valid-residue mask
) -> Tensor:
    """Huber-style bond length deviation in normalized units.

    Sums backbone bonds (N-CA, CA-C, C-O), the CA-CB bond for non-GLY,
    and the peptide C(i)-N(i+1) bond for adjacent valid residues.
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
        d = torch.linalg.norm(
            pred_coords[..., slot_i, :] - pred_coords[..., slot_j, :], dim=-1
        )
        err = (d - ideal).abs()
        losses.append((err * m).sum())
        counts.append(m.sum())

    # Within-residue backbone bonds (all residues)
    _bond(_N,  _CA, IDEAL["N_CA"])
    _bond(_CA, _C,  IDEAL["CA_C"])
    _bond(_C,  _O,  IDEAL["C_O"])

    # CA-CB (skip GLY)
    non_gly = (res_type != _gly_id()) & res_mask
    _bond(_CA, _CB, IDEAL["CA_CB"], extra_mask=non_gly)

    # Peptide bond C(i) - N(i+1)
    if L >= 2:
        c_i  = pred_coords[:, :-1, _C,  :]
        n_j  = pred_coords[:,  1:, _N,  :]
        mi   = atom_mask[:, :-1, _C]  & res_mask[:, :-1]
        mj   = atom_mask[:,  1:, _N]  & res_mask[:,  1:]
        m    = (mi & mj).to(pred_coords.dtype)
        d    = torch.linalg.norm(c_i - n_j, dim=-1)
        err  = (d - IDEAL["C_N"]).abs()
        losses.append((err * m).sum())
        counts.append(m.sum())

    total_err = torch.stack(losses).sum()
    total_cnt = torch.stack(counts).sum().clamp(min=1)
    return total_err / total_cnt


def ca_clash_loss(
    pred_coords: Tensor,           # [B, L, A, 3]
    res_mask: Tensor,              # [B, L]
    chain_id: Tensor | None = None,# [B, L] int; None → treat as single chain
    min_dist_A: float = 3.8,
    seq_sep: int = 2,
) -> Tensor:
    """Cα-Cα steric clash penalty (ReLU(min_dist − d))².

    Intra-chain: exclude sequence-adjacent pairs via `seq_sep`.
    Inter-chain: every pair is a valid clash candidate (no adjacency exception),
    because different chains have no sequence neighbour notion.
    """
    min_d = min_dist_A / COORD_SCALE
    ca = pred_coords[:, :, _CA, :]                                    # [B, L, 3]
    B, L, _ = ca.shape
    d = torch.linalg.norm(ca.unsqueeze(2) - ca.unsqueeze(1), dim=-1)  # [B, L, L]

    idx = torch.arange(L, device=ca.device)
    seq_far = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() > seq_sep   # [L, L]

    if chain_id is None:
        pair = res_mask.unsqueeze(2) & res_mask.unsqueeze(1) & seq_far.unsqueeze(0)
    else:
        same_chain = chain_id.unsqueeze(2) == chain_id.unsqueeze(1)   # [B, L, L]
        intra = same_chain & seq_far.unsqueeze(0)
        inter = ~same_chain                                           # all cross-chain pairs
        pair = res_mask.unsqueeze(2) & res_mask.unsqueeze(1) & (intra | inter)

    m = pair.to(d.dtype)
    violation = torch.relu(min_d - d).pow(2)
    return (violation * m).sum() / m.sum().clamp(min=1)
