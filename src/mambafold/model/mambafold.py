"""MambaFold end-to-end model with Mamba-3 cores and ESM conditioning."""

import torch
import torch.nn as nn
from torch import Tensor

from mambafold.data.constants import MAX_ATOMS_PER_RES
from mambafold.data.types import ProteinBatch
from mambafold.model.embeddings import AtomFeatureEmbedder, SequenceFourierEmbedder
from mambafold.model.bimamba3 import MambaStack


# ── Grouping ───────────────────────────────────────────────────────────────

def group_atoms_to_residues(atom_tok: Tensor, atom_mask: Tensor) -> Tensor:
    """Masked average pooling over the atom dimension.

    Args:
        atom_tok (Tensor): Per-atom token features of shape [B, L, A, D].
        atom_mask (Tensor): Boolean or float atom validity mask of shape [B, L, A].
            Padding atoms (mask == 0) are excluded from the average.

    Returns:
        Tensor: Per-residue token features of shape [B, L, D], computed as the
            masked mean over the A (atom) dimension. Residues with no valid atoms
            return zero vectors (denominator clamped to 1).
    """
    mask_f = atom_mask.unsqueeze(-1).to(atom_tok.dtype)
    return (atom_tok * mask_f).sum(dim=2) / mask_f.sum(dim=2).clamp(min=1)


class ResidueToAtomBroadcast(nn.Module):
    """Project residue tokens and broadcast to atom positions."""

    def __init__(self, d_res: int, d_atom: int):
        """
        Args:
            d_res (int): Residue token dimension. Input shape [B, L, d_res].
            d_atom (int): Atom token dimension. Output shape [B, L, A, d_atom].
                Also the dimension of the per-slot gate embedding [A, d_atom].
        """
        super().__init__()
        self.proj = nn.Linear(d_res, d_atom)
        self.slot_gate = nn.Embedding(MAX_ATOMS_PER_RES, d_atom)
        nn.init.zeros_(self.slot_gate.weight)

    def forward(self, res_tok: Tensor, atom_mask: Tensor) -> Tensor:
        """
        Args:
            res_tok (Tensor): Residue-level token features of shape [B, L, d_res].
            atom_mask (Tensor): Boolean or float atom validity mask of shape [B, L, A].
                Padding atom positions (mask == 0) are zeroed in the output.

        Returns:
            Tensor: Residue features broadcast to atom positions, shape [B, L, A, d_atom].
                Each atom slot is gated by a learnable sigmoid gate of shape [A, d_atom]
                (initialized to 0 so the gate starts near 0.5).
        """
        A = atom_mask.shape[2]
        gate = torch.sigmoid(self.slot_gate(torch.arange(A, device=res_tok.device)))
        broadcast = self.proj(res_tok).unsqueeze(2) * gate.unsqueeze(0).unsqueeze(0)
        return broadcast * atom_mask.unsqueeze(-1).to(res_tok.dtype)


# ── Model ──────────────────────────────────────────────────────────────────

class MambaFoldEqM(nn.Module):
    """Protein folding model with Mamba-3 cores and EqM training.

    Architecture: Atom Encoder → Grouping → Residue Trunk → Ungrouping → Atom Decoder
    """

    def __init__(
        self,
        d_atom: int = 256,
        d_res: int = 768,
        d_plm: int = 1536,
        n_atom_enc: int = 4,
        n_trunk: int = 24,
        n_atom_dec: int = 4,
        use_plm: bool = True,
        d_res_pos: int = 0,
        d_atom_slot: int = 0,
        # Atom encoder/decoder SSM config
        atom_d_state: int = 64,
        atom_mimo_rank: int = 4,
        atom_expand: int = 2,
        atom_headdim: int = 64,
        atom_bidirectional: bool = True,
        # Residue trunk SSM config
        d_state: int = 64,
        mimo_rank: int = 4,
        expand: int = 2,
        headdim: int = 64,
        trunk_bidirectional: bool = True,
    ):
        """
        Args:
            d_atom (int): Atom token dimension throughout encoder/decoder.
                Atom tensors have shape [B, L, A, d_atom] or [B*L, A, d_atom].
                Default: 256.
            d_res (int): Residue token dimension in the trunk.
                Residue tensors have shape [B, L, d_res]. Default: 768.
            d_plm (int): ESM3 embedding dimension. Must match the pre-computed
                embeddings loaded via esm_dir (ESM3-open = 1536). Default: 1536.
            n_atom_enc (int): Number of MambaStack layers in the atom encoder.
                Default: 4.
            n_trunk (int): Number of MambaStack layers in the residue trunk.
                Default: 24.
            n_atom_dec (int): Number of MambaStack layers in the atom decoder.
                Default: 4.
            use_plm (bool): If True, concatenate PLM embeddings [B, L, d_plm]
                into the trunk input. Requires batch.esm to be non-None. Default: True.
            d_res_pos (int): Dimension of the sequence-position Fourier embedding
                [B, L, d_res_pos] broadcast into atom and trunk features.
                Set to 0 to disable. Default: 0.
            d_atom_slot (int): Dimension of per-slot atom-index embedding
                [A, d_atom_slot]. Set to 0 to disable. Default: 0.
            atom_d_state (int): SSM state size for atom encoder/decoder blocks.
                Default: 32.
            atom_mimo_rank (int): MIMO rank for atom encoder/decoder blocks.
                Default: 2.
            atom_expand (int): Inner dimension multiplier for atom encoder/decoder.
                Default: 2.
            atom_headdim (int): Head dimension for atom encoder/decoder SSMs.
                Default: 64.
            atom_bidirectional (bool): If True, atom encoder/decoder use
                BiMamba3Block; otherwise causal Mamba3Block. Default: True.
            d_state (int): SSM state size for residue trunk blocks. Default: 64.
            mimo_rank (int): MIMO rank for residue trunk blocks. Default: 4.
            expand (int): Inner dimension multiplier for residue trunk. Default: 2.
            headdim (int): Head dimension for residue trunk SSMs. Default: 64.
            trunk_bidirectional (bool): If True, trunk uses BiMamba3Block;
                otherwise causal Mamba3Block. Default: True.
        """
        super().__init__()

        self.res_pos_embed = SequenceFourierEmbedder(d_out=d_res_pos) if d_res_pos > 0 else None
        self.atom_embed = AtomFeatureEmbedder(d_atom=d_atom, d_res_pos=d_res_pos, d_atom_slot=d_atom_slot)
        self.atom_encoder = MambaStack(d_atom, n_atom_enc, d_state=atom_d_state, mimo_rank=atom_mimo_rank,
                                       expand=atom_expand, headdim=atom_headdim, bidirectional=atom_bidirectional)

        self.use_plm = use_plm
        self.d_plm = d_plm
        if use_plm:
            self.plm_proj = nn.Linear(d_plm, d_plm)
            trunk_in_dim = d_atom + d_plm + d_res_pos + 2
        else:
            self.plm_proj = None
            trunk_in_dim = d_atom + d_res_pos + 2

        self.trunk_proj = nn.Linear(trunk_in_dim, d_res)
        self.residue_trunk = MambaStack(d_res, n_trunk, d_state=d_state, mimo_rank=mimo_rank,
                                        expand=expand, headdim=headdim, bidirectional=trunk_bidirectional)
        self.res_to_atom = ResidueToAtomBroadcast(d_res, d_atom)

        # Atom decoder: per-residue BiMamba3 stack + gradient head
        self.atom_decoder = MambaStack(d_atom, n_atom_dec, d_state=atom_d_state, mimo_rank=atom_mimo_rank,
                                       expand=atom_expand, headdim=atom_headdim, bidirectional=atom_bidirectional)
        self.grad_head = nn.Sequential(
            nn.LayerNorm(d_atom), nn.Linear(d_atom, d_atom // 2), nn.GELU(), nn.Linear(d_atom // 2, 3),
        )


    def forward(self, batch: ProteinBatch) -> Tensor:
        """
        Args:
            batch (ProteinBatch): Collated protein batch containing:
                - pair_type   [B, L, A]      residue-atom pair type indices
                - x_gamma     [B, L, A, 3]   noisy atom coordinates (x + gamma * noise)
                - atom_mask   [B, L, A]       valid atom mask
                - res_mask    [B, L]          valid residue mask
                - res_seq_nums[B, L]          integer residue sequence numbers
                - res_type    [B, L]          residue type indices (for PLM)
                - valid_mask  [B, L, A]       observed (non-missing) atom mask
                - gamma       [B, 1, 1, 1]    noise level scalar
                - esm         [B, L, d_plm] | None  pre-computed PLM embeddings

        Returns:
            Tensor: Predicted EqM gradient of shape [B, L, A, 3].
                Padding atom positions (atom_mask == 0) are zeroed.
                Pipeline: atom embed [B, L, A, d_atom]
                → atom encoder [B*L, A, d_atom]
                → group [B, L, d_atom]
                → trunk proj + residue trunk [B, L, d_res]
                → res-to-atom broadcast [B, L, A, d_atom]
                → atom decoder [B*L, A, d_atom]
                → grad head [B, L, A, 3].
        """
        B = batch.batch_size
        L = batch.max_len
        A = MAX_ATOMS_PER_RES
        res_pos = self.res_pos_embed(batch.res_seq_nums, batch.res_mask) if self.res_pos_embed is not None else None

        # 1. Atom embedding + encoder
        atom0 = self.atom_embed(batch.pair_type, batch.x_gamma, batch.atom_mask, res_pos)  # [B, L, A, d_atom]
        atom = self.atom_encoder(atom0.reshape(B * L, A, -1),
                                 batch.atom_mask.reshape(B * L, A)).reshape(B, L, A, -1)

        # 2. Grouping: atoms → residues
        res0 = group_atoms_to_residues(atom, batch.atom_mask)  # [B, L, d_atom]

        # 3. PLM conditioning (zero-fill if ESM missing for some examples)
        plm = None
        if self.use_plm:
            esm = batch.esm if batch.esm is not None else torch.zeros(
                B, L, self.d_plm, device=batch.res_type.device, dtype=res0.dtype)
            plm = self.plm_proj(esm)

        # 4. Trunk
        obs_frac = (batch.valid_mask.float().sum(dim=-1) /
                    batch.atom_mask.float().sum(dim=-1).clamp(min=1)).unsqueeze(-1)  # [B, L, 1]
        gamma_feat = batch.gamma.squeeze(-1).squeeze(-1).expand(-1, L).unsqueeze(-1)  # [B, L, 1]
        trunk_parts = [res0, obs_frac, gamma_feat]
        if res_pos is not None:
            trunk_parts.append(res_pos)
        if plm is not None:
            trunk_parts.append(plm)
        res = self.residue_trunk(self.trunk_proj(torch.cat(trunk_parts, dim=-1)), batch.res_mask)

        # 5. Ungrouping + atom decoder
        dec_in = atom + self.res_to_atom(res, batch.atom_mask)  # [B, L, A, d_atom]
        dec_out = self.atom_decoder(dec_in.reshape(B * L, A, -1),
                                    batch.atom_mask.reshape(B * L, A)).reshape(B, L, A, -1)
        return self.grad_head(dec_out) * batch.atom_mask.unsqueeze(-1).to(dec_out.dtype)
