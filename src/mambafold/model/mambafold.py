"""MambaFold end-to-end model with Mamba-3 cores and ESM conditioning."""

import torch
import torch.nn as nn
from torch import Tensor

from mambafold.data.constants import MAX_ATOMS_PER_RES
from mambafold.data.esm import EvolutionaryScalePLM
from mambafold.data.types import ProteinBatch
from mambafold.model.atom_decoder import AtomDecoder
from mambafold.model.embeddings import AtomFeatureEmbedder, SequenceFourierEmbedder
from mambafold.model.grouping import ResidueToAtomBroadcast, group_atoms_to_residues
from mambafold.model.ssm.bimamba3 import MambaStack


class MambaFoldEqM(nn.Module):
    """Protein folding model with Mamba-3 cores and EqM training.

    Architecture: Atom Encoder → Grouping → Residue Trunk → Ungrouping → Atom Decoder

    EqM: model takes only x_γ as input (time-unconditional — γ is NOT given to the model).
    """

    def __init__(
        self,
        d_atom: int = 256,
        d_res: int = 768,
        d_plm: int = 1024,
        n_atom_enc: int = 4,
        n_trunk: int = 24,
        n_atom_dec: int = 4,
        use_plm: bool = True,
        plm_mode: str = "blend",
        esm3_model_name: str = "esm3-open",
        esmc_model_name: str = "esmc_600m",
        d_res_pos: int = 0,
        d_atom_slot: int = 0,
        d_local_frame: int = 0,
        # Atom encoder/decoder SSM config
        atom_d_state: int = 32,
        atom_mimo_rank: int = 2,
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
        super().__init__()

        self.res_pos_embed = SequenceFourierEmbedder(d_out=d_res_pos) if d_res_pos > 0 else None

        self.atom_embed = AtomFeatureEmbedder(
            d_atom=d_atom,
            d_res_pos=d_res_pos,
            d_atom_slot=d_atom_slot,
            d_local_frame=d_local_frame,
        )

        self.atom_encoder = MambaStack(
            d_atom, n_atom_enc,
            d_state=atom_d_state, mimo_rank=atom_mimo_rank,
            expand=atom_expand, headdim=atom_headdim,
            bidirectional=atom_bidirectional,
        )

        self.use_plm = use_plm
        if use_plm:
            self.plm = EvolutionaryScalePLM(
                d_out=d_plm, mode=plm_mode,
                esm3_model_name=esm3_model_name,
                esmc_model_name=esmc_model_name,
            )
            trunk_in_dim = d_atom + d_plm + d_res_pos + 1  # +1 for obs_frac
        else:
            self.plm = None
            trunk_in_dim = d_atom + d_res_pos + 1  # +1 for obs_frac

        self.trunk_proj = nn.Linear(trunk_in_dim, d_res)

        self.residue_trunk = MambaStack(
            d_res, n_trunk,
            d_state=d_state, mimo_rank=mimo_rank,
            expand=expand, headdim=headdim,
            bidirectional=trunk_bidirectional,
        )

        self.res_to_atom = ResidueToAtomBroadcast(d_res, d_atom)

        self.atom_decoder = AtomDecoder(
            d_atom=d_atom, n_layers=n_atom_dec,
            d_state=atom_d_state, mimo_rank=atom_mimo_rank,
            expand=atom_expand, headdim=atom_headdim,
            bidirectional=atom_bidirectional,
        )

    def forward(self, batch: ProteinBatch) -> Tensor:
        """
        Args:
            batch: ProteinBatch with x_gamma, atom_mask, res_type, gamma, etc.

        Returns: [B, L, A, 3] predicted EqM gradient
        """
        B = batch.batch_size
        L = batch.max_len
        A = MAX_ATOMS_PER_RES
        res_pos = self.res_pos_embed(batch.res_seq_nums, batch.res_mask) if self.res_pos_embed is not None else None

        # 1. Atom embedding
        # valid_mask = atom_mask & observed_mask: tells model which atoms are trustworthy
        atom0 = self.atom_embed(
            batch.res_type, res_pos, batch.atom_type, batch.pair_type,
            batch.x_gamma, batch.atom_mask, observed_mask=batch.valid_mask,
        )  # [B, L, A, d_atom]

        # 2. Atom encoder
        atom = self.atom_encoder(
            atom0.reshape(B * L, A, -1),
            batch.atom_mask.reshape(B * L, A),
        ).reshape(B, L, A, -1)  # [B, L, A, d_atom]

        # 3. Grouping: atoms → residues (masked avg pool over observed atoms only)
        # Use valid_mask so unobserved atoms (zero/noisy coords) don't pollute residue tokens
        res0 = group_atoms_to_residues(atom, batch.valid_mask)  # [B, L, d_atom]

        # 4. PLM conditioning (only if model was built with use_plm=True)
        if self.use_plm:
            if batch.esm is not None:
                plm = batch.esm
            elif self.plm is not None:
                plm = self.plm(batch.res_type, batch.res_mask)
            else:
                plm = None
        else:
            plm = None

        # Observed atom fraction: 1.0 = all atoms observed, <1.0 = some missing
        obs_count = batch.valid_mask.float().sum(dim=-1)            # [B, L]
        total_count = batch.atom_mask.float().sum(dim=-1).clamp(min=1)  # [B, L]
        obs_frac = (obs_count / total_count).unsqueeze(-1)           # [B, L, 1]

        trunk_parts = [res0, obs_frac]
        if res_pos is not None:
            trunk_parts.append(res_pos)
        if plm is not None:
            trunk_parts.append(plm)
        trunk_in = self.trunk_proj(torch.cat(trunk_parts, dim=-1))  # [B, L, d_res]

        # 5. Residue trunk (EqM: time-unconditional, no γ input)
        res = self.residue_trunk(trunk_in, batch.res_mask)  # [B, L, d_res]

        # 6. Ungrouping: residues → atoms (broadcast + skip)
        dec_in = atom + self.res_to_atom(res, batch.atom_mask)  # [B, L, A, d_atom]

        # 7. Atom decoder → gradient
        grad = self.atom_decoder(
            dec_in.reshape(B * L, A, -1),
            batch.atom_mask.reshape(B * L, A),
        ).reshape(B, L, A, 3)

        return grad * batch.atom_mask.unsqueeze(-1).to(grad.dtype)
