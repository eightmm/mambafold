"""Direct all-atom flow matching model.

Inputs:
    res_type, PLM, chain_id/entity_id/sym_id, res_seq_nums,
    is_nterm/is_cterm, noised atom-slot coordinates, t

Outputs:
    v_atom        [B, L, A, 3]   — all-atom FM velocity
    trunk_latent  [B, L, d_res]  — residue representation for downstream heads

The main path is atom → residue → atom. The Run A and latent recipes mix
intra-residue atom slots with MLP blocks and run BiMamba along backbone atom
streams before pooling and after residue-to-atom decoding. The residue trunk
is pair-free BiMamba. An intra-residue BiMamba mixer is also supported by
`atom_mixer=mamba` (see `atom_mamba.py`).

An optional latent residual compresses consecutive residue groups, mixes
them with BiMamba and restores residue-specific updates, either at the trunk
midpoint or after the full trunk. It preserves the full-resolution path and is
disabled for Run A.

There is no attention and no pair stack, and neither is reachable by config.
Both were removed rather than switched off: an O(L²) triangle-update path that
merely defaults to disabled still has to be carried, sized and reasoned about,
and a trunk that *could* be given attention cannot support the claim that a
pure SSM trunk suffices. The all-attention control trunk this once promised
was cancelled on cost, so nothing here is a controlled comparison against
attention. Confidence is trained as a separate phase against a frozen folding
model and is not part of this model.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mambafold.data.constants import AA_TO_ID, CA_ATOM_ID, MAX_ATOMS_PER_RES
from mambafold.data.types import ProteinBatch
from mambafold.model.bimamba3 import MambaStack
from mambafold.model.embeddings import (
    CoordinateFourierEmbedder,
    SequenceFourierEmbedder,
    TimeEmbedding,
)
from mambafold.model.fold.atom_mamba import AtomDecoder, AtomEncoder, FiLM
from mambafold.model.fold.latent_mamba import (
    LatentMambaResidual,
    ResidueLatentRefiner,
    SharedTrunkRefiner,
)

NUM_RES_TYPES = len(AA_TO_ID)  # 21 (20 AAs + UNK)


class MambaFoldAllAtom(nn.Module):
    """Direct all-atom flow matching model.

    Args:
        d_res: Residue token dim. Default 1024.
        n_trunk: Number of BiMamba3 layers in the trunk. Default 12.
        d_res_type: Residue-type embedding dim. Default 32.
        d_res_pos: Sequence position (chain/entity/sym + Fourier) embed dim.
        d_plm: External PLM embedding dim. Default 1536.
        d_plm_proj: Internal PLM projection width. Default 256.
        d_ca_emb: Fourier embed dim of per-atom x_t scalar coords. Default 128.
        use_plm: If True, external PLM features are required (loud error otherwise).
        # SSM
        mimo_rank: Mamba MIMO rank. Default 4.
        d_state: Mamba state dim. Default 64.
        expand: Mamba inner-dim multiplier. Default 2.
        headdim: Mamba head dim. Default 64.
        bidirectional: BiMamba (default True) vs causal Mamba.
        latent_compression: Residues per latent (4, 8 or 16); 0 disables the branch.
        d_latent: Width of the optional compressed latent stream.
        n_latent_layers: Number of independent latent BiMamba blocks.
        latent_loops: Passes through the shared latent stack at fixed FM time.
        latent_loop_mode: Plain recurrence or normalized, input-injected,
            continuously depth-conditioned recurrence.
        latent_position: "midpoint" (after n_trunk // 2 blocks) or "post"
            (after the whole trunk, directly before the atom decoder).
        latent_supervision: "final" or training-only LiFT velocity readouts.
        residue_refiner: Add an identity-initialized shared correction after
            the complete trunk. Both residue/latent stacks use d_latent width;
            pretrained trunk conditioning and normalization stay unchanged.
        shared_trunk_layers: Move a full-width suffix into the recurrent core.
            n_trunk then denotes executed full-width depth at configured K;
            the independent prelude has n_trunk - K * shared_trunk_layers blocks.
        latent_hierarchy: Keep the fine latent blocks and add two coarse blocks
            after the first n_latent_layers // 2, with initially zero residual
            restoration.
        latent_coarse_width: Coarse hierarchy width; 0 uses d_latent.
        atom_recycling: "none" preserves the original path. "reset" and "carry"
            run full atom/residue/latent passes, differing only in whether each
            pass reads the initial or preceding decoded hidden atom state.
        recycle_policy: "legacy" preserves the original recurrence. "fixed2",
            "fixed4" and "sampled16_tail2" reinject the live initial residue/PLM
            state; only the sampled policy truncates recurrent gradients.
    """

    def __init__(
        self,
        d_res: int = 1024,
        n_trunk: int = 12,
        d_res_type: int = 32,
        d_res_pos: int = 64,
        d_plm: int = 1536,
        d_plm_proj: int = 256,
        # Width the FM time embedding is carried at. SimpleFold's DiTBlock
        # projects adaLN modulation from the full trunk width
        # (`Linear(hidden_size, 6 * hidden_size)`), so matching means d_temb =
        # d_res. It multiplies through every block, so it is a first-class part
        # of the parameter budget, not a side channel.
        d_temb: int = 128,
        d_ca_emb: int = 128,
        use_plm: bool = True,
        mimo_rank: int = 4,
        d_state: int = 64,
        expand: int = 2,
        headdim: int = 64,
        bidirectional: bool = True,
        bimamba_share: bool = False,
        d_atom: int = 128,
        n_atom_layers: int = 4,
        # The atom levels get their own SSM settings. Inheriting the trunk's is
        # what made `mimo_rank: 1` — chosen so the trunk can reach d_state 128
        # on an RTX 6000 Ada — set chunk_size 64 on a 14-slot sequence, so every
        # atom-level layer processed 64 positions to use 14.
        atom_mixer: str = "mamba",
        atom_d_state: int = 64,
        atom_mimo_rank: int = 2,
        # Cross-residue mixing on the backbone atom streams. 0 disables it and
        # leaves the atom levels strictly intra-residue.
        n_atom_cross_layers: int = 1,
        n_backbone_streams: int = 5,
        self_conditioning: bool = False,
        latent_compression: int = 0,
        d_latent: int = 256,
        n_latent_layers: int = 2,
        latent_loops: int = 2,
        latent_loop_mode: str = "conditioned",
        latent_position: str = "post",
        latent_supervision: str = "final",
        residue_refiner: bool = False,
        atom_recurrence: bool = False,
        shared_trunk_layers: int = 0,
        latent_hierarchy: bool = False,
        latent_coarse_width: int = 0,
        atom_recycling: str = "none",
        recycle_policy: str = "legacy",
    ):
        super().__init__()
        if latent_loop_mode not in ("plain", "conditioned"):
            raise ValueError("latent_loop_mode must be plain or conditioned")
        if type(latent_loops) is not int or latent_loops < 1:
            raise ValueError("latent_loops must be a positive integer")
        if latent_compression not in (0, 4, 8, 16):
            raise ValueError("latent_compression must be 0, 4, 8 or 16")
        if latent_position not in ("midpoint", "post"):
            raise ValueError("latent_position must be midpoint or post")
        if latent_compression and latent_position == "midpoint" and n_trunk < 2:
            raise ValueError("latent residual requires at least two trunk blocks")
        if latent_supervision not in ("final", "lift"):
            raise ValueError("latent_supervision must be final or lift")
        if latent_supervision == "lift" and (
            not latent_compression or latent_position != "post" or latent_loop_mode != "conditioned"
        ):
            raise ValueError("LiFT requires an enabled post-trunk conditioned latent branch")
        if type(residue_refiner) is not bool:
            raise ValueError("residue_refiner must be a boolean")
        if residue_refiner and (not latent_compression or latent_position != "post"
                                or latent_loop_mode != "conditioned"):
            raise ValueError("residue_refiner requires a post conditioned latent branch")
        if type(atom_recurrence) is not bool or (atom_recurrence and not residue_refiner):
            raise ValueError("atom_recurrence requires residue_refiner")
        if (type(shared_trunk_layers) is not int or shared_trunk_layers < 0
                or shared_trunk_layers * latent_loops >= n_trunk and shared_trunk_layers):
            raise ValueError("shared trunk must leave a positive independent prelude")
        if shared_trunk_layers and (not residue_refiner or atom_recurrence):
            raise ValueError("shared trunk requires residue_refiner without atom recurrence")
        if atom_recycling not in ("none", "reset", "carry"):
            raise ValueError("atom_recycling must be none, reset or carry")
        if atom_recycling != "none" and (
                not residue_refiner or atom_recurrence
                or shared_trunk_layers <= 0 or shared_trunk_layers % 2):
            raise ValueError("atom_recycling requires residue_refiner, positive even shared "
                             "trunk layers and no atom_recurrence")
        if type(latent_hierarchy) is not bool or (latent_hierarchy and not shared_trunk_layers):
            raise ValueError("latent hierarchy requires a shared trunk")
        if latent_hierarchy and latent_compression != 4:
            raise ValueError("latent hierarchy requires r4 compression")
        if type(latent_coarse_width) is not int or latent_coarse_width < 0 or (
                latent_coarse_width and not latent_hierarchy):
            raise ValueError("latent_coarse_width must be 0 or a positive hierarchy width")
        if recycle_policy not in ("legacy", "fixed2", "fixed4", "sampled16_tail2"):
            raise ValueError("recycle_policy must be legacy, fixed2, fixed4 or sampled16_tail2")
        if recycle_policy != "legacy" and (
                atom_recycling != "carry" or latent_loops != 2
                or latent_supervision != "lift"):
            raise ValueError("recycle_policy requires atom_recycling=carry, nominal K=2 and LiFT")
        self.shared_trunk_layers = shared_trunk_layers
        # n_trunk retains the nominal executed full-width depth at training K.
        prelude_layers = n_trunk - shared_trunk_layers * latent_loops
        self.backbone_parameter_prefixes = (
            ("latent_residual.residue_stack.",) if shared_trunk_layers else ()
        )
        self.residue_refiner = residue_refiner
        self.atom_recurrence = atom_recurrence
        self.atom_recycling = atom_recycling
        self.recycle_policy = recycle_policy
        self.refinement_training_loops = None
        self.latent_position = latent_position
        self.latent_supervision = latent_supervision
        self.d_res = d_res
        self.use_plm = use_plm
        self.d_plm = d_plm
        self.self_conditioning = self_conditioning

        # ── Residue-side embedders ──────────────────────────────────────
        self.res_type_embed = nn.Embedding(NUM_RES_TYPES, d_res_type)
        self.seq_pos_embed = SequenceFourierEmbedder(d_out=d_res_pos) if d_res_pos > 0 else None
        # The noised structure enters through two embedders on two bands rather
        # than one embedder on one. A single absolute-coordinate embedding has
        # to cover both the whole-crop extent (a 1024-residue chain spans about
        # +-35 A about its centroid) and bond-scale detail (1.2-1.5 A), a range
        # of more than four octaves that no 16-band embedding spans at useful
        # resolution. Splitting them costs nothing and lets each band sit where
        # its quantity actually lives:
        #   ca_abs_embed   the residue's own CA, absolute        128 A -> 4 A
        #   atom_rel_embed the atom minus that CA, relative       16 A -> 1 A
        # The relative half is what makes side-chain geometry expressible at
        # all: as a difference of two absolute embeddings whose finest period
        # was 6.28 A, it was not.
        self.ca_abs_embed = CoordinateFourierEmbedder(
            d_out=d_ca_emb, min_period_A=4.0, max_period_A=128.0
        )
        self.atom_rel_embed = CoordinateFourierEmbedder(
            d_out=d_ca_emb, min_period_A=1.0, max_period_A=16.0, include_norm=True
        )
        if self_conditioning:
            self.self_cond_proj = nn.Linear(d_ca_emb, d_ca_emb, bias=False)
            nn.init.zeros_(self.self_cond_proj.weight)
        else:
            self.self_cond_proj = None

        # FM time/noise-level conditioning. A sinusoidal+MLP embedding of t is
        # broadcast into the trunk (FiLM on the trunk input) and the atom
        # encoder/decoder (FiLM inside), so every level knows the noise level —
        # the atom blocks previously saw no time signal at all.
        self.d_temb = d_temb
        self.time_embed = TimeEmbedding(d_temb)
        # Chain-length conditioning, added into the same embedding the AdaLN
        # blocks read. SimpleFold folds log(num_tokens) into its timestep
        # condition. The length must come from `res_mask.sum()`, never from the
        # padded axis: DDP pads every rank up to one global length, so a padded
        # length would make a protein's conditioning depend on which other
        # proteins its rank-mates happened to draw — and no such length exists
        # at inference.
        self.length_embed = nn.Sequential(
            nn.Linear(1, d_temb), nn.SiLU(), nn.Linear(d_temb, d_temb)
        )
        self.film_trunk = FiLM(d_res, d_temb)

        # Local atom mixing and backbone-stream scans precede residue pooling.
        # Atom SSM state/rank are independent of the full-resolution trunk.
        self.atom_encoder = AtomEncoder(
            d_atom,
            d_ca_emb,
            n_layers=n_atom_layers,
            d_temb=d_temb,
            d_state=atom_d_state,
            mimo_rank=atom_mimo_rank,
            expand=expand,
            headdim=headdim,
            bimamba_share=bimamba_share,
            mixer=atom_mixer,
            n_cross_layers=n_atom_cross_layers,
            n_backbone_streams=n_backbone_streams,
        )

        if use_plm:
            self.plm_norm = nn.LayerNorm(d_plm)
            self.plm_proj = nn.Linear(d_plm, d_plm_proj)
            # Re-injected into the trunk's residual stream part-way down. The
            # PLM is the only evolutionary signal this model has, and a scan
            # cannot re-select it the way attention re-reads every position at
            # every layer: supplied once at the input, it has to survive every
            # residual block to still be usable at layer 10. Zero-init so the
            # trunk starts exactly as it would without it.
            self.plm_reinject = nn.Linear(d_plm_proj, d_res)
            nn.init.zeros_(self.plm_reinject.weight)
            nn.init.zeros_(self.plm_reinject.bias)
        else:
            self.plm_norm = None
            self.plm_proj = None
            self.plm_reinject = None
            d_plm_proj = 0  # contributes 0 to trunk input width

        # Trunk input = atom_token + ca_abs + termini(2) + chain_break(1)
        #             + res_type_emb + seq_pos_emb + plm_proj
        # (time enters via FiLM on the trunk input, not as a concat feature)
        #
        # `ca_abs` is the same embedding the atom encoder already computed, fed
        # to the trunk directly. Without it the trunk's only view of the noised
        # structure is the d_atom-wide token pooled over a residue's atoms —
        # 128 of 1251 input channels, against 1024 for the PLM, in a model
        # whose entire output is a coordinate update. It is a slice of an
        # existing tensor, so it costs one more block of `trunk_proj` and no
        # extra compute.
        trunk_in_dim = d_atom + d_ca_emb + 2 + 1 + d_res_type + d_res_pos + d_plm_proj
        self.trunk_input_norm = nn.LayerNorm(trunk_in_dim)
        self.trunk_proj = nn.Linear(trunk_in_dim, d_res)

        # ── Sequence trunk ──────────────────────────────────────────────
        # Roughly the one-third and two-thirds marks.
        original_inject_at = tuple(sorted({n_trunk // 3, (2 * n_trunk) // 3})) if n_trunk else ()
        self.plm_inject_at = tuple(i for i in original_inject_at if i < prelude_layers)
        self.residue_trunk = MambaStack(
            d_res,
            prelude_layers,
            d_state=d_state,
            mimo_rank=mimo_rank,
            expand=expand,
            headdim=headdim,
            bidirectional=bidirectional,
            d_temb=d_temb,
            bimamba_share=bimamba_share,
        )
        latent_class = ResidueLatentRefiner if residue_refiner else LatentMambaResidual
        if shared_trunk_layers:
            latent_class = SharedTrunkRefiner
        self.latent_residual = (
            latent_class(
                d_res, latent_compression, d_latent, n_latent_layers,
                n_loops=latent_loops,
                loop_mode=latent_loop_mode,
                d_temb=d_temb, d_state=d_state, mimo_rank=mimo_rank,
                expand=expand, headdim=headdim, bimamba_share=bimamba_share,
                **({'d_atom': d_atom} if residue_refiner and atom_recurrence else {}),
                **(dict(shared_layers=shared_trunk_layers, prelude_layers=prelude_layers,
                        plm_inject_at=original_inject_at, hierarchy=latent_hierarchy,
                        coarse_width=latent_coarse_width or None)
                   if shared_trunk_layers else {}),
            )
            if latent_compression else None
        )
        if recycle_policy == "fixed4":
            # Construction K still defines the prelude and PLM injection map.
            # Execution depth changes without adding layers or consuming RNG.
            self.latent_residual.n_loops = 4

        # ── all-atom output decoder ─────────────────────────────────────
        # Atom-specific residue context and the encoder skip feed the decoder.
        self.atom_decoder = AtomDecoder(
            d_res,
            d_atom,
            n_layers=n_atom_layers,
            d_temb=d_temb,
            d_state=atom_d_state,
            mimo_rank=atom_mimo_rank,
            expand=expand,
            headdim=headdim,
            bimamba_share=bimamba_share,
            mixer=atom_mixer,
            n_cross_layers=n_atom_cross_layers,
            n_backbone_streams=n_backbone_streams,
        )

    # ── helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _chain_break(chain_id: Tensor, res_mask: Tensor) -> Tensor:
        """Per-residue flag — True at first residue of each chain visible in the crop."""
        prev = torch.cat(
            [torch.full_like(chain_id[:, :1], -1), chain_id[:, :-1]],
            dim=1,
        )
        return ((chain_id != prev) & res_mask).unsqueeze(-1)  # [B, L, 1]

    def _compose_residue_input(
        self,
        batch: ProteinBatch,
        coord_feat: Tensor,
        ca_feat: Tensor,
        plm: Tensor | None,
    ) -> Tensor:
        dtype = coord_feat.dtype

        terminus_feat = torch.stack(
            [batch.is_nterm.to(dtype), batch.is_cterm.to(dtype)],
            dim=-1,
        )  # [B, L, 2]
        chain_break_feat = self._chain_break(batch.chain_id, batch.res_mask).to(dtype)
        rt_feat = self.res_type_embed(batch.res_type)  # [B, L, d_res_type]
        parts = [coord_feat, ca_feat.to(dtype), terminus_feat, chain_break_feat, rt_feat]
        if self.seq_pos_embed is not None:
            pos_feat = self.seq_pos_embed(
                batch.res_seq_nums,
                batch.res_mask,
                chain_id=batch.chain_id,
                entity_id=batch.entity_id,
                sym_id=batch.sym_id,
            )
            parts.append(pos_feat)
        if plm is not None:
            parts.append(plm)
        trunk_in = torch.cat(parts, dim=-1)
        trunk_in = self.trunk_input_norm(trunk_in)
        return self.trunk_proj(trunk_in)

    def _split_scale_embed(self, coords: Tensor, atom_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Two-band embedding of one coordinate set.

        Returns (per-atom [B, L, A, d_ca_emb], per-residue CA [B, L, d_ca_emb]).

        The per-atom feature is the sum of the coarse CA-absolute embedding and
        the fine CA-relative one. Summing two `Linear`-terminated embedders is
        exactly a single `Linear` over their concatenated raw features, so this
        is the concatenation without materialising a duplicate copy of the CA
        feature in every atom slot.
        """
        mask_f = atom_mask.unsqueeze(-1).to(coords.dtype)
        coords = coords * mask_f
        ca = coords[:, :, CA_ATOM_ID : CA_ATOM_ID + 1, :]  # [B, L, 1, 3]
        ca_feat = self.ca_abs_embed(ca.squeeze(2))  # [B, L, d_ca_emb]
        rel = (coords - ca) * mask_f  # [B, L, A, 3]
        return self.atom_rel_embed(rel) + ca_feat.unsqueeze(2), ca_feat

    def _atom_coord_embed(self, batch: ProteinBatch) -> tuple[Tensor, Tensor]:
        """Embed the noised coordinates.

        Returns (per-atom [B, L, A, d_ca_emb], per-residue CA [B, L, d_ca_emb]).
        The first feeds the atom encoder, the second also goes straight to the
        trunk input.
        """
        x_t = batch.x_t
        if x_t.shape[-2] != MAX_ATOMS_PER_RES:
            raise RuntimeError(
                f"Direct all-atom model expects {MAX_ATOMS_PER_RES} atom slots, "
                f"got {x_t.shape[-2]}."
            )
        coord_emb, ca_feat = self._split_scale_embed(x_t, batch.atom_mask)
        if self.self_cond_proj is not None:
            x_sc = batch.x_self_cond
            if x_sc is None:
                x_sc = torch.zeros_like(x_t)
                present = x_t.new_zeros((x_t.shape[0], 1, 1, 1))
            elif x_sc.shape != x_t.shape:
                raise RuntimeError(
                    f"x_self_cond shape {tuple(x_sc.shape)} != x_t shape {tuple(x_t.shape)}"
                )
            else:
                present = x_t.new_ones((x_t.shape[0], 1, 1, 1))
            # Keep the projection in every autograd graph when self-conditioning
            # is configured, even on batches where the Bernoulli draw is false.
            # The indicator makes the absent branch an exact semantic zero while
            # avoiding intermittent unused parameters under DDP.
            sc_emb, _ = self._split_scale_embed(
                x_sc.to(dtype=x_t.dtype), batch.atom_mask
            )
            coord_emb = coord_emb + self.self_cond_proj(sc_emb) * present
        return coord_emb, ca_feat

    def _embed_plm(self, batch: ProteinBatch, dtype: torch.dtype) -> Tensor | None:
        """Project external PLM features (loud failure if expected but missing)."""
        if not self.use_plm:
            return None
        if batch.esm is None:
            raise RuntimeError(
                "MambaFoldAllAtom built with use_plm=True but batch.esm is None. "
                "Pre-compute PLM features (scripts/precompute_esm.py)."
            )
        if batch.esm.ndim != 3 or batch.esm.shape[1] != batch.res_type.shape[1]:
            raise RuntimeError(
                "batch.esm must have shape [rows, L, d_plm] with the same L as res_type."
            )
        if batch.esm.shape[-1] != self.d_plm:
            raise RuntimeError(f"Expected ESM dim {self.d_plm}, got {batch.esm.shape[-1]}.")
        batch_size = batch.res_type.shape[0]
        rows = batch.esm.shape[0]
        if rows not in (1, batch_size):
            raise RuntimeError(
                f"batch.esm has {rows} rows; expected {batch_size} or 1 (shared across copies)."
            )
        # A one-row embedding means every example in the batch is the same
        # protein — the collator's `copies_per_protein` path. Normalize and
        # project once and broadcast the result: the projection is d_plm ->
        # d_plm_proj (2560 -> 256 here), by far the widest matmul outside the
        # trunk, and doing it per copy computes the same thing every time.
        # `expand` is differentiable; gradients from the copies sum back.
        plm = self.plm_proj(self.plm_norm(batch.esm.to(dtype=dtype)))
        if rows == 1 and batch_size > 1:
            plm = plm.expand(batch_size, -1, -1)
        return plm  # [B, L, d_plm_proj]

    def _decode(
        self,
        batch: ProteinBatch,
        res: Tensor,
        atom_repr: Tensor,
        temb: Tensor,
    ) -> dict:
        """Decode the folding velocity and expose the residue latent."""
        mask_atom = batch.atom_mask.to(res.dtype).unsqueeze(-1)
        v_atom = self.atom_decoder(res, atom_repr, batch, temb) * mask_atom
        return {
            "v_atom": v_atom,
            "trunk_latent": res,
        }

    def _recycle_atoms(
        self, batch: ProteinBatch, res: Tensor, atom_token: Tensor,
        atom_repr: Tensor, temb: Tensor, depths: Tensor, *, trajectory: bool,
        reinject: Tensor | None,
    ) -> dict:
        """Recycle hidden state at fixed coordinates/time within one velocity call."""
        branch = self.latent_residual
        initial_atoms = atom_repr
        predictions = [self._decode(batch, res, initial_atoms, temb)["v_atom"]] if trajectory else []
        atom_mask = batch.atom_mask.unsqueeze(-1)
        res_mask = batch.res_mask.unsqueeze(-1)
        loops = depths.shape[1]
        reinject_initial = self.recycle_policy != "legacy"
        initial_res = (
            (res if reinject is None else res + reinject).masked_fill(~res_mask, 0)
            if reinject_initial else None
        )
        prefix = max(0, loops - 2) if trajectory and self.recycle_policy == "sampled16_tail2" else 0
        for index in range(loops):
            if prefix and index == prefix:
                res, atom_repr = res.detach(), atom_repr.detach()
            # Training coordinates were sampled before entering this context.
            # Only the prefix activations are discarded; live initialization
            # remains available to every supervised pass after the boundary.
            with torch.no_grad() if index < prefix else nullcontext():
                condition = temb + branch.depth_embed(depths[:, index:index + 1].float())
                condition = F.rms_norm(condition, (condition.shape[-1],), eps=1e-6)
                incoming = initial_atoms if self.atom_recycling == "reset" else atom_repr
                token, atoms = self.atom_encoder.remix(incoming, batch, condition)
                if reinject_initial:
                    res = F.rms_norm(res, (res.shape[-1],), eps=1e-6) + initial_res
                # Reuse the original atom-token projection without another bias.
                feedback = F.linear(token - atom_token, self.trunk_proj.weight[:, :token.shape[-1]])
                res = (res + feedback.tanh()).masked_fill(~res_mask, 0)
                res = branch.refine_interleaved_once(
                    res, batch.res_mask, condition, index=index, inject=reinject,
                )
                velocity, atom_repr = self.atom_decoder(
                    res, atoms, batch, condition, return_features=True,
                    predict_velocity=trajectory or index + 1 == loops,
                    head_temb=temb,
                )
                if velocity is not None:
                    velocity = velocity * atom_mask.to(velocity.dtype)
                    if trajectory:
                        predictions.append(velocity)
        output = {"v_atom": velocity, "trunk_latent": res}
        if trajectory:
            output["v_atom_steps"] = tuple(predictions)
            output["latent_depths"] = depths
            if reinject_initial:
                output["latent_supervised_steps"] = tuple(range(prefix + 1, loops + 1))
        return output

    # ── forward ─────────────────────────────────────────────────────────

    def forward(
        self,
        batch: ProteinBatch,
    ) -> dict:
        """Run the single folding path used by training and sampling.

        Returns keys:
            v_atom       [B, L, A, 3]
            trunk_latent [B, L, d_res]
        LiFT training additionally returns initial/recurrent v_atom_steps and
        their latent_depths. Eval and no-grad calls only decode the last state.
        """
        temb = self.time_embed(batch.t)  # [B, d_temb]
        n_res = batch.res_mask.sum(dim=1, keepdim=True).clamp(min=1)
        temb = temb + self.length_embed(torch.log(n_res.to(temb.dtype)))
        coord_emb, ca_feat = self._atom_coord_embed(batch)  # [B,L,A,d], [B,L,d]
        encoder_kwargs = {"safe_padding": True} if self.atom_recycling != "none" else {}
        atom_token, atom_repr = self.atom_encoder(coord_emb, batch, temb, **encoder_kwargs)
        plm = self._embed_plm(batch, atom_token.dtype)

        res = self._compose_residue_input(batch, atom_token, ca_feat, plm)
        res = self.film_trunk(res, temb)  # inject noise level
        reinject = (
            self.plm_reinject(plm) * batch.res_mask.unsqueeze(-1).to(res.dtype)
            if (self.plm_reinject is not None and plm is not None)
            else None
        )
        res = self.residue_trunk(
            res, batch.res_mask, temb, inject=reinject, inject_at=self.plm_inject_at,
            midpoint_residual=(
                self.latent_residual if self.latent_position == "midpoint" else None
            ),
        )
        if self.latent_residual is not None and self.latent_position == "post":
            branch = self.latent_residual
            training_rollout = self.training and torch.is_grad_enabled()
            training_loops = self.refinement_training_loops if self.training else None
            if self.training and self.recycle_policy == "fixed4" and training_loops is None:
                training_loops = 4
            if self.training and self.recycle_policy != "legacy" and training_loops is not None:
                if (type(training_loops) is not int or not 1 <= training_loops <= 16
                        or self.recycle_policy == "fixed2" and training_loops != 2
                        or self.recycle_policy == "fixed4" and training_loops != 4):
                    raise ValueError("recycle policy received an incompatible runtime training K")
            # Both objectives use the same training grids so the objective is
            # the only difference. Sampling and no-grad self-conditioning use
            # a uniform grid and only evaluate the final decoder.
            depths = branch.depth_grid(
                res.shape[0], res.device,
                randomize=training_rollout and branch.loop_mode == "conditioned",
                n_loops=training_loops,
            )
            trajectory = training_rollout and self.latent_supervision == "lift"
            if self.atom_recycling != "none":
                return self._recycle_atoms(
                    batch, res, atom_token, atom_repr, temb, depths,
                    trajectory=trajectory, reinject=reinject,
                )
            atom_kwargs = (
                dict(atom_repr=atom_repr, atom_mask=batch.atom_mask)
                if self.atom_recurrence else {}
            )
            if self.shared_trunk_layers:
                atom_kwargs['inject'] = reinject
            states = branch(res, batch.res_mask, temb, depths=depths,
                            return_states=trajectory, **atom_kwargs)
            if trajectory:
                predictions = tuple(
                    self._decode(batch, state.residue, state.atoms, temb)
                    if self.atom_recurrence else self._decode(batch, state, atom_repr, temb)
                    for state in states
                )
                output = predictions[-1]
                output["v_atom_steps"] = tuple(pred["v_atom"] for pred in predictions)
                output["latent_depths"] = depths
                return output
            if self.atom_recurrence:
                res, atom_repr = states
            else:
                res = states
        return self._decode(batch, res, atom_repr, temb)
