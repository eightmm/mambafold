"""Ordered residue compression and a full-resolution residual readout."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mambafold.model.bimamba3 import MambaStack, RMSNorm
from mambafold.model.embeddings import TimeEmbedding
from mambafold.model.fold.atom_mamba import AtomRecurrentRefiner


class RefinementState(NamedTuple):
    residue: Tensor
    atoms: Tensor


class HierarchicalMambaStack(MambaStack):
    """Keep the learned fine stack and insert one initially zero coarse update.

    The coarse update runs after the first n_layers // 2 fine blocks. Its
    two-block stack defaults to the fine width; a wider coarse_width changes
    only the coarse projections and stack, so the default construction order,
    shapes and RNG draws are those of the original four-block hierarchy.
    """

    def __init__(self, width: int, n_layers: int, *, coarse_width: int | None = None,
                 **kwargs):
        if type(n_layers) is not int or n_layers < 2:
            raise ValueError("hierarchical latent mixing requires at least two fine blocks")
        coarse_width = width if coarse_width is None else coarse_width
        if type(coarse_width) is not int or coarse_width <= 0:
            raise ValueError("coarse width must be a positive integer")
        super().__init__(width, n_layers, **kwargs)
        self.coarse_after = n_layers // 2
        self.coarse_input_norm = RMSNorm(width)
        self.coarse_down = nn.Linear(4 * width, coarse_width)
        self.coarse_stack = MambaStack(coarse_width, 2, **kwargs)
        self.coarse_output_norm = RMSNorm(coarse_width)
        self.coarse_up = nn.Linear(coarse_width, 4 * width)
        nn.init.zeros_(self.coarse_up.weight)
        nn.init.zeros_(self.coarse_up.bias)

    def forward(self, x: Tensor, mask: Tensor, temb: Tensor) -> Tensor:
        for index, layer in enumerate(self.layers):
            x = layer(x, mask, temb)
            if index + 1 == self.coarse_after:
                batch, length, width = x.shape
                pad = (-length) % 4
                fine = self.coarse_input_norm(x).masked_fill(~mask.unsqueeze(-1), 0)
                grouped = F.pad(fine, (0, 0, 0, pad)).reshape(batch, -1, 4 * width)
                count = F.pad(mask, (0, pad)).reshape(batch, -1, 4).sum(-1)
                coarse_mask = count > 0
                coarse = self.coarse_down(grouped / count.clamp_min(1).unsqueeze(-1))
                coarse = coarse.masked_fill(~coarse_mask.unsqueeze(-1), 0)
                coarse = self.coarse_stack(coarse, coarse_mask, temb)
                update = self.coarse_up(self.coarse_output_norm(coarse))
                update = update.reshape(batch, -1, width)[:, :length]
                x = (x + update.tanh()).masked_fill(~mask.unsqueeze(-1), 0)
        return x


class LatentMambaResidual(nn.Module):
    """Mix position-sensitive residue groups alongside the residue stream.

    Masks follow the trunk's contiguous valid-prefix contract. Groups are
    anchored to the crop's first residue, so extra right padding cannot change
    either group membership or the denominator of a partial final group.
    n_layers counts independent blocks; n_loops repeats the same stack without
    adding parameters. Compression runs once; trajectory training also reads the
    initial state and each recurrent state through the same residue readout.
    """

    def __init__(
        self,
        d_res: int,
        compression: int,
        d_latent: int = 256,
        n_layers: int = 2,
        *,
        n_loops: int = 2,
        loop_mode: str = "conditioned",
        d_temb: int = 128,
        d_state: int = 64,
        mimo_rank: int = 4,
        expand: int = 2,
        headdim: int = 64,
        bimamba_share: bool = False,
        stable: bool = False,
    ):
        super().__init__()
        if loop_mode not in ("plain", "conditioned"):
            raise ValueError("latent loop mode must be plain or conditioned")
        if type(n_loops) is not int or n_loops < 1:
            raise ValueError("latent loop count must be a positive integer")
        if compression not in (4, 8, 16):
            raise ValueError("latent compression must be 4, 8 or 16")
        if d_latent <= 0 or n_layers <= 0:
            raise ValueError("latent width and layer count must be positive")
        if headdim <= 0 or (d_latent * expand) % headdim:
            raise ValueError("d_latent * expand must be divisible by headdim")
        self.compression = compression
        self.n_loops = n_loops
        self.loop_mode = loop_mode
        self.input_norm = RMSNorm(d_res)
        self.compress_proj = nn.Linear(compression * d_res, d_latent)
        # Start as a projected masked mean, then learn each offset independently.
        # Keep one 2D Linear weight so the complete projection is Muon-eligible.
        bound = d_res ** -0.5
        with torch.no_grad():
            nn.init.uniform_(self.compress_proj.weight[:, :d_res], -bound, bound)
            self.compress_proj.weight.copy_(
                self.compress_proj.weight[:, :d_res].repeat(1, compression)
            )
            nn.init.uniform_(self.compress_proj.bias, -bound, bound)
        self.stack = MambaStack(
            d_latent, n_layers, d_state=d_state, mimo_rank=mimo_rank,
            expand=expand, headdim=headdim, bidirectional=True,
            bimamba_share=bimamba_share, d_temb=d_temb, stable=stable,
        )
        self.residue_proj = nn.Linear(d_res, d_latent, bias=False)
        self.offset_embed = nn.Embedding(compression, d_latent)
        self.output_norm = RMSNorm(d_latent)
        self.output_proj = nn.Linear(d_latent, d_res, bias=False)
        # One zero readout preserves the pretrained path while allowing its
        # gradient to open the branch; zeroing another outer gate would stall it.
        nn.init.zeros_(self.output_proj.weight)
        self.depth_embed = TimeEmbedding(d_temb) if loop_mode == "conditioned" else None

    def _compress(self, h: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        batch, length, width = h.shape
        pad = (-length) % self.compression
        h = F.pad(h.masked_fill(~mask.unsqueeze(-1), 0), (0, 0, 0, pad))
        grouped_mask = F.pad(mask, (0, pad)).reshape(batch, -1, self.compression)
        count = grouped_mask.sum(dim=-1)
        grouped = h.reshape(batch, -1, self.compression * width)
        # Normalize before the affine projection so partial groups retain the
        # same scale and the bias is added exactly once, even at length one.
        grouped = grouped / count.clamp(min=1).unsqueeze(-1).to(h.dtype)
        latent_mask = count > 0
        z = self.compress_proj(grouped).masked_fill(~latent_mask.unsqueeze(-1), 0)
        return z, latent_mask

    def depth_grid(self, batch_size: int, device: torch.device, *, randomize: bool,
                   n_loops: int | None = None) -> Tensor:
        """Depth coordinates [B,K], with a fixed endpoint and separated interiors."""
        loops = self.n_loops if n_loops is None else n_loops
        if type(loops) is not int or loops < 1:
            raise ValueError("loop budget must be a positive integer")
        if not randomize or loops == 1:
            return (torch.arange(1, loops + 1, device=device, dtype=torch.float32)
                    / loops).expand(batch_size, -1)
        interior = torch.rand(
            batch_size, loops - 1, device=device, dtype=torch.float64,
        ).sort(dim=1).values
        epsilon = min(1e-6, 0.5 / loops)
        interior = interior * (1 - loops * epsilon) + epsilon * torch.arange(
            1, loops, device=device, dtype=torch.float64,
        )
        return torch.cat([interior.float(), torch.ones(batch_size, 1, device=device)], dim=1)

    def forward(
        self, x: Tensor, mask: Tensor, temb: Tensor, *,
        depths: Tensor | None = None, return_states: bool = False,
    ) -> Tensor | tuple[Tensor, ...]:
        length = x.shape[1]
        h = self.input_norm(x)
        z, latent_mask = self._compress(h, mask)
        anchor = z
        if depths is None:
            depths = self.depth_grid(x.shape[0], x.device, randomize=False)
        # Cache the residue-specific part of the shared readout across depths.
        offsets = torch.arange(length, device=x.device) % self.compression
        local = self.residue_proj(h)
        offset = self.offset_embed(offsets).to(h.dtype)

        def restore(state: Tensor) -> Tensor:
            context = state.repeat_interleave(self.compression, dim=1)[:, :length]
            update = self.output_proj(F.silu(self.output_norm(local + context + offset)))
            return (x + update).masked_fill(~mask.unsqueeze(-1), 0)

        states = [restore(z)] if return_states else []
        # Only depth recurs: every pass rescans the updated latent sequence
        # with shared weights, fixed FM time and a fresh internal scan state.
        # Keep the full graph so earlier passes receive later-pass gradients.
        for index in range(self.n_loops):
            loop_temb = temb
            if self.depth_embed is not None:
                z = F.rms_norm(z, (z.shape[-1],), eps=1e-6) + anchor
                # Keep the coordinate in fp32 under bf16 autocast: rounding
                # before the high-frequency embedding changes its phase.
                depth = depths[:, index:index + 1].float()
                loop_temb = temb + self.depth_embed(depth)
            z = self.stack(z, latent_mask, loop_temb)
            if return_states:
                states.append(restore(z))
        return tuple(states) if return_states else restore(z)


class ResidueLatentRefiner(LatentMambaResidual):
    """Identity-initialized residue/latent correction at a shared core width.

    Full residue positions remain distinct. Only channels are projected before
    residue mixing; ordered sequence compression follows that mixing. One zero
    readout preserves the complete pretrained residue stream at initialization.
    """

    def __init__(self, d_res: int, compression: int, d_latent: int, n_layers: int,
                 *, n_residue_layers: int = 3, d_atom: int | None = None, **kwargs):
        # Compression and both shared stacks operate at core width. The outer
        # projections alone cross the pretrained trunk's channel boundary.
        super().__init__(d_latent, compression, d_latent, n_layers, stable=True, **kwargs)
        self.input_norm = RMSNorm(d_res)
        self.residue_proj = nn.Linear(d_res, d_latent, bias=False)
        self.output_proj = nn.Linear(d_latent, d_res, bias=False)
        nn.init.zeros_(self.output_proj.weight)
        self.residue_stack = MambaStack(
            d_latent, n_residue_layers, stable=True,
            **{key: kwargs[key] for key in (
                'd_temb', 'd_state', 'mimo_rank', 'expand', 'headdim', 'bimamba_share'
            ) if key in kwargs},
        )
        self.atom_refiner = (
            AtomRecurrentRefiner(d_res, d_atom, kwargs.get('d_temb', 128))
            if d_atom is not None else None
        )

    def refine_once(self, x: Tensor, mask: Tensor, condition: Tensor) -> Tensor:
        h = self.residue_proj(self.input_norm(x))
        h = self.residue_stack(h, mask, condition)
        return self.latent_update(x, h, mask, condition)

    def latent_update(self, x: Tensor, h: Tensor, mask: Tensor, condition: Tensor) -> Tensor:
        z, latent_mask = self._compress(h, mask)
        z = self.stack(z, latent_mask, condition)
        context = z.repeat_interleave(self.compression, dim=1)[:, :x.shape[1]]
        offsets = torch.arange(x.shape[1], device=x.device) % self.compression
        readout = h + context + self.offset_embed(offsets).to(h.dtype)
        update = self.output_proj(F.silu(self.output_norm(readout)))
        return (x + update.tanh()).masked_fill(~mask.unsqueeze(-1), 0)

    def forward(self, x: Tensor, mask: Tensor, temb: Tensor, *,
                depths: Tensor | None = None, return_states: bool = False,
                atom_repr: Tensor | None = None, atom_mask: Tensor | None = None):
        """Return residue tensors, or paired states when atom recurrence is on."""
        if depths is None:
            depths = self.depth_grid(x.shape[0], x.device, randomize=False)
        if depths.ndim != 2 or depths.shape[0] != x.shape[0] or depths.shape[1] < 1:
            raise ValueError("depths must have shape [B,K] with positive K")
        if self.atom_refiner is not None and (atom_repr is None or atom_mask is None):
            raise ValueError("atom recurrence requires encoder states and atom masks")
        x = x.masked_fill(~mask.unsqueeze(-1), 0)
        def state():
            return RefinementState(x, atom_repr) if self.atom_refiner is not None else x
        states = [state()] if return_states else []
        for index in range(depths.shape[1]):
            condition = temb + self.depth_embed(depths[:, index:index + 1].float())
            condition = F.rms_norm(condition, (condition.shape[-1],), eps=1e-6)
            x = self.refine_once(x, mask, condition)
            if self.atom_refiner is not None:
                x, atom_repr = self.atom_refiner(x, atom_repr, atom_mask, condition)
                x = x.masked_fill(~mask.unsqueeze(-1), 0)
            if return_states:
                states.append(state())
        return tuple(states) if return_states else state()


class SharedTrunkRefiner(ResidueLatentRefiner):
    """Reuse original-width trunk blocks with their learned normalization intact.

    Original trunk blocks receive the original FM condition; depth conditioning
    remains on the learned latent correction. Only full-resolution residue state
    persists between passes. PLM reinjection follows original executed indices.
    """

    def __init__(self, d_res: int, compression: int, d_latent: int, n_layers: int,
                 *, shared_layers: int, prelude_layers: int,
                 plm_inject_at: tuple[int, ...], hierarchy: bool = False,
                 coarse_width: int | None = None, **kwargs):
        if coarse_width is not None and not hierarchy:
            raise ValueError("coarse width requires the latent hierarchy")
        super().__init__(d_res, compression, d_latent, n_layers,
                         n_residue_layers=0, **kwargs)
        block_kwargs = {key: kwargs[key] for key in (
            'd_temb', 'd_state', 'mimo_rank', 'expand', 'headdim', 'bimamba_share'
        ) if key in kwargs}
        self.residue_stack = MambaStack(d_res, shared_layers, **block_kwargs)
        self.prelude_layers = prelude_layers
        self.original_plm_inject_at = plm_inject_at
        if hierarchy:
            self.stack = HierarchicalMambaStack(d_latent, n_layers, stable=True,
                                               coarse_width=coarse_width, **block_kwargs)

    def refine_interleaved_once(
        self, x: Tensor, mask: Tensor, condition: Tensor, *, index: int,
        inject: Tensor | None = None,
    ) -> Tensor:
        """Insert the latent update halfway through one shared residue pass.

        The caller supplies one depth condition to both residue halves and the
        latent stack. Each stack call starts a fresh internal scan; only hidden
        residue and atom representations persist between passes.
        """
        if self.residue_stack.n_layers < 2 or self.residue_stack.n_layers % 2:
            raise ValueError("interleaved refinement requires positive even shared layers")
        offset = self.prelude_layers + index * self.residue_stack.n_layers
        inject_at = tuple(i - offset for i in self.original_plm_inject_at
                          if offset <= i < offset + self.residue_stack.n_layers)

        def latent_midpoint(residue: Tensor, mask: Tensor, condition: Tensor) -> Tensor:
            h = self.residue_proj(self.input_norm(residue))
            return self.latent_update(residue, h, mask, condition)

        return self.residue_stack(
            x, mask, condition, inject=inject, inject_at=inject_at,
            midpoint_residual=latent_midpoint,
        )

    def forward(self, x: Tensor, mask: Tensor, temb: Tensor, *,
                depths: Tensor | None = None, return_states: bool = False,
                inject: Tensor | None = None):
        if depths is None:
            depths = self.depth_grid(x.shape[0], x.device, randomize=False)
        if depths.ndim != 2 or depths.shape[0] != x.shape[0] or depths.shape[1] < 1:
            raise ValueError("depths must have shape [B,K] with positive K")
        x = x.masked_fill(~mask.unsqueeze(-1), 0)
        states = [x] if return_states else []
        for index in range(depths.shape[1]):
            offset = self.prelude_layers + index * self.residue_stack.n_layers
            inject_at = tuple(i - offset for i in self.original_plm_inject_at
                              if offset <= i < offset + self.residue_stack.n_layers)
            x = self.residue_stack(x, mask, temb, inject=inject, inject_at=inject_at)
            condition = temb + self.depth_embed(depths[:, index:index + 1].float())
            condition = F.rms_norm(condition, (condition.shape[-1],), eps=1e-6)
            h = self.residue_proj(self.input_norm(x))
            x = self.latent_update(x, h, mask, condition)
            if return_states:
                states.append(x)
        return tuple(states) if return_states else x
