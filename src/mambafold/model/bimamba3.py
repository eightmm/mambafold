"""Mamba-3 SSM stack: primitives, causal/bidirectional blocks, reusable stack.

Requires mamba-ssm installed from main branch:
    pip install git+https://github.com/state-spaces/mamba --no-build-isolation

Reference: github.com/state-spaces/mamba  |  arXiv:2603.15569
"""

from __future__ import annotations

import importlib
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _load_mamba3_class():
    """Load Mamba3 without requiring its unused legacy selective-scan extension.

    ``mamba_ssm.__init__`` imports the Mamba-1 CUDA extension eagerly even when
    callers only use the Triton/TileLang Mamba-3 kernels.  Cluster nodes expose
    CUDA 13 while the available legacy extension may target CUDA 12; in that
    case, provide a module stub for the unused extension and retry the official
    Mamba3 import. Any other import failure remains fatal.
    """
    try:
        return importlib.import_module("mamba_ssm.modules.mamba3").Mamba3
    except ImportError as exc:
        message = str(exc)
        if "selective_scan_cuda" not in message and "libcudart.so" not in message:
            raise
        for name in list(sys.modules):
            if name == "mamba_ssm" or name.startswith("mamba_ssm."):
                sys.modules.pop(name, None)
        sys.modules["selective_scan_cuda"] = types.ModuleType("selective_scan_cuda")
        return importlib.import_module("mamba_ssm.modules.mamba3").Mamba3


_Mamba3 = _load_mamba3_class()


def _default_chunk_size(mimo_rank: int) -> int:
    """GPU-aware chunk size for Mamba-3 SSD kernels.

    Ampere (A5000, A100): 32 // mimo_rank.
    Hopper+ (H100, B200): 64 // mimo_rank — larger shared mem lets bigger chunks
    reduce kernel-launch overhead without OOM.
    """
    if mimo_rank <= 1:
        return 64
    base = 32
    if torch.cuda.is_available():
        try:
            if torch.cuda.get_device_capability() >= (9, 0):
                base = 64
        except Exception:
            pass
    return max(1, base // mimo_rank)


# ── Primitives ─────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Args:
            d_model (int): Feature dimension size. Initializes learnable scale
                weight of shape [d_model].
            eps (float): Small constant for numerical stability. Default: 1e-6.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape [*, d_model].

        Returns:
            Tensor: RMS-normalized tensor of shape [*, d_model].
        """
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, d_model: int, d_ff: int = None):
        """
        Args:
            d_model (int): Input and output feature dimension [*, d_model].
            d_ff (int | None): Hidden dimension. Defaults to floor(8/3 * d_model)
                rounded up to the nearest multiple of 8.
        """
        super().__init__()
        d_ff = d_ff or int(d_model * 8 / 3)
        d_ff = ((d_ff + 7) // 8) * 8
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape [*, d_model].

        Returns:
            Tensor: Output tensor of shape [*, d_model].
                Computed as w2(SiLU(w1(x)) * w3(x)).
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class AdaLNZero(nn.Module):
    """Time-conditioned AdaLN-Zero modulation for residual branches.

    For each branch, projects the FM time embedding to scale, shift, and gate.
    The projection is zero-initialized, so each branch starts as an identity
    residual path and learns how strongly to activate at each noise level.
    """

    def __init__(self, d_model: int, d_temb: int, n_branches: int = 2):
        super().__init__()
        self.d_model = d_model
        self.n_branches = n_branches
        self.proj = nn.Linear(d_temb, n_branches * 3 * d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, temb: Tensor | None, branch: int) -> tuple[Tensor, Tensor]:
        if temb is None:
            raise RuntimeError("AdaLNZero requires temb.")
        params = self.proj(F.silu(temb)).view(temb.shape[0], self.n_branches, 3, self.d_model)
        scale, shift, gate = params[:, branch].unbind(dim=1)
        scale = scale.to(x.dtype).unsqueeze(1)
        shift = shift.to(x.dtype).unsqueeze(1)
        gate = gate.to(x.dtype).unsqueeze(1)
        return x * (1 + scale) + shift, gate


class Mamba3Layer(nn.Module):
    """Mamba-3 SSM block with padding mask support.

    Wraps the official mamba_ssm.modules.mamba3.Mamba3 and handles
    variable-length sequences by zeroing padding positions.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        expand: int = 2,
        headdim: int = 64,
        mimo_rank: int = 4,
        dtype=None,
        device=None,
        **kwargs,
    ):
        """
        Args:
            d_model (int): Token feature dimension. Input/output shape [B, S, d_model].
            d_state (int): SSM state expansion factor. Default: 128.
            expand (int): Inner dimension multiplier (inner_dim = expand * d_model).
                Default: 2.
            headdim (int): Dimension per attention head inside the SSM. Default: 64.
            mimo_rank (int): MIMO rank; values > 1 enable MIMO mode. Controls
                chunk_size = max(1, 32 // mimo_rank). Default: 4.
            dtype: Floating-point dtype forwarded to the underlying Mamba3 kernel.
            device: Device forwarded to the underlying Mamba3 kernel.
            **kwargs: Extra keyword arguments (ignored, for forward-compatibility).
        """
        super().__init__()
        is_mimo = mimo_rank > 1
        self.chunk_size = _default_chunk_size(mimo_rank)
        self.ssm = _Mamba3(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
            is_mimo=is_mimo,
            mimo_rank=mimo_rank,
            chunk_size=self.chunk_size,
            is_outproj_norm=False,
            dtype=dtype,
            device=device,
        )

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """
        Args:
            x (Tensor): Input token sequence of shape [B, S, d_model].
            mask (Tensor | None): Boolean or float padding mask of shape [B, S].
                Padding positions (mask == 0) are zeroed before and after the SSM.

        Returns:
            Tensor: Output sequence of shape [B, S, d_model] with padding zeroed.
        """
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)

        B, S, D = x.shape
        pad = (self.chunk_size - S % self.chunk_size) % self.chunk_size
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))

        y = self.ssm(x)

        if pad > 0:
            y = y[:, :S]
        if mask is not None:
            y = y * mask.unsqueeze(-1).to(y.dtype)
        return y


# ── Blocks ─────────────────────────────────────────────────────────────────

def _flip_by_mask(x: Tensor, mask: Tensor) -> Tensor:
    """Reverse valid positions along sequence dim, keeping padding at end.

    For each batch element, reverses only the valid (non-padding) tokens so
    that padding tokens stay at the tail. Used by BiMamba3Block to run the
    backward SSM pass.

    Args:
        x (Tensor): Input tensor of shape [B, S, D].
        mask (Tensor): Boolean or integer mask of shape [B, S].
            1 = valid token, 0 = padding.

    Returns:
        Tensor: Sequence-reversed tensor of shape [B, S, D].
            Valid tokens appear in reversed order; padding positions are zeroed.
    """
    lengths = mask.sum(dim=1)
    arange = torch.arange(mask.shape[1], device=x.device).unsqueeze(0).expand(mask.shape[0], -1)
    rev_idx = (lengths.unsqueeze(1) - 1 - arange).clamp(min=0)
    out = torch.gather(x, 1, rev_idx.unsqueeze(-1).expand_as(x))
    return out * mask.unsqueeze(-1).to(x.dtype)


class Mamba3Block(nn.Module):
    """Causal Mamba-3 block: pre-norm SSM + SwiGLU FFN."""

    def __init__(self, d_model: int, d_state: int = 64, mimo_rank: int = 4,
                 expand: int = 2, headdim: int = 64,
                 adaln_zero: bool = False, d_temb: int = 128):
        """
        Args:
            d_model (int): Token feature dimension. Input/output shape [B, S, d_model].
            d_state (int): SSM state expansion factor. Default: 64.
            mimo_rank (int): MIMO rank forwarded to Mamba3Layer. Default: 4.
            expand (int): Inner dimension multiplier inside the SSM. Default: 2.
            headdim (int): Dimension per head inside the SSM. Default: 64.
        """
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.ssm = Mamba3Layer(d_model=d_model, d_state=d_state, expand=expand,
                               headdim=headdim, mimo_rank=mimo_rank)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model)
        self.adaln = AdaLNZero(d_model, d_temb, n_branches=2) if adaln_zero else None

    def forward(self, x: Tensor, mask: Tensor, temb: Tensor | None = None) -> Tensor:
        """
        Args:
            x (Tensor): Input token sequence of shape [B, S, d_model].
            mask (Tensor): Boolean or float padding mask of shape [B, S].

        Returns:
            Tensor: Output tensor of shape [B, S, d_model].
                Residual stream: x + SSM(RMSNorm(x)) + FFN(RMSNorm(x)),
                then padding positions zeroed.
        """
        if self.adaln is None:
            x = x + self.ssm(self.norm1(x), mask)
            x = x + self.ffn(self.norm2(x))
            return x * mask.unsqueeze(-1).to(x.dtype)
        h, gate = self.adaln(self.norm1(x), temb, branch=0)
        x = x + gate * self.ssm(h, mask)
        h, gate = self.adaln(self.norm2(x), temb, branch=1)
        x = x + gate * self.ffn(h)
        return x * mask.unsqueeze(-1).to(x.dtype)


class BiMamba3Block(nn.Module):
    """Bidirectional Mamba-3: forward + backward SSM summed."""

    def __init__(self, d_model: int, d_state: int = 64, mimo_rank: int = 4,
                 expand: int = 2, headdim: int = 64, share_dir: bool = False,
                 adaln_zero: bool = False, d_temb: int = 128):
        """
        Args:
            d_model (int): Token feature dimension. Input/output shape [B, S, d_model].
            d_state (int): SSM state expansion factor shared by both directions.
                Default: 64.
            mimo_rank (int): MIMO rank forwarded to both Mamba3Layers. Default: 4.
            expand (int): Inner dimension multiplier inside each SSM. Default: 2.
            headdim (int): Dimension per head inside each SSM. Default: 64.
            share_dir (bool): weight-tie the two directions — run a single SSM on
                both the forward and reversed sequence. Halves the per-layer SSM
                params/compute. Default: False (separate fwd/bwd SSMs).
        """
        super().__init__()
        self.share_dir = share_dir
        self.norm1 = RMSNorm(d_model)
        self.mamba_f = Mamba3Layer(d_model=d_model, d_state=d_state, expand=expand,
                                   headdim=headdim, mimo_rank=mimo_rank)
        self.mamba_b = None if share_dir else Mamba3Layer(
            d_model=d_model, d_state=d_state, expand=expand,
            headdim=headdim, mimo_rank=mimo_rank)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model)
        self.adaln = AdaLNZero(d_model, d_temb, n_branches=2) if adaln_zero else None

    def forward(self, x: Tensor, mask: Tensor, temb: Tensor | None = None) -> Tensor:
        """Residual: x + mamba_f(h) + flip(mamba_b(flip(h))) + FFN(RMSNorm(x)),
        padding zeroed. With share_dir, mamba_b is the (weight-tied) mamba_f."""
        h = self.norm1(x)
        gate = None
        if self.adaln is not None:
            h, gate = self.adaln(h, temb, branch=0)
        mamba_b = self.mamba_f if self.share_dir else self.mamba_b
        y_f = self.mamba_f(h, mask)
        y_b = _flip_by_mask(mamba_b(_flip_by_mask(h, mask), mask), mask)
        y = y_f + y_b
        x = x + (gate * y if gate is not None else y)
        h = self.norm2(x)
        if self.adaln is not None:
            h, gate = self.adaln(h, temb, branch=1)
            x = x + gate * self.ffn(h)
        else:
            x = x + self.ffn(h)
        return x * mask.unsqueeze(-1).to(x.dtype)


class TimeFiLM(nn.Module):
    """Zero-init feature-wise modulation from the FM time embedding."""

    def __init__(self, d_model: int, d_temb: int):
        super().__init__()
        self.proj = nn.Linear(d_temb, 2 * d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, temb: Tensor) -> Tensor:
        scale, shift = self.proj(temb).chunk(2, dim=-1)
        return x * (1 + scale.to(x.dtype).unsqueeze(1)) + shift.to(x.dtype).unsqueeze(1)


class MambaStack(nn.Module):
    """Reusable stack of Mamba-3 blocks.

    There is no attention layer to intersperse. The hybrid Nemotron-style
    attention path this stack used to carry was removed: this project's claim is
    that a pure SSM trunk suffices, so an attention escape hatch reachable by
    config is a claim the code cannot make. An all-attention control arm is a
    separate trunk, written for the purpose — not a flag on this one.
    """

    def __init__(self, d_model: int, n_layers: int, d_state: int = 64,
                 mimo_rank: int = 4, expand: int = 2, headdim: int = 64,
                 bidirectional: bool = True,
                 bimamba_share: bool = False,
                 layerwise_time_film: bool = False,
                 adaln_zero: bool = False,
                 d_temb: int = 128):
        """
        Args:
            d_model, n_layers, d_state, mimo_rank, expand, headdim, bidirectional:
                Mamba block hyperparameters.
            bimamba_share: weight-tie the two BiMamba directions (halves SSM params).
            layerwise_time_film: inject the FM time embedding before every block.
            adaln_zero: add time-conditioned scale/shift/gates inside each
                residual branch, zero-initialized so the stack starts as identity.
        """
        super().__init__()
        self.n_layers = n_layers
        self.adaln_zero = adaln_zero

        layers = []
        for _ in range(n_layers):
            if bidirectional:
                layers.append(BiMamba3Block(d_model=d_model, d_state=d_state,
                                            mimo_rank=mimo_rank, expand=expand,
                                            headdim=headdim, share_dir=bimamba_share,
                                            adaln_zero=adaln_zero, d_temb=d_temb))
            else:
                layers.append(Mamba3Block(d_model=d_model, d_state=d_state,
                                          mimo_rank=mimo_rank, expand=expand,
                                          headdim=headdim,
                                          adaln_zero=adaln_zero, d_temb=d_temb))
        self.layers = nn.ModuleList(layers)
        self.time_films = nn.ModuleList([
            TimeFiLM(d_model, d_temb) for _ in range(n_layers)
        ]) if layerwise_time_film else None

    def forward(self, x: Tensor, mask: Tensor, temb: Tensor | None = None) -> Tensor:
        """Pass through all blocks (sequential unit-weight residual).
        [B, S, d_model] → same."""
        if (self.time_films is not None or self.adaln_zero) and temb is None:
            raise RuntimeError("MambaStack time conditioning requires temb.")
        for i, layer in enumerate(self.layers):
            if self.time_films is not None:
                x = self.time_films[i](x, temb)
            x = layer(x, mask, temb)
        return x
