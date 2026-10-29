"""Mamba-3 SSM stack: primitives, causal/bidirectional blocks, reusable stack.

Requires the Mamba-3 revision pinned in pyproject.toml / uv.lock, including
heavy-tail data-dependent decay and the updated MIMO backward kernels.

Reference: github.com/state-spaces/mamba  |  arXiv:2603.15569
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import sys
import types
from functools import wraps
from pathlib import Path

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
        module = importlib.import_module("mamba_ssm.modules.mamba3")
    except ImportError as exc:
        message = str(exc)
        if "selective_scan_cuda" not in message and "libcudart.so" not in message:
            raise
        for name in list(sys.modules):
            if name == "mamba_ssm" or name.startswith("mamba_ssm."):
                sys.modules.pop(name, None)
        sys.modules["selective_scan_cuda"] = types.ModuleType("selective_scan_cuda")
        module = importlib.import_module("mamba_ssm.modules.mamba3")
    if not callable(getattr(module, "heavy_tail_activation", None)):
        raise RuntimeError(
            "This latent model requires Mamba-3 with heavy-tail decay. "
            "Install the mamba-ssm revision pinned in pyproject.toml / uv.lock "
            "and use this project's Python environment."
        )
    return module.Mamba3


def _stable_symbolic_factory(factory):
    """Reuse symbolic identities so TileLang's own JIT cache can hit.

    Upstream creates fresh TIR Vars for B/S/H/G on each dense kernel call.
    TileLang 0.1.8 keys its first cache by those objects, rebuilding the TIR
    before its second cache finds the already compiled kernel. Canonicalize
    symbols only; leave static options and kernel/target caching to TileLang.
    """
    from tvm import tir

    symbols = {}

    def canonical(value):
        if isinstance(value, tir.Var):
            return symbols.setdefault((value.name, str(value.dtype)), value)
        return value

    @wraps(factory)
    def call(*args, **kwargs):
        return factory(
            *(canonical(arg) for arg in args),
            **{key: canonical(value) for key, value in kwargs.items()},
        )

    return call


_Mamba3 = _load_mamba3_class()

# Limit the adapter to the three dense factories used by this folding model.
# No activation, parameter, output or autograd state is cached here.
for _module_name, _factory_names in (
    ("mamba3_mimo_fwd", ("mamba_mimo_fwd",)),
    ("mamba3_mimo_bwd", ("mamba_mimo_bwd_fwd", "mamba_mimo_bwd_bwd")),
):
    _module = importlib.import_module(f"mamba_ssm.ops.tilelang.mamba3.{_module_name}")
    for _name in _factory_names:
        setattr(_module, _name, _stable_symbolic_factory(getattr(_module, _name)))


def mamba3_runtime_info(model: nn.Module | None = None) -> dict:
    """Record the external implementation that determines checkpoint dynamics."""
    distribution = importlib.metadata.distribution("mamba-ssm")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    module = importlib.import_module(_Mamba3.__module__)
    layers = (
        [module for module in model.modules() if isinstance(module, Mamba3Layer)]
        if model is not None else []
    )
    modes = {layer.ssm.is_outproj_norm for layer in layers}
    outproj_norm = "mixed" if len(modes) > 1 else next(iter(modes), False)
    return {
        "version": distribution.version,
        "revision": direct_url.get("vcs_info", {}).get("commit_id"),
        "module_sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        "decay": "heavy_tail",
        "outproj_norm": outproj_norm,
        "stable_kernel_symbols": True,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
    }


def _default_chunk_size(mimo_rank: int) -> int:
    """GPU-aware chunk size for Mamba-3 SSD kernels.

    Use 64 // rank only on newer GPUs with at least 160 KiB opt-in shared
    memory. Blackwell workstation GPUs have an Ada-sized per-block budget;
    their generation alone does not make the larger backward tile launchable.
    """
    if mimo_rank <= 1:
        return 64
    base = 32
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(None)
            if (torch.cuda.get_device_capability() >= (9, 0)
                    and props.shared_memory_per_block_optin >= 160 * 1024):
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

    def __init__(self, d_model: int, d_temb: int, n_branches: int = 2, *, stable: bool = False):
        super().__init__()
        self.d_model = d_model
        self.n_branches = n_branches
        self.stable = stable
        self.proj = nn.Linear(d_temb, n_branches * 3 * d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self, x: Tensor, temb: Tensor, branch: int, repeat: int = 1
    ) -> tuple[Tensor, Tensor]:
        """Modulate one residual branch.

        `repeat` covers the atom stacks, which flatten [B, L, A, d] to
        [B*L, A, d]: the projection runs on the B rows of `temb` and only its
        output is expanded. Expanding `temb` first instead would run a
        d_temb -> 6*d_model matmul B*L times to produce B distinct results.
        """
        return self.modulate(x, self.project(temb), branch, repeat)

    def project(self, temb: Tensor) -> Tensor:
        """Compute both branches once per block invocation, retaining gradients."""
        if temb is None:
            raise RuntimeError("AdaLNZero requires temb.")
        if self.stable:
            temb = F.rms_norm(temb, (temb.shape[-1],), eps=1e-6)
        params = self.proj(F.silu(temb)).view(temb.shape[0], self.n_branches, 3, self.d_model)
        return params.tanh() if self.stable else params

    def modulate(
        self, x: Tensor, params: Tensor, branch: int, repeat: int = 1
    ) -> tuple[Tensor, Tensor]:
        scale, shift, gate = params[:, branch].unbind(dim=1)
        if repeat > 1:
            scale = scale.repeat_interleave(repeat, dim=0)
            shift = shift.repeat_interleave(repeat, dim=0)
            gate = gate.repeat_interleave(repeat, dim=0)
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
        outproj_norm: bool = False,
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
            outproj_norm: Enable the native headwise pre-gate output norm.
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
            is_outproj_norm=outproj_norm,
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
                 expand: int = 2, headdim: int = 64, d_temb: int = 128,
                 stable: bool = False):
        """
        Args:
            d_model (int): Token feature dimension. Input/output shape [B, S, d_model].
            d_state (int): SSM state expansion factor. Default: 64.
            mimo_rank (int): MIMO rank forwarded to Mamba3Layer. Default: 4.
            expand (int): Inner dimension multiplier inside the SSM. Default: 2.
            headdim (int): Dimension per head inside the SSM. Default: 64.
        """
        super().__init__()
        self.stable = stable
        self.norm1 = RMSNorm(d_model)
        self.ssm = Mamba3Layer(d_model=d_model, d_state=d_state, expand=expand,
                               headdim=headdim, mimo_rank=mimo_rank, outproj_norm=stable)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model)
        self.adaln = AdaLNZero(d_model, d_temb, n_branches=2, stable=stable)

    def forward(
        self, x: Tensor, mask: Tensor, temb: Tensor, temb_repeat: int = 1
    ) -> Tensor:
        """x: [B, S, d_model], mask: [B, S], temb: [B, d_temb].

        Both residual branches are AdaLN-Zero modulated by the flow-matching
        time. Returns [B, S, d_model] with padding zeroed.
        """
        modulation = self.adaln.project(temb)
        h, gate = self.adaln.modulate(self.norm1(x), modulation, branch=0, repeat=temb_repeat)
        update = self.ssm(h, mask)
        if self.stable:
            update = F.rms_norm(update, (update.shape[-1],), eps=1e-6)
        x = x + gate * update
        h, gate = self.adaln.modulate(self.norm2(x), modulation, branch=1, repeat=temb_repeat)
        update = self.ffn(h)
        if self.stable:
            update = F.rms_norm(update, (update.shape[-1],), eps=1e-6)
        x = x + gate * update
        return x * mask.unsqueeze(-1).to(x.dtype)


class BiMamba3Block(nn.Module):
    """Bidirectional Mamba-3: forward + backward SSM summed."""

    def __init__(self, d_model: int, d_state: int = 64, mimo_rank: int = 4,
                 expand: int = 2, headdim: int = 64, share_dir: bool = False,
                 d_temb: int = 128, stable: bool = False):
        """
        Args:
            d_model (int): Token feature dimension. Input/output shape [B, S, d_model].
            d_state (int): SSM state expansion factor shared by both directions.
                Default: 64.
            mimo_rank (int): MIMO rank forwarded to both Mamba3Layers. Default: 4.
            expand (int): Inner dimension multiplier inside each SSM. Default: 2.
            headdim (int): Dimension per head inside each SSM. Default: 64.
            share_dir (bool): weight-tie the two directions — run a single SSM on
                both the forward and reversed sequence. Halves SSM parameters;
                both scans still execute. Default: False (separate fwd/bwd SSMs).
        """
        super().__init__()
        self.share_dir = share_dir
        self.stable = stable
        self.norm1 = RMSNorm(d_model)
        self.mamba_f = Mamba3Layer(d_model=d_model, d_state=d_state, expand=expand,
                                   headdim=headdim, mimo_rank=mimo_rank, outproj_norm=stable)
        self.mamba_b = None if share_dir else Mamba3Layer(
            d_model=d_model, d_state=d_state, expand=expand,
            headdim=headdim, mimo_rank=mimo_rank, outproj_norm=stable)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model)
        self.adaln = AdaLNZero(d_model, d_temb, n_branches=2, stable=stable)
        # Per-channel weight on each scan direction. A protein reads N->C, so
        # the two directions are not interchangeable and a channel should be
        # able to prefer one. Both weights start at 1, which is exactly the
        # plain sum this replaces — a convex `g*y_f + (1-g)*y_b` would instead
        # have started at half the magnitude, and would force every channel to
        # trade one direction against the other rather than scale them
        # independently.
        self.w_fwd = nn.Parameter(torch.ones(d_model))
        self.w_bwd = nn.Parameter(torch.ones(d_model))

    def forward(
        self, x: Tensor, mask: Tensor, temb: Tensor, temb_repeat: int = 1
    ) -> Tensor:
        """Residual: x + g0*(bi-SSM(mod(RMSNorm(x)))) + g1*FFN(mod(RMSNorm(x))).

        Both branches are AdaLN-Zero modulated by the flow-matching time, so
        every block knows the noise level rather than only the stack's input.
        With share_dir, mamba_b is the (weight-tied) mamba_f."""
        modulation = self.adaln.project(temb)
        h, gate = self.adaln.modulate(self.norm1(x), modulation, branch=0, repeat=temb_repeat)
        mamba_b = self.mamba_f if self.share_dir else self.mamba_b
        y_f = self.mamba_f(h, mask)
        y_b = _flip_by_mask(mamba_b(_flip_by_mask(h, mask), mask), mask)
        update = self.w_fwd.to(y_f.dtype) * y_f + self.w_bwd.to(y_b.dtype) * y_b
        if self.stable:
            # Bound the residual after the SSM gate as well as its native
            # pre-gate norm: a large learned Z can otherwise amplify it again.
            update = F.rms_norm(update, (update.shape[-1],), eps=1e-6)
        x = x + gate * update
        h, gate = self.adaln.modulate(self.norm2(x), modulation, branch=1, repeat=temb_repeat)
        update = self.ffn(h)
        if self.stable:
            update = F.rms_norm(update, (update.shape[-1],), eps=1e-6)
        x = x + gate * update
        return x * mask.unsqueeze(-1).to(x.dtype)


class MambaStack(nn.Module):
    """Reusable stack of Mamba-3 blocks.

    There is no attention layer to intersperse. The hybrid Nemotron-style
    attention path this stack used to carry was removed: this project's claim is
    that a pure SSM trunk suffices, so an attention escape hatch reachable by
    config is a claim the code cannot make. That reasoning stands on its own:
    it is why attention is absent rather than disabled. It once also promised an
    all-attention control trunk written for the purpose; that arm was cancelled
    on cost, so this stack has no counterpart to be compared against.
    """

    def __init__(self, d_model: int, n_layers: int, d_state: int = 64,
                 mimo_rank: int = 4, expand: int = 2, headdim: int = 64,
                 bidirectional: bool = True,
                 bimamba_share: bool = False,
                 d_temb: int = 128, stable: bool = False):
        """
        Args:
            d_model, n_layers, d_state, mimo_rank, expand, headdim, bidirectional:
                Mamba block hyperparameters.
            bimamba_share: weight-tie the two BiMamba directions (halves SSM params).
            d_temb: width of the FM time embedding driving AdaLN-Zero. Every
                block is time-modulated; this is not optional.
        """
        super().__init__()
        self.n_layers = n_layers

        layers = []
        for _ in range(n_layers):
            if bidirectional:
                layers.append(BiMamba3Block(d_model=d_model, d_state=d_state,
                                            mimo_rank=mimo_rank, expand=expand,
                                            headdim=headdim, share_dir=bimamba_share,
                                            d_temb=d_temb, stable=stable))
            else:
                layers.append(Mamba3Block(d_model=d_model, d_state=d_state,
                                          mimo_rank=mimo_rank, expand=expand,
                                          headdim=headdim, d_temb=d_temb, stable=stable))
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        temb: Tensor,
        temb_repeat: int = 1,
        inject: Tensor | None = None,
        inject_at: tuple[int, ...] = (),
        midpoint_residual: nn.Module | None = None,
    ) -> Tensor:
        """Pass through all blocks. [B, S, d_model] → same.

        `inject` is added to the residual stream before each block index in
        `inject_at`. Attention re-reads every position at every layer with
        content-dependent weights; a scan cannot, so a per-position signal
        supplied only at the stack's input has to survive every intervening
        residual block to be usable deep in the trunk. Re-adding it is the
        cheapest way to keep it available.

        `temb_repeat` lets a caller that flattened a leading axis into the batch
        — the atom stacks turn [B, L, A, d] into [B*L, A, d] — pass `temb` with
        its original B rows and have the modulation broadcast.

        `midpoint_residual`, when supplied for a residue trunk, runs after
        n_layers // 2 blocks. Existing layer indices and checkpoint keys stay
        unchanged; the caller must provide at least two blocks.
        """
        if temb is None:
            raise RuntimeError("MambaStack is time-conditioned and requires temb.")
        if midpoint_residual is not None and self.n_layers < 2:
            raise ValueError("midpoint residual requires at least two trunk blocks")
        inject_at = set(inject_at)
        for i, layer in enumerate(self.layers):
            if inject is not None and i in inject_at:
                x = x + inject
            x = layer(x, mask, temb, temb_repeat)
            if midpoint_residual is not None and i + 1 == self.n_layers // 2:
                x = midpoint_residual(x, mask, temb)
        return x
