"""Standalone residue-level pLDDT prediction head.

The confidence head is deliberately separate from the folding model.  It can be
trained, checkpointed, and calibrated without changing a folding checkpoint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class PLDDTHeadConfig:
    """Architecture contract for :class:`PLDDTHead`.

    Defaults match the 1024-channel MambaFold residue trunk.  Small values may
    be supplied for tests or ablations as long as ``d_model`` is divisible by
    ``n_heads``.
    """

    d_model: int = 1024
    n_bins: int = 50
    n_layers: int = 4
    n_heads: int = 16
    ff_mult: int = 4
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for name in ("d_model", "n_bins", "n_layers", "n_heads", "ff_mult"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.n_bins < 2:
            raise ValueError(f"n_bins must be at least 2, got {self.n_bins}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)):
            raise ValueError(f"dropout must be a real number in [0, 1), got {self.dropout!r}")
        dropout = float(self.dropout)
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout!r}")
        object.__setattr__(self, "dropout", dropout)

    def to_dict(self) -> dict[str, int | float]:
        """Return a checkpoint-safe mapping containing only plain scalar types."""

        return asdict(self)

    @classmethod
    def from_dict(cls, value: Any) -> PLDDTHeadConfig:
        """Construct a config from an exact, plain mapping schema."""

        if not isinstance(value, dict):
            raise ValueError(f"config must be a dict, got {type(value).__name__}")
        expected = {field.name for field in cls.__dataclass_fields__.values()}
        actual = set(value)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(f"invalid config keys: missing={missing}, extra={extra}")
        if not all(isinstance(key, str) for key in value):
            raise ValueError("config keys must be strings")
        return cls(**value)


class _TransformerBlock(nn.Module):
    """Pre-norm Transformer block used by the confidence-only residue stack."""

    def __init__(self, config: PLDDTHeadConfig) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.ff_norm = nn.LayerNorm(config.d_model)
        self.ff_in = nn.Linear(config.d_model, config.d_model * config.ff_mult)
        self.ff_out = nn.Linear(config.d_model * config.ff_mult, config.d_model)
        self.dropout = config.dropout

    def forward(self, x: Tensor, *, key_padding_mask: Tensor, res_mask: Tensor) -> Tensor:
        normed = self.attn_norm(x)
        attn, _ = self.attn(
            normed,
            normed,
            normed,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + F.dropout(attn, p=self.dropout, training=self.training)
        ff = self.ff_in(self.ff_norm(x))
        ff = F.gelu(ff, approximate="none")
        ff = F.dropout(ff, p=self.dropout, training=self.training)
        ff = self.ff_out(ff)
        x = x + F.dropout(ff, p=self.dropout, training=self.training)
        return x.masked_fill(~res_mask.unsqueeze(-1), 0.0)


class PLDDTHead(nn.Module):
    """Transformer-style pLDDT head over frozen folding residue latents.

    Args:
        config: Serializable head architecture.

    Inputs:
        latent: Residue representation with shape ``[B, L, D]``.
        res_mask: Boolean valid-residue mask with shape ``[B, L]``.

    Returns:
        Unnormalized lDDT-bin logits with shape ``[B, L, n_bins]``.  Logits at
        padding positions are zero; callers must still use ``res_mask`` in the
        loss and when exporting scores.
    """

    def __init__(self, config: PLDDTHeadConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else PLDDTHeadConfig()
        self.blocks = nn.ModuleList(
            [_TransformerBlock(self.config) for _ in range(self.config.n_layers)]
        )
        self.final_norm = nn.LayerNorm(self.config.d_model)
        self.to_logits = nn.Linear(self.config.d_model, self.config.n_bins)

    def forward(self, latent: Tensor, res_mask: Tensor) -> Tensor:
        _validate_inputs(latent, res_mask, self.config.d_model)

        # MultiheadAttention produces NaNs when every key in a row is masked.
        # Give an empty row one zero-valued dummy key, then mask its output back
        # to zero.  Real rows and padding positions retain their usual masks.
        safe_res_mask = res_mask
        empty_rows = ~res_mask.any(dim=1)
        if empty_rows.any():
            safe_res_mask = res_mask.clone()
            safe_res_mask[empty_rows, 0] = True
        key_padding_mask = ~safe_res_mask

        x = latent.masked_fill(~res_mask.unsqueeze(-1), 0.0)
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask, res_mask=res_mask)
        logits = self.to_logits(self.final_norm(x))
        return logits.masked_fill(~res_mask.unsqueeze(-1), 0.0)


def _validate_inputs(latent: Tensor, res_mask: Tensor, d_model: int) -> None:
    if not isinstance(latent, Tensor) or not isinstance(res_mask, Tensor):
        raise TypeError("latent and res_mask must be torch.Tensor instances")
    if latent.ndim != 3:
        raise ValueError(f"latent must have shape [B, L, D], got {tuple(latent.shape)}")
    if res_mask.ndim != 2:
        raise ValueError(f"res_mask must have shape [B, L], got {tuple(res_mask.shape)}")
    if latent.shape[:2] != res_mask.shape:
        raise ValueError(
            f"latent and res_mask batch/length shapes differ: "
            f"{tuple(latent.shape[:2])} vs {tuple(res_mask.shape)}"
        )
    if latent.shape[1] == 0:
        raise ValueError("sequence length must be positive")
    if latent.shape[-1] != d_model:
        raise ValueError(f"latent width must be {d_model}, got {latent.shape[-1]}")
    if not latent.is_floating_point():
        raise TypeError(f"latent must have a floating dtype, got {latent.dtype}")
    if res_mask.dtype is not torch.bool:
        raise TypeError(f"res_mask must have dtype torch.bool, got {res_mask.dtype}")
    if latent.device != res_mask.device:
        raise ValueError(
            f"latent and res_mask must share a device, got {latent.device}/{res_mask.device}"
        )
