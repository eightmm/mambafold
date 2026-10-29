"""Exponential Moving Average of model parameters."""

import copy

import torch
import torch.nn as nn


class EMA:
    """Maintains an exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update shadow params: shadow = decay * shadow + (1-decay) * model."""
        groups = {}
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            key = (s_param.device, s_param.dtype, m_param.device, m_param.dtype)
            shadow, current = groups.setdefault(key, ([], []))
            shadow.append(s_param)
            current.append(m_param)
        for shadow, current in groups.values():
            # Preserve the two-operation rounding and parameter-only checkpoint
            # contract; batching avoids two CUDA launches for every parameter.
            torch._foreach_mul_(shadow, self.decay)
            torch._foreach_add_(shadow, current, alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.shadow.load_state_dict(state_dict, strict=strict)
