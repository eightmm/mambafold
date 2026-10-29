"""Native Muon for hidden linear weights, AdamW for the remaining parameters."""

from __future__ import annotations

import math
from copy import deepcopy

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel


def parameter_groups(model: nn.Module) -> dict[str, list[tuple[str, nn.Parameter]]]:
    """Partition by module role, not dimensionality alone; shared weights appear once."""
    raw = model.module if isinstance(model, DistributedDataParallel) else model
    # The coordinate readout is the model's output head, not a hidden projection.
    head = getattr(getattr(raw, "atom_decoder", None), "out", None)
    excluded = {id(p) for module in raw.modules() if isinstance(module, nn.Embedding)
                for p in module.parameters(recurse=False)}
    if head is not None:
        excluded.update(id(p) for p in head.parameters())
    matrices = {id(module.weight) for module in raw.modules()
                if isinstance(module, nn.Linear) and module.weight.ndim == 2}
    matrices -= excluded
    groups = {"muon": [], "adamw": []}
    for name, parameter in raw.named_parameters():
        if parameter.requires_grad:
            kind = "muon" if id(parameter) in matrices else "adamw"
            groups[kind].append((name, parameter))
    return groups


def _group_specs(groups, backbone_lr_scale, backbone_prefixes=()):
    specs = []
    for kind, pairs in groups.items():
        roles = [(None, pairs)] if backbone_lr_scale == 1 else [
            (role, [(name, p) for name, p in pairs
                    if (name.startswith("latent_residual.")
                        and not name.startswith(backbone_prefixes)) == (role == "refiner")])
            for role in ("backbone", "refiner")
        ]
        for role, selected in roles:
            if selected:
                spec = {"params": [p for _, p in selected],
                        "param_names": [n for n, _ in selected], "optimizer_kind": kind}
                if role:
                    spec.update(parameter_role=role,
                                lr_scale=backbone_lr_scale if role == "backbone" else 1.0)
                specs.append(spec)
    return specs


class MuonAdamW(torch.optim.Optimizer):
    """One scheduler/checkpoint interface over two disjoint native optimizers.

    Both children see the final accumulated, DDP-reduced gradient. The parent's
    parameter-group dictionaries are the children's actual dictionaries so a
    single LambdaLR schedules both, including after checkpoint restoration.
    """

    def __init__(self, groups, *, lr: float, weight_decay: float, fused: bool,
                 muon_lr: float | None = None, muon_adjust_lr_fn: str = "match_rms_adamw",
                 backbone_lr_scale: float = 1.0, backbone_prefixes=()):
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError("optimizer=muon requires PyTorch with torch.optim.Muon")
        if not groups["muon"]:
            raise ValueError("optimizer=muon requires at least one trainable hidden linear weight")
        specs = _group_specs(groups, backbone_lr_scale, backbone_prefixes)
        super().__init__(specs, {"lr": lr, "weight_decay": weight_decay})
        self._layout = [
            {"kind": group["optimizer_kind"], "names": list(group["param_names"]),
             "shapes": [tuple(p.shape) for p in group["params"]]}
            for group in self.param_groups
        ]
        self.optimizers = {}
        for kind in groups:
            selected = [group for group in self.param_groups if group["optimizer_kind"] == kind]
            if not selected:
                continue
            for group in selected:
                peak = (lr if muon_lr is None else muon_lr) if kind == "muon" else lr
                group["lr"] = peak * group.get("lr_scale", 1.0)
            if kind == "muon":
                child = torch.optim.Muon(
                    selected, lr=lr, weight_decay=weight_decay, momentum=0.95,
                    nesterov=True, ns_steps=5, adjust_lr_fn=muon_adjust_lr_fn,
                )
            else:
                child = torch.optim.AdamW(selected, lr=lr, weight_decay=weight_decay, fused=fused)
            self.optimizers[kind] = child

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for optimizer in self.optimizers.values():
            optimizer.step()
        return loss

    def state_dict(self):
        return {
            "optimizer_kind": "muon", "format_version": 1,
            "layout": deepcopy(self._layout),
            "children": {kind: optimizer.state_dict()
                         for kind, optimizer in self.optimizers.items()},
        }

    def load_state_dict(self, state_dict):
        if state_dict.get("optimizer_kind") != "muon":
            raise ValueError("Cannot load AdamW state into Muon; use --reset_optimizer")
        if state_dict.get("format_version") != 1 or state_dict.get("layout") != self._layout:
            raise ValueError("Muon parameter layout changed; use --reset_optimizer")
        children = state_dict.get("children", {})
        if set(children) != set(self.optimizers):
            raise ValueError("Muon optimizer groups changed; use --reset_optimizer")
        # Validate every group before allowing either child to mutate its state.
        for kind, optimizer in self.optimizers.items():
            saved = children[kind].get("param_groups", [])
            expected = [layout for layout in self._layout if layout["kind"] == kind]
            if len(saved) != len(expected):
                raise ValueError("Muon checkpoint parameter groups mismatch; use --reset_optimizer")
            for group, layout, current in zip(saved, expected, optimizer.param_groups, strict=True):
                if (group.get("param_names") != layout["names"]
                        or len(group["params"]) != len(layout["names"])
                        or group.get("lr_scale", 1.0) != current.get("lr_scale", 1.0)):
                    raise ValueError(
                        "Muon checkpoint parameter groups mismatch; use --reset_optimizer"
                    )
        for kind, optimizer in self.optimizers.items():
            optimizer.load_state_dict(children[kind])
        self.param_groups = [group for optimizer in self.optimizers.values()
                             for group in optimizer.param_groups]


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    kind = cfg.get("optimizer", "adamw")
    parameters = [p for p in model.parameters() if p.requires_grad]
    fused = bool(parameters) and all(p.device.type == "cuda" for p in parameters)
    kwargs = dict(lr=cfg.get("lr", 1e-4), weight_decay=cfg.get("weight_decay", 0.0))
    scale = cfg.get("backbone_lr_scale", 1.0)
    if not isinstance(scale, (int, float)) or not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError("backbone_lr_scale must be finite and in (0,1]")
    raw = model.module if isinstance(model, DistributedDataParallel) else model
    if scale != 1 and not getattr(raw, "residue_refiner", False):
        raise ValueError("backbone_lr_scale requires a residue refiner")
    if kind == "adamw":
        if scale != 1:
            specs = _group_specs({"adamw": [(n, p) for n, p in raw.named_parameters()
                                           if p.requires_grad]}, scale,
                                 getattr(raw, "backbone_parameter_prefixes", ()))
            for spec in specs:
                spec["lr"] = kwargs["lr"] * spec["lr_scale"]
            return torch.optim.AdamW(specs, fused=fused, **kwargs)
        return torch.optim.AdamW(parameters, fused=fused, **kwargs)
    if kind == "muon":
        return MuonAdamW(
            parameter_groups(model), fused=fused, **kwargs,
            muon_lr=cfg.get("muon_lr"),
            muon_adjust_lr_fn=cfg.get("muon_adjust_lr_fn", "match_rms_adamw"),
            backbone_lr_scale=scale,
            backbone_prefixes=getattr(raw, "backbone_parameter_prefixes", ()),
        )
    raise ValueError(f"Unknown optimizer: {kind!r}")


def load_optimizer_state(optimizer: torch.optim.Optimizer, state: dict) -> None:
    current = "muon" if isinstance(optimizer, MuonAdamW) else "adamw"
    saved = state.get("optimizer_kind", "adamw")
    if current != saved:
        raise ValueError(
            f"Optimizer changed from {saved} to {current}; use --reset_optimizer"
        )
    if current == "adamw" and any("parameter_role" in g for g in optimizer.param_groups):
        groups = state.get("param_groups", [])
        if len(groups) != len(optimizer.param_groups) or any(
            old.get("param_names") != new.get("param_names")
            or old.get("lr_scale", 1.0) != new.get("lr_scale", 1.0)
            for old, new in zip(groups, optimizer.param_groups, strict=True)
        ):
            raise ValueError("AdamW backbone LR groups changed; use --reset_optimizer")
    optimizer.load_state_dict(state)


def optimizer_summary(optimizer: torch.optim.Optimizer) -> dict:
    children = (optimizer.optimizers if isinstance(optimizer, MuonAdamW)
                else {"adamw": optimizer})
    return {
        kind: {"lr": child.param_groups[0]["lr"],
               "adjust_lr_fn": child.param_groups[0].get("adjust_lr_fn"),
               "parameters": sum(p.numel() for g in child.param_groups for p in g["params"]),
               "tensors": sum(len(g["params"]) for g in child.param_groups),
               "role_lrs": {g.get("parameter_role", "all"): g["lr"]
                            for g in child.param_groups}}
        for kind, child in children.items()
    }
