"""Direct all-atom ODE/SDE samplers."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Callable

import numpy as np
import torch
from torch import Tensor

from mambafold.data.constants import CA_ATOM_ID, COORD_SCALE
from mambafold.utils.geometry import masked_centroid

BatchFn = Callable[[Tensor, float], "object"]
_T_END = 0.99


@torch.no_grad()
def sample(
    model,
    example,
    batch_fn: BatchFn,
    *,
    n_steps: int = 50,
    seed: int = 0,
    device: str = "cuda",
    sampler: str = "ode",
    sde_tau: float = 0.01,
    sde_eps: float = 0.01,
    sde_w_cutoff: float = 0.99,
    sde_log_timesteps: bool = True,
):
    """Sample all atom slots with one flow-matching trajectory.

    ``sampler="ode"`` is the default Euler flow path. ``sampler="sde"`` follows
    SimpleFold's Euler-Maruyama solver for the linear flow path.

    Returns:
        final_ca: [L, 3] in Angstrom
        final_aa: [L, A, 3] in Angstrom
        traj_ca: [steps, L, 3] in Angstrom
        sched: time schedule
        conf: [L] predicted per-residue confidence (pLDDT in [0, 1])
    """
    model.eval()
    atom_mask_f = example.atom_mask.unsqueeze(-1).float().to(device)
    L, A = example.atom_mask.shape
    torch.manual_seed(seed)
    # Match training noise: flow_corrupt centers ε to zero-mean over valid atoms,
    # so the prior here must be centered the same way or t→0 is off-distribution.
    x = torch.randn(L, A, 3, device=device)
    valid = example.atom_mask.reshape(-1).to(device)
    x = x - masked_centroid(x.reshape(-1, 3), valid).unsqueeze(0)
    x = x * atom_mask_f

    if sampler not in {"ode", "sde"}:
        raise ValueError(f"unknown sampler: {sampler}")

    if sampler == "sde" and sde_log_timesteps:
        sched = 1.0 - torch.logspace(-2, 0, n_steps + 1, device=device).flip(0)
        sched = sched - sched.min()
        sched = (sched / sched.max()).clamp(min=1e-4, max=1.0)
    elif sampler == "sde":
        sched = torch.linspace(1e-4, 1.0, n_steps + 1, device=device)
    else:
        sched = torch.linspace(0.0, _T_END, n_steps + 1, device=device)
    traj = []
    amp_enabled = str(device).startswith("cuda")
    x_self_cond = None
    for i in range(n_steps):
        ti = float(sched[i].clamp(min=1e-4))
        dt = float(sched[i + 1] - sched[i])
        x = (x - masked_centroid(x.reshape(-1, 3), valid).unsqueeze(0)) * atom_mask_f
        batch = batch_fn(x, ti)
        if x_self_cond is not None:
            batch = replace(batch, x_self_cond=x_self_cond.unsqueeze(0))
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
            out = model(batch, return_aux=False)
            v_atom = out["v_atom"].squeeze(0)
        x_self_cond = (x + (1.0 - ti) * v_atom).detach() * atom_mask_f
        if sampler == "sde":
            if ti >= sde_w_cutoff:
                w = 0.0
            else:
                w = (1.0 - ti) / (ti + sde_eps)
            score = ((ti * v_atom) - x) / max(1.0 - ti, 1e-6)
            drift = v_atom + w * score
            x = (x + dt * drift) * atom_mask_f
            noise_scale = math.sqrt(max(2.0 * dt * w * sde_tau, 0.0))
            if noise_scale > 0.0 and i < n_steps - 1:
                noise = torch.randn_like(x) * atom_mask_f
                noise = (
                    noise - masked_centroid(noise.reshape(-1, 3), valid).unsqueeze(0)
                ) * atom_mask_f
                x = (x + noise_scale * noise) * atom_mask_f
        else:
            x = (x + dt * v_atom) * atom_mask_f
        # Re-center to zero CoM each step: training data/noise are centroid-centered,
        # and the model is not translation-equivariant, so accumulated Euler drift
        # would walk the input off-distribution.
        x = (x - masked_centroid(x.reshape(-1, 3), valid).unsqueeze(0)) * atom_mask_f
        traj.append(x[:, CA_ATOM_ID, :].float().cpu().numpy() * COORD_SCALE)

    batch = batch_fn(x, float(sched[-1]))
    if x_self_cond is not None:
        batch = replace(batch, x_self_cond=x_self_cond.unsqueeze(0))
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
        out = model(batch, return_aux=False)
        v_final = out["v_atom"].squeeze(0)
        conf = out["conf"].squeeze(0).float().cpu().numpy()  # [L] predicted pLDDT
    if sampler == "sde":
        x_clean = x * atom_mask_f
    else:
        x_clean = (x + (1.0 - float(sched[-1])) * v_final) * atom_mask_f

    final_aa = x_clean.float().cpu().numpy() * COORD_SCALE
    final_ca = final_aa[:, CA_ATOM_ID, :]
    return final_ca, final_aa, np.asarray(traj, dtype=np.float32), sched.cpu().numpy(), conf
