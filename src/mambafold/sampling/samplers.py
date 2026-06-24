"""Direct all-atom Euler sampler."""

from __future__ import annotations

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
):
    """Sample all atom slots with one flow-matching Euler trajectory.

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

    sched = torch.linspace(0.0, _T_END, n_steps + 1, device=device)
    traj = []
    amp_enabled = str(device).startswith("cuda")
    for i in range(n_steps):
        ti = float(sched[i].clamp(min=1e-4))
        dt = float(sched[i + 1] - sched[i])
        batch = batch_fn(x, ti)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
            out = model(batch, return_aux=False)
            v_atom = out["v_atom"].squeeze(0)
        x = (x + dt * v_atom) * atom_mask_f
        # Re-center to zero CoM each step: training data/noise are centroid-centered,
        # and the model is not translation-equivariant, so accumulated Euler drift
        # would walk the input off-distribution.
        x = (x - masked_centroid(x.reshape(-1, 3), valid).unsqueeze(0)) * atom_mask_f
        traj.append(x[:, CA_ATOM_ID, :].float().cpu().numpy() * COORD_SCALE)

    batch = batch_fn(x, float(sched[-1]))
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
        out = model(batch, return_aux=False)
        v_final = out["v_atom"].squeeze(0)
        conf = out["conf"].squeeze(0).float().cpu().numpy()        # [L] predicted pLDDT
    x_clean = (x + (1.0 - float(sched[-1])) * v_final) * atom_mask_f

    final_aa = x_clean.float().cpu().numpy() * COORD_SCALE
    final_ca = final_aa[:, CA_ATOM_ID, :]
    return final_ca, final_aa, np.asarray(traj, dtype=np.float32), sched.cpu().numpy(), conf
