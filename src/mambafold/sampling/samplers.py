"""coarse-to-fine inference samplers.

Two-stage Euler ODE:
    Stage 1: noise → CA (3 channels per residue), conditioned on residue
             features only. Yields `x_ca_final` and `s1_latent`.
    Stage 2: initialised so the CA slot equals `x_ca_final` and the rest
             are fresh noise; conditioned on `s1_latent` and `s1_ca`. Runs
             Euler over all atom slots, allowing a small learned CA residual.

Both stages support inference-time recycling (B1 / SDEdit variant) — the
default `n_recycle=0` reproduces a single Euler trajectory per stage.

External API:
    sample(two_stage, example, batch_fn_factory, *, n_steps_s1, n_steps_s2,
              n_recycle_s1, n_recycle_s2, recycle_t_start, seed, device)
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch import Tensor

from mambafold.data.constants import CA_ATOM_ID, COORD_SCALE

_T_END = 0.99


def _run_euler_segment(
    model,
    example,
    batch_fn,
    *,
    t_start: float,
    t_end: float,
    n_steps: int,
    x_init: Tensor,
    device: str,
):
    """Run one FM Euler segment and return final reconstruction."""
    mask_f = example.atom_mask.unsqueeze(-1).float().to(device)
    sched = torch.linspace(t_start, t_end, n_steps + 1, device=device)
    x = x_init * mask_f
    traj = []
    for i in range(n_steps):
        ti = float(sched[i].clamp(min=1e-4))
        dt = float(sched[i + 1] - sched[i])
        batch = batch_fn(x, ti)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            v = model(batch).squeeze(0)
        x = (x + dt * v) * mask_f
        traj.append(x[:, CA_ATOM_ID, :].float().cpu().numpy() * COORD_SCALE)

    batch = batch_fn(x, float(sched[-1]))
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
        v_final = model(batch).squeeze(0)
    x_clean = (x + (1.0 - float(sched[-1])) * v_final) * mask_f
    return x, x_clean, np.asarray(traj, dtype=np.float32), sched.cpu().numpy()


# A factory that, given Stage 1 outputs, returns a `batch_fn(x_atom, t) → batch`
# for Stage 2 sampling. The factory pattern keeps Stage 2's per-step batch
# construction shared between Stage 1 and Stage 2.
BatchFnFactory = Callable[[Tensor, Tensor], Callable[[Tensor, float], "object"]]


# ── Stage 1 sub-sampler — Euler on CA coordinates only ────────────────────


@torch.no_grad()
def _stage1_run(
    stage1,
    example,
    batch_fn_s1,
    *,
    n_steps: int,
    n_recycle: int,
    recycle_t_start: float,
    seed: int,
    device: str,
):
    """Run Stage 1's CA-only Euler ODE.

    Returns:
        x_ca_final  [L, 3]              — predicted CA in normalised units
        s1_latent   [L, d_res]           — trunk latent from final Stage 1 pass
        traj_ca     [steps_total, L, 3] — CA trajectory in Å
        sched_total [...]               — concatenated schedule values
    """
    # Stage 1 model wraps the same forward signature: stage1(batch) → (v_ca, latent).
    # Adapt `_run_euler_segment` by writing a tiny shim that produces
    # a [B, L, A, 3] velocity with only the CA slot filled.
    class _CaModelShim(torch.nn.Module):
        """Adapter: wrap Stage 1 as an atom-shaped velocity model."""

        def __init__(self, inner, n_atoms):
            super().__init__()
            self.inner = inner
            self.n_atoms = n_atoms

        def forward(self, batch):
            v_ca, latent = self.inner(batch)  # v_ca: [B, L, 3]
            shim = v_ca.new_zeros(v_ca.shape[0], v_ca.shape[1], self.n_atoms, 3)
            shim[..., CA_ATOM_ID, :] = v_ca
            # cache the latent for the caller to retrieve later
            self.last_latent = latent
            return shim

    n_atoms = example.atom_mask.shape[1]
    shim = _CaModelShim(stage1, n_atoms).to(device)

    # Initial noise: only on CA slot (others are kept at 0 — Stage 1 ignores them anyway).
    atom_mask_f = example.atom_mask.unsqueeze(-1).float().to(device)
    torch.manual_seed(seed)
    x = torch.zeros(example.seq_len, n_atoms, 3, device=device)
    x[..., CA_ATOM_ID, :] = torch.randn(example.seq_len, 3, device=device)
    x = x * atom_mask_f

    # First Euler segment
    _, x_clean, traj, sched = _run_euler_segment(
        shim, example, batch_fn_s1,
        t_start=0.0, t_end=_T_END, n_steps=n_steps,
        x_init=x, device=device,
    )

    trajs = [traj]
    scheds = [sched]
    last_latent = shim.last_latent[0]  # [L, d_res]

    # Optional re-noise / re-denoise recycles (B1 variant)
    n_sub = max(1, int(round(n_steps * (_T_END - recycle_t_start) / _T_END)))
    for r in range(n_recycle):
        torch.manual_seed(seed + r + 1)
        eps = torch.zeros_like(x)
        eps[..., CA_ATOM_ID, :] = torch.randn(example.seq_len, 3, device=device)
        eps = eps * atom_mask_f
        x_t = (recycle_t_start * x_clean + (1.0 - recycle_t_start) * eps) * atom_mask_f
        _, x_clean, traj_r, sched_r = _run_euler_segment(
            shim, example, batch_fn_s1,
            t_start=recycle_t_start, t_end=_T_END, n_steps=n_sub,
            x_init=x_t, device=device,
        )
        trajs.append(traj_r)
        scheds.append(sched_r)
        last_latent = shim.last_latent[0]

    x_ca_final = x_clean[..., CA_ATOM_ID, :]
    return x_ca_final, last_latent, np.concatenate(trajs, axis=0), np.concatenate(scheds, axis=0)


# ── Stage 2 sub-sampler — Euler on atoms with CA residual refinement ─────


def _stage2_step(model_two_stage, batch_fn, state, ti, dt, device):
    """One Euler step for Stage 2 within TwoStageMambaFold.

    The Stage 1 outputs (s1_ca, s1_latent) are pre-cached on the wrapper so
    each step only re-runs Stage 2. The CA slot is initialized from s1_ca but
    may move by the learned residual field.
    """
    x, mask_f = state["x"], state["mask_f"]
    batch = batch_fn(x, ti)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                            enabled=(device == "cuda")):
        v_atom = model_two_stage.stage2(
            batch,
            s1_ca=state["s1_ca"],
            s1_latent=state["s1_latent"],
        ).squeeze(0)                                # [L, A, 3]
    x_new = x + dt * v_atom
    x_new = x_new * mask_f
    return {**state, "x": x_new, "x_prev": x}


@torch.no_grad()
def _stage2_run(
    two_stage,
    example,
    batch_fn_s2,
    *,
    s1_ca: Tensor,
    s1_latent: Tensor,
    n_steps: int,
    n_recycle: int,
    recycle_t_start: float,
    seed: int,
    device: str,
):
    """Run Stage 2 Euler ODE with learned CA residual refinement."""
    atom_mask_f = example.atom_mask.unsqueeze(-1).float().to(device)
    n_atoms = example.atom_mask.shape[1]
    L = example.seq_len

    torch.manual_seed(seed + 7919)  # decorrelated seed for Stage 2
    x = torch.randn(L, n_atoms, 3, device=device)
    x[..., CA_ATOM_ID, :] = s1_ca
    x = x * atom_mask_f

    sched = torch.linspace(0.0, _T_END, n_steps + 1, device=device)
    state = {
        "x": x, "x_prev": x.clone(), "mask_f": atom_mask_f,
        "s1_ca": s1_ca, "s1_latent": s1_latent.unsqueeze(0),  # [1, L, d_res]
    }

    traj_pieces = []
    for i in range(n_steps):
        ti = float(sched[i].clamp(min=1e-4))
        dt = float(sched[i + 1] - sched[i])
        state = _stage2_step(two_stage, batch_fn_s2, state, ti, dt, device)
        traj_pieces.append(state["x"][:, CA_ATOM_ID, :].float().cpu().numpy() * COORD_SCALE)

    # One-step recon at the schedule end → final atom positions.
    batch = batch_fn_s2(state["x"], float(sched[-1]))
    with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                            enabled=(device == "cuda")):
        v_final = two_stage.stage2(
            batch, s1_ca=state["s1_ca"], s1_latent=state["s1_latent"],
        ).squeeze(0)
    one_minus_t = 1.0 - float(sched[-1])
    x_clean = state["x"] + one_minus_t * v_final
    x_clean = x_clean * atom_mask_f

    trajs = [np.array(traj_pieces, dtype=np.float32)]
    scheds = [sched.cpu().numpy()]

    # B1 recycles for Stage 2 (same SDEdit pattern)
    n_sub = max(1, int(round(n_steps * (_T_END - recycle_t_start) / _T_END)))
    for r in range(n_recycle):
        torch.manual_seed(seed + 7919 + r + 1)
        eps = torch.randn(L, n_atoms, 3, device=device) * atom_mask_f
        x_t = (recycle_t_start * x_clean + (1.0 - recycle_t_start) * eps) * atom_mask_f

        sched_r = torch.linspace(recycle_t_start, _T_END, n_sub + 1, device=device)
        state = {
            "x": x_t, "x_prev": x_t.clone(), "mask_f": atom_mask_f,
            "s1_ca": s1_ca, "s1_latent": s1_latent.unsqueeze(0),
        }
        traj_pieces_r = []
        for i in range(n_sub):
            ti = float(sched_r[i].clamp(min=1e-4))
            dt = float(sched_r[i + 1] - sched_r[i])
            state = _stage2_step(two_stage, batch_fn_s2, state, ti, dt, device)
            traj_pieces_r.append(
                state["x"][:, CA_ATOM_ID, :].float().cpu().numpy() * COORD_SCALE,
            )
        batch = batch_fn_s2(state["x"], float(sched_r[-1]))
        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(device == "cuda")):
            v_final = two_stage.stage2(
                batch, s1_ca=state["s1_ca"], s1_latent=state["s1_latent"],
            ).squeeze(0)
        one_minus_t = 1.0 - float(sched_r[-1])
        x_clean = state["x"] + one_minus_t * v_final
        x_clean = x_clean * atom_mask_f

        trajs.append(np.array(traj_pieces_r, dtype=np.float32))
        scheds.append(sched_r.cpu().numpy())

    return x_clean, np.concatenate(trajs, axis=0), np.concatenate(scheds, axis=0)


# ── Public entry point ────────────────────────────────────────────────────


def sample(
    two_stage,
    example,
    batch_fn_factory: BatchFnFactory,
    *,
    n_steps_s1: int = 50,
    n_steps_s2: int = 50,
    n_recycle_s1: int = 0,
    n_recycle_s2: int = 0,
    recycle_t_start: float = 0.5,
    seed: int = 0,
    device: str = "cuda",
):
    """Run a full inference: Stage 1 CA-Euler → Stage 2 atom-Euler.

    Args:
        two_stage: `TwoStageMambaFold` instance, eval mode.
        example: ProteinExample (residue features + atom_mask).
        batch_fn_factory: function `(x_t_atom, t) → ProteinBatch` builder.
            the sampler passes the *full* atom tensor and time scalar each Euler step.
        n_steps_s1, n_steps_s2: Euler steps per stage.
        n_recycle_s1, n_recycle_s2: SDEdit-style recycle iterations per stage.
        recycle_t_start: t value to re-noise to during recycling.
        seed: RNG seed for Stage 1 (Stage 2 derives a decorrelated seed).
        device: "cuda" or "cpu".

    Returns:
        (final_ca, final_aa, traj_ca, sched)
            final_ca:  [L, 3]      Stage 2 refined CA prediction (Å)
            final_aa:  [L, A, 3]   Stage 2's full atom prediction (Å)
            traj_ca:   [steps, L, 3]  concatenated CA trajectory across stages
            sched:     1D schedule values per step
    """
    two_stage.eval()
    batch_fn_s1 = batch_fn_factory  # caller-built; passes the full atom tensor

    x_ca, s1_latent, traj_ca_s1, sched_s1 = _stage1_run(
        two_stage.stage1, example, batch_fn_s1,
        n_steps=n_steps_s1, n_recycle=n_recycle_s1,
        recycle_t_start=recycle_t_start, seed=seed, device=device,
    )

    # Stage 2 uses the same batch_fn — the model itself routes via stage2.
    x_atom, traj_ca_s2, sched_s2 = _stage2_run(
        two_stage, example, batch_fn_factory,
        s1_ca=x_ca.unsqueeze(0), s1_latent=s1_latent,
        n_steps=n_steps_s2, n_recycle=n_recycle_s2,
        recycle_t_start=recycle_t_start, seed=seed, device=device,
    )

    final_aa = x_atom.float().cpu().numpy() * COORD_SCALE
    final_ca = final_aa[:, CA_ATOM_ID, :]
    return (
        final_ca,
        final_aa,
        np.concatenate([traj_ca_s1, traj_ca_s2], axis=0),
        np.concatenate([sched_s1, sched_s2], axis=0),
    )
