"""EqM samplers: Nesterov Accelerated Gradient and Euler ODE."""

import torch
from torch import Tensor

from mambafold.data.constants import COORD_SCALE
from mambafold.data.types import ProteinBatch
from mambafold.losses.eqm import eqm_reconstruction_scale
from mambafold.utils.geometry import remove_translation


class EqMNAGSampler:
    """NAG gradient descent sampler for Equilibrium Matching.

    Generates protein structures by descending the learned energy landscape.
    """

    def __init__(
        self,
        model,
        eta: float = 0.1,
        mu: float = 0.3,
        g_min: float = 5e-3,
        max_steps: int = 128,
        max_disp: float = 0.5,
    ):
        self.model = model
        self.eta = eta            # step size
        self.mu = mu              # NAG momentum factor
        self.g_min = g_min        # gradient norm stopping threshold
        self.max_steps = max_steps
        self.max_disp = max_disp  # per-step displacement clamp (normalized)

    @torch.no_grad()
    def sample(self, batch: ProteinBatch) -> tuple[Tensor, int]:
        """Generate structures from noise via NAG gradient descent.

        Args:
            batch: ProteinBatch with sequence info (res_type, atom_type, atom_mask,
                   res_mask, esm, gamma). x_gamma is overwritten each step.

        Returns:
            coords: [B, L, A, 3] generated coordinates in Angstrom
            n_steps: number of steps taken
        """
        self.model.eval()
        device = batch.device
        dtype = next(self.model.parameters()).dtype
        shape = batch.atom_mask.shape + (3,)  # [B, L, A, 3]

        # Initialize from noise
        x = torch.randn(shape, device=device, dtype=dtype)
        x = remove_translation(x, batch.atom_mask)
        x = x * batch.atom_mask.unsqueeze(-1).to(dtype)
        x_prev = x.clone()

        n_steps = 0
        for k in range(self.max_steps):
            # NAG lookahead position
            look = x if k == 0 else x + self.mu * (x - x_prev)

            # Model prediction at lookahead
            look_batch = batch.with_coords(look)
            grad = self.model(look_batch)  # [B, L, A, 3]

            # Stopping criterion (gradient RMS)
            n_valid = batch.atom_mask.sum().clamp(min=1)
            grad_rms = (grad.pow(2).sum() / n_valid / 3).sqrt()

            n_steps = k + 1
            if grad_rms.item() < self.g_min:
                break

            # NAG step per EqM paper Eq.(9): x_{k+1} = x_k - η·f(lookahead)
            step = self.eta * grad
            step_norm = step.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            step = step * (step_norm.clamp(max=self.max_disp) / step_norm)
            x_next = x - step
            x_next = x_next * batch.atom_mask.unsqueeze(-1).to(dtype)

            x_prev = x
            x = x_next

        # Convert back to Angstrom
        coords = x * COORD_SCALE
        return coords, n_steps


class EqMEulerSampler:
    """Euler ODE sampler for Equilibrium Matching.

    Integrates the probability flow ODE dx/dγ = (x_hat - x) / (1-γ)
    from γ=0 (pure noise) to γ≈1 (clean structure).
    x_hat = x - scale(γ)·f(x) is the one-step clean prediction.
    """

    def __init__(
        self,
        model,
        n_steps: int = 50,
        a: float = 0.8,
        lam: float = 4.0,
    ):
        self.model = model
        self.n_steps = n_steps
        self.a = a
        self.lam = lam

    def _make_batch(
        self,
        batch: ProteinBatch,
        x: Tensor,
        gamma_val: float,
    ) -> ProteinBatch:
        device = batch.device
        dtype = x.dtype
        B = batch.batch_size
        gamma_t = torch.full((B, 1, 1, 1), gamma_val, device=device, dtype=dtype)
        return ProteinBatch(
            res_type=batch.res_type,
            res_seq_nums=batch.res_seq_nums,
            atom_type=batch.atom_type,
            pair_type=batch.pair_type,
            res_mask=batch.res_mask,
            atom_mask=batch.atom_mask,
            valid_mask=batch.valid_mask,
            ca_mask=batch.ca_mask,
            x_clean=batch.x_clean,
            x_gamma=x,
            eps=torch.zeros_like(x),
            gamma=gamma_t,
            esm=batch.esm,
        )

    @torch.no_grad()
    def sample(self, batch: ProteinBatch) -> tuple[Tensor, int]:
        """Generate structures via Euler integration of the EqM ODE.

        Args:
            batch: ProteinBatch with sequence info (res_type, atom_type, atom_mask,
                   res_mask, esm). x_gamma and gamma are overwritten each step.

        Returns:
            coords: [B, L, A, 3] generated coordinates in Angstrom
            n_steps: number of steps taken (always self.n_steps)
        """
        self.model.eval()
        device = batch.device
        dtype = next(self.model.parameters()).dtype
        shape = batch.atom_mask.shape + (3,)  # [B, L, A, 3]
        mask_f = batch.atom_mask.unsqueeze(-1).to(dtype)

        # Initialize from centered noise (matches training distribution)
        x = torch.randn(shape, device=device, dtype=dtype) * mask_f
        x = remove_translation(x, batch.atom_mask)

        sched = torch.linspace(0.0, 0.99, self.n_steps + 1, device=device)
        amp_on = str(device).startswith("cuda")

        for i in range(self.n_steps):
            gamma_cur = float(sched[i].clamp(min=1e-4))
            dg = float(sched[i + 1] - sched[i])

            step_batch = self._make_batch(batch, x, gamma_cur)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_on):
                pred = self.model(step_batch)

            scale = eqm_reconstruction_scale(step_batch.gamma, a=self.a, lam=self.lam)
            x_hat = x - scale * pred

            # ODE velocity: (x_hat - x) / (1 - γ)
            velocity = (x_hat - x) / max(1.0 - gamma_cur, 1e-4)
            x = (x + dg * velocity) * mask_f

        # Final reconstruction step at γ=sched[-1] to remove residual noise floor
        final_batch = self._make_batch(batch, x, float(sched[-1]))
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_on):
            pred_final = self.model(final_batch)

        scale_final = eqm_reconstruction_scale(final_batch.gamma, a=self.a, lam=self.lam)
        x_hat_final = x - scale_final * pred_final

        coords = x_hat_final * COORD_SCALE
        return coords, self.n_steps
