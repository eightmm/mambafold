"""EqM Nesterov Accelerated Gradient sampler with adaptive compute."""

import torch
from torch import Tensor

from mambafold.data.constants import COORD_SCALE
from mambafold.data.types import ProteinBatch
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
            step = (self.eta * grad).clamp(-self.max_disp, self.max_disp)
            x_next = x - step
            x_next = remove_translation(x_next, batch.atom_mask)
            x_next = x_next * batch.atom_mask.unsqueeze(-1).to(dtype)

            x_prev = x
            x = x_next

        # Convert back to Angstrom
        coords = x * COORD_SCALE
        return coords, n_steps
