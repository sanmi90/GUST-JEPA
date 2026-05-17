"""VICReg anti-collapse regularizer (without the invariance term).

Reference:
    Bardes, Ponce, LeCun. "VICReg: Variance-Invariance-Covariance
    Regularization for Self-Supervised Learning." arXiv:2105.04906, ICLR
    2022, Section 3.

The variance term penalises any latent dimension whose standard deviation
drops below ``gamma``; the covariance term penalises off-diagonal
correlations between dimensions. The invariance term is the only one that
requires a second augmented view of each sample; JEPA without paired
augmentations has only one view (HANDOFF.md D6), so the invariance term
is dropped (HANDOFF.md D22). ``lambda_`` is retained in the public API for
forward compatibility with future symmetry-augmentation ablations.

VICReg is the auto-fallback regularizer for SIGReg (HANDOFF.md D5,
CLAUDE.md "Risk-management"): when the participation ratio and probe R^2
diagnostics both drop below threshold at iter >= 20k, the training loop
swaps ``SIGReg`` for ``VICReg`` and continues without restarting.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class VICReg(nn.Module):
    """Variance-Covariance regularization (invariance term dropped).

    Args (Bardes ICLR 2022 defaults per D22):
        d: Latent dimension (must match ``z.shape[-1]``).
        mu: Weight of the variance term. ICLR 2022 default 25.
        lambda_: Weight of the invariance term. Kept in the API but ignored
            in the forward pass because JEPA has no second view here.
        nu: Weight of the covariance term. ICLR 2022 default 1.
        gamma: Variance hinge target (standard deviation). ICLR 2022 default 1.
        eps: Numerical-stability term added inside the sqrt to avoid
            infinite gradients when the per-dim variance approaches zero.
    """

    def __init__(
        self,
        d: int,
        mu: float = 25.0,
        lambda_: float = 25.0,
        nu: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        if d <= 0:
            raise ValueError(f"d must be positive, got {d}")
        self.d = int(d)
        self.mu = float(mu)
        self.lambda_ = float(lambda_)
        self.nu = float(nu)
        self.gamma = float(gamma)
        self.eps = float(eps)

    def forward(self, z: Tensor) -> Tensor:
        """Computes the variance + covariance loss for a batch of embeddings.

        Args:
            z: Batch of shape ``(N, d)``. Any floating dtype is accepted;
                the computation is performed in fp32 internally to match the
                SIGReg numerical-stability convention.

        Returns:
            Scalar fp32 tensor ``mu * L_var + nu * L_cov``.
        """
        if z.dim() != 2 or z.shape[-1] != self.d:
            raise ValueError(f"expected z of shape (N, {self.d}), got {tuple(z.shape)}")
        if z.shape[0] < 2:
            raise ValueError(f"VICReg requires at least 2 samples, got {z.shape[0]}")

        with torch.amp.autocast(device_type=z.device.type, enabled=False):
            z32 = z.float()
            mean = z32.mean(dim=0, keepdim=True)
            z_centered = z32 - mean

            var = z_centered.pow(2).mean(dim=0)
            std = torch.sqrt(var + self.eps)
            loss_var = torch.relu(self.gamma - std).mean()

            n = z32.shape[0]
            cov = (z_centered.t() @ z_centered) / (n - 1)
            off_diag = cov - torch.diag(torch.diagonal(cov))
            loss_cov = off_diag.pow(2).sum() / self.d

            return self.mu * loss_var + self.nu * loss_cov
