"""Sketched Isotropic Gaussian Regularizer (SIGReg).

References:
    Balestriero, LeCun. "LeJEPA: Provable and Scalable Self-Supervised
    Learning Without the Heuristics." arXiv:2511.08544, 2025.
    Maes, Le Lidec, Scieur, LeCun, Balestriero. "LeWorldModel: Stable
    End-to-End Joint-Embedding Predictive Architecture from Pixels."
    arXiv:2603.19312, 2026 (appendix A defines T^(m) without an N
    multiplier; see HANDOFF.md D13).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SIGReg(nn.Module):
    """SIGReg anti-collapse regularizer.

    Projects the input batch onto ``num_projections`` unit-norm directions and
    averages the univariate Epps-Pulley statistic for each projection against
    the standard Gaussian characteristic function. With ``M`` directions and
    ``B`` samples the cost is ``O(B * M)`` per batch (Cramer-Wold sketching).

    Numerical notes (see SESSION2_MODEL_PRIMITIVES.md):
        - The body runs in fp32 even if the caller is in bf16 autocast; this
          avoids instability in the complex exponential reduction.
        - Per LeWM appendix A, the Epps-Pulley test statistic has no leading
          ``N`` multiplier; the LeJEPA reference PyTorch listing adds one but
          this project follows LeWM (HANDOFF.md D13).
    """

    def __init__(
        self,
        dim: int,
        num_projections: int = 256,
        num_knots: int = 17,
        knot_range: tuple[float, float] = (0.2, 4.0),
        resample_each_step: bool = True,
        weight_lambda: float = 1.0,
    ) -> None:
        """Sets up the quadrature grid and optionally caches projection directions.

        Args:
            dim: Latent dimension ``d`` (must match ``z.shape[-1]``).
            num_projections: Number of random unit directions ``M`` per call.
            num_knots: Number of trapezoidal quadrature knots.
            knot_range: Inclusive endpoints of the integration interval.
            resample_each_step: If ``True``, draw fresh directions on every
                ``forward``; if ``False``, draw once at construction time and
                cache them as a non-trainable buffer.
            weight_lambda: Bandwidth ``sigma`` of the Gaussian quadrature
                weight ``w(t) = exp(-t**2 / (2 * weight_lambda**2))``.
        """
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_projections <= 0:
            raise ValueError(f"num_projections must be positive, got {num_projections}")
        if num_knots < 2:
            raise ValueError(f"num_knots must be >= 2, got {num_knots}")

        self.dim = dim
        self.num_projections = num_projections
        self.num_knots = num_knots
        self.knot_range = knot_range
        self.resample_each_step = resample_each_step
        self.weight_lambda = float(weight_lambda)

        knots = torch.linspace(knot_range[0], knot_range[1], num_knots, dtype=torch.float32)
        spacing = (knot_range[1] - knot_range[0]) / (num_knots - 1)
        trap_weights = torch.full((num_knots,), spacing, dtype=torch.float32)
        trap_weights[0] *= 0.5
        trap_weights[-1] *= 0.5
        gauss_window = torch.exp(-knots.pow(2) / (2.0 * self.weight_lambda**2))
        quad_weights = trap_weights * gauss_window
        phi_0 = torch.exp(-knots.pow(2) / 2.0)

        self.register_buffer("knots", knots)
        self.register_buffer("quad_weights", quad_weights)
        self.register_buffer("phi_0", phi_0)

        if not resample_each_step:
            u = torch.randn(num_projections, dim)
            u = u / u.norm(dim=-1, keepdim=True)
            self.register_buffer("directions", u)
        else:
            self.directions = None

    def _sample_directions(self, device: torch.device) -> Tensor:
        u = torch.randn(self.num_projections, self.dim, device=device, dtype=torch.float32)
        return u / u.norm(dim=-1, keepdim=True)

    def forward(self, z: Tensor) -> Tensor:
        """Computes the SIGReg statistic for a batch of latent embeddings.

        Args:
            z: Batch of shape ``(B, dim)``. Any floating dtype is accepted;
                the computation is performed in fp32 internally.

        Returns:
            A scalar fp32 tensor: the mean over ``M`` projections of the
            univariate Epps-Pulley statistic.
        """
        if z.dim() != 2 or z.shape[-1] != self.dim:
            raise ValueError(f"expected z of shape (B, {self.dim}), got {tuple(z.shape)}")

        with torch.amp.autocast(device_type=z.device.type, enabled=False):
            z32 = z.float()
            if self.resample_each_step:
                u = self._sample_directions(z32.device)
            else:
                u = self.directions.to(device=z32.device, dtype=torch.float32)

            h = z32 @ u.t()
            total = torch.zeros(self.num_projections, device=z32.device, dtype=torch.float32)
            for k in range(self.num_knots):
                tk = self.knots[k]
                angle = tk * h
                phi_n = torch.complex(angle.cos(), angle.sin()).mean(dim=0)
                phi_0_k = torch.complex(self.phi_0[k], torch.zeros_like(self.phi_0[k]))
                diff = phi_n - phi_0_k
                total = total + self.quad_weights[k] * (diff.real.pow(2) + diff.imag.pow(2))
            return total.mean()
