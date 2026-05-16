"""Adaptive LayerNorm conditioning with zero-init (AdaLN-Zero).

Reference:
    Peebles, Xie. "Scalable Diffusion Models with Transformers."
    arXiv:2212.09748, 2023, figure 3.

Used in the JEPA predictor blocks to inject the static episode descriptor
``c = (G, D, Y)`` and the time-varying phase ``phi_t`` (see CLAUDE.md D6).
The zero-init guarantees the predictor starts as identity-on-residual.
"""

from __future__ import annotations

from torch import Tensor, nn


class AdaLN(nn.Module):
    """Adaptive LayerNorm conditioning with zero-init.

    Returns the ``(shift, scale, gate)`` tuple used inside a DiT-style block.
    The consumer applies them as::

        x_in = layer_norm(x) * (1 + scale) + shift
        x_out = x + gate * sublayer(x_in)

    Attributes:
        linear: Final ``nn.Linear(cond_dim, 3 * hidden_dim)`` whose weights and
            bias are zero-initialized so that ``(shift, scale, gate)`` are all
            zero at construction time.
    """

    def __init__(self, hidden_dim: int, cond_dim: int) -> None:
        """Builds the projection.

        Args:
            hidden_dim: Width of the transformer residual stream.
            cond_dim: Width of the conditioning vector.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.act = nn.SiLU()
        self.linear = nn.Linear(cond_dim, 3 * hidden_dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, cond: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Computes the per-token modulation parameters.

        Args:
            cond: Conditioning tensor of shape ``(B, cond_dim)`` or
                ``(B, T, cond_dim)``.

        Returns:
            Three tensors ``(shift, scale, gate)`` each of shape
            ``(B, hidden_dim)`` or ``(B, T, hidden_dim)`` matching the input
            rank. At initialization they are exactly zero.
        """
        h = self.linear(self.act(cond))
        shift, scale, gate = h.chunk(3, dim=-1)
        return shift, scale, gate
