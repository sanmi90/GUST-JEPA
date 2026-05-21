"""CoordMLPDecoder: coordinate neural field decoder for the diagnostic audit.

A coordinate MLP maps ``(x, y, z)`` -> ``omega_z(x, y; z)`` for each
spatial location independently. Two activation modes:

- ``"sine"``: SIREN (Sitzmann et al., arXiv:2006.09661). Sinusoidal
  activations with the SIREN initialisation scheme. The first layer
  uses a frequency scaling ``omega_0`` (default 30) that sets the
  bandwidth of the implicit representation.
- ``"gelu_fourier"``: GELU on Fourier-encoded coordinates (Tancik et
  al., arXiv:2006.10739). Coordinates are pre-encoded with sin/cos at
  geometrically spaced frequencies; the body of the MLP is a vanilla
  GELU stack.

Either mode is appreciably better at high-frequency signals than a
plain CNN-pyramid decoder. Used in Session 10's E4 run as a
LATENT-INFORMATION-CONTENT diagnostic: if the coord-MLP audit also
fails to recover wake-scale structure on the frozen JEPA latent, the
limitation is in the encoder, not the decoder (see
SESSION10_MULTISCALE_DECODER.md, decision string ``COORD_MLP_BEST``
vs ``ALL_THREE_FAIL`` branch).

Forward signature: takes ``z`` of shape ``(B, latent_dim)`` or
``(B, T, latent_dim)``; returns ``omega_hat`` of shape
``(B, 1, H, W)`` or ``(B, T, 1, H, W)``.

Pixels are processed in chunks of ``chunk_pixels`` to keep the
intermediate tensor size bounded for large grids and batches. The
chunking is bitwise-equivalent to a single full pass because each
pixel is processed independently (no batch-norm or cross-pixel ops).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _build_full_grid(h: int, w: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Return (h*w, 2) coordinate grid in [-1, 1]; column 0 = x, column 1 = y."""
    ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)


def _fourier_encode(coords: Tensor, n_bands: int) -> Tensor:
    """Encode (N, 2) coords -> (N, 2 + 4*n_bands) with sin/cos at geometric freqs.

    The raw (x, y) are kept as the first two channels and the Fourier
    components are appended at frequencies ``pi, 2 pi, ..., 2^(n_bands-1) pi``.
    """
    if n_bands == 0:
        return coords
    freqs = (2.0 ** torch.arange(n_bands, device=coords.device, dtype=coords.dtype)) * math.pi
    ang = coords[:, :, None] * freqs[None, None, :]
    sin = ang.sin().reshape(coords.shape[0], -1)
    cos = ang.cos().reshape(coords.shape[0], -1)
    return torch.cat([coords, sin, cos], dim=-1)


class _SineLayer(nn.Module):
    """SIREN sine-activated linear layer with the appendix initialisation.

    First layer uses ``Uniform(-1/in, 1/in)``. Hidden layers use
    ``Uniform(-sqrt(6/in)/omega_0, +sqrt(6/in)/omega_0)`` so that the
    pre-activation distribution stays close to N(0, 1) after the
    omega_0 scaling.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        is_first: bool = False,
        omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.omega_0 = omega_0
        self.is_first = is_first
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1.0 / in_features, 1.0 / in_features)
            else:
                bound = math.sqrt(6.0 / in_features) / omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class CoordMLPDecoder(nn.Module):
    """Coordinate neural field decoder.

    Args:
        latent_dim: Dimension of the per-frame latent ``z``.
        hidden: Hidden width of the MLP.
        layers: Number of hidden layers. The total stack is
            ``[input_proj] + [layers - 1 hidden blocks] + [output_proj]``.
        fourier_bands: Number of Fourier bands for the ``"gelu_fourier"``
            activation mode. Ignored for ``"sine"`` (SIREN does its own
            implicit Fourier encoding through the first sine layer).
        activation: ``"sine"`` (SIREN) or ``"gelu_fourier"``.
        chunk_pixels: Pixels per forward-pass chunk. Memory optimisation
            only; output is bitwise identical for any chunk size.
        H, W: Default output resolution when ``coords=None``.
        siren_omega_0: First-layer frequency multiplier for SIREN.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden: int = 128,
        layers: int = 5,
        fourier_bands: int = 8,
        activation: str = "sine",
        chunk_pixels: int = 4096,
        H: int = 192,
        W: int = 96,
        siren_omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        if activation not in ("sine", "gelu_fourier"):
            raise ValueError(f"unknown activation {activation!r}")
        if layers < 2:
            raise ValueError("layers must be >= 2 (input projection + output)")

        self.latent_dim = latent_dim
        self.hidden = hidden
        self.n_layers = layers
        self.fourier_bands = fourier_bands
        self.activation = activation
        self.chunk_pixels = chunk_pixels
        self.H = H
        self.W = W
        self.siren_omega_0 = siren_omega_0

        if activation == "sine":
            coord_dim = 2
            self.coord_encoder = None
        else:
            coord_dim = 2 + 4 * fourier_bands
            self.coord_encoder = "fourier"

        input_dim = latent_dim + coord_dim

        if activation == "sine":
            blocks: list[nn.Module] = []
            blocks.append(_SineLayer(input_dim, hidden, is_first=True,
                                     omega_0=siren_omega_0))
            for _ in range(layers - 2):
                blocks.append(_SineLayer(hidden, hidden, is_first=False,
                                         omega_0=siren_omega_0))
            self.body = nn.ModuleList(blocks)
            self.out = nn.Linear(hidden, 1)
            with torch.no_grad():
                bound = math.sqrt(6.0 / hidden) / siren_omega_0
                self.out.weight.uniform_(-bound, bound)
        else:
            body: list[nn.Module] = [nn.Linear(input_dim, hidden), nn.GELU()]
            for _ in range(layers - 2):
                body.append(nn.Linear(hidden, hidden))
                body.append(nn.GELU())
            self.body = nn.Sequential(*body)
            self.out = nn.Linear(hidden, 1)

    def _encode_coords(self, coords: Tensor) -> Tensor:
        if self.coord_encoder == "fourier":
            return _fourier_encode(coords, self.fourier_bands)
        return coords

    def _forward_chunk(self, z_chunk: Tensor, coord_feat: Tensor) -> Tensor:
        """Run one chunk: ``z_chunk`` is (N, latent_dim) and
        ``coord_feat`` is (N, coord_dim). Returns (N, 1)."""
        inp = torch.cat([z_chunk, coord_feat], dim=-1)
        if self.activation == "sine":
            h = inp
            for layer in self.body:
                h = layer(h)
            return self.out(h)
        return self.out(self.body(inp))

    def forward(self, z: Tensor, coords: Optional[Tensor] = None) -> Tensor:
        """Decode latents to vorticity via a coordinate MLP.

        Args:
            z: ``(B, latent_dim)`` or ``(B, T, latent_dim)``.
            coords: Optional ``(N, 2)`` coordinate grid in [-1, 1].
                When None, uses the full ``(H, W)`` grid.

        Returns: ``(B, 1, H, W)`` (or ``(B, T, 1, H, W)``) if
        ``coords`` is None, otherwise ``(B, 1, N)``.
        """
        squeeze_T = False
        B_out, T_out = z.shape[0], 1
        if z.dim() == 3:
            B_out, T_out, _ = z.shape
            z = z.reshape(B_out * T_out, -1)
            squeeze_T = True

        N_batch = z.shape[0]
        if coords is None:
            grid = _build_full_grid(self.H, self.W, z.device, z.dtype)
            return_grid = True
        else:
            grid = coords
            return_grid = False

        n_pixels = grid.shape[0]
        coord_feat = self._encode_coords(grid)
        out = torch.empty(N_batch, n_pixels, 1, device=z.device, dtype=z.dtype)

        chunk = max(1, self.chunk_pixels)
        for start in range(0, n_pixels, chunk):
            end = min(start + chunk, n_pixels)
            cf = coord_feat[start:end]
            # Broadcast z to (N_batch, end-start, latent_dim) and cf to
            # (N_batch, end-start, coord_dim) without an explicit expand
            # so the concat is contiguous in the right order.
            z_b = z[:, None, :].expand(N_batch, end - start, -1).reshape(-1, z.shape[1])
            cf_b = cf[None, :, :].expand(N_batch, end - start, -1).reshape(-1, cf.shape[1])
            y = self._forward_chunk(z_b, cf_b)
            out[:, start:end, :] = y.view(N_batch, end - start, 1)

        if return_grid:
            out = out.view(N_batch, self.H, self.W, 1).permute(0, 3, 1, 2).contiguous()
            if squeeze_T:
                out = out.view(B_out, T_out, *out.shape[-3:])
        else:
            if squeeze_T:
                out = out.view(B_out, T_out, n_pixels, 1)
        return out
