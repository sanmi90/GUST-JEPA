"""PLDM 5-term loss (Sobal et al., arXiv:2502.14819, 2025).

Reference (verified against the paper text on 2026-05-18; see
HANDOFF.md D30 for the discrepancy versus D8's original 7-term
reading):

    Sobal, V., Zhang, W., Cho, K., Balestriero, R., Rudner, T. G. J.,
    LeCun, Y. "Learning from Reward-Free Offline Data: A Case for
    Planning with Latent Dynamics Models." arXiv:2502.14819,
    February 2025. Section 3.3 (L_sim, Eq. 3); Appendix D.1.1
    (collapse-prevention block and combined objective).

The paper's loss has FIVE terms (not seven as D8 originally claimed):

    L_JEPA = L_sim
           + alpha * L_var
           + beta  * L_cov
           + delta * L_time_sim
           + omega * L_IDM

- L_sim:      multi-step rollout MSE between predictor output and
              ground-truth encoder latents (Eq. 3, paper Section 3.3).
              In our setup this is computed by the wrapper that owns
              the predictor and the encoder; this module just takes
              the two latent streams ``z`` and ``z_hat`` and computes
              the MSE.
- L_var:      VICReg-style variance hinge applied PER-TIME-SLICE then
              averaged over (time, dim). Bardes et al. ICLR 2022
              (arXiv:2105.04906) Equation 1, gamma defaults to 1.
- L_cov:      Off-diagonal covariance Frobenius PER-TIME-SLICE then
              averaged over time.
- L_time_sim: Temporal-smoothness MSE ||z_t - z_{t+1}||^2 averaged
              across (B, T-1, d).
- L_IDM:      Inverse-dynamics MLP regression. The paper predicts the
              per-step action ``a_t`` from ``(z_t, z_{t+1})``. We
              adapt to our static episode descriptor ``c = (G, D, Y)``
              by replacing the action target with c, broadcast across
              all (T-1) consecutive pairs. This is the D8 adaptation,
              kept after D30.

There are NO var/cov terms applied to the temporal-difference signal
``dz = z_{t+1} - z_t`` (the spurious "7-term" addition in D8). See
HANDOFF.md D30 for the full discrepancy resolution.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PLDMLoss(nn.Module):
    """PLDM five-term anti-collapse + prediction + IDM loss.

    The wrapper owns the encoder + predictor; this module is just the
    loss head. It expects pre-computed latents ``z`` and ``z_hat`` and
    the static episode descriptor ``c``.

    Args:
        d: Latent dimension.
        c_dim: Static descriptor dimension (3 for ``(G, D, Y)``).
        lambda_var: Weight on the variance hinge term (default 1.0;
            the paper uses alpha in [4.0, 35.0] across environments,
            see Appendix J.2 Tables 13-17).
        lambda_cov: Weight on the off-diagonal covariance term
            (default 1.0; paper beta in [0.5, 12.0]).
        lambda_time_sim: Weight on the temporal-smoothness term
            (default 1.0; paper delta in [0.1, 8.1]).
        lambda_idm: Weight on the inverse-dynamics term (default 1.0;
            paper omega in [0.0, 5.4], with omega=0 in Two-Rooms).
        gamma: Variance-hinge target (default 1.0 per Bardes et al.).
        eps: Numerical stabiliser inside ``sqrt(var + eps)`` (default
            1e-4 per VICReg).
        idm_hidden: Inverse-dynamics MLP hidden width (default 128).

    Note:
        Default weights are all 1.0 as placeholders. The paper-derived
        defaults vary by environment; the smoke entrypoint should
        override via CLI flags. A future paper-grade run should
        bisect or grid-search these on Test B.
    """

    def __init__(
        self,
        d: int,
        c_dim: int = 3,
        lambda_var: float = 1.0,
        lambda_cov: float = 1.0,
        lambda_time_sim: float = 1.0,
        lambda_idm: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
        idm_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.d = d
        self.c_dim = c_dim
        self.lambda_var = float(lambda_var)
        self.lambda_cov = float(lambda_cov)
        self.lambda_time_sim = float(lambda_time_sim)
        self.lambda_idm = float(lambda_idm)
        self.gamma = float(gamma)
        self.eps = float(eps)

        self.idm = nn.Sequential(
            nn.Linear(2 * d, idm_hidden),
            nn.GELU(),
            nn.Linear(idm_hidden, idm_hidden),
            nn.GELU(),
            nn.Linear(idm_hidden, c_dim),
        )

    def _variance_hinge(self, z_bt: Tensor) -> Tensor:
        """VICReg variance hinge applied per-time-slice, averaged over (T, d).

        Args:
            z_bt: Latent of shape ``(B, T, d)`` in fp32.

        Returns:
            Scalar tensor.
        """
        B, T, D = z_bt.shape
        var = z_bt.var(dim=0, unbiased=True)
        std = torch.sqrt(var + self.eps)
        hinge = (self.gamma - std).clamp_min(0.0)
        return hinge.mean()

    def _covariance_offdiag(self, z_bt: Tensor) -> Tensor:
        """Per-time-slice off-diagonal covariance Frobenius squared, averaged over T.

        Args:
            z_bt: Latent of shape ``(B, T, d)`` in fp32.

        Returns:
            Scalar tensor.
        """
        B, T, D = z_bt.shape
        if B < 2:
            return torch.zeros((), device=z_bt.device, dtype=z_bt.dtype)
        cov_total = torch.zeros((), device=z_bt.device, dtype=z_bt.dtype)
        eye = torch.eye(D, device=z_bt.device, dtype=z_bt.dtype)
        for t in range(T):
            zt = z_bt[:, t, :]
            zt = zt - zt.mean(dim=0, keepdim=True)
            cov = (zt.T @ zt) / (B - 1)
            offdiag = cov - cov * eye
            cov_total = cov_total + offdiag.pow(2).sum() / D
        return cov_total / T

    def _temporal_smoothness(self, z_bt: Tensor) -> Tensor:
        """MSE of consecutive frame differences, averaged over (B, T-1, d).

        Args:
            z_bt: Latent of shape ``(B, T, d)`` in fp32. Requires T >= 2.

        Returns:
            Scalar tensor.
        """
        if z_bt.shape[1] < 2:
            return torch.zeros((), device=z_bt.device, dtype=z_bt.dtype)
        dz = z_bt[:, 1:, :] - z_bt[:, :-1, :]
        return dz.pow(2).mean()

    def _idm_loss(self, z_bt: Tensor, c: Tensor) -> Tensor:
        """Inverse-dynamics MSE: predict c from each consecutive (z_t, z_{t+1}) pair.

        Args:
            z_bt: Latent ``(B, T, d)`` in fp32.
            c:    Static descriptor ``(B, c_dim)`` in fp32. Broadcast
                  across the ``T - 1`` consecutive pairs per batch sample.

        Returns:
            Scalar MSE.
        """
        if z_bt.shape[1] < 2:
            return torch.zeros((), device=z_bt.device, dtype=z_bt.dtype)
        B, T, D = z_bt.shape
        pairs = torch.cat([z_bt[:, :-1, :], z_bt[:, 1:, :]], dim=-1)
        c_pred = self.idm(pairs.reshape(-1, 2 * D))
        c_tile = c.unsqueeze(1).expand(-1, T - 1, -1).reshape(-1, c.shape[-1])
        return (c_pred - c_tile).pow(2).mean()

    def forward(
        self,
        z: Tensor,
        z_hat: Tensor,
        c: Tensor,
    ) -> dict[str, Tensor]:
        """Compute all five PLDM terms and the weighted total.

        Args:
            z:     Encoder latents, ``(B, T, d)``. The regularisation
                   terms (var, cov, time_sim, idm) are computed on
                   this stream.
            z_hat: Predictor rollout, ``(B, T, d)``. Used by L_sim only.
                   The wrapper should produce ``z_hat`` via the predictor's
                   open-loop rollout from a single anchor frame.
            c:     Static episode descriptor, ``(B, c_dim)``.

        Returns:
            Dict with keys ``L_sim``, ``L_var``, ``L_cov``,
            ``L_time_sim``, ``L_idm``, ``L_total``. All are zero-dim
            scalars in fp32.
        """
        z32 = z.float()
        z_hat32 = z_hat.float()
        c32 = c.float()

        L_sim = (z_hat32 - z32).pow(2).mean()
        L_var = self._variance_hinge(z32)
        L_cov = self._covariance_offdiag(z32)
        L_time_sim = self._temporal_smoothness(z32)
        L_idm = self._idm_loss(z32, c32)

        L_total = (
            L_sim
            + self.lambda_var * L_var
            + self.lambda_cov * L_cov
            + self.lambda_time_sim * L_time_sim
            + self.lambda_idm * L_idm
        )

        return {
            "L_total": L_total,
            "L_sim": L_sim,
            "L_var": L_var,
            "L_cov": L_cov,
            "L_time_sim": L_time_sim,
            "L_idm": L_idm,
        }
