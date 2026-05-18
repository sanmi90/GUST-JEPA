"""Tests for ``src/models/pldm_wrapper.py``.

The wrapper composes ``HybridCNNViTEncoder`` + ``AutoregressivePredictor``
+ ``PLDMLoss`` into one forward pass, in parallel to ``JEPA`` but with
the PLDM 5-term loss (Sobal et al., arXiv:2502.14819) instead of the
two-term ``L_pred + L_roll + lambda * L_sigreg`` composition.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.baselines.pldm import PLDMLoss
from src.models.encoder import HybridCNNViTEncoder
from src.models.pldm_wrapper import PLDMWrapper
from src.models.predictor import AutoregressivePredictor
from src.utils.device import NoRTX6000Error, require_rtx6000


def _tiny_pldm(
    *,
    latent_dim: int = 32,
    prediction_horizon: int = 4,
    predictor_dropout: float = 0.0,
    max_seq_len: int = 32,
) -> PLDMWrapper:
    """Build a small but real PLDM wrapper for unit tests."""
    encoder = HybridCNNViTEncoder(latent_dim=latent_dim)
    predictor = AutoregressivePredictor(
        latent_dim=latent_dim,
        cond_dim=3,
        hidden_dim=384,
        depth=2,
        heads=8,
        dropout=predictor_dropout,
        max_seq_len=max_seq_len,
    )
    loss = PLDMLoss(d=latent_dim, c_dim=3)
    return PLDMWrapper(encoder=encoder, predictor=predictor, loss=loss,
                       prediction_horizon=prediction_horizon)


def _tiny_batch(B: int = 2, T: int = 8, H: int = 192, W: int = 96) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {"omega": torch.randn(B, T, 1, H, W), "c": torch.randn(B, 3)}


def test_pldm_wrapper_shape_contract() -> None:
    """Batch with omega (2, 8, 1, 192, 96) and c (2, 3) returns a dict with
    all five loss terms plus L_total plus z (the cached encoder output)."""
    torch.manual_seed(0)
    wrapper = _tiny_pldm()
    out = wrapper(_tiny_batch(B=2, T=8))
    expected = {"L_total", "L_sim", "L_var", "L_cov", "L_time_sim", "L_idm", "z"}
    assert expected.issubset(set(out.keys()))
    for k in ("L_total", "L_sim", "L_var", "L_cov", "L_time_sim", "L_idm"):
        assert out[k].dim() == 0
        assert torch.isfinite(out[k]), f"{k} not finite"
    assert out["z"].shape == (2, 8, 32)


def test_pldm_wrapper_loss_decomposition() -> None:
    """L_total equals the weighted sum of the five terms with the configured lambdas."""
    torch.manual_seed(0)
    encoder = HybridCNNViTEncoder(latent_dim=32)
    predictor = AutoregressivePredictor(
        latent_dim=32, cond_dim=3, hidden_dim=384, depth=2,
        heads=8, dropout=0.0, max_seq_len=32,
    )
    loss = PLDMLoss(d=32, c_dim=3,
                    lambda_var=4.0, lambda_cov=6.9, lambda_time_sim=0.75, lambda_idm=1.0)
    wrapper = PLDMWrapper(encoder=encoder, predictor=predictor, loss=loss, prediction_horizon=4)
    out = wrapper(_tiny_batch(B=2, T=8))
    expected = (
        out["L_sim"].item()
        + 4.0 * out["L_var"].item()
        + 6.9 * out["L_cov"].item()
        + 0.75 * out["L_time_sim"].item()
        + 1.0 * out["L_idm"].item()
    )
    assert math.isclose(out["L_total"].item(), expected, rel_tol=1e-5, abs_tol=1e-5)


def test_pldm_wrapper_gradient_flows_through_all_modules() -> None:
    """A backward on L_total produces nonzero gradient on encoder, predictor,
    and IDM-MLP parameters (the three trainable param groups in PLDM)."""
    torch.manual_seed(0)
    wrapper = _tiny_pldm()
    out = wrapper(_tiny_batch(B=2, T=8))
    wrapper.zero_grad(set_to_none=True)
    out["L_total"].backward()

    enc_first = next(wrapper.encoder.parameters())
    pred_first = next(wrapper.predictor.parameters())
    idm_first = next(wrapper.loss.idm.parameters())
    assert enc_first.grad is not None and enc_first.grad.abs().sum().item() > 0.0
    assert pred_first.grad is not None and pred_first.grad.abs().sum().item() > 0.0
    assert idm_first.grad is not None and idm_first.grad.abs().sum().item() > 0.0


def test_pldm_wrapper_rollout_is_full_horizon() -> None:
    """z_hat lines up with z along the full prediction_horizon window.

    At init the predictor is AdaLN-Zero (identity-on-residual) so z_hat
    won't match z, but the shapes and the wrapper's slice indices should
    be self-consistent. We assert the shapes here, not the values.
    """
    torch.manual_seed(0)
    H = 4
    wrapper = _tiny_pldm(prediction_horizon=H)
    out = wrapper(_tiny_batch(B=2, T=8))
    assert out["z"].shape == (2, 8, 32)
    # The wrapper internally uses z[:, :H+1, :] as the comparison window.
    # We don't expose z_hat as a top-level key, so smoke-test L_sim is finite.
    assert torch.isfinite(out["L_sim"])


def test_pldm_wrapper_bf16_autocast_smoke() -> None:
    """One forward+backward under bf16 autocast on the RTX 6000."""
    try:
        device = require_rtx6000()
    except NoRTX6000Error as e:
        pytest.skip(f"no RTX 6000: {e}")
    torch.manual_seed(0)
    wrapper = _tiny_pldm(prediction_horizon=2).to(device)
    batch = {
        "omega": torch.randn(2, 8, 1, 192, 96, device=device),
        "c": torch.randn(2, 3, device=device),
    }
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = wrapper(batch)
    out["L_total"].backward()
    enc_first = next(wrapper.encoder.parameters())
    assert torch.isfinite(out["L_total"])
    assert enc_first.grad is not None
    assert torch.isfinite(enc_first.grad).all()
