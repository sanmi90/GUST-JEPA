"""Tests for ``src.models.jepa.JEPA``."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from src.models.encoder import HybridCNNViTEncoder
from src.models.jepa import JEPA
from src.models.predictor import AutoregressivePredictor
from src.models.sigreg import SIGReg
from src.models.vicreg import VICReg
from src.utils.device import NoRTX6000Error, require_rtx6000


def _tiny_jepa(
    *,
    latent_dim: int = 32,
    H_roll: int = 2,
    rollout_start_strategy: str = "fixed_zero",
    rollout_weight: float = 0.5,
    lambda_anticollapse: float = 0.1,
    predictor_dropout: float = 0.0,
    anticollapse: torch.nn.Module | None = None,
) -> JEPA:
    """Build a small but real JEPA for unit testing.

    Default ``predictor_dropout=0`` so tests that compare values exactly
    are deterministic in train mode.
    """
    encoder = HybridCNNViTEncoder(latent_dim=latent_dim)
    predictor = AutoregressivePredictor(
        latent_dim=latent_dim,
        cond_dim=3,
        hidden_dim=384,
        depth=2,
        heads=8,
        dropout=predictor_dropout,
        max_seq_len=32,
    )
    if anticollapse is None:
        anticollapse = SIGReg(dim=latent_dim, num_projections=64, num_knots=9)
    return JEPA(
        encoder=encoder,
        predictor=predictor,
        anticollapse=anticollapse,
        lambda_anticollapse=lambda_anticollapse,
        rollout_weight=rollout_weight,
        H_roll=H_roll,
        rollout_start_strategy=rollout_start_strategy,
    )


def _tiny_batch(B: int = 2, T: int = 8, H: int = 192, W: int = 96) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "omega": torch.randn(B, T, 1, H, W),
        "c": torch.randn(B, 3),
    }


def test_jepa_shape_contract() -> None:
    torch.manual_seed(0)
    jepa = _tiny_jepa()
    batch = _tiny_batch()
    out = jepa(batch)
    assert set(out.keys()) == {
        "loss_total", "loss_pred", "loss_roll", "loss_anticollapse", "loss_obs", "z"
    }
    for k in ("loss_total", "loss_pred", "loss_roll", "loss_anticollapse", "loss_obs"):
        assert out[k].dim() == 0
        assert torch.isfinite(out[k])
    assert out["loss_obs"].item() == 0.0, "loss_obs should be 0 when no observable head"
    assert out["z"].shape == (2, 8, 32)


def test_jepa_loss_decomposition() -> None:
    """L_total == L_pred + rollout_weight * L_roll + lambda * L_anticollapse."""
    torch.manual_seed(0)
    jepa = _tiny_jepa(rollout_weight=0.5, lambda_anticollapse=0.1)
    batch = _tiny_batch()
    out = jepa(batch)
    expected = (
        out["loss_pred"].item()
        + 0.5 * out["loss_roll"].item()
        + 0.1 * out["loss_anticollapse"].item()
    )
    assert math.isclose(out["loss_total"].item(), expected, rel_tol=1e-5, abs_tol=1e-5)


def test_jepa_pred_loss_nonzero_at_init() -> None:
    """AdaLN-Zero makes the predictor identity-on-residual, but embed and
    out_proj are not identity, so L_pred is not zero at init."""
    torch.manual_seed(0)
    jepa = _tiny_jepa()
    batch = _tiny_batch()
    out = jepa(batch)
    L_pred = out["loss_pred"].item()
    assert L_pred > 0.0
    assert 0.01 < L_pred < 10.0


def test_jepa_anticollapse_swap_takes_effect() -> None:
    """After set_anticollapse(VICReg(...)) the loss value changes and no
    SIGReg-specific buffers remain in state_dict."""
    torch.manual_seed(0)
    jepa = _tiny_jepa()
    batch = _tiny_batch()

    torch.manual_seed(1)
    out_sigreg = jepa(batch)
    L_anti_sigreg = out_sigreg["loss_anticollapse"].item()

    sigreg_keys_before = {k for k in jepa.state_dict() if k.startswith("anticollapse.")}
    assert sigreg_keys_before == {"anticollapse.knots", "anticollapse.quad_weights", "anticollapse.phi_0"}

    jepa.set_anticollapse(VICReg(d=32, mu=25.0, lambda_=25.0, nu=1.0))

    torch.manual_seed(1)
    out_vicreg = jepa(batch)
    L_anti_vicreg = out_vicreg["loss_anticollapse"].item()

    assert not math.isclose(L_anti_sigreg, L_anti_vicreg, rel_tol=1e-3)
    sigreg_keys_after = {k for k in jepa.state_dict() if k.startswith("anticollapse.")}
    assert sigreg_keys_after == set(), f"unexpected remaining keys: {sigreg_keys_after}"


def test_jepa_gradient_flows_through_each_loss() -> None:
    """Each of L_pred, L_roll, L_anticollapse separately produces nonzero
    encoder gradients."""
    torch.manual_seed(0)
    jepa = _tiny_jepa()
    batch = _tiny_batch()

    enc_first_param = next(jepa.encoder.parameters())

    out = jepa(batch)
    jepa.zero_grad()
    out["loss_pred"].backward(retain_graph=True)
    g_pred = enc_first_param.grad.detach().abs().sum().item()
    assert g_pred > 0.0

    out = jepa(batch)
    jepa.zero_grad()
    out["loss_roll"].backward(retain_graph=True)
    g_roll = enc_first_param.grad.detach().abs().sum().item()
    assert g_roll > 0.0

    out = jepa(batch)
    jepa.zero_grad()
    out["loss_anticollapse"].backward(retain_graph=True)
    g_anti = enc_first_param.grad.detach().abs().sum().item()
    assert g_anti > 0.0


def test_jepa_rollout_strategies_produce_different_t0() -> None:
    """fixed_zero is deterministic; uniform_random depends on the torch seed."""
    torch.manual_seed(0)
    jepa = _tiny_jepa(rollout_start_strategy="fixed_zero", H_roll=2)
    assert jepa._sample_t0(T=8) == 0
    assert jepa._sample_t0(T=8) == 0

    jepa.rollout_start_strategy = "uniform_random"
    torch.manual_seed(1)
    a = jepa._sample_t0(T=32)
    torch.manual_seed(2)
    b = jepa._sample_t0(T=32)
    assert 0 <= a <= 32 - 1 - 2
    assert 0 <= b <= 32 - 1 - 2
    assert a != b


def test_jepa_rollout_loss_matches_hand_computation() -> None:
    """fixed_zero rollout MSE matches a direct computation on B=2, T=10, H_roll=2.

    B=2 is the minimum so the rollout's seed (T_init=1) survives the
    predictor's BatchNorm1d in train mode.
    """
    torch.manual_seed(0)
    jepa = _tiny_jepa(H_roll=2, rollout_start_strategy="fixed_zero", predictor_dropout=0.0)
    omega = torch.randn(2, 10, 1, 192, 96)
    c = torch.randn(2, 3)
    batch = {"omega": omega, "c": c}

    torch.manual_seed(0)
    out = jepa(batch)
    L_roll_wrapper = out["loss_roll"].item()

    torch.manual_seed(0)
    z = jepa.encoder(omega)
    _ = jepa.predictor(z, c)
    z_full = jepa.predictor.rollout(z[:, :1, :], c, steps=2)
    expected = F.mse_loss(z_full[:, 1:3, :].float(), z[:, 1:3, :].float()).item()
    assert math.isclose(L_roll_wrapper, expected, rel_tol=1e-5, abs_tol=1e-5)


def test_jepa_c_dropout_zeros_cond_in_train_mode() -> None:
    """With c_dropout_prob=1.0 in train mode, cond is replaced with zeros
    before the predictor sees it. The predictor outputs are therefore the
    same as if the caller had passed cond=0 directly.
    """
    from src.models.encoder import HybridCNNViTEncoder
    from src.models.predictor import AutoregressivePredictor
    from src.models.sigreg import SIGReg

    torch.manual_seed(0)
    encoder = HybridCNNViTEncoder(latent_dim=32)
    predictor = AutoregressivePredictor(
        latent_dim=32, cond_dim=3, hidden_dim=384, depth=2, heads=8,
        dropout=0.0, max_seq_len=32,
    )
    sigreg = SIGReg(dim=32, num_projections=64, num_knots=9)
    jepa = JEPA(
        encoder=encoder, predictor=predictor, anticollapse=sigreg,
        H_roll=2, rollout_weight=0.0, c_dropout_prob=1.0,
    )
    jepa.train()
    batch = _tiny_batch()
    torch.manual_seed(0)
    out_drop = jepa(batch)
    # reference: same batch with c manually zeroed and c_dropout_prob=0
    jepa.c_dropout_prob = 0.0
    batch_zero = {"omega": batch["omega"], "c": torch.zeros_like(batch["c"])}
    torch.manual_seed(0)
    out_zero = jepa(batch_zero)
    assert torch.allclose(
        out_drop["loss_pred"], out_zero["loss_pred"], atol=1e-5
    ), "c_dropout_prob=1.0 should match cond=0 reference (train mode)"


def test_jepa_c_dropout_eval_mode_preserves_cond() -> None:
    """In eval mode, c_dropout never fires regardless of c_dropout_prob."""
    from src.models.encoder import HybridCNNViTEncoder
    from src.models.predictor import AutoregressivePredictor
    from src.models.sigreg import SIGReg

    torch.manual_seed(0)
    encoder = HybridCNNViTEncoder(latent_dim=32)
    predictor = AutoregressivePredictor(
        latent_dim=32, cond_dim=3, hidden_dim=384, depth=2, heads=8,
        dropout=0.0, max_seq_len=32,
    )
    jepa = JEPA(
        encoder=encoder, predictor=predictor,
        anticollapse=SIGReg(dim=32, num_projections=64, num_knots=9),
        H_roll=2, rollout_weight=0.0, c_dropout_prob=1.0,
    )
    jepa.eval()
    batch = _tiny_batch()
    torch.manual_seed(0)
    out_dropout_on = jepa(batch)
    # reference: c_dropout_prob=0 in eval mode (no zeroing either way)
    jepa.c_dropout_prob = 0.0
    torch.manual_seed(0)
    out_no_dropout = jepa(batch)
    assert torch.allclose(
        out_dropout_on["loss_pred"], out_no_dropout["loss_pred"], atol=1e-5
    ), "c_dropout should be a no-op in eval mode"


def test_jepa_observable_head_adds_loss_and_grad() -> None:
    """With an ObservableHead and a 'cl_future' batch tensor, loss_obs is
    nonzero, loss_total includes the eta * loss_obs term, and the encoder
    receives a gradient component from L_obs.
    """
    from src.models.encoder import HybridCNNViTEncoder
    from src.models.observable_head import ObservableHead
    from src.models.predictor import AutoregressivePredictor
    from src.models.sigreg import SIGReg

    torch.manual_seed(0)
    encoder = HybridCNNViTEncoder(latent_dim=32)
    predictor = AutoregressivePredictor(
        latent_dim=32, cond_dim=3, hidden_dim=384, depth=2, heads=8,
        dropout=0.0, max_seq_len=32,
    )
    head = ObservableHead(latent_dim=32, hidden_dim=64, n_deltas=3)
    jepa = JEPA(
        encoder=encoder, predictor=predictor,
        anticollapse=SIGReg(dim=32, num_projections=64, num_knots=9),
        H_roll=2, rollout_weight=0.0, lambda_anticollapse=0.0,
        observable_head=head, observable_weight=0.01,
    )
    batch = _tiny_batch()
    batch["cl_future"] = torch.randn(2, 8, 3)
    out = jepa(batch)
    assert out["loss_obs"].item() > 0.0, "loss_obs should be > 0 with random head + targets"
    # loss_total = loss_pred + 0 * loss_roll + 0 * loss_anti + 0.01 * loss_obs
    expected = out["loss_pred"].item() + 0.01 * out["loss_obs"].item()
    assert math.isclose(out["loss_total"].item(), expected, rel_tol=1e-5, abs_tol=1e-5)

    # gradient flow from loss_obs alone reaches the encoder
    enc_first = next(jepa.encoder.parameters())
    jepa.zero_grad()
    out["loss_obs"].backward()
    assert enc_first.grad is not None and enc_first.grad.detach().abs().sum().item() > 0.0


def test_jepa_observable_head_missing_batch_key_errors() -> None:
    """When the head is configured but the batch lacks 'cl_future', the
    wrapper raises a KeyError early."""
    from src.models.encoder import HybridCNNViTEncoder
    from src.models.observable_head import ObservableHead
    from src.models.predictor import AutoregressivePredictor
    from src.models.sigreg import SIGReg

    torch.manual_seed(0)
    encoder = HybridCNNViTEncoder(latent_dim=32)
    predictor = AutoregressivePredictor(
        latent_dim=32, cond_dim=3, hidden_dim=384, depth=2, heads=8,
        dropout=0.0, max_seq_len=32,
    )
    head = ObservableHead(latent_dim=32, hidden_dim=64, n_deltas=3)
    jepa = JEPA(
        encoder=encoder, predictor=predictor,
        anticollapse=SIGReg(dim=32, num_projections=64, num_knots=9),
        H_roll=2, observable_head=head, observable_weight=0.01,
    )
    with pytest.raises(KeyError, match="cl_future"):
        jepa(_tiny_batch())


def test_jepa_bf16_autocast_smoke() -> None:
    """One forward+backward under bf16 autocast on the RTX 6000."""
    try:
        device = require_rtx6000()
    except NoRTX6000Error as e:
        pytest.skip(f"No RTX 6000 GPU: {e}")

    torch.manual_seed(0)
    jepa = _tiny_jepa(H_roll=2).to(device)
    batch = {
        "omega": torch.randn(2, 8, 1, 192, 96, device=device),
        "c": torch.randn(2, 3, device=device),
    }
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = jepa(batch)
    out["loss_total"].backward()
    enc_first_param = next(jepa.encoder.parameters())
    assert enc_first_param.grad is not None
    assert torch.isfinite(enc_first_param.grad).all()
    assert torch.isfinite(out["loss_total"])
