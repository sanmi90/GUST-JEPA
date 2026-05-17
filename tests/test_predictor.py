"""Unit tests for src/models/predictor.py.

Covers the AutoregressivePredictor shape contract, AdaLN-Zero identity at
init, AdaLN module count / zero-init, causal mask enforcement, RoPE-on-Q-K-
only (not V), open-loop rollout, divergence between rollout and teacher
forcing, parameter-count bounds, and gradient flow under the AdaLN-Zero
coupling.
"""

from __future__ import annotations

import torch

import src.models.predictor as pred_mod
from src.models.adaln import AdaLN
from src.models.predictor import AutoregressivePredictor


def test_predictor_shape_contract() -> None:
    """Input z: (2, 8, 32), cond: (2, 3) -> output: (2, 8, 32)."""
    torch.manual_seed(0)
    predictor = AutoregressivePredictor()
    z = torch.randn(2, 8, 32)
    cond = torch.randn(2, 3)
    z_hat = predictor(z, cond)
    assert z_hat.shape == (2, 8, 32)


def test_predictor_identity_at_init() -> None:
    """At init, the block stack returns the residual unchanged.

    All AdaLN gate outputs are zero at init, so every block is
    ``x -> x + 0 * sublayer(...) = x``. The input embedding is NOT identity
    (it lifts 32 -> 384 with random weights); the identity holds INSIDE the
    block stack only.
    """
    torch.manual_seed(0)
    predictor = AutoregressivePredictor()
    z = torch.randn(2, 8, 32)
    cond = torch.randn(2, 3)

    x_after_embed = predictor.embed(z)
    c_seq = predictor.cond_mlp(cond).unsqueeze(1).expand(-1, z.shape[1], -1)
    x_after_blocks = x_after_embed
    for block in predictor.blocks:
        x_after_blocks = block(x_after_blocks, c_seq)

    assert torch.allclose(x_after_blocks, x_after_embed, atol=1e-6)


def test_predictor_all_adaln_modules_zero_init() -> None:
    """Exactly 12 AdaLN modules (2 per block x 6 blocks), each zero-init."""
    torch.manual_seed(0)
    predictor = AutoregressivePredictor()
    adalns = [m for m in predictor.modules() if isinstance(m, AdaLN)]
    assert len(adalns) == 12, f"expected 12 AdaLN modules, got {len(adalns)}"
    for i, adaln in enumerate(adalns):
        assert torch.equal(
            adaln.linear.weight, torch.zeros_like(adaln.linear.weight)
        ), f"AdaLN {i} weight is not zero"
        assert torch.equal(
            adaln.linear.bias, torch.zeros_like(adaln.linear.bias)
        ), f"AdaLN {i} bias is not zero"


def test_predictor_causal_mask() -> None:
    """Perturbing z[:, t, :] leaves outputs at positions < t unchanged.

    Predictor is put in eval mode so the BatchNorm1d at out_proj uses
    running statistics (init: mean 0, var 1) rather than batch statistics,
    which would otherwise couple all positions through the BN reduction
    and mask any genuine causal-attention property.
    """
    torch.manual_seed(0)
    predictor = AutoregressivePredictor()
    predictor.eval()
    z = torch.randn(2, 8, 32)
    cond = torch.randn(2, 3)
    with torch.no_grad():
        z_hat_base = predictor(z, cond)
        for t_perturb in (0, 3, 6):
            z_perturbed = z.clone()
            z_perturbed[:, t_perturb, :] += torch.randn(2, 32)
            z_hat_pert = predictor(z_perturbed, cond)
            if t_perturb > 0:
                assert torch.allclose(
                    z_hat_pert[:, :t_perturb, :],
                    z_hat_base[:, :t_perturb, :],
                    atol=1e-5,
                ), f"causal mask violated: positions < {t_perturb} changed"


def test_predictor_rope_q_k_only_not_v() -> None:
    """apply_rope is called exactly 12 times (Q and K of each of 6 blocks).

    Monkey-patches ``src.models.predictor.apply_rope`` to count calls and
    capture input shapes. V passes straight into scaled_dot_product_attention
    without going through apply_rope, so the call count == 2 * depth.
    """
    torch.manual_seed(0)
    predictor = AutoregressivePredictor()
    captured: list[torch.Tensor] = []
    original = pred_mod.apply_rope

    def wrapped(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        captured.append(x.detach().clone())
        return original(x, cos, sin)

    pred_mod.apply_rope = wrapped
    try:
        z = torch.randn(2, 8, 32)
        cond = torch.randn(2, 3)
        predictor(z, cond)
    finally:
        pred_mod.apply_rope = original

    # 2 calls per block (q, k) * 6 blocks = 12.
    assert len(captured) == 12, f"expected 12 apply_rope calls, got {len(captured)}"
    # Default heads=16, head_dim=384//16=24; input shape (B, heads, T, head_dim).
    expected_shape = (2, 16, 8, 24)
    for i, t in enumerate(captured):
        assert (
            t.shape == expected_shape
        ), f"apply_rope call {i}: got {tuple(t.shape)}, expected {expected_shape}"


def test_predictor_rollout_shape_and_seed_match() -> None:
    """rollout(z_init=(2,1,32), cond=(2,3), steps=4) -> (2, 5, 32); seed preserved."""
    torch.manual_seed(0)
    predictor = AutoregressivePredictor()
    z_init = torch.randn(2, 1, 32)
    cond = torch.randn(2, 3)
    z_full = predictor.rollout(z_init, cond, steps=4)
    assert z_full.shape == (2, 5, 32)
    assert torch.equal(z_full[:, 0, :], z_init[:, 0, :])


def test_predictor_rollout_diverges_from_teacher_at_later_steps() -> None:
    """Rollout matches teacher at first prediction but diverges at later steps.

    Both predictors are put in eval mode (so BN1d uses running stats and the
    test is not perturbed by batch-statistics coupling). At init the AdaLN
    gates are zero, so the block stack is identity; the predictor's only
    nonlinearity in the latent path is the random embed -> identity ->
    out_proj projection. Even so, the prediction of z[:, 2, :] differs
    between rollout (fed its own prediction at position 1) and teacher
    forcing (fed ground-truth z[:, 1, :]), and the divergence grows.
    """
    torch.manual_seed(0)
    predictor = AutoregressivePredictor()
    predictor.eval()
    T = 8
    z = torch.randn(2, T, 32)
    cond = torch.randn(2, 3)

    with torch.no_grad():
        z_full = predictor.rollout(z[:, :1, :], cond, steps=T - 1)
        teacher = predictor(z, cond)

    # First rollout prediction equals teacher's prediction at position 0
    # (both functions of z[:, :1] only, because of the causal mask).
    assert torch.allclose(z_full[:, 1, :], teacher[:, 0, :], atol=1e-5)
    # Last rollout prediction differs from teacher's last prediction.
    assert not torch.allclose(z_full[:, T - 1, :], teacher[:, T - 2, :], atol=1e-5)


def test_predictor_parameter_count_in_range() -> None:
    """14M < total params < 18M for the default config."""
    torch.manual_seed(0)
    predictor = AutoregressivePredictor()
    n_params = sum(p.numel() for p in predictor.parameters())
    assert 1.4e7 < n_params < 1.8e7, f"got {n_params} params, expected in (1.4e7, 1.8e7)"


def test_predictor_gradient_flows() -> None:
    """After one warm-up optimizer step, all parameters get nonzero gradients.

    AdaLN-Zero gates are zero at init, so the sublayer (attention, MLP) and
    the cond_mlp (which feeds AdaLN) get zero gradient on the first
    backward. One SGD step on the gated parameters makes the gates non-zero;
    the subsequent backward then produces non-zero gradients on every
    trainable parameter.
    """
    torch.manual_seed(0)
    predictor = AutoregressivePredictor()
    z = torch.randn(2, 8, 32)
    cond = torch.randn(2, 3)
    opt = torch.optim.SGD(predictor.parameters(), lr=1e-2)

    # Warm-up step to break AdaLN-Zero identity (sublayer params start with
    # zero gradient because gate=0).
    z_hat = predictor(z, cond)
    z_hat.pow(2).sum().backward()
    opt.step()
    opt.zero_grad()

    # Fresh backward now propagates gradients into every parameter.
    z_hat = predictor(z, cond)
    z_hat.pow(2).sum().backward()
    for name, p in predictor.named_parameters():
        assert p.grad is not None, f"{name}: no gradient"
        assert torch.any(p.grad != 0), f"{name}: zero gradient"
