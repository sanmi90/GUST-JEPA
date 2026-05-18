"""Session 5 pre-variant sanity checks.

Five small gates that must pass before any 5k-iter variant smoke run
lands. The checks rule out trivial wiring bugs (BatchNorm running stats
going pathological, predictor not learning, SIGReg gradient not
reaching the encoder, rollout disagreeing with teacher-forced at
horizon 1, data loader emitting wrong shapes) so that a TRIVIAL outcome
from the actual training run is a methodological finding, not a bug
masquerading as collapse.

Reference: SESSION5_MEANINGFUL_SMOKE_5K.md "Pre-run sanity checks".

Run:
    python -m src.training.sanity_checks --all                 # checks 1-5
    python -m src.training.sanity_checks --check 3             # one check only

Each check returns a :class:`SanityCheckResult` with a PASS/FAIL flag,
a one-line human-readable message, and an optional diagnostics dict.
The CLI exits non-zero if any selected check fails.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch
import yaml
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.data.episode_dataset import EpisodeDataset
from src.models.adaln import AdaLN
from src.models.encoder import HybridCNNViTEncoder
from src.models.jepa import JEPA
from src.models.predictor import AutoregressivePredictor
from src.models.sigreg import SIGReg
from src.utils.device import NoRTX6000Error, require_rtx6000


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SanityCheckResult:
    """Outcome of one sanity check.

    Attributes:
        name: Short identifier (``check_1``, ``check_2``, ...).
        passed: True iff the check's assertions all held.
        message: Single-line human-readable summary.
        diagnostics: Optional dict of metric name to scalar value.
    """

    name: str
    passed: bool
    message: str
    diagnostics: dict[str, float] = field(default_factory=dict)


def _projection_batchnorm(encoder: HybridCNNViTEncoder) -> nn.BatchNorm1d:
    """Return ``encoder.proj[-1]``, asserting it is a ``nn.BatchNorm1d``.

    The sanity checks that touch BatchNorm running stats are
    BatchNorm-specific (HANDOFF.md D17). If a future LayerNorm variant
    runs these checks the caller is responsible for skipping them or
    routing them to a LayerNorm-equivalent diagnostic.
    """
    final = encoder.proj[-1]
    if not isinstance(final, nn.BatchNorm1d):
        raise TypeError(
            f"sanity checks 1 and 3 require encoder.proj[-1] to be nn.BatchNorm1d, "
            f"got {type(final).__name__}. Skip these checks for LayerNorm variants."
        )
    return final


def check_1_batchnorm_running_stats(
    encoder: HybridCNNViTEncoder,
    batches: Sequence[Tensor],
) -> SanityCheckResult:
    """Verify the projection BN running stats stay healthy in train mode.

    Phases:
        1. At init, ``running_mean`` is approximately 0 and ``running_var``
           is approximately 1 (PyTorch defaults).
        2. Feed ``batches`` through ``encoder.train()`` forward-only
           (no backward); the BN running statistics update.
        3. After warmup, the running stats are finite, ``|mean|.max() < 10``,
           and ``var`` lies in ``(1e-4, 100)``.

    These bounds are loose; they catch obvious wiring bugs (NaN inputs,
    BN running stats not updating) but do NOT diagnose the deeper "case
    label leaking through batch statistics" concern from HANDOFF
    "Warnings and pitfalls" -- that is theoretical and is checked by the
    linear probe diagnostic during training, not here.

    Args:
        encoder: Fresh ``HybridCNNViTEncoder`` instance.
        batches: Iterable of ``(B, T, 1, H, W)`` tensors to feed in train
            mode. The check is fast; a handful of small batches suffices.

    Returns:
        :class:`SanityCheckResult` with ``name='check_1'``.
    """
    bn = _projection_batchnorm(encoder)

    init_mean = bn.running_mean.detach().clone()
    init_var = bn.running_var.detach().clone()
    if not torch.allclose(init_mean, torch.zeros_like(init_mean), atol=1e-5):
        return SanityCheckResult(
            name="check_1",
            passed=False,
            message=(
                "BN running_mean at init is not zero "
                f"(max abs {init_mean.abs().max().item():.2e})"
            ),
        )
    if not torch.allclose(init_var, torch.ones_like(init_var), atol=1e-5):
        return SanityCheckResult(
            name="check_1",
            passed=False,
            message=(
                "BN running_var at init is not one "
                f"(max |var-1| {(init_var - 1).abs().max().item():.2e})"
            ),
        )

    encoder.train()
    with torch.no_grad():
        for b in batches:
            _ = encoder(b)

    final_mean = bn.running_mean.detach()
    final_var = bn.running_var.detach()
    if not (torch.isfinite(final_mean).all() and torch.isfinite(final_var).all()):
        return SanityCheckResult(
            name="check_1",
            passed=False,
            message="BN running stats are not finite (NaN/Inf) after warmup",
        )
    max_abs_mean = final_mean.abs().max().item()
    max_var = final_var.max().item()
    min_var = final_var.min().item()
    if max_abs_mean > 10.0:
        return SanityCheckResult(
            name="check_1",
            passed=False,
            message=f"BN running_mean too large after warmup: max |mean|={max_abs_mean:.3f} > 10.0",
        )
    if not (1e-4 < min_var and max_var < 100.0):
        return SanityCheckResult(
            name="check_1",
            passed=False,
            message=(
                f"BN running_var out of expected range after warmup: "
                f"[{min_var:.4g}, {max_var:.4g}] not in (1e-4, 100)"
            ),
        )

    return SanityCheckResult(
        name="check_1",
        passed=True,
        message=(
            f"BN running stats healthy after {len(batches)} warmup batches "
            f"(|mean| <= {max_abs_mean:.3f}, var in [{min_var:.3f}, {max_var:.3f}])"
        ),
        diagnostics={
            "max_abs_mean": max_abs_mean,
            "min_var": min_var,
            "max_var": max_var,
        },
    )


def check_2_predictor_identity_then_moves(
    jepa: JEPA,
    batch: dict[str, Tensor],
    n_overfitting_steps: int = 10,
    lr: float = 1e-3,
) -> SanityCheckResult:
    """Verify AdaLN-Zero init and that one batch's overfit decreases L_pred.

    Phases:
        1. At iter 0 ``L_pred`` is in ``[0.01, 10.0]``. AdaLN-Zero makes the
           predictor identity-on-residual, but ``embed`` and ``out_proj`` are
           not identity, so ``L_pred`` is bounded but nonzero.
        2. After one Adam step on this batch, at least one ``AdaLN.linear.weight``
           has at least one nonzero element.
        3. After ``n_overfitting_steps`` further Adam steps on the same batch,
           ``L_pred`` is below its iter-0 value.

    This is the cheapest possible test that learning is wired correctly.

    Args:
        jepa: A fresh :class:`JEPA` wrapper with ``predictor_dropout=0`` so
            the iter-0 forward is deterministic.
        batch: ``{'omega', 'c'}`` dict matching the JEPA contract.
        n_overfitting_steps: Number of additional Adam steps after the
            init step. The check overfits on a single batch on purpose.
        lr: Adam learning rate.

    Returns:
        :class:`SanityCheckResult` with ``name='check_2'``.
    """
    jepa.train()

    out0 = jepa(batch)
    L_pred_init = out0["loss_pred"].item()
    if not (0.01 < L_pred_init < 10.0):
        return SanityCheckResult(
            name="check_2",
            passed=False,
            message=(
                f"L_pred at init is {L_pred_init:.4f}, outside expected "
                f"[0.01, 10.0]. AdaLN-Zero init may be broken."
            ),
        )

    optimizer = AdamW([p for p in jepa.parameters() if p.requires_grad], lr=lr)

    optimizer.zero_grad(set_to_none=True)
    out0["loss_total"].backward()
    optimizer.step()

    moved = False
    for m in jepa.predictor.modules():
        if isinstance(m, AdaLN) and m.linear.weight.detach().abs().sum().item() > 0.0:
            moved = True
            break
    if not moved:
        return SanityCheckResult(
            name="check_2",
            passed=False,
            message=(
                "After one Adam step, no AdaLN.linear.weight has moved off zero. "
                "Predictor is not learning."
            ),
        )

    L_pred_trace = [L_pred_init]
    for _ in range(n_overfitting_steps):
        out = jepa(batch)
        L_pred_trace.append(out["loss_pred"].item())
        optimizer.zero_grad(set_to_none=True)
        out["loss_total"].backward()
        optimizer.step()
    L_pred_final = L_pred_trace[-1]

    if not (L_pred_final < L_pred_init):
        return SanityCheckResult(
            name="check_2",
            passed=False,
            message=(
                f"L_pred did not decrease after {n_overfitting_steps} overfitting "
                f"steps on one batch: init={L_pred_init:.4f}, final={L_pred_final:.4f}"
            ),
        )

    return SanityCheckResult(
        name="check_2",
        passed=True,
        message=(
            f"Predictor learns: L_pred {L_pred_init:.4f} -> {L_pred_final:.4f} over "
            f"{n_overfitting_steps + 1} Adam steps; AdaLN gates moved off zero."
        ),
        diagnostics={
            "L_pred_init": L_pred_init,
            "L_pred_final": L_pred_final,
        },
    )


def check_3_sigreg_gradient_reaches_encoder(
    encoder: HybridCNNViTEncoder,
    sigreg: SIGReg,
    omega: Tensor,
    min_grad_norm: float = 1e-8,
) -> SanityCheckResult:
    """Verify ``SIGReg(z).backward()`` produces a finite nonzero gradient on
    the encoder projection's BatchNorm bias.

    The deeper concern (SESSION5_MEANINGFUL_SMOKE_5K.md): "if SIGReg's
    gradient on real encoder output is orders of magnitude different from
    the synthetic-Gaussian unit test, that is a signal that the encoder
    output distribution is so degenerate that even the regulariser is
    degenerate." This check only verifies the gradient is finite and
    nonzero; the magnitude comparison is left to the training diagnostics.

    Args:
        encoder: ``HybridCNNViTEncoder`` instance (typically fresh).
        sigreg: ``SIGReg(dim=latent_dim, ...)`` instance.
        omega: ``(B, T, 1, H, W)`` input batch.
        min_grad_norm: Floor on the BN-bias gradient L2 norm; values
            below this are treated as effectively zero.

    Returns:
        :class:`SanityCheckResult` with ``name='check_3'``.
    """
    bn = _projection_batchnorm(encoder)

    encoder.zero_grad(set_to_none=True)
    z = encoder(omega)
    loss = sigreg(z.flatten(0, 1))
    if not torch.isfinite(loss):
        return SanityCheckResult(
            name="check_3",
            passed=False,
            message=f"SIGReg loss is not finite: {loss.item()}",
        )
    if not loss.requires_grad:
        return SanityCheckResult(
            name="check_3",
            passed=False,
            message=(
                "SIGReg loss has no grad_fn (encoder is fully frozen or graph is "
                "detached); BN bias cannot receive a gradient."
            ),
        )
    loss.backward()

    grad = bn.bias.grad
    if grad is None:
        return SanityCheckResult(
            name="check_3",
            passed=False,
            message=(
                "encoder.proj BatchNorm bias.grad is None after SIGReg.backward(). "
                "Either the encoder is frozen or the autograd graph is broken."
            ),
        )
    if not torch.isfinite(grad).all():
        return SanityCheckResult(
            name="check_3",
            passed=False,
            message="encoder.proj BatchNorm bias.grad contains NaN/Inf",
        )
    grad_norm = grad.norm().item()
    if grad_norm < min_grad_norm:
        return SanityCheckResult(
            name="check_3",
            passed=False,
            message=(
                f"encoder.proj BatchNorm bias.grad L2 norm too small: "
                f"{grad_norm:.2e} < {min_grad_norm:.2e}"
            ),
        )

    return SanityCheckResult(
        name="check_3",
        passed=True,
        message=(
            f"SIGReg gradient reaches encoder BN bias: ||grad||={grad_norm:.4f}, "
            f"loss={loss.item():.4f}"
        ),
        diagnostics={
            "sigreg_loss": float(loss.item()),
            "bn_bias_grad_norm": grad_norm,
        },
    )


def check_4_rollout_matches_teacher_at_horizon_1(
    jepa: JEPA,
    batch: dict[str, Tensor],
    atol: float = 1e-4,
) -> SanityCheckResult:
    """Verify ``predictor.rollout(steps=1)`` agrees with teacher-forced
    prediction at the corresponding position.

    Both methods predict frame 1 from frame 0. On the same eval-mode encoder
    output they must agree within numerical tolerance. Disagreement here
    would indicate a Session 3 bug missed by the existing tests.

    Args:
        jepa: The wrapper composing encoder + predictor.
        batch: ``{'omega', 'c'}`` dict.
        atol: Max-abs tolerance. Default ``1e-4`` (fp32 path); use ``2e-2``
            under bf16 autocast.

    Returns:
        :class:`SanityCheckResult` with ``name='check_4'``.
    """
    jepa.eval()
    with torch.no_grad():
        z = jepa.encoder(batch["omega"])
        cond = batch["c"]
        z_tf = jepa.predictor(z, cond)
        z_roll = jepa.predictor.rollout(z[:, :1, :], cond, steps=1)

    pred_tf_frame1 = z_tf[:, 0, :].float()
    pred_roll_frame1 = z_roll[:, 1, :].float()
    diff = (pred_tf_frame1 - pred_roll_frame1).abs().max().item()
    if diff > atol:
        return SanityCheckResult(
            name="check_4",
            passed=False,
            message=(
                f"rollout(steps=1)[:, 1, :] disagrees with predictor(z)[:, 0, :] by "
                f"max-abs {diff:.6e} > atol {atol:.6e}"
            ),
            diagnostics={"max_abs_diff": diff},
        )
    return SanityCheckResult(
        name="check_4",
        passed=True,
        message=(
            f"rollout agrees with teacher-forced at horizon 1 within atol "
            f"{atol:.2e} (diff={diff:.2e})"
        ),
        diagnostics={"max_abs_diff": diff},
    )


def check_5_data_shapes(
    partition: str,
    case_subset: Sequence[str],
    batch_size: int = 16,
    T: int = 32,
    max_abs_omega: float = 10_000.0,
) -> SanityCheckResult:
    """Verify the data loader emits the expected shapes on the 5-case subset.

    Asserts on one collated batch:
        - ``omega.shape == (B, T, 1, 192, 96)``
        - ``c.shape == (B, 3)``
        - ``omega`` is finite (no NaN, no Inf)
        - ``omega.abs().max() < max_abs_omega``. The cache stores
          unnormalised vorticity; at Re=5000 a sample survey across 14
          encounters shows the bulk of values in ``[-1400, +2010]``, with
          peaks inside vortex cores or boundary-layer separations. The
          default 10,000 ceiling catches obvious corruption (e.g., a
          stray Inf surviving NaN filling) without false-failing
          legitimate intense vorticity events.

    Args:
        partition: Partition tag, typically ``'v1'``.
        case_subset: List of ``case_id`` strings to filter the train split.
        batch_size: Number of sub-trajectories per batch (Session 5 uses 16).
        T: Sub-trajectory length (Session 5 uses 32).
        max_abs_omega: Bound on ``omega`` magnitude; raise if exceeded.

    Returns:
        :class:`SanityCheckResult` with ``name='check_5'``.
    """
    ds = EpisodeDataset(partition=partition, split="train", subtraj_len=T)
    wanted = set(case_subset)
    ds.samples = [s for s in ds.samples if s[0] in wanted]
    if len(ds.samples) < batch_size:
        return SanityCheckResult(
            name="check_5",
            passed=False,
            message=(
                f"only {len(ds.samples)} train samples for cases={sorted(wanted)} in "
                f"partition={partition}; need >= {batch_size}"
            ),
        )

    from src.training.train_jepa import jepa_collate

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=jepa_collate,
        drop_last=False,
    )
    batch = next(iter(loader))
    omega = batch["omega"]
    c = batch["c"]
    expected_omega = (batch_size, T, 1, 192, 96)
    if tuple(omega.shape) != expected_omega:
        return SanityCheckResult(
            name="check_5",
            passed=False,
            message=f"omega shape {tuple(omega.shape)} != expected {expected_omega}",
        )
    if tuple(c.shape) != (batch_size, 3):
        return SanityCheckResult(
            name="check_5",
            passed=False,
            message=f"c shape {tuple(c.shape)} != expected ({batch_size}, 3)",
        )
    if not torch.isfinite(omega).all():
        return SanityCheckResult(
            name="check_5",
            passed=False,
            message="omega contains NaN/Inf in the first collated batch",
        )
    max_abs = omega.abs().max().item()
    if not (max_abs < max_abs_omega):
        return SanityCheckResult(
            name="check_5",
            passed=False,
            message=(
                f"omega max abs {max_abs:.2f} >= {max_abs_omega:.2f}; cache may be "
                f"corrupted or preprocessing went wrong"
            ),
        )
    return SanityCheckResult(
        name="check_5",
        passed=True,
        message=(
            f"data shapes verified on {len(ds.samples)} samples: omega "
            f"{tuple(omega.shape)} in [{omega.min().item():.1f}, "
            f"{omega.max().item():.1f}], c {tuple(c.shape)}"
        ),
        diagnostics={
            "omega_min": float(omega.min().item()),
            "omega_max": float(omega.max().item()),
            "n_samples": float(len(ds.samples)),
        },
    )


def _run_check_1(seed: int) -> SanityCheckResult:
    torch.manual_seed(seed)
    encoder = HybridCNNViTEncoder(latent_dim=32)
    batches = [torch.randn(4, 4, 1, 192, 96) for _ in range(5)]
    return check_1_batchnorm_running_stats(encoder, batches)


def _run_check_2(seed: int) -> SanityCheckResult:
    torch.manual_seed(seed)
    encoder = HybridCNNViTEncoder(latent_dim=32)
    predictor = AutoregressivePredictor(latent_dim=32, cond_dim=3, depth=2, max_seq_len=32, dropout=0.0)
    sigreg = SIGReg(dim=32, num_projections=64, num_knots=9)
    jepa = JEPA(encoder, predictor, sigreg, H_roll=2, rollout_weight=0.5)
    batch = {"omega": torch.randn(2, 8, 1, 192, 96), "c": torch.randn(2, 3)}
    return check_2_predictor_identity_then_moves(jepa, batch)


def _run_check_3(seed: int) -> SanityCheckResult:
    torch.manual_seed(seed)
    encoder = HybridCNNViTEncoder(latent_dim=32)
    sigreg = SIGReg(dim=32, num_projections=64, num_knots=9)
    omega = torch.randn(2, 8, 1, 192, 96)
    return check_3_sigreg_gradient_reaches_encoder(encoder, sigreg, omega)


def _run_check_4(seed: int, partition: str, case_subset: Sequence[str], T: int) -> SanityCheckResult:
    torch.manual_seed(seed)
    encoder = HybridCNNViTEncoder(latent_dim=32)
    predictor = AutoregressivePredictor(latent_dim=32, cond_dim=3, depth=2, max_seq_len=T, dropout=0.0)
    sigreg = SIGReg(dim=32, num_projections=64, num_knots=9)
    jepa = JEPA(encoder, predictor, sigreg, H_roll=2, rollout_weight=0.5)
    ds = EpisodeDataset(partition=partition, split="train", subtraj_len=T)
    wanted = set(case_subset)
    ds.samples = [s for s in ds.samples if s[0] in wanted]
    if not ds.samples:
        return SanityCheckResult(
            name="check_4",
            passed=False,
            message=f"no samples for cases {sorted(wanted)} in partition {partition}",
        )
    from src.training.train_jepa import jepa_collate

    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=jepa_collate)
    batch = next(iter(loader))
    return check_4_rollout_matches_teacher_at_horizon_1(jepa, batch)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Session 5 pre-variant sanity checks")
    p.add_argument("--all", action="store_true", help="Run all five checks")
    p.add_argument(
        "--check",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=None,
        help="Run a single check by number (1-5).",
    )
    p.add_argument("--partition", type=str, default="v1")
    p.add_argument(
        "--cases-from",
        type=str,
        default="configs/cases/smoke_5cases.yaml",
        help="YAML path with a 'cases' list (relative to repo root).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--T", type=int, default=32, help="Sub-trajectory length for checks 4, 5.")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size for check 5.")
    p.add_argument(
        "--require-gpu",
        action="store_true",
        help="Require the RTX 6000; skip with NoRTX6000Error otherwise.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.all and args.check is None:
        print("error: pass --all or --check N", file=sys.stderr)
        sys.exit(2)
    if args.require_gpu:
        try:
            require_rtx6000()
        except NoRTX6000Error as e:
            print(f"[sanity_checks] {e}", file=sys.stderr)
            sys.exit(2)

    cases_path = REPO_ROOT / args.cases_from
    case_subset: list[str] = []
    if args.all or args.check in (4, 5):
        if not cases_path.exists():
            print(
                f"error: --cases-from {cases_path} does not exist; "
                f"required for checks 4 and 5",
                file=sys.stderr,
            )
            sys.exit(2)
        with open(cases_path) as f:
            case_subset = list(yaml.safe_load(f)["cases"])

    run_n = (
        {1, 2, 3, 4, 5}
        if args.all
        else ({args.check} if args.check is not None else set())
    )

    results: list[SanityCheckResult] = []
    if 1 in run_n:
        results.append(_run_check_1(args.seed))
    if 2 in run_n:
        results.append(_run_check_2(args.seed))
    if 3 in run_n:
        results.append(_run_check_3(args.seed))
    if 4 in run_n:
        results.append(_run_check_4(args.seed, args.partition, case_subset, args.T))
    if 5 in run_n:
        results.append(
            check_5_data_shapes(
                partition=args.partition,
                case_subset=case_subset,
                batch_size=args.batch_size,
                T=args.T,
            )
        )

    n_fail = 0
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"[{r.name}] {status}: {r.message}", flush=True)
        if not r.passed:
            n_fail += 1

    n_total = len(results)
    print(f"\n{n_total - n_fail}/{n_total} sanity checks passed", flush=True)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
