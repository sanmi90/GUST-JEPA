# SESSION4_JEPA_WRAPPER_AND_TRAINING_SCAFFOLD.md

Session 4 plan for the vortex-jepa project.

Last updated: 2026-05-17.

## Session goal

Make the codebase training-ready. Build the JEPA wrapper that composes
encoder + predictor + anti-collapse loss, build the VICReg fallback that
the auto-fallback rule needs, build the scheduled-sampling utility for
multi-step rollout, build the diagnostics module, and build a minimal
argparse training entrypoint with full W&B integration and a 200 to 500
iteration smoke run that proves the wiring works end-to-end on 3 cases.

This session does NOT answer the question “does the JEPA learn anything
useful?” That question is for Session 5’s meaningful 5k-iter smoke run
on 5 cases (per HANDOFF step 4). Session 4’s question is narrower: “can
the training script run without crashing, produce finite losses, log
everything the paper needs, and exit cleanly?”

|Module                              |Purpose                                   |
|------------------------------------|------------------------------------------|
|`src/models/jepa.py`                |JEPA wrapper composing encoder + predictor|
|`src/models/vicreg.py`              |VICReg fallback for SIGReg                |
|`src/training/scheduled_sampling.py`|V-JEPA 2-AC-faithful 2-loss recipe        |
|`src/training/diagnostics.py`       |PR, linear probe R^2, variance histogram  |
|`src/training/auto_fallback.py`     |The SIGReg -> VICReg toggle rule          |
|`src/training/train_jepa.py`        |argparse training entrypoint, W&B wired   |
|`src/utils/device.py`               |require_rtx6000(), used everywhere        |

Pass criteria: all new unit tests pass, all Session 2 (15) and Session 3
unit tests remain green, AND the smoke run command

```
python -m src.training.train_jepa \
    --partition v1 \
    --cases G+0.00_D0.00_Y+0.00 G+1.00_D0.50_Y+0.10 G-1.00_D1.00_Y-0.20 \
    --max-iters 200 \
    --seed 0
```

completes in under 10 minutes on the RTX 6000 Blackwell, produces a finite
final loss, logs all required W&B keys, and writes one checkpoint to
`outputs/checkpoints/smoke_iter200.pt`.

## Why these modules together

The JEPA wrapper, VICReg, scheduled sampling, and diagnostics are not
independently meaningful: the JEPA wrapper does not compile without the
anti-collapse loss; the auto-fallback rule cannot be tested without VICReg;
the training entrypoint cannot be tested without all of them. Bundling them
into one session reflects this coupling.

The minimal training entrypoint is included rather than deferred to Session 5
because the W&B logging contract is the artifact the paper depends on: every
run that appears in the paper must log the four required keys plus the seven
paper-grade keys (CLAUDE.md “Logging (W&B)”). Wiring this up correctly is a
deliverable in its own right, and the 200-iter smoke run is the test that
the wiring works.

Hydra configs are NOT in scope for Session 4. The training entrypoint uses
argparse with sensible defaults. Hydra lands in Session 5 when the meaningful
5k-iter smoke and the lambda bisection mechanics need its config-group
structure. Postponing Hydra keeps Session 4 bounded.

## arXiv MCP plugin

The arXiv MCP plugin is enabled in this session. If a specification below is
ambiguous or contradicts your prior reading, consult the primary source
directly rather than guessing. Recommended primary sources:

|Reference                                        |arXiv ID  |Used for                                                 |
|-------------------------------------------------|----------|---------------------------------------------------------|
|LeWM: Maes, Le Lidec, Scieur, LeCun, Balestriero |2603.19312|Loss composition, training recipe                        |
|VICReg: Bardes, Ponce, LeCun (ICLR 2022)         |2105.04906|Three-term loss with mu, lambda, nu                      |
|V-JEPA 2 / V-JEPA 2-AC: Assran et al.            |2506.09985|Scheduled sampling implementation (Section 6, appendices)|
|LeJEPA: Balestriero, LeCun                       |2511.08544|Auto-fallback diagnostic logic, participation ratio      |
|PLDM: Sobal, Jyothir, Jalagam, Carion, Cho, LeCun|2211.10831|VICReg-derived 7-term loss (used for the baseline only)  |

LeWM remains the direct architectural template; where LeWM and others disagree,
default to LeWM (carrying forward D17). If after consulting the source there is
still genuine ambiguity, record the decision and rationale as a new D-entry in
HANDOFF.md (next available is D21) before proceeding with the code.

## Files to create

```
src/models/jepa.py
src/models/vicreg.py
src/training/__init__.py
src/training/scheduled_sampling.py
src/training/diagnostics.py
src/training/auto_fallback.py
src/training/train_jepa.py
src/utils/__init__.py
src/utils/device.py
tests/test_jepa.py
tests/test_vicreg.py
tests/test_scheduled_sampling.py
tests/test_diagnostics.py
tests/test_auto_fallback.py
tests/test_device.py
tests/test_train_jepa_smoke.py    # marked slow, skipped by default
```

No changes to `src/data/`, no changes to the Session 2 primitives, no changes
to the Session 3 encoder or predictor or their unit tests. If a Session 4
change forces a Session 2 or 3 module to evolve, stop, update that module’s
unit tests first, confirm they pass, then continue.

## Locked decisions baked into Session 4

The following decisions are locked in CLAUDE.md and HANDOFF.md. Do not revisit
without explicit user approval.

1. **Loss composition** (CLAUDE.md “Locked decisions, Training”):
   `L_total = L_pred + 0.5 * L_roll + 0.1 * L_anticollapse`
   where `L_anticollapse` is SIGReg by default and switches to VICReg if the
   auto-fallback rule fires.
1. **Anti-collapse default**: SIGReg with M=256 projections, 17 knots in
   `[0.2, 4.0]`, no `*N` multiplier per D13.
1. **Auto-fallback condition** (CLAUDE.md “Risk-management”):
   `iter >= 20k AND PR(z) < 0.3 * d AND probe_R2(c | z_T) < 0.7` on a held-out
   Test B sub-batch. When the condition fires, switch anti-collapse to VICReg
   with the coefficients in D22. Log the event prominently to W&B and stdout.
   Continue training; do not restart.
1. **VICReg coefficients** (D22, see below): `mu=25, lambda=25, nu=1`. The
   Bardes ICLR 2022 defaults.
1. **Scheduled sampling recipe** (D21, see below): V-JEPA 2-AC-faithful,
   `L_pred = teacher-forced one-step MSE averaged over T-1 next-frame predictions`, `L_roll = open-loop rollout MSE averaged over H_roll steps`,
   `H_roll = 8` per CLAUDE.md. NOT Bengio probabilistic mixing.
1. **Hardware** (D19): RTX 6000 Blackwell (sm_120) only. `require_rtx6000()`
   is called at the top of every training entrypoint and any test that
   actually exercises CUDA.
1. **W&B contract** (CLAUDE.md “Logging”): every run logs four required keys
   (`preprocessing_version`, `partition_version`, `lambda_sigreg`, `seed`)
   plus seven paper-grade keys (`split_sha256`, `inventory_sha256`,
   `code_sha256` or `git_commit`, `auto_fallback_triggered`, `wandb_run_id`,
   `gpu_name`, training run group `partition_v1`). A run missing any required
   key is considered untraceable.
1. **Optimizer** (CLAUDE.md): AdamW (0.9, 0.95), weight decay 0.05, linear
   warmup 5% + cosine to 0.05 * peak LR. Encoder LR 1.5e-4, predictor LR
   5e-4. bf16 mixed precision. Gradient clip 1.0.
1. **Conditioning** (D16): predictor receives c=(G, D, Y) only, no phi_t.
   Predictor’s cond input is `(B, T, cond_dim=3)` with c broadcast across t.
   Data loader batch is `{'omega': (B, T, 1, H, W), 'c': (B, 3)}`. No `phi`
   field.

## Module 1: `src/models/jepa.py`

### Class signature

```python
class JEPA(nn.Module):
    """End-to-end Joint-Embedding Predictive Architecture wrapper.

    Composes the encoder (`src.models.encoder.HybridCNNViTEncoder`) and the
    predictor (`src.models.predictor.AutoregressivePredictor`) with an
    anti-collapse loss (SIGReg by default, VICReg after auto-fallback).
    Computes the three-term loss

        L_total = L_pred + 0.5 * L_roll + lambda * L_anticollapse

    where L_pred is the one-step teacher-forced MSE, L_roll is the open-loop
    H_roll-step rollout MSE, and L_anticollapse is applied to the encoder
    output before the predictor.

    Reference recipe:
        Maes et al., "LeWorldModel", arXiv:2603.19312, Section 3.1 and
        equation (training objective).
    """

    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        anticollapse: nn.Module,
        lambda_anticollapse: float = 0.1,
        rollout_weight: float = 0.5,
        H_roll: int = 8,
        rollout_start_strategy: str = "uniform_random",
    ) -> None:
        ...

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Forward pass on one training batch.

        Args:
            batch: dictionary with keys
                'omega': (B, T, 1, H, W) input vorticity, T == L == 32
                'c':     (B, 3)         static episode descriptor

        Returns:
            Dictionary with keys:
                'loss_total':       scalar, used for backward()
                'loss_pred':        scalar, teacher-forced one-step MSE
                'loss_roll':        scalar, H_roll-step rollout MSE
                'loss_anticollapse': scalar, SIGReg or VICReg on z
                'z':                (B, T, d) cached for diagnostics
        """
        ...

    def set_anticollapse(self, new_module: nn.Module) -> None:
        """Swap the anti-collapse module (used by the auto-fallback).
        After this call, future forward passes use `new_module`.
        """
        ...
```

### Internal structure

The wrapper holds three submodules:

```python
self.encoder = encoder
self.predictor = predictor
self.anticollapse = anticollapse  # initially a SIGReg instance
```

The forward pass:

1. **Encode**: `z = self.encoder(batch['omega'])` produces `(B, T, d)`. Both
   the teacher-forced and rollout passes use this same `z`.
1. **Teacher-forced prediction**: `z_hat_tf = self.predictor(z, batch['c'])`
   produces `(B, T, d)`. The MSE between `z_hat_tf[:, :-1, :]` and
   `z[:, 1:, :]` is `L_pred`. Note the slicing: position t’s output predicts
   position t+1’s encoder latent, so we align `z_hat_tf[:, t, :]` with
   `z[:, t+1, :]` for `t in [0, T-2]`.
1. **Rollout prediction**: pick a rollout start position `t0` per
   `rollout_start_strategy`. Slice the seed `z_init = z[:, :t0+1, :]`. Call
   `self.predictor.rollout(z_init, batch['c'], steps=H_roll)`. The result
   has shape `(B, t0+1+H_roll, d)`. The MSE between the rolled-out
   `z_full[:, t0+1:t0+1+H_roll, :]` and `z[:, t0+1:t0+1+H_roll, :]` is
   `L_roll`.
1. **Anti-collapse**: `L_anticollapse = self.anticollapse(z.flatten(0, 1))`.
   This flattens `(B, T, d)` to `(B*T, d)` and applies the regularizer to
   the full set of latents. SIGReg sees a batch of size `B * T`; VICReg
   sees the same.
1. **Total**: `L_total = L_pred + 0.5 * L_roll + lambda * L_anticollapse`.

The `rollout_start_strategy` enum values are:

- `"fixed_zero"`: always `t0 = 0`. Deterministic; biases the rollout to
  early-trajectory dynamics. Simplest. Use for unit tests.
- `"uniform_random"`: sample `t0` uniformly from `[0, T-1-H_roll]` per
  forward pass per episode. Default for training. With `T=32, H_roll=8`,
  this samples `t0` from `[0, 23]`.
- `"impact_aware"`: bias `t0` toward the impact window. Specifically, with
  probability 0.7 sample `t0` from `[24-H_roll, 24]` (so the rollout ends
  near the impact frame ~40 of the original trajectory, accounting for the
  sub-trajectory’s `impact_overlap_start_range`); with probability 0.3,
  sample uniformly. Optional, not used in Session 4 smoke.

The `rollout_start_strategy` is a constructor argument so that the unit
tests can use `"fixed_zero"` for determinism while the training script
uses `"uniform_random"`. The `"impact_aware"` mode is implemented but not
exercised in Session 4; Session 5+ can ablate it.

### Numerical notes (mandatory)

- The rollout uses the predictor’s `rollout()` method, which returns
  shape `(B, t0+1+H_roll, d)` per Session 3 spec.
- The MSE is computed in fp32 (cast inside the loss; the surrounding model
  runs bf16 under autocast). Use `F.mse_loss(reduction="mean")`.
- The anti-collapse loss receives `z.flatten(0, 1)` which is `(B*T, d)`.
  At `B=16, T=32, d=32`, the SIGReg batch is 512. This is well above the
  SIGReg recommended minimum of 256 from CLAUDE.md.
- Note: `set_anticollapse()` swaps the module reference. Make sure the
  optimizer is rebuilt or the parameter list is consulted before the
  optimizer step that follows; alternatively, ensure SIGReg and VICReg
  have NO trainable parameters (they don’t; both are pure loss functions).
- The `z` cached in the returned dict is detached BEFORE returning if
  needed by external diagnostics that should not affect autograd. But for
  the loss itself, do NOT detach: gradients flow back through the encoder
  via L_pred, L_roll, and L_anticollapse all simultaneously.

### Unit tests for the JEPA wrapper

`tests/test_jepa.py`:

```python
def test_jepa_shape_contract():
    """Input batch with omega (2, 32, 1, 192, 96) and c (2, 3) produces a
    dict with the five expected keys, all loss scalars are finite, and
    z is (2, 32, 32)."""

def test_jepa_loss_decomposition():
    """L_total == L_pred + 0.5 * L_roll + lambda * L_anticollapse within
    fp32 tolerance (1e-5)."""

def test_jepa_identity_predictor_gives_zero_pred_loss_at_init():
    """At init, AdaLN-Zero makes the predictor identity-on-residual. The
    embedded input passes through unchanged, so z_hat_tf[:, t, :] equals
    the input embedding of z[:, t, :], NOT z[:, t+1, :]. Therefore L_pred
    is NOT zero at init (this is the correct behaviour, not a bug). This
    test asserts L_pred > 0 at init to document the expected non-trivial
    starting point. Compare against random initialization to check the
    magnitude is in a reasonable range (0.01 < L_pred < 10.0 with
    torch.manual_seed(0))."""

def test_jepa_anticollapse_swap_takes_effect():
    """Build a JEPA with SIGReg. Compute one forward, record L_anticollapse.
    Call set_anticollapse(VICReg(d=32, mu=25, lambda_=25, nu=1)). Compute
    a forward on the same batch (under torch.manual_seed reset for
    deterministic z). Verify the new L_anticollapse value is different and
    that no SIGReg-specific parameters survive in the JEPA module's
    state_dict()."""

def test_jepa_gradient_flows_through_all_three_terms():
    """Disable L_roll (rollout_weight=0), check encoder gradients are
    nonzero. Disable L_pred (set wrapper variant), check encoder gradients
    nonzero via L_roll alone. Disable both (set both to 0), check encoder
    gradients still nonzero via L_anticollapse. This verifies all three
    paths back-propagate independently."""

def test_jepa_bf16_autocast_smoke():
    """Under torch.autocast(device_type='cuda', dtype=torch.bfloat16),
    one forward+backward pass on B=2, T=8 completes without dtype errors
    and produces finite gradients. Skip the test if require_rtx6000()
    raises (no Blackwell card available)."""

def test_jepa_rollout_strategies_produce_different_losses():
    """With rollout_start_strategy='fixed_zero', call forward twice with
    the same seed. L_roll is deterministic. Switch to 'uniform_random'
    with a different random seed; L_roll changes. Verify the per-batch
    L_roll computed under 'fixed_zero' matches a hand-computed reference
    on a tiny B=1, T=10, H_roll=2 batch."""
```

Use `torch.manual_seed(0)` at the top of every test. Default test batch
`B=2, T=8, H_roll=2` (not the production `T=32, H_roll=8`) to keep tests
under one second each. The `T=32, H_roll=8` configuration is exercised
only in the integration smoke test below.

## Module 2: `src/models/vicreg.py`

### Class signature

```python
class VICReg(nn.Module):
    """Variance-Invariance-Covariance Regularization (Bardes, Ponce, LeCun,
    ICLR 2022).

    Three-term loss applied to a batch of embeddings z of shape (N, d):
        L_VIC = mu * L_var(z) + lambda_ * L_inv(z) + nu * L_cov(z)

    where
        L_var(z) = mean(hinge(gamma - std(z, dim=0)))     # variance hinge
        L_inv(z) = mean( ||z_a - z_b||^2 )                # not used here
        L_cov(z) = (1 / d) * sum_{i != j} (Cov(z)[i, j])^2  # off-diag Frob.

    For our use case there is no second view (no z_b), so the invariance term
    L_inv is dropped. The loss reduces to mu * L_var + nu * L_cov.

    Reference:
        Bardes, Ponce, LeCun, "VICReg: Variance-Invariance-Covariance
        Regularization for Self-Supervised Learning",
        arXiv:2105.04906 (ICLR 2022), Section 3.
    """

    def __init__(
        self,
        d: int,
        mu: float = 25.0,
        lambda_: float = 25.0,   # not used when there is no second view
        nu: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
    ) -> None:
        ...

    def forward(self, z: Tensor) -> Tensor:
        """z: (N, d) -> scalar loss.

        L_var(z) = mean over d of hinge(gamma - sqrt(var(z, dim=0) + eps))
        L_cov(z) = (1 / d) * Frobenius_off_diag( (1 / (N - 1)) * (z - mean).T @ (z - mean) )^2

        Total: mu * L_var + nu * L_cov.
        """
        ...
```

### Algorithm

The standard VICReg three-term loss, with the invariance term dropped because
JEPA without a second view has no `z_a, z_b` pair to enforce invariance over.
The variance and covariance terms suffice as an anti-collapse signal.

The `lambda_` argument is kept in the constructor so that future ablations
that introduce a second view (e.g., the symmetry-augmentation pair) can re-
enable the invariance term without changing the public API.

The hinge is `F.relu(gamma - sqrt(var(z) + eps))`. The `eps` prevents
infinite gradients when variance approaches zero.

### Unit tests for VICReg

`tests/test_vicreg.py`:

```python
def test_vicreg_low_on_isotropic_unit_variance_gaussian():
    """N(0, I_32) with N=1024 gives total VICReg below 1.0. Both var hinge
    and cov term should be small (variance ~1 satisfies the hinge;
    independent dims give near-zero covariance)."""

def test_vicreg_high_on_collapsed_z():
    """All-zero z gives variance term mu * gamma^2 = 25.0. Covariance term
    is zero (no off-diagonal correlation in a constant matrix). Total
    should be in [25, 26]. This is the "complete collapse" case."""

def test_vicreg_high_on_low_rank_z():
    """z of shape (1024, 32) where only the first 4 dimensions are
    non-trivial (the rest are exactly zero). The variance hinge fires on
    28 of 32 dimensions for total mu * 28 / 32 * gamma^2 ~ 21.9. The
    covariance term is exactly zero on the zero dimensions. This is the
    "dimensional collapse to a 4-dim subspace" case; VICReg should rate it
    much higher than the unit-variance Gaussian."""

def test_vicreg_high_on_correlated_z():
    """z = [u, u, u, ..., u] where u is a single random vector of shape
    (1024,) replicated across 32 dimensions. Variance is fine, but covariance
    is maximum: every off-diagonal entry is the on-diagonal entry. Total
    should be dominated by the covariance term."""

def test_vicreg_gradient_flows():
    """Backward pass produces non-zero gradients on z. Deterministic via
    torch.manual_seed(0)."""

def test_vicreg_dtype_promotion():
    """Input in bf16 under autocast produces fp32 output and the backward
    pass runs without overflow. Gradient on the bf16 input is bf16. (Same
    convention as SIGReg.)"""
```

## Module 3: `src/training/scheduled_sampling.py`

### Function signatures

```python
def teacher_forced_prediction_loss(
    z_target: Tensor,
    z_hat: Tensor,
) -> Tensor:
    """One-step teacher-forced MSE.

    Args:
        z_target: (B, T, d) ground-truth encoder latents.
        z_hat:    (B, T, d) predictor output, where z_hat[:, t, :] is the
                  prediction of position t+1 from positions [0, t].

    Returns:
        scalar = mean over (B, t in [0, T-2], d) of (z_hat[:, t, :] -
                  z_target[:, t+1, :])^2.
    """
    ...

def open_loop_rollout_loss(
    predictor: nn.Module,
    z_target: Tensor,
    cond: Tensor,
    start_t: int | Tensor,
    horizon: int,
) -> Tensor:
    """Open-loop rollout MSE over `horizon` steps starting from `start_t`.

    Args:
        predictor: the AutoregressivePredictor instance.
        z_target: (B, T, d) ground-truth encoder latents.
        cond:     (B, cond_dim) static episode descriptor.
        start_t:  int or (B,) tensor of starting indices per batch element.
        horizon:  number of steps to roll out (H_roll, e.g. 8).

    Returns:
        scalar = mean MSE between rolled-out predictions
                  z_rollout[:, start_t+1:start_t+1+horizon, :] and
                  z_target [:, start_t+1:start_t+1+horizon, :].
    """
    ...
```

The two losses are kept as free functions, not methods on the JEPA wrapper,
so they can be unit-tested in isolation against a stub predictor.

### V-JEPA 2-AC-faithful design decision (D21)

V-JEPA 2-AC (Assran et al., arXiv:2506.09985) sums a teacher-forced one-step
loss over 15 positions with a two-step rollout loss with fixed coefficients.
Our design transposes that recipe to our setting:

- Teacher-forced loss is one-step prediction over T-1 = 31 positions
  (we have full sub-trajectory access; V-JEPA 2-AC uses 15 positions because
  their architecture exposes 16 frame slots at a time).
- Rollout loss is H_roll = 8 step rollout from one random start position
  per episode per batch (we have a longer sub-trajectory; H_roll = 2 is too
  short for impact dynamics that last 5 to 20 t/c).
- Coefficients are fixed: L_total includes 1.0 * L_pred + 0.5 * L_roll.

This is the V-JEPA 2-AC recipe in spirit (two-loss sum, fixed coefficients,
no Bengio probabilistic teacher-student mixing), parameterized to our
sub-trajectory length and impact-dynamics time scale.

### Unit tests for scheduled sampling

`tests/test_scheduled_sampling.py`:

```python
def test_teacher_forced_loss_shape_check():
    """Inputs (2, 8, 32), output is a scalar."""

def test_teacher_forced_loss_zero_on_perfect_prediction():
    """If z_hat[:, t, :] = z_target[:, t+1, :] for all valid t (cheat),
    the loss is exactly zero."""

def test_teacher_forced_loss_value_on_known_input():
    """Hand-construct z_target and z_hat with known offsets, compute the
    expected MSE by hand, assert allclose."""

def test_rollout_loss_shape_check():
    """Inputs (2, 8, 32), start_t=2, horizon=3. Output is a scalar."""

def test_rollout_loss_uses_predictions_not_ground_truth():
    """Build a stub predictor that always outputs a known constant.
    Verify the rollout MSE depends on the predictor's output sequence,
    NOT on the ground-truth z_target except at the seed."""

def test_rollout_with_perfect_predictor_gives_zero_loss():
    """Build a stub predictor that always outputs the next ground-truth
    frame (cheating). Rollout MSE is zero."""
```

## Module 4: `src/training/diagnostics.py`

### Function signatures

```python
def participation_ratio(z_batch: Tensor) -> float:
    """Compute PR = (sum_i s_i)^2 / sum_i s_i^2 where s_i are the
    singular values of z_batch.

    Args:
        z_batch: (N, d). Typically N = B * T from a held-out batch.

    Returns:
        PR as a Python float in (0, d]. PR = d means perfect isotropy
        (all singular values equal). PR = 1 means complete collapse
        (one singular value dominates).
    """
    ...

def linear_probe_r2(
    z: Tensor,
    c: Tensor,
    fit_indices: Tensor,
    eval_indices: Tensor,
) -> dict[str, float]:
    """Fit a linear regression z -> c on fit_indices, evaluate on
    eval_indices, return R^2 per output dimension and overall.

    Args:
        z: (N, d). Latents.
        c: (N, c_dim). Targets, e.g. (G, D, Y).
        fit_indices, eval_indices: 1D long tensors disjoint, summing to
            a subset of [0, N).

    Returns:
        dict with keys 'r2_overall' (float), 'r2_G', 'r2_D', 'r2_Y'
        (floats). R^2 can be negative (regression is worse than the mean).
    """
    ...

def per_dim_variance_histogram(
    z_batch: Tensor,
    n_bins: int = 20,
) -> tuple[Tensor, Tensor]:
    """Compute a histogram of the per-dimension variances of z_batch.

    Args:
        z_batch: (N, d).
        n_bins: histogram bin count.

    Returns:
        (counts, bin_edges). counts is (n_bins,), bin_edges is (n_bins+1,).
        Useful for visualizing dimensional collapse.
    """
    ...
```

The diagnostics are pure functions; the training loop calls them every
1000 iterations on a held-out batch and logs the results to W&B.

### Unit tests for diagnostics

`tests/test_diagnostics.py`:

```python
def test_pr_isotropic_gaussian_is_close_to_d():
    """N(0, I_32) with N=8192 gives PR > 0.85 * 32 = 27.2 (Marcenko-Pastur
    correction notwithstanding, isotropic samples have PR close to d for
    large N). Use torch.manual_seed(0)."""

def test_pr_complete_collapse_is_one():
    """z = ones(8192, 32) (rank 1 matrix) gives PR == 1.0 within
    numerical tolerance."""

def test_pr_partial_collapse_low():
    """z constructed so the top 4 singular values dominate (rank-4
    structure plus tiny noise) gives PR in [4, 5]. This is the
    "intrinsic dim ~4" case."""

def test_linear_probe_perfect_on_linear_data():
    """Synthesize z = A @ c + epsilon for known A and small epsilon.
    Linear probe should give R^2 > 0.95 on the held-out indices."""

def test_linear_probe_zero_on_independent_data():
    """z independent of c (both drawn from independent Gaussians).
    R^2 should be near zero (slightly negative on small samples is fine;
    assert -0.5 < r2_overall < 0.1)."""

def test_variance_histogram_collapsed_z_concentrates_at_zero():
    """z constructed with 28 collapsed dimensions and 4 active ones. The
    histogram should have most counts in the zero bin."""
```

## Module 5: `src/training/auto_fallback.py`

### Class signature

```python
class AutoFallbackController:
    """State machine that decides when to switch SIGReg -> VICReg.

    Encapsulates the rule
        if iter >= threshold_iter AND PR < pr_threshold * d AND probe_R2 < r2_threshold:
            fire fallback
    so that the rule can be unit-tested in isolation from the training loop.

    Reference: CLAUDE.md "Risk-management" section.
    """

    def __init__(
        self,
        d: int,
        threshold_iter: int = 20_000,
        pr_threshold: float = 0.3,
        r2_threshold: float = 0.7,
    ) -> None:
        self.fired = False
        ...

    def step(self, iteration: int, pr: float, probe_r2: float) -> bool:
        """Check whether the fallback should fire NOW.

        Args:
            iteration: current training iteration (0-indexed).
            pr: latest participation ratio computed on the held-out batch.
            probe_r2: latest linear-probe R^2 for c, computed on Test B.

        Returns:
            True if the controller is firing the fallback this step.
            (Idempotent: once fired, returns False on subsequent calls.)
        """
        ...
```

The training loop holds one `AutoFallbackController` instance. Each time
diagnostics are computed (every 1000 iterations), the loop calls
`controller.step(iter, pr, probe_r2)`. If the return is True, the loop
calls `jepa.set_anticollapse(VICReg(...))`, logs the event to W&B and
stdout, and continues training without restarting.

### Unit tests for the auto-fallback

`tests/test_auto_fallback.py`:

```python
def test_fallback_does_not_fire_before_threshold():
    """At iter < 20_000, even with PR=0 and R^2=0, fallback does not fire."""

def test_fallback_does_not_fire_when_pr_healthy():
    """At iter >= 20_000, with PR=0.5*d (healthy) and R^2=0 (low),
    fallback does not fire because the PR condition fails."""

def test_fallback_does_not_fire_when_r2_healthy():
    """At iter >= 20_000, with PR=0.1*d (low) and R^2=0.9 (healthy),
    fallback does not fire because the R^2 condition fails."""

def test_fallback_fires_on_both_conditions_met():
    """At iter >= 20_000, with PR=0.1*d AND R^2=0.3, the controller fires
    once. step() returns True on this call."""

def test_fallback_is_idempotent_once_fired():
    """After step() returns True once, subsequent calls return False even
    if the conditions remain true. The controller's `fired` attribute is
    True after the first firing."""

def test_fallback_threshold_at_exactly_20k():
    """At iter == 20_000 (boundary), with both conditions met, fallback
    fires. At iter == 19_999, it does not."""
```

## Module 6: `src/utils/device.py`

### Function signature

```python
def require_rtx6000() -> torch.device:
    """Find the first RTX 6000 Blackwell GPU and return its device.

    Walks torch.cuda.device_count(), picks the first device whose
    torch.cuda.get_device_name(i) contains both 'RTX' and '6000'. Runs
    a tiny probe kernel (`torch.zeros(4, device=d) + 1`) to confirm the
    installed PyTorch wheel actually ships kernels for sm_120 (Blackwell).

    Returns:
        torch.device('cuda:<idx>') where idx is the matching device.

    Raises:
        NoRTX6000Error: if no matching GPU is found OR if the probe kernel
            fails. The error message lists all visible GPUs and includes
            the suggested reinstall command if it appears to be a wheel
            mismatch (sm_120 not in supported list).
    """
    ...

class NoRTX6000Error(RuntimeError):
    """Raised when require_rtx6000() cannot find a usable Blackwell GPU."""
```

### Unit tests for the device helper

`tests/test_device.py`:

```python
def test_require_rtx6000_on_workstation():
    """Skip if CUDA not available. Otherwise call require_rtx6000() and
    assert the returned device's name contains 'RTX' and '6000'. This is a
    CUDA-only test; it does not run in CI."""

def test_no_rtx6000_error_message_lists_visible_gpus():
    """Mock torch.cuda.get_device_name to return ['L40S', 'L40S']. Call
    require_rtx6000(), catch NoRTX6000Error, assert the message contains
    'L40S' so the user can see what torch did see."""
```

## Module 7: `src/training/train_jepa.py`

### Entrypoint signature

```python
def main() -> None:
    """argparse-based training entrypoint for the JEPA.

    Required CLI arguments:
        --partition: partition version (default 'v1')
        --cases: case_ids to include, space-separated. Smoke runs use 2-3.
        --max-iters: int, total training iterations (default 80_000 for
            full runs; 200 for smoke).
        --seed: int

    Optional CLI arguments (Hydra lands in Session 5; for now these are
    plain CLI flags):
        --d: latent dim (default 32)
        --B: batch size (default 16)
        --T: sub-trajectory length (default 32)
        --H-roll: rollout horizon (default 8)
        --lambda-sigreg: anti-collapse weight (default 0.1)
        --lr-encoder: (default 1.5e-4)
        --lr-predictor: (default 5e-4)
        --weight-decay: (default 0.05)
        --warmup-frac: (default 0.05)
        --output-dir: (default 'outputs/runs/smoke')
        --log-every: int, W&B log frequency in iters (default 100)
        --diagnostic-every: int (default 1000)
        --checkpoint-every: int (default 10000; smoke run sets to max-iters)
        --wandb-mode: 'online' | 'offline' | 'disabled' (default 'online'
            for full runs; smoke uses 'offline' to avoid polluting the
            project space).
    """
    ...
```

### Required behaviour

1. **Hardware check**: first thing inside `main()`, call
   `require_rtx6000()` and assign the returned device. Move all subsequent
   tensors to this device. Do NOT call `torch.cuda.current_device()` or
   hardcode ‘cuda:0’.
1. **Seed**: set every random source: `torch.manual_seed(seed)`,
   `np.random.seed(seed)`, `random.seed(seed)`,
   `torch.cuda.manual_seed_all(seed)`. Log the seed to W&B.
1. **Data**: instantiate `EpisodeDataset` with the partition manifest and
   the case subset. DataLoader with `batch_size=B`, `shuffle=True`,
   `num_workers=4`, `pin_memory=True`.
1. **Model**: encoder + predictor + SIGReg (default), composed via
   `JEPA(...)`. Move to device. Wrap in `torch.compile()` if available
   (Session 4 can skip this for simplicity; Session 5 enables it).
1. **Optimizer**: two parameter groups, `lr=1.5e-4` for encoder and
   `lr=5e-4` for predictor, `weight_decay=0.05`, `betas=(0.9, 0.95)`.
   Use `AdamW`. Note: anti-collapse modules have no parameters so they
   are not in any parameter group.
1. **LR scheduler**: linear warmup over `warmup_frac * max_iters` then
   cosine decay to `0.05 * peak_lr` over the remaining iterations. A
   `torch.optim.lr_scheduler.SequentialLR` of LinearLR + CosineAnnealingLR.
1. **AMP**: `torch.amp.GradScaler` for bf16. Wrap forward in
   `torch.autocast(device_type='cuda', dtype=torch.bfloat16)`. Gradient
   clip at 1.0. Standard AMP pattern.
1. **Auto-fallback controller**: instantiate at init. Call its `step()`
   method every `diagnostic_every` iterations.
1. **W&B init**: as the first action after device and seed setup, call
   
   ```python
   wandb.init(
       project=os.environ['WANDB_PROJECT'],   # 'vortex-jepa'
       group=f'partition_{partition}',
       tags=['hybrid_cnn_vit', 'sigreg'],     # ablations tag differently
       mode=wandb_mode,
       config={
           # Required
           'preprocessing_version': cfg.preprocessing_version,
           'partition_version': partition,
           'lambda_sigreg': lambda_sigreg,
           'seed': seed,
           # Paper-grade
           'split_sha256': read_split_manifest_sha256(),
           'inventory_sha256': read_inventory_sha256(),
           'code_sha256': read_git_commit_hash(),
           'auto_fallback_triggered': False,  # updated by controller
           'gpu_name': torch.cuda.get_device_name(device.index),
       },
   )
   wandb.run.summary['wandb_run_id'] = wandb.run.id
   ```
   
   Assert that `gpu_name` contains both `RTX` and `6000`; if not, fail
   immediately with a clear message (defence in depth alongside
   `require_rtx6000()`).
1. **Training loop**: for `iter in range(max_iters)`:
- Get a batch from the dataloader (infinite stream; loop with
  `itertools.cycle()` if needed).
- `loss_dict = jepa(batch)`. Backward `loss_dict['loss_total']`.
  Gradient clip. Optimizer step. Scheduler step.
- Every `log_every` iters: log all loss components and the current
  LR to W&B.
- Every `diagnostic_every` iters: compute diagnostics on a held-out
  Test B sub-batch. Call `controller.step(...)`. If it returns True,
  swap to VICReg and log the event.
- Every `checkpoint_every` iters: save `jepa.state_dict()` and the
  optimizer state to `output_dir/checkpoint_iter{iter}.pt`.
1. **Exit**: clean shutdown. Final W&B summary update with the final
   loss values and `auto_fallback_triggered`.

### Pass criteria for the smoke run

The smoke run command in the “Session goal” section above (`--max-iters 200`,
three cases) must:

- Complete in under 10 minutes on the RTX 6000 Blackwell.
- Exit with code 0.
- Produce a final `loss_total` that is finite (not NaN, not Inf).
- Log all four required W&B keys plus all seven paper-grade keys.
- Write one checkpoint at iter 200.
- Compute diagnostics at least once (200 / 1000 ~ 0 by default; for the
  smoke run set `--diagnostic-every 100` so PR and probe R^2 are computed
  twice during the run).
- The auto-fallback controller is instantiated and stepped twice; both
  times it returns False because iter < 20_000. This is the negative case
  for the controller, which is fine: the smoke run is not testing whether
  fallback works in production, only that it is wired in and does not
  raise.

### Unit test for the training entrypoint

`tests/test_train_jepa_smoke.py` (marked slow, skipped by default):

```python
@pytest.mark.slow
def test_train_jepa_smoke_runs_to_completion(tmp_path):
    """Programmatically invoke main() with smoke-run args and assert it
    exits cleanly. Use --wandb-mode offline so no network is involved.
    Use a tiny 1-case subset to keep the test under 2 minutes on the
    RTX 6000. Assert the output dir contains a final checkpoint.
    This test is opt-in via `pytest -m slow`; CI does not run it."""
```

The smoke test is genuine integration: it instantiates the full model,
runs the full data loader, calls the full optimizer, hits the full
diagnostics path, and writes a checkpoint. If anything in the wiring is
broken, this test catches it. It is the most important new test of the
session, even though it is slow.

## Coding conventions (recap from CLAUDE.md and prior sessions)

- Python 3.10+, PyTorch 2.x.
- One module per file. No catch-all `utils.py` (the new `src/utils/device.py`
  is the device helper specifically, not a junk drawer).
- Type hints everywhere. Google-style docstrings on every public class and
  function. Cite the paper and arXiv ID in each module’s top-level docstring.
- Imports: `torch`, `torch.nn`, `torch.nn.functional`, `math`, Python stdlib,
  the Session 2 primitives, the Session 3 encoder/predictor, and (for the
  training entrypoint only) `wandb`, `numpy`, `argparse`, `os`, `pathlib`,
  `random`. No new third-party dependencies beyond what is already in
  `requirements.txt`.
- `ruff check src/ tests/` and `black --check --line-length 100 src/ tests/`
  must pass before commit. `mypy --strict src/models src/training` should
  pass; if it cannot pass on the training entrypoint (Hydra-style configs
  often trip mypy), narrow the `mypy --strict` invocation to
  `src/models src/training/diagnostics.py src/training/scheduled_sampling.py src/training/auto_fallback.py` and exclude `train_jepa.py` until
  Hydra-typed configs land in Session 5.
- `torch.manual_seed(0)` at the top of each unit test. Default test batch
  B=2, sub-trajectory T=8 (not the production T=32, to keep tests under
  one second each). The slow smoke test is the exception.

## Out of scope for Session 4

- **Hydra configs** (`configs/encoder/`, `configs/predictor/`, `configs/loss/`,
  `configs/data/`, `configs/sweep/`). Session 5. Session 4 uses argparse.
- **The meaningful 5k-iter smoke run on 5 cases** per HANDOFF step 4.
  Session 5.
- **Lambda bisection** mechanics (`scripts/sweep_lambda.py`). Session 6.
- **Full 80k training run**. Session 7.
- **Visualization decoder** (trained separately on a frozen encoder).
  Separate session after the first reportable JEPA checkpoint exists.
- **Baselines** (POD, Fukami AE, Solera-Rico beta-VAE, PLDM). Parallel work,
  not blocking the main path.
- **The impact-aware rollout start strategy** (`"impact_aware"` in the JEPA
  wrapper). Implemented but not exercised. Session 5 can ablate it.
- **`torch.compile()`** on the JEPA wrapper. Session 5 enables it once the
  smoke baseline is established.
- **The 4-stage shallow-stem encoder ablation** (72 tokens). Deferred per
  D3 alternative; not part of Session 4.

## Expected duration

Four to six hours, TDD-style. The JEPA wrapper and the training entrypoint
are the bulk of the time. VICReg and scheduled sampling are smaller. The
diagnostics and the auto-fallback controller are the smallest.

Suggested order:

1. `src/utils/device.py` + `tests/test_device.py`. Smallest, unblocks the
   training entrypoint.
1. `src/models/vicreg.py` + `tests/test_vicreg.py`. Unblocks the auto-
   fallback controller test.
1. `src/training/diagnostics.py` + `tests/test_diagnostics.py`.
1. `src/training/auto_fallback.py` + `tests/test_auto_fallback.py`.
1. `src/training/scheduled_sampling.py` + `tests/test_scheduled_sampling.py`.
1. `src/models/jepa.py` + `tests/test_jepa.py`. Pulls everything together.
1. `src/training/train_jepa.py` + the slow integration test
   `tests/test_train_jepa_smoke.py`. Finally, run the 200-iter smoke on the
   3-case subset and confirm it lands.

## If something is unclear

The arXiv MCP plugin is enabled. Recommended consultation order if you have
doubts:

1. **VICReg three terms and coefficient defaults**: arXiv:2105.04906
   (Bardes, Ponce, LeCun, ICLR 2022), Section 3. The three coefficients
   `mu, lambda, nu` and the variance hinge `gamma` are defined there.
   Their ICLR 2022 defaults are `mu=25, lambda=25, nu=1, gamma=1`. We drop
   the invariance term because we have no second view (no `z_a, z_b` pair),
   keeping `mu` and `nu` only. Recorded as D22.
1. **V-JEPA 2-AC training recipe**: arXiv:2506.09985 (Assran et al.,
   2025), Section 6 and appendices. The actual recipe is teacher-forcing
   over T=15 positions PLUS two-step rollout, summed with fixed
   coefficients. NOT Bengio probabilistic teacher-student mixing. Our
   choice (D21) transposes this to our setting (full sub-trajectory
   teacher forcing + H_roll=8 rollout).
1. **LeWM loss composition**: arXiv:2603.19312 (Maes et al., 2026),
   Section 3.1 (training objective). The two-term `L = L_pred + lambda * L_sigreg`. We extend with the rollout term per V-JEPA 2-AC.
1. **Auto-fallback diagnostic logic**: there is no direct reference for
   “switch SIGReg to VICReg at iter 20k if PR and probe R^2 are both
   low”. This is our own pre-registered methodology (CLAUDE.md
   “Risk-management”) and is part of the paper’s methodological
   contribution. Do not deviate from the locked thresholds without
   recording a new D-entry.
1. **VICReg without second view**: VICReg’s invariance term requires two
   augmented views of the same sample. JEPA without contrastive pairs has
   only one view. The standard solution (Bardes et al., section 5.4
   “Without invariance”) is to drop the invariance term and rely on
   variance + covariance. The H-JEPA reference (Wiggins, 2026,
   `github.com/jonwiggins/H-JEPA`) and PLDM (Sobal et al., 2022) both
   confirm this is the standard practice for JEPA-style models.

If after consulting the source there is genuine ambiguity, record the
decision and rationale as D21, D22, D23, … in HANDOFF.md before
proceeding with the code.

## After Session 4 lands

Carlos triggers Session 5 with one message. Session 5 introduces:

- Hydra configs for the full hyperparameter surface.
- The meaningful 5k-iter smoke run on 5 cases (HANDOFF step 4 pass
  criteria: SIGReg loss < 5.0, PR > 0.5 * d, probe R^2 for c > 0.5).
- Wandb sweep configuration for the lambda bisection (Session 6).
- `torch.compile()` enabled on the JEPA wrapper.

Session 2, 3, and 4 unit tests must remain green throughout Session 5.

## Decisions to record in HANDOFF.md (new)

### D21: Scheduled sampling is V-JEPA 2-AC-faithful with H_roll=8

Use a two-loss sum with fixed coefficients (`L_total = L_pred + 0.5 * L_roll + lambda * L_sigreg`), NOT Bengio-style probabilistic teacher-student
mixing. The rollout horizon `H_roll = 8` (CLAUDE.md “Locked decisions,
Training”) rather than V-JEPA 2-AC’s H_roll = 2, because vortex impact
dynamics last 5 to 20 t/c which is too long for a 2-step rollout to
capture. Recorded in `SESSION4_JEPA_WRAPPER_AND_TRAINING_SCAFFOLD.md`
Section “V-JEPA 2-AC-faithful design decision”.

Alternative considered: Bengio scheduled sampling with `p_tf` annealed from
1.0 to 0.5 over 30 percent of training (the architectural spec proposal).
Rejected because (a) it adds a hyperparameter axis the V-JEPA 2-AC recipe
does not need, (b) it has no published precedent for JEPA-style models,
(c) the two-loss sum is simpler to ablate against (just turn off
`rollout_weight`).

### D22: VICReg coefficients are mu=25, lambda=25, nu=1, gamma=1 with the invariance term dropped

CLAUDE.md “Risk-management” specifies VICReg with `mu = 25.0, nu = 1.0` but
omits `lambda`. The full Bardes ICLR 2022 default is `mu=25, lambda=25, nu=1, gamma=1`. The invariance term parameterized by `lambda` requires two
views of the same sample, which JEPA-without-augmentation-pairs does not
have. Per the H-JEPA and PLDM precedents, drop the invariance term and
keep `mu * L_var + nu * L_cov` only. `gamma = 1` is the variance hinge
target.

Effect: `src/models/vicreg.py` constructor takes all four arguments
(`mu, lambda_, nu, gamma`) for forward-compatibility with future
ablations that introduce a second view, but the default forward pass
ignores `lambda_` and uses only `mu * L_var + nu * L_cov`. A unit test
asserts that `lambda_ != 0` does not change the loss output when no
second view is provided.

## Decision references (existing)

- D5 (HANDOFF): SIGReg with auto-fallback to VICReg.
- D6 (HANDOFF): encoder is unconditional, c enters only the predictor.
- D13 (HANDOFF): SIGReg follows LeWM appendix A, no `*N` multiplier.
- D14, D15, D20 (HANDOFF): partition v1 now has 37 train cases / 126
  train encounters / 52 Test A / 28 Test B / 24 Test C / 47 total cases
  / 230 total encounters.
- D16 (HANDOFF): predictor cond_dim = 3 (no phi_t).
- D17 (HANDOFF): BatchNorm at the encoder projection per LeWM, LayerNorm
  is the first diagnostic if H4 (partial SIGReg collapse) bites at
  iter 20k.
- D19 (HANDOFF): RTX 6000 Blackwell only, `require_rtx6000()` is the
  canonical accessor.
- D21 (this session): V-JEPA 2-AC-faithful scheduled sampling, H_roll = 8.
- D22 (this session): VICReg coefficients mu=25, lambda=25, nu=1, gamma=1,
  invariance term dropped.
