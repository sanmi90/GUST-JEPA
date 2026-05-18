# SESSION5_PLDM_BASELINE.md

Conditional session plan for the vortex-jepa project. Triggered if and only if
Session 5 lands TRIVIAL (per the Session 5 decision tree). If Session 5 lands
HEALTHY, PARTIAL, WEAK, or DEAD, this session does not execute and PLDM
remains parallel work alongside the other three baselines.

Last updated: 2026-05-17.

## Trigger condition

Read Session 5’s final analysis notebook output before starting. The required
condition is:

```
Session 5 outcome: TRIVIAL
```

If the outcome is anything else, STOP and report back. PLDM is conditional
priority per D29; outside the TRIVIAL outcome there is no reason to bring it
forward of the other baselines.

## Session goal

Train the PLDM baseline (Sobal et al., arXiv:2502.14819, 2025) on the
identical 5-case subset that Session 5 ran, with the same encoder and
predictor architecture, the same seed, the same diagnostic cadence, the same
W&B contract. Compare PLDM’s PR(z) and probe R^2 on Test B against Session 5
Run A’s results. The single methodological question the session answers:

**Does the LeWM Two-Room precedent (PLDM outperforms LeWM at low intrinsic
dimensionality, arXiv:2603.19312 Section 5) replicate on our physics data?**

There are two informative answers and one uninformative one:

|PLDM at iter 5000                       |Methodological reading                                                                                                                                                       |Next session                                                                                                                |
|----------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
|PR > 16 and probe_R^2 in (0.5, 0.7)     |Yes, regime-dependent contrast confirmed. PLDM wins on low-intrinsic-dim physics. Paper’s contribution claim 3 sharpens to the regime-dependent SIGReg-PR diagnostic.        |Session 6 with PLDM as the primary trained model; SIGReg becomes the comparison baseline. The full ablation suite continues.|
|PR <= 16 (any probe_R^2)                |No, failure is not regulariser-specific. Both SIGReg and PLDM collapse on 5 cases. Failure is data-scale-bound.                                                              |Session 5.5 (expand to 10-12 cases) on both SIGReg and PLDM.                                                                |
|PR > 16 and probe_R^2 outside (0.5, 0.7)|Ambiguous. PLDM is anti-collapsed but does not encode c well, or encodes it pathologically well (collapse-to-c). Diagnose with the same combinatorial 2x2 table as Session 5.|Depends on which quadrant; see Session 5’s decision tree.                                                                   |

The session is methodological. The deliverable is the answer to the question
above plus a HANDOFF D-entry recording it.

## Why this is harder than it sounds

Three points worth surfacing before writing code:

**PLDM’s loss is genuinely seven terms, not one.** LeWM’s HuggingFace summary
explicitly states “PLDM uses 7 terms” with “six loss hyperparameters”
(arXiv:2603.19312). The seventh term has weight 1 by convention; the other
six are tunable. Implementing all seven faithfully is substantive code and
roughly 8 to 12 unit tests, comparable to Session 4’s `vicreg.py` plus
`scheduled_sampling.py` work. This is not a “swap one anti-collapse module”
change; it is a parallel training pipeline.

**The PLDM paper does not target physics data.** arXiv:2502.14819 trains on
navigation tasks (offline reward-free trajectories in grid worlds). The loss
formulation should transfer, but two adaptations are mandatory:

- Term 7 (inverse-dynamics MSE: predict action `a_t` from `z_t, z_{t+1}`).
  Our setup has no per-step action; we adapt by predicting the static
  descriptor `c = (G, D, Y)` from `(z_t, z_{t+1})` instead. This is the
  adaptation already locked in D8.
- PLDM was trained on a CNN encoder, not a hybrid CNN+ViT. We hold the
  encoder fixed (use ours, identical to Session 5) so the comparison
  isolates the LOSS as the only difference. This is the standard ablation
  framing: same architecture, different objective.

**The 7-term loss has a hidden gotcha: variance/covariance on temporal
differences.** Terms 5 and 6 are VICReg variance and covariance applied to
the temporal difference signal `dz_t = z_{t+1} - z_t`, not to `z_t` itself.
This matters because a perfectly anti-collapsed z (terms 2 and 3 satisfied)
can still have a degenerate dz (terms 5 and 6 high). Our encoder, paired
with the LeWM-style architecture, is not obviously designed to satisfy both
simultaneously. The unit tests for `pldm.py` should specifically construct
“good z, bad dz” and “good dz, bad z” synthetic inputs to verify the loss
discriminates them.

## arXiv references (verified, explicit)

The arXiv MCP plugin is enabled in this session. The references below are
the verified primary sources. Use them rather than guessing.

|Reference                                                                                                                                                              |arXiv ID      |Use for                                                                                                                                                                                                                                                                                                                                               |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|**PLDM** (PRIMARY): Sobal, Zhang, Cho, Balestriero, Rudner, LeCun, “Learning from Reward-Free Offline Data: A Case for Planning with Latent Dynamics Models” (Feb 2025)|**2502.14819**|The loss formulation. Verified from the paper main body: Section 3.3 Equation 3.3 gives `L_sim = sum_{t=0..H} (1/N) * sum_b                                                                                                                                                                                                                           |
|PLDM precursor: Sobal, S V, Jalagam, Carion, Cho, LeCun, “Joint Embedding Predictive Architectures Focus on Slow Features” (NeurIPS 2022 SSL workshop, 4 pages)        |**2211.10831**|Theoretical analysis of WHY JEPA without anti-collapse focuses on slow features and fails on fixed-noise distractors. Useful for the paper’s introduction discussion; the 2022 paper is NOT the source of the PLDM name or the multi-term loss formulation. Note: HANDOFF D8 originally cited 2211.10831 as “PLDM”; this was incorrect, see D32 below.|
|VICReg: Bardes, Ponce, LeCun (ICLR 2022)                                                                                                                               |**2105.04906**|Variance hinge `max(0, gamma - sqrt(var(z) + eps))` and covariance Frobenius off-diagonal definitions. Equation 1 of the paper gives the canonical form. Section 5.4 covers the “no second view” case.                                                                                                                                                |
|LeWM Section 5: Maes et al. (March 2026)                                                                                                                               |**2603.19312**|The Two-Room comparison (“PLDM and DINO-WM outperform LeWM”), Figure 6 (planning performance), and the explanation grounded in low intrinsic dimensionality. This is the published precedent our session replicates.                                                                                                                                  |
|LeJEPA: Balestriero, LeCun (Nov 2025)                                                                                                                                  |**2511.08544**|The SIGReg theory, for comparison framing. Cite only when discussing why SIGReg’s Gaussian prior fails on low-dim data. Sections 4 and 6.                                                                                                                                                                                                             |

If after consulting arXiv:2502.14819 the exact loss formulation differs from
the seven-term decomposition in D8 (HANDOFF), the PAPER is authoritative.
Update `pldm.py` to match the paper, then record the discrepancy and the
resolution as a new HANDOFF D-entry. D8 was Carlos’s reading at project
bootstrap and was not re-verified against the paper at that time.

## Files to create

```
src/baselines/__init__.py
src/baselines/pldm.py                # 7-term loss module
src/models/pldm_wrapper.py           # composes encoder + predictor + PLDMLoss
src/training/train_baseline.py       # argparse entrypoint with --baseline pldm
tests/test_pldm_loss.py              # 8 to 12 unit tests for the 7 terms
tests/test_pldm_wrapper.py           # 4 to 6 integration tests
configs/cases/smoke_5cases.yaml      # already exists from Session 5
SESSION5_PLDM_BASELINE.md            # this file
SESSION_REPORT_5PLDM_*.md            # written at session close
```

No changes to:

- The encoder, predictor, AdaLN, RoPE, SIGReg, VICReg, JEPA modules.
- The data loader.
- The Session 5 entrypoint `train_jepa.py`. PLDM gets its own entrypoint
  (`train_baseline.py`) to keep the JEPA training path stable.
- The Session 5 analysis notebook is EXTENDED but not modified destructively;
  the PLDM analysis is added as a new section, not as in-place edits to the
  Session 5 sections.

## Module 1: `src/baselines/pldm.py`

### Class signature

```python
class PLDMLoss(nn.Module):
    """PLDM 7-term loss (Sobal et al., arXiv:2502.14819, 2025).

    Implements the loss composition

        L_total = L_pred                                    # term 1
                + lambda_var_z   * L_var(z)                 # term 2
                + lambda_cov_z   * L_cov(z)                 # term 3
                + lambda_smooth  * L_smooth(z)              # term 4
                + lambda_var_dz  * L_var(dz)                # term 5
                + lambda_cov_dz  * L_cov(dz)                # term 6
                + lambda_idm     * L_idm(z_t, z_{t+1}, c)   # term 7

    where dz_t = z_{t+1} - z_t is the temporal difference and L_idm is
    the inverse-dynamics MSE: a small MLP predicts c from (z_t, z_{t+1})
    and the loss is MSE between predicted and true c.

    Verify all seven terms against arXiv:2502.14819 directly before
    implementation. D8 in HANDOFF.md gives the decomposition as understood
    at project bootstrap; the paper is authoritative if it differs.

    Reference:
        Sobal et al., "Learning from Reward-Free Offline Data: A Case for
        Planning with Latent Dynamics Models", arXiv:2502.14819, 2025.
        Variance hinge and covariance terms follow Bardes et al., ICLR
        2022 (arXiv:2105.04906) Equation 1.
    """

    def __init__(
        self,
        d: int,
        c_dim: int = 3,
        lambda_var_z: float = 1.0,
        lambda_cov_z: float = 1.0,
        lambda_smooth: float = 1.0,
        lambda_var_dz: float = 1.0,
        lambda_cov_dz: float = 1.0,
        lambda_idm: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
        idm_hidden: int = 128,
    ) -> None:
        super().__init__()
        ...
        # Inverse-dynamics MLP: 2 * d -> idm_hidden -> idm_hidden -> c_dim
        self.idm = nn.Sequential(
            nn.Linear(2 * d, idm_hidden),
            nn.GELU(),
            nn.Linear(idm_hidden, idm_hidden),
            nn.GELU(),
            nn.Linear(idm_hidden, c_dim),
        )

    def forward(
        self,
        z_seq: Tensor,
        z_hat_seq: Tensor,
        c: Tensor,
    ) -> dict[str, Tensor]:
        """Compute all 7 terms and the weighted total.

        Args:
            z_seq:     (B, T, d) encoder latents (the target stream).
            z_hat_seq: (B, T, d) predictor outputs aligned so that
                       z_hat_seq[:, t, :] predicts z_seq[:, t+1, :].
            c:         (B, c_dim) static episode descriptor.

        Returns:
            dict with keys 'L_total', 'L_pred', 'L_var_z', 'L_cov_z',
            'L_smooth', 'L_var_dz', 'L_cov_dz', 'L_idm'. All values are
            scalars except none. 'L_total' is the weighted sum used for
            backward.
        """
        ...
```

### Defaults rationale

The default weights (all 1.0) are placeholders. The PLDM paper’s
hyperparameters should be the actual defaults. Until verified against the
paper, the code should warn (via a one-time log message at construction)
that the defaults are unverified placeholders and the agent should consult
arXiv:2502.14819 for the recommended values. The lambda bisection (Session
6 scope) will eventually tune these anyway.

The inverse-dynamics MLP architecture (2*d -> 128 -> 128 -> c_dim) is a
reasonable default. The PLDM paper specifies its IDM architecture; verify
against the paper before fixing.

### Numerical notes

- All seven losses are computed in fp32 even under bf16 autocast (same
  convention as SIGReg and VICReg from earlier sessions).
- The variance hinge requires `eps > 0` to avoid infinite gradients when
  variance approaches zero.
- The covariance term uses the unbiased estimator with denominator `N - 1`.
- For terms 5 and 6, `dz = z[:, 1:, :] - z[:, :-1, :]` has shape
  `(B, T-1, d)`. Flatten the time axis: `dz.reshape(-1, d)` before passing
  to the variance/covariance helpers, same as for `z`.
- The IDM forward expects `concat([z_t, z_{t+1}], dim=-1)` of shape
  `(B*(T-1), 2*d)`.

### Unit tests for `pldm.py`

`tests/test_pldm_loss.py`:

```python
def test_pldm_shape_contract():
    """z_seq (2, 8, 32), z_hat_seq (2, 8, 32), c (2, 3) -> dict with 8
    scalar values (7 terms + total). torch.manual_seed(0)."""

def test_pldm_pred_loss_zero_on_perfect_prediction():
    """If z_hat_seq[:, :-1, :] == z_seq[:, 1:, :], term 1 (L_pred) is zero.
    The other six terms are nonzero (they depend on z alone)."""

def test_pldm_var_z_low_on_unit_variance_gaussian():
    """z ~ N(0, I_32) -> term 2 (L_var_z) is below 0.1 because the variance
    hinge is satisfied. Term 3 (L_cov_z) is also low (independent dims)."""

def test_pldm_var_z_high_on_collapsed_z():
    """z = zeros(1024, 32) -> term 2 is mu * gamma^2 = 1.0 (per default
    lambda_var_z=1, gamma=1). Same construction as in tests/test_vicreg.py;
    consistency check between the two anti-collapse paths."""

def test_pldm_smooth_low_on_static_z():
    """z[:, t, :] == z[:, 0, :] for all t (no temporal variation) ->
    term 4 (L_smooth) is exactly zero. This is "perfectly smooth" but is
    actually pathological (the latent does not capture dynamics); the
    PLDM loss does NOT penalize this by itself, which is the failure
    mode terms 5 and 6 are designed to catch."""

def test_pldm_var_dz_high_on_static_z():
    """For the same construction as above (z[:, t, :] == z[:, 0, :]),
    dz is exactly zero, so var(dz) is zero, and term 5 (L_var_dz) fires
    at maximum (gamma^2 = 1.0 default). This confirms terms 5 and 6 catch
    the "perfectly smooth but static" failure mode that term 4 misses."""

def test_pldm_idm_zero_on_perfect_predictor():
    """Manually set the IDM MLP weights so it predicts c exactly from
    (z_t, z_{t+1}). Term 7 (L_idm) is then zero. This is the test that
    the IDM module is wired correctly."""

def test_pldm_idm_high_on_random_initialization():
    """At init, the IDM MLP has random weights. Term 7 is on the order of
    Var(c), which for c uniform in [-3, 3] x [0, 1.5] x [-0.2, 0.2] is
    roughly 3.0 + 0.2 + 0.013 ~ 3.2. Test that the unweighted L_idm is
    between 0.5 and 10.0 at init (loose bracket)."""

def test_pldm_total_equals_weighted_sum():
    """L_total == sum over the 7 terms with the configured weights.
    Tolerance 1e-5 in fp32. This is the test that the wrapper composition
    is correct."""

def test_pldm_gradient_flows_through_all_seven_terms():
    """Zero out six of the seven term weights, verify the gradient on z
    is nonzero from each remaining term independently. Repeat for all
    seven terms (seven runs)."""

def test_pldm_dtype_promotion():
    """Input in bf16 under autocast produces fp32 output, no overflow,
    backward pass succeeds. Same convention as SIGReg and VICReg."""
```

Use `torch.manual_seed(0)` at the top of every test. Default test sizes
B=2, T=8, d=32.

## Module 2: `src/models/pldm_wrapper.py`

### Class signature

```python
class PLDMWrapper(nn.Module):
    """PLDM training wrapper composing encoder + predictor + 7-term loss.

    Parallel to src.models.jepa.JEPA. The difference is the loss
    composition: JEPA uses (L_pred + 0.5 * L_roll + lambda * L_sigreg);
    PLDM uses the 7-term loss in src.baselines.pldm.PLDMLoss.

    Architecturally identical to JEPA: same encoder, same predictor, same
    bf16 path. This is so the SIGReg-vs-PLDM comparison isolates the LOSS
    as the only difference.

    Reference:
        Sobal et al., arXiv:2502.14819, 2025.
    """

    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        loss: PLDMLoss,
        rollout_weight: float = 0.0,
        H_roll: int = 8,
        rollout_start_strategy: str = "uniform_random",
    ) -> None:
        ...
```

### Design question RESOLVED: PLDM’s L_sim is already multi-step

Earlier drafts of this plan posed a design question about whether to include
a separate rollout loss for PLDM. The PLDM paper main body Equation 3.3
resolves this: `L_sim = sum over t=0..H of ||z_hat_t - z_t||^2`, summed
over the prediction horizon H. This is **already** a multi-step prediction
loss, structurally equivalent to combining our `L_pred + L_roll`.

Therefore the correct implementation is:

```python
class PLDMWrapper(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        loss: PLDMLoss,
        prediction_horizon: int = 8,   # PLDM's H, matches our H_roll = 8
    ) -> None:
        ...
```

The wrapper does not have separate `L_pred` and `L_roll` terms. It has one
`L_sim` term that runs the rollout over `prediction_horizon` steps and
averages the prediction error across all horizon positions. This is the
PLDM-faithful single-loss interpretation that matches arXiv:2502.14819
Eq 3.3, and it is structurally equivalent to our V-JEPA 2-AC-faithful
sum (D21).

The architectural comparison is thus:

- **SIGReg JEPA**: `L_pred + 0.5 * L_roll + lambda * L_sigreg`. Two
  prediction loss components (teacher-forced one-step plus rollout), one
  anti-collapse term.
- **PLDM**: `L_sim(horizon=H) + lambda_var_z * L_var(z) + lambda_cov_z * L_cov(z) + [additional terms from C.1.1] + lambda_idm * L_idm`. One
  prediction loss (multi-step), several anti-collapse terms, one inverse
  dynamics term.

The contrast the paper reports is therefore “one prediction stream +
SIGReg” vs “one prediction stream + multi-term VICReg-derived
anti-collapse + IDM”. The prediction-loss part is comparable across both
recipes (both are multi-step, both use the same H = 8 horizon); the
anti-collapse part is what differs. This isolates the methodological
question cleanly.

### Forward signature

```python
def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
    """Forward pass on one training batch.

    Args:
        batch: dictionary with keys 'omega' (B, T, 1, H, W) and 'c' (B, 3).

    Returns:
        Dictionary with all 7 PLDM terms (L_pred, L_var_z, L_cov_z,
        L_smooth, L_var_dz, L_cov_dz, L_idm) PLUS L_roll (if
        rollout_weight > 0) PLUS L_total. Also includes 'z' for
        diagnostics (detached cache, as in the JEPA wrapper).
    """
    ...
```

### Unit tests for the wrapper

`tests/test_pldm_wrapper.py`:

```python
def test_pldm_wrapper_shape_contract():
    """Batch with omega (2, 32, 1, 192, 96) and c (2, 3) -> dict with all
    expected keys, all losses finite, z is (2, 32, 32)."""

def test_pldm_wrapper_loss_decomposition():
    """L_total == sum of 7 terms weighted by their lambdas. Same
    consistency check as test_pldm_total_equals_weighted_sum but at the
    wrapper level so encoder and predictor are exercised."""

def test_pldm_wrapper_gradient_flows():
    """One backward pass on L_total produces non-zero gradients on every
    trainable parameter (encoder, predictor, IDM MLP, no others). The
    PLDMLoss has only the IDM MLP as parameters; SIGReg/VICReg have none."""

def test_pldm_wrapper_bf16_autocast_smoke():
    """Under bf16 autocast, one forward+backward on B=2, T=8 succeeds.
    Skip if require_rtx6000() raises."""

def test_pldm_wrapper_with_rollout_zero_equals_no_rollout_path():
    """With rollout_weight=0, the wrapper output should not depend on
    H_roll (the rollout path is skipped). Verify by running with H_roll=2
    and H_roll=8 and asserting allclose on all losses."""
```

## Module 3: `src/training/train_baseline.py`

argparse entrypoint, mirroring `train_jepa.py` but dispatching on a
`--baseline` flag. Pseudocode:

```python
def main() -> None:
    args = parse_args()
    require_rtx6000()
    set_all_seeds(args.seed)
    cases = load_cases_yaml(args.cases_from)

    encoder = HybridCNNViTEncoder(...)
    predictor = AutoregressivePredictor(...)

    if args.baseline == "pldm":
        loss = PLDMLoss(d=32, c_dim=3, ...)
        wrapper = PLDMWrapper(encoder, predictor, loss, ...)
        wandb_tags = ["hybrid_cnn_vit", "pldm_vicreg_7term"]
    elif args.baseline == "fukami_ae":
        raise NotImplementedError("Fukami AE: separate session")
    elif args.baseline == "solera_rico":
        raise NotImplementedError("Solera-Rico beta-VAE: separate session")
    elif args.baseline == "pod":
        raise NotImplementedError("POD: separate session, not a training run")
    else:
        raise ValueError(f"unknown baseline: {args.baseline}")

    dataset = EpisodeDataset(...)
    loader = DataLoader(...)
    optimizer = build_optimizer(wrapper)  # same two-group AdamW as JEPA
    scheduler = build_scheduler(...)      # same warmup+cosine as JEPA

    wandb.init(
        project=os.environ['WANDB_PROJECT'],
        group=f'partition_{args.partition}',
        tags=wandb_tags,
        config={
            'preprocessing_version': ...,
            'partition_version': args.partition,
            'baseline': args.baseline,
            'lambda_var_z': args.lambda_var_z,
            'lambda_cov_z': args.lambda_cov_z,
            'lambda_smooth': args.lambda_smooth,
            'lambda_var_dz': args.lambda_var_dz,
            'lambda_cov_dz': args.lambda_cov_dz,
            'lambda_idm': args.lambda_idm,
            'rollout_weight': args.rollout_weight,
            'seed': args.seed,
            'split_sha256': ...,
            'inventory_sha256': ...,
            'code_sha256': ...,
            'gpu_name': torch.cuda.get_device_name(device.index),
        },
    )

    train_loop(wrapper, loader, optimizer, scheduler, ...)
    save_checkpoint(...)
    wandb.finish()
```

Notes:

- Same W&B contract as `train_jepa.py`. Required keys still apply.
  `lambda_sigreg` is replaced by the six PLDM lambdas in the config.
- Auto-fallback controller is NOT instantiated for PLDM (PLDM has its own
  anti-collapse machinery; the SIGReg -> VICReg fallback is irrelevant).
- The training loop and diagnostics interface is shared with `train_jepa.py`.
  If the diagnostics module needs adapting (e.g., the AutoFallbackController
  field is missing from PLDM wrapper outputs), guard with a `hasattr` check
  or pass a no-op controller.

## Run protocol

### Run PLDM-A: faithful 5-case run

```bash
python -m src.training.train_baseline \
    --baseline pldm \
    --partition v1 \
    --cases-from configs/cases/smoke_5cases.yaml \
    --max-iters 5000 \
    --seed 0 \
    --diagnostic-every 250 \
    --checkpoint-every 1000 \
    --log-every 25 \
    --output-dir outputs/runs/smoke5k/run_pldm_a \
    --wandb-mode online \
    --tag-suffix run_pldm_a_seed0
```

This is the matched-protocol PLDM run that compares directly against Session
5’s Run A (SIGReg baseline). Same seed, same cases, same diagnostic cadence.

### Run PLDM-B: PLDM + LayerNorm (optional, conditional)

Run PLDM-B fires only if Run PLDM-A shows the PR <= 16 collapse mode. Per
the Session 5 D17 contingency, the LayerNorm intervention is the cheapest
first attempt. PLDM-B uses LayerNorm at the encoder projection while
keeping all PLDM-specific loss weights at defaults.

```bash
python -m src.training.train_baseline \
    --baseline pldm \
    [...same as Run PLDM-A...] \
    --projection-norm layernorm \
    --output-dir outputs/runs/smoke5k/run_pldm_b_layernorm \
    --tag-suffix run_pldm_b_layernorm_seed0
```

Optional, not part of the minimum session. Run it only if PLDM-A is
ambiguous and a quick LayerNorm test would clarify.

### Stopping conditions

Each run completes 5000 iterations even on diagnostic anomalies. No early
termination. Total wall-clock for PLDM-A alone is roughly 15 to 25 minutes
on the RTX PRO 6000 Blackwell (similar to a Session 5 variant; the 7-term
loss is slightly more expensive than SIGReg but the dominant cost is the
encoder+predictor forward, which is identical).

## Analysis: extending the Session 5 notebook

Add a new section to `notebooks/01_smoke_5k_analysis.ipynb` titled
“Section 7: PLDM Comparison”. Do not modify Sections 1 through 6.

### Section 7.1: PLDM loss trajectories

One figure with seven panels (one per PLDM term) showing the loss curve
over 5000 iters. Overlay the SIGReg L_anticollapse curve from Session 5
Run A for visual comparison; their numerical scales are not directly
comparable but the trajectory shape is informative.

### Section 7.2: PLDM diagnostics over time

Same 2x2 figure as Session 5 Section 3, with PLDM-A added as a new line.
Highlight PLDM-A in a contrasting colour.

### Section 7.3: The methodological table

Compute and report the same 2x2 outcome table as Session 5 Section 4, for
PLDM-A specifically. Then place SIGReg Run A, SIGReg Run B (LayerNorm),
SIGReg Run C (VICReg), and PLDM-A side by side in a single 2x2 grid:

|        |probe_R^2 > 0.5        |probe_R^2 <= 0.5|
|--------|-----------------------|----------------|
|PR > 16 |HEALTHY                |WEAK            |
|PR <= 16|TRIVIAL (collapse to c)|DEAD            |

Each cell lists which runs landed there. This is the headline figure of the
session.

### Section 7.4: Decision string update

Extend the decision string from Session 5 with the PLDM outcome:

```
Session 5.PLDM outcome: <one of>

  REGIME_CONFIRMED  - PLDM-A clears PR > 16 and 0.5 < probe_R^2 < 0.7.
                      The LeWM Two-Room precedent replicates on physics
                      data. SIGReg fails, PLDM succeeds, at low intrinsic
                      dimensionality.
                      -> Session 6 with PLDM as the primary trained model;
                         SIGReg becomes the comparison baseline.
                      -> Paper contribution claim 3 sharpens: "the
                         regime-dependent SIGReg-PR diagnostic plus the
                         PLDM-as-fallback recommendation for low-intrinsic-
                         dim domains".

  DATA_SCALE_BOUND  - PLDM-A also fails the PR criterion (PR <= 16).
                      Both regularisers collapse on 5 cases. The failure is
                      not regulariser-specific; it is data-scale-bound.
                      -> Session 5.5 (expand to 10-12 cases) on BOTH SIGReg
                         and PLDM.
                      -> Defer the regime-dependent claim until 5.5 data.

  PLDM_PARTIAL      - PLDM-A clears PR but probe R^2 is outside (0.5, 0.7).
                      Either PLDM is anti-collapsed but uninformative
                      (probe < 0.5, "WEAK") or PLDM also memorises c
                      (probe > 0.7, "PLDM TRIVIAL"). Diagnose with the
                      combinatorial table.
                      -> Outcome-dependent next session, see Session 5
                         decision tree.
```

## Pass criteria for Session 5.PLDM as a session

Three things must be true at the end:

1. All `test_pldm_loss.py` and `test_pldm_wrapper.py` tests pass. Sessions
   2, 3, 4, and 5 tests must remain green.
1. Run PLDM-A completes 5000 iterations with finite final loss and a clean
   W&B upload.
1. The analysis notebook’s Section 7 is populated and the decision string
   is printed.

The numerical pass criteria (PR, probe R^2) are NOT pass criteria for the
session. They are the methodological finding the session reports. A clean
“DATA_SCALE_BOUND” outcome is a successful session: it tells us what
Session 5.5 must do.

## Out of scope for Session 5.PLDM

- **The other three baselines** (POD, Fukami AE, Solera-Rico). Each is its
  own session; PLDM is the priority comparator under the LeWM precedent,
  the others can be trained later in any order.
- **Hyperparameter tuning of PLDM’s six weights**. Use defaults from
  arXiv:2502.14819 (or all 1.0 if the paper does not specify, with a
  warning in the W&B config). Tuning is a Session 6 concern alongside the
  SIGReg lambda bisection.
- **The decoder**. The visualisation decoder is a separate stage, deferred
  until a non-collapsed JEPA or PLDM checkpoint exists.
- **Multi-seed variance**. A single PLDM-A run is enough to answer the
  methodological question. Variance can be a Session 6 or paper-prep
  concern.
- **Hydra**. Same reasoning as Session 5; argparse is enough.

## Expected duration

- Reading arXiv:2502.14819 and verifying the 7-term loss against the paper:
  30 to 60 minutes. This is the most important pre-coding step.
- Implementing `pldm.py` and its tests (TDD style): 90 to 120 minutes.
- Implementing `pldm_wrapper.py` and its tests: 30 to 45 minutes.
- Implementing `train_baseline.py`: 30 to 45 minutes (mostly mirroring
  `train_jepa.py`).
- Run PLDM-A (5k iters): 15 to 25 minutes.
- Run PLDM-B (optional, conditional on PLDM-A outcome): 15 to 25 minutes.
- Analysis notebook Section 7: 30 to 60 minutes.
- HANDOFF entries + session report: 30 minutes.

Total realistic span: 4 to 7 hours. Plan accordingly; this is a substantive
session, not a “swap one module” change.

## D-entries to record (always, in HANDOFF.md)

**D30**: Session 5.PLDM executed. Triggered by Session 5 TRIVIAL outcome
per D29. The PLDM prediction loss is multi-step (Equation 3.3 of
arXiv:2502.14819), structurally equivalent to our `L_pred + L_roll` with
H = 8; the wrapper implements it as a single `L_sim` over the horizon
rather than as separate teacher-forced + rollout components. The
collapse-prevention term count and weights are read directly from
arXiv:2502.14819 Appendix C.1.1 and cross-referenced against
github.com/vladisai/PLDM `pldm/` source code; record any discrepancy
between the paper’s prose and the code as part of D30’s resolution.
The IDM adaptation: predicts c = (G, D, Y) from (z_t, z_{t+1}) since our
setup has no per-step action; this is locked in D8 (with the correction
in D32) and restated here for traceability.

**D31 (if Session 5.PLDM outcome is REGIME_CONFIRMED)**: The regime-
dependent contrast is confirmed on physics data. Numbers are <PR, probe
R^2> for PLDM-A and Run A (SIGReg). Session 6 proceeds with PLDM as the
primary trained model. Paper contribution claim 3 sharpens.

**D31 (if Session 5.PLDM outcome is DATA_SCALE_BOUND)**: Both SIGReg and
PLDM collapse on 5 cases. Session 5.5 (expand to 10-12 cases) follows.

**D31 (if Session 5.PLDM outcome is PLDM_PARTIAL)**: PLDM clears PR but
probe R^2 outside the healthy range. Outcome-dependent next session per
Session 5 decision tree.

**D32 (CORRECTION to D8)**: D8 in HANDOFF.md cites PLDM as
“Sobal, Jyothir, Jalagam, Carion, Cho, LeCun (2022), arXiv:2211.10831”
with the title “Joint Embedding Predictive Architectures Focus on Slow
Features”. This citation is INCORRECT. The 2022 paper is a 4-page NeurIPS
SSL workshop precursor by a partially overlapping author group; it is
useful as theoretical background but is NOT the source of the PLDM name
or the multi-term loss formulation. The actual PLDM paper is:

Sobal, Zhang, Cho, Balestriero, Rudner, LeCun, “Learning from
Reward-Free Offline Data: A Case for Planning with Latent Dynamics
Models”, arXiv:2502.14819, February 2025.

Update D8 in HANDOFF.md to cite arXiv:2502.14819 as the primary reference,
with arXiv:2211.10831 listed separately as the workshop precursor for
theoretical background. Also update CLAUDE.md’s “Baselines to implement”
section item 4 (which still references 2211.10831). The “7-term loss”
language in D8 is approximate; the actual term count and weight set are
to be read directly from arXiv:2502.14819 Appendix C.1.1 and the official
code at github.com/vladisai/PLDM, and the D8 description updated to match
once verified.

## After Session 5.PLDM lands

The next session depends on the outcome:

- REGIME_CONFIRMED -> Session 6 with PLDM as primary, SIGReg as comparison.
  Hydra + lambda bisection (on PLDM’s six tunable weights, not SIGReg’s
  single lambda) plus torch.compile. The SIGReg-vs-PLDM contrast is now
  the central headline.
- DATA_SCALE_BOUND -> Session 5.5: expand cases to 10-12, rerun BOTH
  SIGReg and PLDM, repeat the analysis.
- PLDM_PARTIAL -> follow the Session 5 decision tree for the relevant
  quadrant.

All earlier-session unit tests must remain green throughout.

## Decision references (existing)

- D5 (HANDOFF): SIGReg auto-fallback to VICReg.
- D8 (HANDOFF): PLDM as the fourth matched-capacity baseline, with the
  7-term loss decomposition Carlos read from his understanding at project
  bootstrap. Section 5.PLDM should verify this against arXiv:2502.14819
  directly and update D8 if the paper’s formulation differs.
- D17 (HANDOFF): BatchNorm projection per LeWM; LayerNorm is the first
  diagnostic intervention. Run PLDM-B exercises this.
- D19 (HANDOFF): RTX 6000 Blackwell only.
- D21 (HANDOFF): V-JEPA 2-AC-faithful scheduled sampling with H_roll = 8.
  Note that PLDM does NOT use this recipe; whether to include the rollout
  loss for PLDM is the design question above.
- D22 (HANDOFF): VICReg coefficients mu=25, lambda=25, nu=1, gamma=1, no
  invariance term. PLDM’s variance and covariance terms (terms 2, 3, 5, 6)
  follow the same VICReg derivation; the PLDM defaults may differ.
- D29 (HANDOFF): PLDM is conditional priority on Session 5 TRIVIAL.
  Session 5.PLDM is the operationalisation.
- D30 and D31 (this session): see above.
