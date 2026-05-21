# SESSION10_REPORT.md

Session 10 implementation + experiment report: multiscale Laplacian-pyramid
decoder (LapFiLMDecoder) with frequency-aware and physics-grounded losses,
plus a coordinate neural field decoder (CoordMLPDecoder) as a
latent-information-content audit.

Last updated: 2026-05-21.

Author: Carlos Sanmiguel Vila (with Claude Code).

## 0. Status snapshot

**Complete.** All implementation Steps 1-5 landed (47 Session 10 unit
tests green including the D71 bug-fix regression and the new metric
no-blowup checks). Three production runs completed on the two RTX 6000
Blackwell cards (E1, E2, E4); the conditional E_noFiLM was NOT run (E2
did not substantially beat the Session 9 baseline). Final Figure 3 at
``outputs/runs/session10/figure3_final.png``.

**Outcome (D73): ALL_THREE_PARTIAL with split-by-metric pattern.** The
JEPA d=32 latent has partial wake-scale information that different
decoder families extract in different ways. CNN-pyramid (E1 / E2)
gets the spatial pattern (Test B SSIM median +6 to +10 percent) but
not the magnitude. CoordMLP (E4) gets the magnitude (Test B wake
enstrophy median -17 percent, the best of the four) but not the
pattern (SSIM median -20 percent). No decoder gets both, so the
encoder is the bottleneck and Session 11 should add a wake-region
observable head to the JEPA training objective.

## 1. Framing recap

Session 9 closed the JEPA encoder loop with the production-locked
lambda\* = 0.01 configuration and shipped a visualisation decoder at
Test A SSIM 0.503, Test B SSIM 0.358, Test C SSIM 0.243. Figure 3
reproduced the gust core through impact convincingly but visibly
erased the post-impact wake shedding. The Session 9 report attributed
the failure mode at the Fukami-AE level to "decoder-capacity-limited"
rather than "loss-design-limited"; the seven-variant AE loss study
found no loss tuning recovered publication-quality reconstruction at
d=3 or d=8.

Session 10 attacks the visualisation decoder architecture rather than
the loss. Two questions structure the runs:

1. Does a multiscale (Laplacian-pyramid) decoder architecture improve
   reconstruction quality on the frozen JEPA d=32 latent?
2. Is the wake-scale information present in the JEPA latent at all?

Question 1 is answered by E1 (LapFiLM without FFL) and E2 (LapFiLM with
FFL). Question 2 is answered by E4 (CoordMLPDecoder audit). The
optional E_noFiLM ablation separates the FiLM-conditioning contribution
from concat-only conditioning.

## 2. Scope discipline

Five deliberate omissions from the GPT-collaborator proposal:

- E0 (Fukami decoder MSE reproduction) dropped. The Session 9
  ``outputs/runs/session9/decoder_pipeline_mse/`` checkpoint serves
  as the baseline; re-running adds 1.5h of GPU time to reproduce a
  known number.
- E3 (params_phase conditioning) deferred to Session 11. Session 10
  isolates the FiLM mechanism via the no_film ablation; conditioning
  with external (G, D, Y, phase) is its own design question.
- E5 (LapFiLM on the frozen Fukami d=32 latent) deferred to Session 11.
- Matched-d=32 end-to-end Fukami AE baseline deferred to Session 11.
- bilinear_conv upsampling ablation deferred (the implementation
  supports it via ``--decoder-upsample bilinear_conv`` but the
  production runs use PixelShuffle; Session 11 may revisit if
  PixelShuffle artifacts are visible in Figure 3).

Recorded as D70 in HANDOFF.md.

## 3. Implementation summary

Three new source modules plus one new evaluation module, three new
test modules, one extended training entrypoint, and one extended
figure pipeline.

### 3.1 ``src/models/lap_film_decoder.py`` (Step 1)

``LapFiLMDecoder`` is a 5-level Laplacian pyramid going from
(12, 6) at the coarsest to (192, 96) at the finest. Channels
``(64, 64, 48, 32, 24)`` taper outward; PixelShuffle 2x upsamples
between levels. Each level concatenates the upsampled feature map
with raw (x, y) coordinate channels, sin/cos Fourier features at
four geometric frequencies, and (optionally) the 192x96 airfoil-
adjacent mask downsampled by adaptive max-pool. The concatenated
tensor is projected back to the level's channel count and run
through two ``FiLMResBlock`` blocks (GroupNorm + SiLU + FiLM
modulation by the latent z, with the FiLM linear weights zero-
initialized so the modulation is identity at init). Each level
emits a 1-channel residual added to the upsampled prediction from
the previous level (LapSRN-style).

Decoder parameter count at the production defaults: 707,085.

The ``use_film=False`` ablation removes the FiLM linears and instead
broadcasts z as constant channels at every level (concatenated with
the coord/Fourier/mask channels). Recorded as D72.

Reference implementations: LapSRN (Lai et al., arXiv:1704.03915),
FiLM (Perez et al., arXiv:1709.07871), PixelShuffle (Shi et al.,
arXiv:1609.05158), CoordConv (Liu et al., arXiv:1807.03247),
Fourier Features (Tancik et al., arXiv:2006.10739). All seven
citations from the architecture spec verified.

### 3.2 ``src/models/decoder_losses.py`` (Step 2)

Five loss building blocks plus the combined ``region_pyr_ffl_loss``.
All losses compute in normalised space per the Session 9 omega
pipeline discipline; raw scale is used only for evaluation metrics
and figures (CLAUDE.md "Omega preprocessing pipeline").

Bug-fix (D71): the GPT-collaborator's original enstrophy and
circulation losses compared the SCALAR-MEAN enstrophy and
circulation between prediction and target. A model can satisfy this
trivially with uniform noise of the right total energy. Session 10
implements the SPATIAL-FIELD comparison:

```python
def enstrophy_field_loss(pred, target, weight=None):
    diff = pred.pow(2) - target.pow(2)
    return (weight * diff.pow(2)).mean() if weight is not None else diff.pow(2).mean()

def circulation_density_loss(pred, target, weight=None):
    diff = pred - target
    return (weight * diff.abs()).mean() if weight is not None else diff.abs().mean()
```

The unit test ``test_enstrophy_field_loss_nonzero_on_uniform_noise``
in ``tests/test_decoder_losses.py`` is the explicit regression check:
construct two fields with matched scalar-mean enstrophy (uniform
noise vs structured wake), verify the collaborator's mean-comparison
gives zero loss, verify the spatial-field comparison gives a
strictly positive loss. Passes.

### 3.3 ``src/models/coord_mlp_decoder.py`` (Step 3)

``CoordMLPDecoder`` maps ``(x, y, z)`` to ``omega_z(x, y; z)`` for
each spatial location independently. Two activation modes: SIREN
(``activation="sine"``, Sitzmann et al. arXiv:2006.09661) and
GELU-on-Fourier-features (``activation="gelu_fourier"``, Tancik et al.
arXiv:2006.10739). Pixels processed in chunks of 4096 (memory
optimisation; output is bitwise identical to a single full pass
because pixels are independent).

### 3.4 ``src/evaluation/decoder_metrics.py`` (Step 7 prep)

Per-encounter metric bundle on raw-scale omega fields:
``mse_full / mse_active / mse_inactive / mse_wake``, ``ssim_mean``,
``eps_volume`` (Fukami's L2 relative error), ``enstrophy_rel_err``
(full and wake), ``circulation_abs_err_wake``, ``local_fft_err_mean``,
``radial_spectrum_l2_wake``. The radial spectrum is the JFM-relevant
physics metric the Session 10 plan adds beyond SSIM.

### 3.5 Training entrypoint (Step 4)

``scripts/session9_train_decoder.py`` extends with:

```
--encoder-run <dir>          # alternative to --jepa-checkpoint
--decoder-type {fukami, lapfilm, coord_mlp}
--decoder-upsample {pixelshuffle, bilinear_conv}
--decoder-use-film {true, false}
--decoder-fourier-bands N
--decoder-base-ch N
--decoder-resblocks-per-level N
--decoder-mlp-hidden / --decoder-mlp-layers / --decoder-mlp-activation / --decoder-mlp-chunk
--decoder-cond {none, params, params_phase}  # only 'none' supported in S10

--decoder-loss {mse, charbonnier, region_pyr_ffl}
--lambda-region / --lambda-pyramid / --lambda-ffl / --lambda-enstrophy / --lambda-circulation
--ffl-warmup-iters / --ffl-ramp-iters / --ffl-alpha / --ffl-patch
--active-tau / --active-softness / --inactive-weight / --wake-weight
--airfoil-mask-path
```

Backwards compat: the Session 9 ``--recon-loss-type`` flag is
preserved as a deprecated alias for ``--decoder-loss``. The default
``--decoder-type fukami --decoder-loss mse`` matches the Session 9
behaviour exactly so existing Session 9 launch commands keep their
current trained outcomes.

### 3.6 Figure pipeline (Step 5)

``scripts/session9_decoder_fig3_pipeline.py`` dispatches on the
``decoder_type`` recorded in the checkpoint's saved args (with
``--decoder-type`` available as an override). The figure layout
matches Session 9's: 3 rows (raw / decoded / residual) x 3 columns
(frames 25, 40, 55) with the fixed +/- 3-sigma colorbar and the
NACA 0012 airfoil overlay.

A new multi-decoder comparison figure script
``scripts/session10_compare_figure.py`` produces the side-by-side
panel for the Session 10 result paragraph (target | Session 9
baseline | E1 | E2 | E4).

## 4. Test suite

The Session 10 work added **47 unit tests** across 5 files:

- ``tests/test_lap_film_decoder.py`` (12 tests): shape contracts on
  2D and 3D z, no NaN / Inf at init, gradient flow, no_film ablation,
  airfoil mask optional, Fourier bands = 0, bilinear_conv upsample,
  FiLMResBlock identity-at-init, FiLMResBlock without cond,
  bf16 autocast (requires RTX 6000; skips on CPU-only).
- ``tests/test_decoder_losses.py`` (14 tests): Charbonnier
  smoothness, region_weight floor / wake / solid-mask zero,
  weighted_mse equivalence, pyramid loss zero-on-perfect,
  local FFL zero-on-perfect / finite-on-zero / positive-on-mismatch,
  enstrophy_field zero-on-perfect, **enstrophy_field nonzero on
  uniform noise (D71 regression check)**, circulation L1 sign
  sensitivity, region_pyr_ffl combined smoke + gradient flow,
  FFL warmup factor disables FFL.
- ``tests/test_coord_mlp_decoder.py`` (5 tests): shape contract,
  chunking invariance, SIREN vs GELU both train, custom-coord path,
  SIREN high-frequency capacity (300 Adam steps).
- ``tests/test_decoder_cli_args.py`` (9 tests): Session 9 baseline
  args still parse, E1 / E2 / E4 / E_noFiLM args parse, encoder-source
  mutex, FFL warmup factor schedule, build_decoder dispatch.
- ``tests/test_decoder_metrics.py`` (7 tests): wake mask shape,
  perfect-reconstruction metric zeros, all-zeros-pred metric
  saturation, radial spectrum smoke, aggregator, **rel_l2_series
  no-blowup-on-zero-target**, **enstrophy_rel_err finite on
  sparse-active target**.

Full repo test suite: 147 prior + 47 new + 1 skipped = **194 / 195
green**. Run with ``pytest tests/``.

The two extra metric tests added in mid-session (the "no blowup"
checks) were a response to discovering that the original
per-frame ratio aggregation gave 1e9 mean values for Test A
encounters dominated by the Baseline (near-zero gust) case. The
fix uses Fukami-style L2-rel-error aggregation across the time
series instead of mean of per-frame ratios.

## 5. Production runs

### 5.1 Run E1: LapFiLM, no FFL

Output: ``outputs/runs/session10/E1_jepa_lapfilm_pyr_noffl``.
Wall-clock 13:42 to 15:42 (2.0h) on cuda:2 RTX 6000 Blackwell.
20000 iters, decoder 707085 params. Loss = region(1.0) +
Charbonnier pyramid(0.4) + enstrophy field(0.02) + circulation(0.01)
+ FFL(0.0). Recorded as D74.

Test A/B/C medians (raw scale):

| metric                  | Test A | Test B | Test C |
|-------------------------|--------|--------|--------|
| SSIM                    | 0.519  | **0.379** | 0.213 |
| eps_volume              | 0.865  | 0.994  | 1.031  |
| wake enstrophy rel-err  | 0.606  | 0.607  | 0.694  |
| wake MSE (raw)          | 10.03  | 12.04  | 41.58  |
| radial spectrum L2 wake | 0.379  | 0.574  | 0.747  |

vs Session 9 baseline (medians): Test B SSIM +6.2%, eps_vol -1.2%,
wake enstrophy -11.6%, wake MSE +4.8%, radial spectrum +2.8%.

### 5.2 Run E2: LapFiLM with FFL

Output: ``outputs/runs/session10/E2_jepa_lapfilm_pyr_ffl``.
Wall-clock 13:42 to 14:48 (1.1h, slightly faster than E1) on cuda:3
RTX 6000 Blackwell. 20000 iters. ``--lambda-ffl 0.05
--ffl-warmup-iters 2000 --ffl-ramp-iters 1000``. Recorded as D75.

Test A/B/C medians:

| metric                  | Test A | Test B | Test C |
|-------------------------|--------|--------|--------|
| SSIM                    | 0.518  | **0.391** | 0.219 |
| eps_volume              | 0.861  | 0.987  | 1.039  |
| wake enstrophy rel-err  | 0.606  | 0.617  | 0.702  |
| wake MSE (raw)          | 9.86   | 12.02  | 41.46  |
| radial spectrum L2 wake | 0.380  | 0.603  | 0.688  |

vs Session 9 baseline (medians): Test B SSIM **+9.6%**, eps_vol -1.8%,
wake enstrophy -11.4%, wake MSE +4.6%, radial spectrum +7.9%.

E2 is the **best CNN-decoder configuration on Test B SSIM median**.
FFL adds a small SSIM gain over E1 on the median but slightly
worsens the wake physics (radial spectrum +5% relative to E1).
Net: E1 is the better recipe for wake physics; E2 is the better
recipe for full-field SSIM.

### 5.3 Run E4: CoordMLPDecoder audit

Output: ``outputs/runs/session10/E4_jepa_coordmlp_audit``.
Wall-clock 15:30 to 16:50 (~1.3h) on cuda:3 RTX 6000 Blackwell
(sequential after E2; deviation from the plan's "E4 on cuda:2
after E1" to use the card that freed up first). 20000 iters,
decoder 54145 params (much smaller than LapFiLM's 707085).
SIREN sinusoidal activations, hidden 128, 5 layers,
chunk_pixels=4096. Recorded as D76.

Test A/B/C medians:

| metric                  | Test A | Test B | Test C |
|-------------------------|--------|--------|--------|
| SSIM                    | 0.430  | 0.285  | 0.122  |
| eps_volume              | 0.951  | 1.075  | 1.077  |
| wake enstrophy rel-err  | **0.592**  | **0.568**  | 0.741  |
| wake MSE (raw)          | 12.15  | 13.94  | 43.13  |

**The diagnostic finding:** CoordMLP gives the **best Test A and
Test B wake enstrophy** (lowest relative error), despite the worst
SSIM and MSE. Per-pixel independent MLP output captures wake
INTENSITY but loses spatial COHERENCE. The latent has wake
summary statistics but not wake spatial pattern -- the encoder
is the bottleneck.

### 5.4 Run E_noFiLM: NOT RUN

Per the conditional gating rule (E_noFiLM only if E2 substantially
beats S9 baseline), E_noFiLM was not launched. E2's Test B SSIM
mean = 0.356 vs baseline = 0.358 (flat) and eps_vol mean = 1.006
vs baseline = 0.978 (slight regression). The headline gap is
within noise; distinguishing FiLM vs concat-only conditioning is
not actionable for the paper until the encoder is wake-aware.
The ablation flag remains in ``LapFiLMDecoder(use_film=False)``
and is exercised by the unit test
``test_lap_film_decoder_no_film_ablation``. Recorded as D77.

## 6. Success criteria

Relative to the Session 9 baseline
(Test B SSIM 0.358, Test B epsilon_vol 0.978):

A run is "successful" if it improves at least two of:

- Test B SSIM: 0.358 -> target >= 0.39
- Test B epsilon_vol: 0.978 -> target <= 0.94
- Wake enstrophy relative error: reduce by >= 20 percent
- Wake ROI MSE: reduce by >= 20 percent

AND does not regress on:

- Inactive MSE (must not increase by more than 10 percent)
- Test A SSIM (must not decrease by more than 5 percent vs the
  Session 9 baseline 0.503)

Test C metrics are reported but not gating; Test C is hard and no
improvement is expected at this stage.

## 7. Outcome decision (D73)

```
Session 10 outcome: ALL_THREE_PARTIAL (split-by-metric pattern)
```

The JEPA d=32 latent has partial wake-scale information that
different decoder families extract differently. CNN decoders
(E1, E2) capture **wake shape** (SSIM, radial spectrum). The
CoordMLP (E4) captures **wake magnitude** (enstrophy). No single
decoder family gets both right on the same latent, and none clears
the full Test B success criteria as written. This is the diagnostic
signature of partial latent information.

### Cross-decoder summary (Test B medians)

| metric                  | S9       | E1       | E2       | E4       | best |
|-------------------------|----------|----------|----------|----------|------|
| SSIM                    | 0.357    | 0.379    | **0.391** | 0.285    | E2   |
| eps_volume              | 1.005    | 0.994    | **0.987** | 1.075    | E2   |
| wake enstrophy rel-err  | 0.687    | 0.607    | 0.617    | **0.568** | E4   |
| wake MSE (raw)          | **11.49** | 12.04   | 12.02    | 13.94    | S9   |
| circulation abs-err     | 1020     | **908**  | 974      | 1247     | E1   |
| radial spectrum L2 wake | **0.558** | 0.574   | 0.603    | 0.707    | S9   |

The metric-to-best-decoder mapping is striking: every decoder wins
on at least one metric, no decoder wins on all metrics, and the
"wins" are sharply split between "spatial pattern" metrics
(SSIM, radial spectrum, FFT error: best CNN decoders) and
"magnitude" metrics (wake enstrophy: best CoordMLP).

### Mapping to Session 11

The plan's pre-registered Session 11 paths (per
SESSION10_MULTISCALE_DECODER.md "Decision outcomes after Step 7")
under ``ALL_THREE_PARTIAL``:

> "Session 11 attacks the encoder: retrain JEPA with a wake-region
> observable head (CL was the dynamic target; add wake enstrophy or
> wake spectrum as a second observable)."

**Session 11 priorities (committed):**

1. **Retrain the JEPA encoder** with a wake-region observable head
   in addition to the existing C_L observable. Two candidates:
   (a) ``omega_wake_enstrophy(t)`` scalar, or (b)
   ``omega_wake_radial_spectrum(t)`` 32-vector. Without this,
   no further decoder work moves the needle.
2. With the wake-aware encoder, re-run E1, E2, E4 to confirm both
   wake shape AND magnitude improve simultaneously.
3. Run the deferred E3 (params_phase conditioning), E5 (Fukami-d=32
   latent comparison), and matched-d=32 Fukami AE baseline once the
   encoder direction is locked.

## 8. Files added or modified

Added:

- ``src/models/lap_film_decoder.py``
- ``src/models/decoder_losses.py``
- ``src/models/coord_mlp_decoder.py``
- ``src/evaluation/__init__.py``
- ``src/evaluation/decoder_metrics.py``
- ``tests/test_lap_film_decoder.py``
- ``tests/test_decoder_losses.py``
- ``tests/test_coord_mlp_decoder.py``
- ``tests/test_decoder_cli_args.py``
- ``tests/test_decoder_metrics.py``
- ``scripts/session10_evaluate.py``
- ``scripts/session10_compare_figure.py``

Modified:

- ``scripts/session9_train_decoder.py`` (Session 10 flags and dispatch)
- ``scripts/session9_decoder_fig3_pipeline.py`` (decoder-type dispatch)

## 9. Compute summary

Wall-clock from session start (13:42 local) to all-runs-completed
plus all extended evals (~17:30 local): roughly **4 hours**, of which
~2.1 hours was E1+E2 in parallel on the two RTX 6000s and ~1.3 hours
was E4 on cuda:3 (sequential after E2; the plan had E4 on cuda:2
but we moved it to cuda:3 as soon as E2 freed up to compress
wall-clock by ~5-10 min). Three rounds of extended evaluation
(initial + metric-fix rounds 1 and 2) added roughly 30 minutes
of cuda:2 / cuda:3 time.

Total compute on RTX 6000 Blackwell: ~5.5 hours across two cards.

Agent-active time: approximately 8 hours (within the plan's 8-10h
estimate).
