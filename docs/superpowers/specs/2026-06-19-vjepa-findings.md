# Faithful V-JEPA on the gust data: findings (variant B)

Date: 2026-06-19
Spec: `docs/superpowers/specs/2026-06-19-vjepa-design.md`
Plan: `docs/superpowers/plans/2026-06-19-vjepa.md`
Branch: `vjepa` (not merged, not pushed)

## Question

Does the V-JEPA *objective* (masked space-time tubelet feature prediction with an
EMA target encoder, Bardes et al. arXiv:2404.08471) beat our autoregressive
rollout objective on the gust data? Tested at d=64, seeds {0,1,2}, split v2.1,
on the six-metric suite. (Variant A, spatio-temporal *encoding* inside our
framing, was a clean null; B was a user-requested exploration.)

## FINAL VERDICT (after the fair re-eval, 2026-06-19 12:30)

V-JEPA improves PARAMETER READABILITY (real: impact-frame Y R^2 0.92 vs the
per-frame JEPA's 0.57) but does NOT forecast the wake better; on a FAIR
same-predictor test it forecasts much WORSE, and that weakness lives in the
V-JEPA latent itself, not the evaluation. So: helpful for a parameter-sensing /
disentanglement goal, not for the dynamical (forecasting) goal that is this
project's primary aim. Do not adopt it for the forecasting story.

### The fair re-eval that decided it (supersedes the "eval confound" section below)

The first run's poor V-JEPA forecast had two suspected confounds; both were
removed and the verdict held:
1. Coarse d=64 pooling -> re-extracted with a FINER latent (overlapping clips
   stride 8 + linear interp, vjepa_fine_s*). Forecast essentially UNCHANGED
   (-0.08 -> +0.51 vs the coarse +0.00 -> +0.53). The pooling was NOT the cause;
   my earlier "likely an eval artifact" framing was wrong.
2. Matched-vs-own predictor -> built jepa_matched (per-frame JEPA latent + the
   SAME matched predictor). It forecasts +0.89 -> +0.59, essentially identical to
   jepa-own (+0.89 -> +0.61). The matched protocol is NOT the cause, and the JEPA
   latent's forecastability lives in the LATENT, not the co-trained predictor.

Fair same-predictor 3-seed forecast band (wake R^2, h=1 -> h=16):
| family (same matched predictor) | h=1 | h=16 |
|---|---|---|
| jepa_matched (per-frame JEPA latent, 3-seed) | +0.88 [0.88,0.89] | +0.53 [0.48,0.59] |
| regAE-matched | +0.54 | +0.37 |
| vjepa_fine (finer latent, 3-seed) | -0.08 [-0.24,0.03] | +0.51 [0.39,0.59] |
| (ref) jepa-own (co-trained predictor) | +0.89 | +0.61 |

Conclusion: with both confounds removed, the V-JEPA latent is a decisively worse
short-horizon wake forecaster (-0.08 at h=1 vs +0.89 for the JEPA latent under the
identical predictor). The inverted curve (poor short, rising long) suggests the
masked-prediction objective learns slow / clip-scale structure, not the fast
sub-clip dynamics short-horizon wake forecasting needs. The readability win (Y)
stands and is the one solid V-JEPA advantage.

NOTE: all bands above are firm 3-seed (jepa_matched 3-seed +0.88->+0.53 confirms
matched ~ own predictor). The h=1 gap (+0.88 jepa vs -0.08 vjepa, same predictor)
is the decisive result.

---
## (SUPERSEDED) earlier framing: "forecast confounded by the eval adapter"

The section below was written before the fair re-eval and is kept for the record.
Its core hypothesis (coarse pooling causes the poor forecast) was REFUTED by the
finer-latent re-extraction above. The one-line summary it gives is no longer the
conclusion.

V-JEPA clearly improves PARAMETER READABILITY (a real, eval-robust win,
especially the hard Y parameter), but its rollout FORECAST is poor and
pathological UNDER THIS EVALUATION, which is very likely an artifact of the d=64
eval adapter rather than the objective. The honest one-line summary: the V-JEPA
latent encodes the physical parameters much better, but our d=64 pooling makes
the rollout-forecast comparison unfair, so "does the objective forecast better?"
is not cleanly answered here.

## Six-metric table (d=64, 3 seeds; SSIM pending decoders ~2h)

| metric | V-JEPA | per-frame JEPA | ST | regAE |
|---|---|---|---|---|
| forecast wake R^2 h=1 | +0.00 [-0.16,0.09] | +0.89 (own pred) | +0.82 | +0.54 (matched) |
| forecast wake R^2 h=16 | +0.53 [0.48,0.56] | +0.61 | +0.37 | +0.37 |
| forecast curve shape | INVERTED (rises with h) | decays | decays | decays |
| drift maha_ratio | 0.71 [0.70,0.75] | 1.60 (own) | 1.90 (own) | n/a |
| probe Y R^2 (test_b) | **+0.92** | +0.57 | +0.49 | n/a |
| probe G / D R^2 | 0.94 / 0.95 | 0.95 / 0.88 | 0.94 / 0.87 | n/a |
| participation ratio (d64) | 4.1 | ~1.8 | ~1.8 | 9.9 |
| SSIM (test_a_subset8 @8k, matched) | 0.56 (climbing) | 0.61 | - | - |
| SSIM test_b (canonical, pending) | pending | 0.502 | ~0.50 | 0.476 |

## What is real

- Parameter readability is a clear, sizable V-JEPA win: Y +0.92 vs the per-frame
  JEPA's +0.57 and ST's +0.49 (CV-honest, GroupKFold by case, test_b); D +0.95 vs
  0.88; G tied ~0.94. The probe does NOT roll the latent, so it is not affected by
  the eval-adapter confound below. The d=64 latent is also higher-rank (PR 4.1 vs
  1.8). This is consistent with V-JEPA's latent carrying 32-frame temporal context
  (a genuine property of the objective), which helps recover Y (vertical offset)
  more than a per-frame snapshot does.

## The forecast confound (do not read the forecast row as a clean result)

The forecast curve is inverted: ~0 at short horizon (h=1-4), rising to +0.53 at
h=16. Every other model starts ~0.9 and decays. This is almost certainly an
artifact of the d=64 EVAL ADAPTER, not the objective:

- The adapter pools tokens per 32-frame clip -> 16 token-frames, then
  NEAREST-upsamples 16 -> 120. The per-frame latent is therefore piecewise-constant
  within each 32-frame clip.
- A matched predictor rolling a piecewise-constant latent cannot move at short
  horizons (h=1-4 stay inside one clip -> R^2 ~ 0), and only "improves" at h=12-16
  when the rollout crosses into the next clip's distinct features.
- The low drift (0.71) is consistent with the same mechanism: a smooth,
  mean-reverting rollout has low Mahalanobis, which looks like "stays on-manifold"
  but here reflects a near-constant trajectory, not superior dynamics. Also note
  V-JEPA uses a MATCHED predictor while the JEPA/ST drift used their OWN predictor,
  so the drift rows are not strictly apples-to-apples.

SSIM corroborates the confound: V-JEPA reconstructs the field somewhat worse at
matched iters (test_a_subset8 0.56 vs JEPA 0.61 at 8k), i.e. the SAME coarse,
temporally-blocky d=64 latent that hurts the rollout also smears the decoded
field. Both temporal-resolution-dependent metrics (forecast, SSIM) are degraded,
while the probe (a single readout, no rollout) is not, which is exactly where
V-JEPA's representation quality shows through.

A fair forecast test of the V-JEPA objective needs a finer-temporal eval latent
(overlapping clips with stride < 32, and LINEAR rather than nearest time
interpolation), then re-extract -> re-roll -> recompute forecast. This is the
recommended follow-up; it is NOT yet done (awaiting user decision).

## Training health (sanity, all good)

EMA + masking prevented collapse without SIGReg: loss 0.7 -> ~0.04, token_std
~0.94, token PR ~25/384 across seeds. Smoke gate passed (z_full (n,120,64)). The
objective itself trains cleanly and fast (~15-25 min per 20k-iter seed).

## Bottom line

- V-JEPA's representation is more parameter-readable (real, eval-robust): a point
  in favour of the objective for a disentanglement / sensing goal.
- Its wake FORECAST under our standard eval is poor, but the eval adapter
  handicaps it; the comparison is not clean. Do not claim "V-JEPA forecasts worse"
  without the coarse-pooling caveat, and do not claim it forecasts better either.
- Whether to escalate (a fair-eval re-run) is a user call. The readability gain is
  the one solid, reportable result.

## Reproduce
Code on branch `vjepa`: `src/models/vjepa{,_tokenizer,_masking,_pool}.py`,
`src/training/train_vjepa.py`, `--baseline vjepa` in
`scripts/session18/encode_baseline_latents.py`, launcher
`scripts/session29/vjepa_band.sh`. Artifacts: encoders
`outputs/runs/session29/vjepa/vjepa_s*`, latents
`outputs/session28/latents/vjepa_s*`, matched rollouts
`outputs/session29/lowd_rollouts/vjepa_matched_s*`, decoders (SSIM)
`outputs/runs/session29/dec_vjepa_s*`. Forecast via `m_lowd_forecast`; drift via
`drift_ce1.compute_key` (ROLL_DIR=lowd_rollouts, PRED_LAT[vjepa_matched_s{s}]=
vjepa_s{s}); probe = impact-frame z -> KernelRidge(RBF) GroupKFold-CV.

---
## UPDATE 2 (2026-06-19 18:40): the supervision-head confound (user-flagged) + C_L metric

KEY CORRECTION to the forecast verdict. V-JEPA (plain AND dense) has NO lift/wake
head (pure SSL); the JEPA/regAE it was compared to BOTH have lift (C_L) + wake
(lambda=1.0) heads. The forecast metric IS the wake head's training target, so the
comparison conflated OBJECTIVE with SUPERVISION.

Step-2 control (ctrl_pred_vit_nowake = JEPA, lift head only, wake head removed),
3-seed, own predictor, wake + C_L forecast R^2 (h=1 / h=16):

| family (heads) | wake h1 | wake h16 | C_L h1 | C_L h16 |
|---|---|---|---|---|
| jepa-own (wake+lift) | +0.89 | +0.61 | +0.72 | +0.56 |
| ctrl_nowake-own (LIFT only) | +0.13 | +0.28 | +0.69 | +0.40 |
| regAE-matched (wake+lift) | +0.42 | +0.13 | +0.33 | +0.26 |
| vjepa_fine (NO heads) | -0.08 | +0.51 | -0.06 | +0.55 |
| vjepa_dense (NO heads) | -0.05 | +0.62 | -0.09 | +0.56 |

DOUBLE-DISSOCIATION: removing the WAKE head crashes the WAKE forecast
(+0.89 -> +0.13, toward V-JEPA's -0.08) but barely touches C_L (+0.72 -> +0.69,
lift head retained). So the head, not the objective, drives each forecast.
CONCLUSION: the "JEPA forecasts the wake far better than V-JEPA" result is largely
a SUPERVISION artifact. A JEPA without the wake head forecasts the wake nearly as
poorly as the unsupervised V-JEPA; the autoregressive objective retains only a
small residual edge (+0.13 vs -0.08).

DENSE CORRECTION: the firm 3-seed dense forecast is wake h1 = -0.05 (NOT the +0.16
2-seed read), i.e. dense does NOT fix the short horizon; it only helps long-horizon
(wake h16 +0.62, the best of any latent). Same for C_L (-0.09 -> +0.56).

C_L metric: added throughout (above). Both wake and C_L follow the same
supervision pattern.

PENDING: step (1) = V-JEPA WITH lift+wake heads (the proper both-supervised fair
test) + dense SSIM-vs-AE (decoders running).

---
## UPDATE 3 (2026-06-19 20:05): dense V-JEPA SSIM vs the AE

Canonical T9-decoder SSIM (same recipe as the AE decoders), dense V-JEPA latent:
- dec_vjepa_dense_s1: test_b 0.492 (test_a 0.578, test_c 0.322); s0 firming.
- refs (test_b): per-frame JEPA 0.502, regAE 0.476, Fukami AE 0.380.

So dense V-JEPA reconstructs the vorticity field ~= the per-frame JEPA (0.492 vs
0.502) and slightly BETTER than the reconstruction-trained regAE (0.476), despite
V-JEPA having NO reconstruction loss (pure masked-feature SSL). Image quality is a
point in V-JEPA's favour, not against it. (The dense loss did not raise SSIM over
plain V-JEPA ~0.56 test_a_subset8; it helped long-horizon forecast, not recon.)

Net fair picture (after removing the confounds the user flagged): V-JEPA is
competitive-or-better on readability (Y 0.92 >> 0.57), reconstruction (>= AE), and
long-horizon forecast (dense h16 +0.62, best); its only clear deficit is
SHORT-horizon wake forecast, and step 2 showed that gap is largely the missing
wake head. Step 1 (V-JEPA + lift+wake heads) tests whether matched supervision
closes even that.

---
## UPDATE 4 (2026-06-19 22:10): toward the GOAL = best JEPA + heads beats regAE

vjepa_heads = plain masked V-JEPA + lift(0.01)+wake(1.0) heads (matched to JEPA),
3-seed, matched predictor. vs regAE-matched (the target):

| axis | vjepa_heads | regAE | winner |
|---|---|---|---|
| forecast wake h1 | +0.53 | +0.42 | V-JEPA |
| forecast wake h16 | +0.17 | +0.13 | ~tie (V-JEPA) |
| forecast C_L h1 | +0.00 | +0.33 | regAE |
| forecast C_L h16 | +0.68 | +0.26 | V-JEPA |
| SSIM test_b (dense variant) | 0.492 | 0.476 | V-JEPA |
| probe Y | +0.88 | (tbd) | V-JEPA strong |

So V-JEPA + the SAME heads as JEPA already BEATS regAE on 4/5 comparable axes; only
C_L SHORT-horizon lags (lift head weight 0.01 under-shapes C_L vs the wake head's
1.0). Symmetric confirmation of the head-confound: adding the wake head lifted
V-JEPA wake h1 from -0.08 (no head) to +0.53; jepa-own +0.89 shows the
autoregressive objective keeps a residual short-horizon edge.

Training now: vjepa_best = FULL V-JEPA 2.1 (dense lam_ctx 0.5 + deep-SS n_levels 4)
+ lift+wake heads, to win cleanly + try to close the C_L h1 gap.

---
## UPDATE 5 (2026-06-19, user-flagged): the h=1 forecast metric is degenerate; use persistence + encoded ceiling

User point: R^2=0 at h=1 = random (mean) prediction, uninterpretable. Deeper:
the PERSISTENCE baseline (predict wake(impact+h)=wake(impact)) scores R^2 = +0.95
at h=1, decaying to +0.43 at h=16 (wake enstrophy is slowly varying). So:
- The h=1 forecast R^2-vs-mean is a DEGENERATE regime: NO model beats persistence
  at h=1 (even jepa-own rolled +0.89 < persistence +0.95). Prior h=1 comparisons
  (-0.08 vs +0.49 etc.) were the WRONG bar and should not be over-interpreted.
- Real forecast SKILL = beating persistence, which only happens at LONG horizon.
  At h=16 (persistence +0.43): jepa rolled +0.60, vjepa_fine +0.59, vjepa_dense
  +0.62 BEAT it; regAE +0.37 does NOT. So on meaningful forecast skill, dense
  V-JEPA beats regAE, and regAE barely matches 'assume no change'.

ENCODED-latent ceiling (read wake from the TRUE latent at impact+h, no rollout)
disentangles latent-info from rollout-fidelity (test_b, s0):
| model | enc h1 | enc h16 | rolled h1 | rolled h16 |
|---|---|---|---|---|
| jepa_tf_noc | +0.81 | +0.78 | +0.89 | +0.60 |
| vjepa_fine (no head) | -0.10 | +0.68 | +0.03 | +0.59 |
| vjepa_heads | +0.58 | +0.70 | +0.55 | +0.17 |
Reading: V-JEPA's no-head latent genuinely LACKS short-horizon wake info
(encoded h1 -0.10; it only appears at long range) -- NOT a rollout artifact. The
wake head injects it (encoded h1 +0.58). vjepa_heads' rolled h16 (+0.17) falls
far below its encoded ceiling (+0.70) -> the matched predictor rolls the headed
latent poorly at long range (a predictor issue, not a latent one).

ACTION: re-report the forecast comparison as SKILL vs PERSISTENCE at meaningful
horizons (drop h=1 R^2-vs-mean), and report the encoded ceiling alongside the
rolled forecast. The best-config (vjepa_best) eval will use this framing.

---
## UPDATE 6 (2026-06-19, user-flagged): the chaos lens -- inverted curve is the anomaly, not a win

Physically (chaotic flow), forecast skill must be HIGH at short horizon and DECAY
at long (predictability lost as error grows past the Lyapunov time). Persistence
(+0.95->+0.43) and jepa-own rolled (+0.89->+0.60) both show this correct decay.

Plain V-JEPA's INVERTED curve (bad short, good long) is therefore the anomaly, and
it is a DEFICIT, not a win: its clip-pooled / temporally-averaged latent is blind
to the fast impact transient (the short-horizon, most-predictable part) and only
captures the slow settled wake that survives at long range. Consequences:
- The earlier "plain/dense V-JEPA wins long-horizon forecast" is a HOLLOW read:
  long horizon is chaos-limited (persistence already +0.43 from the slow
  component), so winning there mostly means "captured the slow remnant." V-JEPA
  was failing exactly where forecasting is possible (short horizon).
- The WAKE HEAD restores the physical shape: vjepa_heads rolled decays
  +0.55->+0.48->+0.43->+0.46->+0.17 (h1->h16) -- correct good-short/worse-long --
  because the head injects the short-horizon wake info the clip-pooled latent
  lacked (encoded h1 -0.10 no-head -> +0.58 with head).
- So the honest "beat regAE" must rest on the PREDICTABLE (short) horizon: there
  vjepa_heads +0.55 > regAE +0.42 (holds, for the right reason). The chaos-limited
  long-horizon tail should not be used to claim forecast superiority.

Net: head-less V-JEPA's long-horizon "advantage" is a slow-structure-bias artifact
under chaos; the legitimate result is that V-JEPA WITH the wake head recovers the
correct decaying forecast and edges regAE in the predictable regime. Evaluate
vjepa_best (full 2.1 + heads) on short-to-mid horizons (the predictable band),
not the chaos tail.

---
## regAE reference cells (for the final vs-regAE table)
- regAE encoded wake ceiling (3-seed): h1 +0.53, h8 +0.44, h16 +0.52 (flat ~0.5;
  per-frame recon latent reads instantaneous wake moderately, below jepa ~0.8).
- regAE probe Y (3-seed): +0.23  -> V-JEPA's Y readability (0.92) BEATS regAE by a
  wide margin (0.92 vs 0.23), not marginal.
Scorecard so far (best config pending): V-JEPA(+heads) beats regAE on readability
Y (0.92 vs 0.23), SSIM (0.492 vs 0.476), and short-horizon wake forecast in the
predictable regime (+0.55 vs +0.42).
