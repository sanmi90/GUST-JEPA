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
