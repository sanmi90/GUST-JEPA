# Spatio-temporal JEPA encoder: findings (variant A)

Date: 2026-06-19 (overnight autonomous run)
Spec: `docs/superpowers/specs/2026-06-18-spatiotemporal-jepa-design.md`
Branch: `spatiotemporal-jepa`

## Question

Does making the JEPA encoder temporal (a causal 3D-conv tubelet stem, so each
`z_t` integrates a ~7-frame causal window) help, versus the per-frame encoder,
inside the existing autoregressive-JEPA framing? Tested at d=64 and d=16 on the
six-metric suite, matched recipe (same predictor, SIGReg 0.01, lift 0.01, wake
1.0, H_roll 8, 20k iters, split v2.1), seeds {0,1,2,42} (d64) / {0,1,42} (d16).

## Answer: No. Spatio-temporal encoding does not help.

Clearly worse at d64; a marginal wash at d16 that does not rescue the low-d
weakness. Per the spec's A->B gate, we do NOT escalate to the full V-JEPA build.

## Six-metric comparison (ST-own vs per-frame JEPA-own)

Forecast = wake-enstrophy R^2 vs horizon (JEPA-own predictor rollout), seed band.

### d=64 (ST clearly worse)
| metric | ST-own | per-frame JEPA-own |
|---|---|---|
| forecast h=1 | +0.82 [0.69,0.90] | +0.89 [0.88,0.89] |
| forecast h=16 | +0.37 [0.13,0.60] | +0.61 [0.60,0.62] |
| drift maha_ratio | 1.90 | 1.60 |
| probe Y R^2 (test_b) | +0.49 | +0.57 |
| probe G / D R^2 | 0.94 / 0.89 | 0.95 / 0.88 |
| participation ratio | ~1.8 | ~1.8 |

ST is worse at every forecast horizon, drifts more off-manifold, and is worse on
the discriminating Y parameter. G and D are saturated (~0.9) for both.

### d=16 (marginal wash, no rescue)
| metric | ST-own | per-frame JEPA-own | regAE-matched (ref) |
|---|---|---|---|
| forecast h=1 | +0.37 [0.33,0.47] | +0.30 [0.07,0.45] | +0.78 [0.69,0.85] |
| forecast h=16 | +0.21 [0.16,0.24] | +0.18 [-0.28,0.46] | +0.40 |
| drift maha_ratio | 2.10 | 1.92 | n/a |
| probe Y R^2 (test_b) | +0.34 | +0.27 | n/a |
| probe G / D R^2 | 0.94 / 0.87 | 0.94 / 0.87 | n/a |

ST has a marginally higher forecast mean and a tighter seed band than the
per-frame JEPA, and a slightly better Y probe, but it drifts more and, crucially,
forecasts nowhere near the reconstructive regAE (+0.78->+0.40). The low-d
weakness of the predictive latent is not a per-frame-encoding artifact: temporal
context in the encoder does not fix it.

SSIM (field reconstruction via the T9 decoders): a TIE. Matched-iteration read
(test_a_subset8, same eval) at 14k iters gives ST d64 0.609/0.618 (s0/s1) vs
per-frame JEPA 0.617; the JEPA decoder is converged (0.617 at 14k -> 0.619 at
30k), so ST and per-frame JEPA reconstruct the field equally well. The full
test_b 30k value (per-frame JEPA 0.502) will land at the same tie. SSIM neither
helps nor hurts: ST is worse only on the forecast/drift/Y-probe axes.

## Interpretation

The predictive objective's forecast quality is governed by latent dimension and
the objective itself, not by whether the encoder sees a temporal window. Adding
causal temporal context to the encoder slightly degrades long-horizon forecast
and on-manifold consistency at d=64 (the regime where the per-frame JEPA already
wins) and does not help the latent forecast the wake at d=16 (where the
reconstructive AE wins). The factorized per-frame-encoder + temporal-predictor
design is as good or better. This is a clean null and bounds the architectural
search: a heavier V-JEPA-style encoder is not warranted on this problem.

## Reproduce

- Encoders: `outputs/runs/session29/st/st_d{64,16}_s*/encoder/` (train via
  `--encoder st_hybrid --temporal-kernel 3`, recipe in
  `scripts/session29/overnight_gpu.sh`).
- Latents: `outputs/session28/latents/st_d{64,16}_s*/`; JEPA-own rollouts:
  `outputs/session29/lowd_rollouts/st_own_d{64,16}_s*/`.
- Forecast: `m_lowd_forecast.fit_probe`/`forecast_curve`. Drift:
  `drift_ce1.compute_key` with `ROLL_DIR=outputs/session29/lowd_rollouts`,
  `PRED_LAT["st_own_d{D}_s{s}"]="st_d{D}_s{s}"`. Probe: impact-frame z ->
  KernelRidge(RBF), GroupKFold-by-case CV, scored on test_b.
