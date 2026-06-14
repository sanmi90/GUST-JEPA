# Track D: probe-class robustness (v2.1, wake enstrophy, H=16, test_b)

Readout-frame probe, nested grouped-CV by case, case-clustered bootstrap CI.

| family | probe | R^2 | 95% CI | MAE/case |
|---|---|---|---|---|
| jepa_tf_noc | linear | +0.584 | [-4.517, +0.856] | 94.883 |
| jepa_tf_noc | kernel_ridge_rbf | +0.677 | [-3.881, +0.920] | 75.278 |
| jepa_tf_noc | mlp | +0.423 | [-4.836, +0.713] | 121.775 |
| jepa_tf_noc | gbm | +0.356 | [-1.336, +0.661] | 92.429 |
| fukami | linear | +0.339 | [-8.010, +0.793] | 113.177 |
| fukami | kernel_ridge_rbf | +0.445 | [-5.182, +0.830] | 100.230 |
| fukami | mlp | +0.076 | [-4.445, +0.516] | 134.429 |
| fukami | gbm | +0.518 | [-0.752, +0.858] | 90.555 |
| pod | linear | +0.112 | [-8.837, +0.767] | 135.439 |
| pod | kernel_ridge_rbf | +0.323 | [-6.875, +0.780] | 120.342 |
| pod | mlp | +0.250 | [-4.862, +0.914] | 96.856 |
| pod | gbm | +0.543 | [-2.174, +0.806] | 88.545 |

## Matched probe-class verdict

| probe | jepa | fukami | pod | jepa leads? |
|---|---|---|---|---|
| linear | +0.584 | +0.339 | +0.112 | yes |
| kernel_ridge_rbf | +0.677 | +0.445 | +0.323 | yes |
| mlp | +0.423 | +0.076 | +0.250 | yes |
| gbm | +0.356 | +0.518 | +0.543 | NO (reversed) |

**Verdict: WEAK.** Predictive latent leads under ['linear', 'kernel_ridge_rbf', 'mlp']; reversed under ['gbm'].

STRONG = predictive latent leads under EVERY probe class (broad readability). WEAK = a nonlinear probe closes/reverses the ordering => pin claim to LINEAR decodability + organization. NB: case-clustered R^2 CIs are very wide at 10 test_b cases, so the robust signal is the SIGN consistency across probe classes and the per-case MAE, not the individual R^2 magnitudes.
