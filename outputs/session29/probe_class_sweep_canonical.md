# Track D: probe-class robustness (v2.1, wake enstrophy, H=16, test_b)

Readout-frame probe, nested grouped-CV by case, case-clustered bootstrap CI.

| family | probe | R^2 | 95% CI | MAE/case |
|---|---|---|---|---|
| jepa_tf_noc | linear | +0.796 | [-0.334, +0.847] | 28.624 |
| jepa_tf_noc | kernel_ridge_rbf | +0.846 | [-0.300, +0.894] | 22.426 |
| jepa_tf_noc | mlp | +0.402 | [-5.239, +0.254] | 55.078 |
| jepa_tf_noc | gbm | +0.681 | [-1.424, +0.798] | 28.298 |
| fukami | linear | +0.095 | [-3.663, +0.248] | 50.128 |
| fukami | kernel_ridge_rbf | +0.009 | [-2.816, +0.055] | 57.201 |
| fukami | mlp | -0.398 | [-5.049, -0.170] | 65.184 |
| fukami | gbm | +0.069 | [-2.501, +0.139] | 50.413 |
| pod | linear | -0.201 | [-5.279, -0.007] | 64.990 |
| pod | kernel_ridge_rbf | -0.249 | [-8.309, +0.035] | 67.963 |
| pod | mlp | +0.172 | [-3.091, +0.321] | 49.831 |
| pod | gbm | -0.138 | [-6.629, +0.196] | 53.049 |

## Matched probe-class verdict

| probe | jepa | fukami | pod | jepa leads? |
|---|---|---|---|---|
| linear | +0.796 | +0.095 | -0.201 | yes |
| kernel_ridge_rbf | +0.846 | +0.009 | -0.249 | yes |
| mlp | +0.402 | -0.398 | +0.172 | yes |
| gbm | +0.681 | +0.069 | -0.138 | yes |

**Verdict: STRONG.** Predictive latent leads under ['linear', 'kernel_ridge_rbf', 'mlp', 'gbm']; reversed under [].

STRONG = predictive latent leads under EVERY probe class (broad readability). WEAK = a nonlinear probe closes/reverses the ordering => pin claim to LINEAR decodability + organization. NB: case-clustered R^2 CIs are very wide at 10 test_b cases, so the robust signal is the SIGN consistency across probe classes and the per-case MAE, not the individual R^2 magnitudes.
