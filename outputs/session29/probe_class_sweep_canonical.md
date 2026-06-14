# Track D: probe-class robustness (v2.1, wake enstrophy, H=16, test_b)

Readout-frame probe, nested grouped-CV by case, case-clustered bootstrap CI.

| family | probe | R^2 | 95% CI | MAE/case |
|---|---|---|---|---|
| jepa_tf_noc | linear | +0.796 | [-0.334, +0.847] | 28.624 |
| jepa_tf_noc | kernel_ridge_rbf | +0.846 | [-0.300, +0.894] | 22.426 |
| supervised_only | linear | +0.918 | [-0.165, +0.962] | 20.064 |
| supervised_only | kernel_ridge_rbf | +0.884 | [-0.481, +0.940] | 21.939 |
| ctrl_recon | linear | -0.488 | [-19.453, +0.280] | 81.047 |
| ctrl_recon | kernel_ridge_rbf | -0.528 | [-12.372, -0.038] | 70.673 |
| fukami | linear | +0.095 | [-3.663, +0.248] | 50.128 |
| fukami | kernel_ridge_rbf | +0.009 | [-2.816, +0.055] | 57.201 |
| pod | linear | -0.201 | [-5.279, -0.007] | 64.990 |
| pod | kernel_ridge_rbf | -0.249 | [-8.309, +0.035] | 67.963 |

## Matched probe-class verdict

| probe | jepa | fukami | pod | jepa leads? |
|---|---|---|---|---|
| linear | +0.796 | +0.095 | -0.201 | yes |
| kernel_ridge_rbf | +0.846 | +0.009 | -0.249 | yes |

**Verdict: STRONG.** Predictive latent leads under ['linear', 'kernel_ridge_rbf']; reversed under [].

STRONG = predictive latent leads under EVERY probe class (broad readability). WEAK = a nonlinear probe closes/reverses the ordering => pin claim to LINEAR decodability + organization. NB: case-clustered R^2 CIs are very wide at 10 test_b cases, so the robust signal is the SIGN consistency across probe classes and the per-case MAE, not the individual R^2 magnitudes.
