# Track G: stronger physical floors vs the predictive latent (v2.1, wake enstrophy, H=16, test_b)

Readout-frame fit at impact+H, fit on train, evaluated on test_b, GroupKFold(5)-by-case alpha selection, case-clustered bootstrap CI. Per row the best of ['ridge', 'kernel_ridge_rbf'] is reported.

| kind | name | R^2 | 95% CI | MAE/case | probe | n_feat |
|---|---|---|---|---|---|---|
| floor | gdy | +0.441 | [-1.004, +0.675] | 94.341 | kernel_ridge_rbf | 3 |
| floor | gdy_history | +0.250 | [-2.339, +0.555] | 121.778 | kernel_ridge_rbf | 23 |
| floor | persistence | +0.151 | [-3.433, +0.327] | 148.544 | ridge | 1 |
| floor | pressure_only | +0.242 | [-6.873, +0.909] | 105.964 | kernel_ridge_rbf | 192 |
| predictive-latent | jepa_tf_noc | +0.677 | [-3.881, +0.920] | 75.278 | kernel_ridge_rbf | 64 |
| latent (context) | fukami | +0.445 | [-5.182, +0.830] | 100.230 | kernel_ridge_rbf | 64 |
| latent (context) | pod | +0.323 | [-6.875, +0.780] | 120.342 | kernel_ridge_rbf | 64 |

## Skipped floors

- `pod_dmd_phase`: POD/DMD phase reconstruction is out of scope for this track; the linear POD floor is already represented by the 'pod' latent family context row.

## Verdict

Predictive latent (`jepa_tf_noc`) R^2 = +0.677. Best floor = `gdy` at R^2 = +0.441. Latent advantage over the best floor = +0.236.

Floors the latent clears: ['gdy', 'gdy_history', 'persistence', 'pressure_only'].
Floors the latent does NOT clear: [].

**Verdict: STRONG.**

STRONG = the predictive latent (jepa_tf_noc) wake-enstrophy R^2 exceeds EVERY physical floor; the latent advantage is the latent minus the best floor. WEAK = some cheap floor (most likely gdy_history or persistence) matches or beats the latent, so the latent's wake claim shrinks to the increment over that floor. NB: case-clustered R^2 CIs are wide at 10 test_b cases; weigh the sign consistency and per-case MAE alongside the point R^2.
