# Track I: strictly causal pressure -> state recovery (F6, v2.1, test_b)

Pressure window (K=8 qDEIM taps, W=30 frames) -> readout latent z(impact+16); KernelRidge(RBF), (alpha,gamma) by GroupKFold(5)-by-case, fit on train, evaluated once on test_b with a case-clustered bootstrap CI. The qDEIM placement is target-blind and computed on the strictly pre-impact window-mean snapshot (causal), then reused for every window so only the window POSITION varies.

Targets: **latent_state** (PRIMARY, variance-weighted R^2 over d outputs), params [G,D,Y], impact C_L (context), and wake enstrophy via the frozen closure probe (a known NON-RESULT, reported with no recovery claim).

## latent_state R^2 (PRIMARY) -- family x window, test_b case-clustered CI

| window | strictly pre-impact | jepa_tf_noc | fukami | pod | predictive most recoverable? |
|---|---|---|---|---|---|
| `preimpact_m30_to_m1` | True | +0.822 [+0.660,+0.870] | +0.765 [+0.301,+0.892] | +0.224 [+0.020,+0.317] | YES |
| `through_impact_m29_to_0` | False | +0.830 [+0.662,+0.881] | +0.769 [+0.291,+0.891] | +0.230 [+0.017,+0.331] | YES |
| `readout_m13_to_p16` | False | +0.625 [+0.220,+0.737] | +0.762 [+0.361,+0.863] | +0.362 [+0.108,+0.550] | no |

## context + non-result targets (test_b, R^2 [CI]) at each window

| window | family | params [G,D,Y] | impact C_L | wake via latent (NON-RESULT) |
|---|---|---|---|---|
| `preimpact_m30_to_m1` | jepa_tf_noc | +0.934 [+0.818,+0.963] | +0.778 [+0.362,+0.956] | +0.216 [-2.139,+0.386] |
| `preimpact_m30_to_m1` | fukami | +0.934 [+0.818,+0.963] | +0.778 [+0.362,+0.956] | +0.250 [-6.764,+0.725] |
| `preimpact_m30_to_m1` | pod | +0.934 [+0.818,+0.963] | +0.778 [+0.362,+0.956] | -0.042 [-8.246,+0.335] |
| `through_impact_m29_to_0` | jepa_tf_noc | +0.936 [+0.826,+0.963] | +0.780 [+0.371,+0.957] | +0.226 [-1.637,+0.379] |
| `through_impact_m29_to_0` | fukami | +0.936 [+0.826,+0.963] | +0.780 [+0.371,+0.957] | +0.273 [-6.517,+0.744] |
| `through_impact_m29_to_0` | pod | +0.936 [+0.826,+0.963] | +0.780 [+0.371,+0.957] | -0.065 [-8.374,+0.312] |
| `readout_m13_to_p16` | jepa_tf_noc | +0.787 [+0.552,+0.959] | +0.919 [+0.759,+0.988] | +0.229 [-0.646,+0.362] |
| `readout_m13_to_p16` | fukami | +0.787 [+0.552,+0.959] | +0.919 [+0.759,+0.988] | +0.434 [-3.152,+0.735] |
| `readout_m13_to_p16` | pod | +0.787 [+0.552,+0.959] | +0.919 [+0.759,+0.988] | -0.438 [-14.392,+0.186] |

## Verdict

Predictive family: `jepa_tf_noc`. Strictly-causal window: `preimpact_m30_to_m1`.

Predictive-latent ordering (predictive family has the strictly highest latent_state R^2) holds per window: `preimpact_m30_to_m1`=True, `through_impact_m29_to_0`=True, `readout_m13_to_p16`=False.

Ordering holds under the strictly-causal window: **True**.

**Verdict: STRONG.** Pressure -> state recovery stays a MAIN result, now reported with the strictly-causal preimpact_m30_to_m1 window.

STRONG iff the predictive latent (jepa_tf_noc) has the strictly highest variance-weighted latent-state R^2 (test_b) under the strictly-causal preimpact_m30_to_m1 window. case-clustered R^2 CIs are wide at 10 test_b cases; weigh the sign/ordering consistency alongside the point R^2.

qDEIM taps (K=8, causal placement): [7, 10, 11, 12, 26, 69, 106, 167].
