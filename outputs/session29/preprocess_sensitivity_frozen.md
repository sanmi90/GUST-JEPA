# Track B0.5: frozen-encoder preprocessing sensitivity of the wake result (v2.1, wake enstrophy, H=16, test_b)

Frozen latents, NO retrain. The wake-enstrophy TARGET at readout frame impact+16 is recomputed under three TRAINING-ONLY clip treatments of the raw-cache omega field (pipeline drop mask applied in all three), then the frozen-latent -> wake Ridge probe (GroupKFold(5)-by-case alpha selection, fit train, eval test_b, case-clustered CI) is re-fit per family.

Clip treatments: `none` (no amplitude clip), `per_encounter` (current pipeline p99.99, future-dependent), `training_global` (one leak-free p99.99 over all TRAIN encounters; train-pooled threshold = 94.7026).

## test_b wake-enstrophy R^2 (family x clip treatment)

| family | none | per_encounter | training_global |
|---|---|---|---|
| **jepa_tf_noc** (predictive) | +0.746 [-0.948,+0.853] | +0.796 [-0.334,+0.847] | +0.842 [-0.142,+0.887] |
| fukami | +0.115 [-2.809,+0.265] | +0.095 [-3.663,+0.248] | +0.076 [-3.240,+0.198] |
| pod | -0.271 [-4.695,-0.013] | -0.050 [-3.722,+0.031] | -0.002 [-3.259,+0.080] |

## JEPA-minus-best-baseline wake advantage per treatment

| treatment | JEPA R^2 | best baseline | baseline R^2 | advantage |
|---|---|---|---|---|
| none | +0.746 | fukami | +0.115 | +0.631 |
| per_encounter | +0.796 | fukami | +0.095 | +0.700 |
| training_global | +0.842 | fukami | +0.076 | +0.767 |

Reference advantage (current pipeline, per_encounter): +0.700. Max |advantage shift| across treatments: 0.070 (tolerance 0.1). Sign preserved: True. Magnitude preserved: True.

## Target sanity (enstrophy lowers monotonically as clip tightens)

| split | stat | none | per_encounter | training_global |
|---|---|---|---|---|
| train | mean | 153.5 | 146.2 | 142.2 |
| train | median | 118.2 | 116 | 114.8 |
| train | max | 885.8 | 817.2 | 794.4 |
| test_b | mean | 129.8 | 124.7 | 123.2 |
| test_b | median | 110.1 | 108.6 | 107.9 |
| test_b | max | 546.6 | 533.6 | 501.5 |

## Verdict

**STRONG.** The JEPA-vs-baseline held-out wake advantage keeps its sign and approximate magnitude across all three training-only clip treatments (`none`, `per_encounter`, `training_global`). The wake headline is preprocessing-robust; the Track B0 clip leak does not move the result. A B1 retrain under a leak-free clip is NOT required.
