# Track E auxiliary leakage: shuffled-label probe sentinel (canonical wake, H=16, test_b)

| family | real R^2 | shuffled R^2 | gap |
|---|---|---|---|
| jepa_tf_noc | +0.796 | +0.008 | +0.787 |
| supervised_only | +0.918 | -0.328 | +1.246 |
| fukami | +0.095 | -0.122 | +0.218 |
| pod | -0.201 | -0.201 | +0.001 |

**Sentinel PASS.** shuffled-label probe sentinel: permute TRAIN wake labels, refit, eval on REAL test_b. shuffled R^2 ~0 confirms the real R^2 is genuine latent->wake structure, not probe overfitting. A training-time shuffled-label encoder control is the GPU complement (deferred).
