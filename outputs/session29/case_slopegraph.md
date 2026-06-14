# SESSION29 Track C-min: per-case wake-enstrophy slopegraph

Headline cell: wake_enstrophy representational readout (ridge) at H=16 on test_b (split v2.1). Predictive = `jepa_tf_noc_d64_s42`; reconstructive = `fukami_d64_s42`. Source: dump.

**VERDICT: WEAK**  (case-level confirms = False; encounter-level confirms = True)

## Per-case mean absolute error (normalised wake-enstrophy units)

| case | n_enc | reconstructive | predictive (JEPA) | delta (fuk-jepa) | JEPA wins |
|---|---|---|---|---|---|
| G+0.50_D1.50_Y+0.00 | 4 | 102.090 | 14.074 | +88.016 | yes |
| G+1.00_D0.50_Y+0.40 | 4 | 47.459 | 13.091 | +34.368 | yes |
| G+1.50_D1.50_Y+0.10 | 4 | 86.007 | 38.765 | +47.242 | yes |
| G+2.00_D0.50_Y+0.10 | 6 | 57.447 | 15.373 | +42.073 | yes |
| G+3.00_D1.00_Y+0.10 | 4 | 33.168 | 72.973 | -39.805 | no |
| G-0.50_D1.00_Y-0.40 | 4 | 36.055 | 14.354 | +21.701 | yes |
| G-1.00_D1.00_Y-0.20 | 4 | 41.475 | 43.433 | -1.958 | no |
| G-1.50_D0.50_Y-0.20 | 4 | 7.193 | 28.795 | -21.602 | no |
| G-2.00_D1.00_Y-0.40 | 4 | 88.796 | 30.891 | +57.905 | yes |
| G-3.00_D1.50_Y-0.10 | 4 | 154.171 | 50.733 | +103.437 | yes |

## Case-level paired test (delta = reconstructive - predictive, >0 favours JEPA)

- cases where JEPA is better: 7/10
- exact one-sided sign test p = 0.1719
- Wilcoxon signed-rank (one-sided) stat = 47.0, p = 0.0244140625
- median per-case delta = 38.221
- mean per-case delta = 33.138

## Encounter-level sign test (context only; 42 encounters)

- JEPA better in 30/42 encounters, p = 0.0040
- mean per-encounter delta = 33.563

We do NOT use the case-permutation p here: it is a dependence test, degenerate for a paired location shift (the B6 lesson).

Provenance: git 033ea11d2cd8, 2026-06-14T12:28:42.190710+00:00.
