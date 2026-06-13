# Physics Track P1: similarity collapse and its latent image (v2.1)

Generated 2026-06-13T20:24:21.140151+00:00 (CPU, read-only).

## GP1 verdict: **WEAK**

best force collapse (candidate s4_MMF) only reaches held-out R2=0.45 < 0.8; Y modulation dominates

Winning candidate (highest pooled held-out force R^2): **s4_MMF** (Martinez-Muriel & Flores (2020) induced-velocity ratio).

Pooled power-law held-out test_b R^2 of |dCL| vs each candidate:
  - s4_MMF (Martinez-Muriel & Flores (2020) induced-velocity ratio): 0.45
  - s3_Gamma_g (Taylor-vortex core circulation Gamma_g(G,D), numeric): 0.45
  - s2_GD (G*D (proportional to Taylor-profile gust circulation)): 0.45
  - s1_G (G (Kussner-like gust ratio)): 0.40

## Headline numbers (winning candidate, pooled, power-law)
  force |dCL|: exponent p = 1.10 CI ['1.00', '1.19'], held-out R^2 = 0.45 CI ['-0.76', '0.80'], test_c R^2 = -0.02
  latent excursion: exponent p = 0.51 CI ['0.40', '0.62'], held-out R^2 = -0.87 CI ['-4.54', '0.09'], test_c R^2 = -148.39

## Full grid (pooled, both models): held-out R^2 [VRR] / exponent

| response | model | s1_G | s2_GD | s3_Gamma_g | s4_MMF |
|---|---|---|---|---|---|
| force_dcl | linear | 0.41 [0.42] | 0.32 [0.36] | 0.32 [0.36] | 0.36 [0.40] |
| force_dcl | powerlaw | 0.40 [0.41] / p=0.99 | 0.45 [0.45] / p=1.00 | 0.45 [0.45] / p=1.00 | 0.45 [0.45] / p=1.10 |
| wake_enstrophy | linear | 0.35 [0.52] | -0.32 [-0.05] | -0.32 [-0.05] | 0.32 [0.55] |
| wake_enstrophy | powerlaw | 0.56 [0.57] / p=0.92 | 0.38 [0.43] / p=0.89 | 0.38 [0.43] / p=0.89 | 0.67 [0.68] / p=0.92 |
| latent_maha | linear | -0.38 [-0.36] | -0.30 [-0.28] | -0.30 [-0.28] | -0.37 [-0.35] |
| latent_maha | powerlaw | -0.90 [-0.83] / p=0.60 | -1.10 [-1.07] / p=0.54 | -1.10 [-1.07] / p=0.54 | -0.87 [-0.76] / p=0.51 |

## Per-|Y| stratified (power-law held-out R^2, winning candidate)
  force_dcl: absY_0.00: R2=-4.80 (n_tb=4); absY_0.10: R2=0.10 (n_tb=18); absY_0.20: R2=-0.94 (n_tb=8); absY_0.40: R2=0.74 (n_tb=12)
  wake_enstrophy: absY_0.00: R2=-10.02 (n_tb=4); absY_0.10: R2=0.02 (n_tb=18); absY_0.20: R2=-0.55 (n_tb=8); absY_0.40: R2=-3.65 (n_tb=12)
  latent_maha: absY_0.00: R2=-146.08 (n_tb=4); absY_0.10: R2=-9.76 (n_tb=18); absY_0.20: R2=-65.09 (n_tb=8); absY_0.40: R2=0.06 (n_tb=12)

## Latent Mahalanobis orbit geometry (caveat)
  baseline case: Baseline; orbit points: 80
  orbit effective dim: 3.57 / 64; cov condition number: 219680.5
  Mahalanobis on a rank-deficient (~80-pt, eff-dim ~3 of 64) orbit cloud inflates off-orbit directions; reported per the master-plan P1 spec (the s46 HEADLINE return metric is Euclidean). Treat the latent excursion as ordinal across encounters, not as an absolute whitened distance.

## Method notes
  - Candidates are SIGNED strength variables; responses are non-negative magnitudes; fits use |s| (documented in the module docstring).
  - s3 (Gamma_g) is exactly proportional to s2 (G*D); their power-law exponents are identical and their held-out R^2 coincide. The exponent differs from s1 (G) only through the D leverage.
  - test_c (|G|=4) is reported as a secondary extrapolation R^2; it is NEVER used for fitting or candidate selection.
  - Held-out R^2 uses closure_matrix.r2_heldout (SST about the held-out mean); CIs are case-clustered (stats_lib convention, resampling test_b cases).
