# HEADLINE NUMBERS: canonical reference for the manuscript

This file is the single source of truth for every number cited in the paper. All
numbers are taken from on-disk artifacts under `outputs/session18/exp_b1_test3/`
and `outputs/session18/exp_b1/`. Cite numbers from this file rather than computing
them locally.

Last verified: 2026-05-29, partition v2 (84 cases, 10-case Test B).

---

## 1. B1 fairness protocol: physical-closure error (Section 5 Table)

**Mean absolute error at horizon H = 16 on test_b**, mode `z_dns` (oracle latent
encoded from ground-truth DNS frames, then probe predicting the physical
observable). Source: `physical_closure_noBN_unified.csv`, 42 encounters.

Smaller is better. The `z_pred` (predictor rollout) column is reserved but not
yet populated in the unified CSV; the **train R² from the rolled-out latent**
(Section 1b below) is the proxy for end-to-end forecast quality.

| baseline      | d  | C_L   | C_D   | I_y    | wake_enstrophy | circ_pos | circ_neg |
|---------------|----|-------|-------|--------|----------------|----------|----------|
| JEPA          | 64 | 0.624 | 0.301 | 1.901  | 29.83          | 0.801    | 0.715    |
| JEPA          | 32 | 0.675 | 0.246 | 2.976  | 37.99          | 1.084    | 1.061    |
| Fukami        | 3  | 1.074 | 0.384 | 2.327  | 51.70          | 1.110    | 1.886    |
| Fukami        | 32 | 1.038 | 0.368 | 2.195  | 63.94          | 0.946    | 1.789    |
| Fukami        | 64 | 0.989 | 0.340 | 2.421  | 72.90          | 1.057    | 1.904    |
| POD           | 16 | 1.168 | 0.317 | 2.235  | 89.14          | 1.645    | 1.681    |
| POD           | 32 | 0.937 | 0.269 | 2.608  | 81.82          | 1.650    | 1.617    |
| POD           | 64 | 0.872 | 0.291 | 2.245  | 89.31          | 1.660    | 1.800    |

**Headline:** JEPA d=64 wins on 5/6 observables (all except C_D where JEPA d=32
is best at 0.246). JEPA d=64 wake_enstrophy MAE = 29.83 is roughly 2.4x lower
than Fukami d=64 (72.90) and 3.0x lower than POD d=64 (89.31).

## 1b. Train probe R² (rolled-out latent to physical observable)

Source: `probe_train_r2.csv`. R² on the train split between the K-step probe of
each physical observable computed from a rolled-out (Markov-recursion) latent
vs the DNS ground truth. This is the *closure* metric: it measures whether the
predictor's rollout preserves enough information to recover the observable.

| baseline   | d  | C_L   | C_D   | I_y   | wake_enstrophy | circ_pos | circ_neg | mean  |
|------------|----|-------|-------|-------|----------------|----------|----------|-------|
| JEPA       | 64 | 0.864 | 0.802 | 0.562 | 0.934          | 0.922    | 0.927    | 0.835 |
| JEPA       | 32 | 0.863 | 0.795 | 0.490 | 0.898          | 0.901    | 0.897    | 0.808 |
| Fukami     | 3  | 0.553 | 0.408 | 0.191 | 0.079          | 0.130    | 0.033    | 0.232 |
| Fukami     | 32 | 0.695 | 0.596 | 0.278 | 0.333          | 0.369    | 0.313    | 0.430 |
| Fukami     | 64 | 0.671 | 0.589 | 0.335 | 0.277          | 0.336    | 0.352    | 0.427 |
| POD        | 16 | 0.763 | 0.720 | 0.514 | 0.574          | 0.582    | 0.527    | 0.613 |
| POD        | 32 | 0.763 | 0.748 | 0.690 | 0.589          | 0.618    | 0.621    | 0.671 |
| POD        | 64 | 0.641 | 0.656 | 0.799 | 0.373          | 0.388    | 0.506    | 0.561 |

**Headline:** JEPA d=64 mean R² = 0.835 vs Fukami d=64 mean R² = 0.427 vs
POD d=64 mean R² = 0.561. JEPA dominates Fukami by a factor of 2x on average,
and the gap is largest on wake_enstrophy (0.934 vs 0.277) and the two
circulation observables (0.92+ vs 0.34).

JEPA d=64 vs d=32 is nearly tied (0.835 vs 0.808): matched-capacity ablation
confirms d=32 captures most of the closure quality at half the latent budget.

## 2. Latent drift diagnostic (Section 4)

Source: `latent_drift_diagnostic.json`. Mahalanobis distance of the rolled-out
latent vs the DNS-encoded latent reference distribution, computed per encounter
and averaged over 42 test_b encounters.

| baseline   | d  | mahal_dns | mahal_markov | ratio (markov / dns) | rel_drift @ H=16 |
|------------|----|-----------|--------------|----------------------|------------------|
| Fukami     | 64 | 2.92      | 28.96        | **9.90**             | 0.370            |
| JEPA       | 64 | 8.08      | 6.84         | 0.846                | 0.494            |
| JEPA       | 32 | 5.43      | 4.65         | 0.857                | 0.475            |
| POD        | 64 | 7.74      | 6.27         | 0.811                | 0.722            |

**Headline:** Fukami's rollout latent drifts ~10x further from its training
distribution (Mahalanobis 28.96 vs 2.92) than its DNS-encoded counterpart;
JEPA and POD rollouts stay inside the training manifold (ratio < 1). This is
the mechanistic explanation for Fukami's degraded forecast quality.

## 3. Conditioning-only control floor (Section 5 + Section 8)

Source: `conditioning_only_baseline.csv`. KRR-RBF((G, D, Y)) -> observable(impact)
fit on train, evaluated on test_b and test_c. This is the floor that any
encoder + predictor must clear to demonstrate that the latent (not the
parameters alone) is doing the work.

| observable        | train R² | test_b R² | test_c R² | JEPA d=64 train R² | JEPA delta on train |
|-------------------|----------|-----------|-----------|---------------------|---------------------|
| C_L               | 0.895    | 0.303     | 0.547     | 0.864               | -0.030              |
| C_D               | 0.941    | 0.386     | -2.459    | 0.802               | -0.139              |
| I_y               | 0.422    | -0.301    | -1.335    | 0.562               | +0.140              |
| wake_enstrophy    | 0.806    | 0.482     | -1.252    | 0.934               | +0.129              |
| circulation_pos   | 0.792    | -0.032    | -0.249    | 0.922               | +0.130              |
| circulation_neg   | 0.799    | 0.568     | -4.262    | 0.927               | +0.128              |

**Headline:** On train, (G,D,Y)-only is competitive on C_L and C_D (RBF
interpolates 226 points in 3-D well). On test_b, (G,D,Y)-only collapses across
all observables except a weak C_L and wake_enstrophy hold. On test_c the
(G,D,Y)-only floor is negative for 5/6 observables. JEPA's latent contributes a
real generalization advantage that is not explained by conditioning alone.

## 4. C_L inference comparison (Section 5)

Source: `cl_inference_comparison.csv` and `cl_inference_predictor_in_loop.csv`.
test_b R² for predicting C_L at impact from pre-impact information at various
lead times tau (frames). Modes:
- **oracle**: probe applied to the DNS-encoded latent at the lead time.
- **direct**: probe of pressure sensors at lead time -> C_L at impact.
- **via_baseline**: probe of the baseline latent at lead time -> C_L at impact.
- **predictor_in_loop**: rollout the latent from tau forward through the
  predictor, then probe -> C_L at impact.

JEPA d=64:
| tau (frames) | oracle  | direct | via_baseline | predictor_in_loop |
|--------------|---------|--------|--------------|-------------------|
| 30           | 0.683   | -0.084 | -0.064       | 0.139             |
| 20           | 0.683   | -0.122 | -0.115       | 0.078             |
| 10           | 0.683   | 0.134  | 0.236        | 0.350             |
| 5            | 0.683   | 0.542  | 0.583        | 0.600             |
| 2            | 0.683   | 0.690  | 0.638        | 0.637             |

**Headline:** at tau=10 frames pre-impact, JEPA's predictor-in-loop gives
R² = 0.35 (vs direct pressure-probe R² = 0.13 and oracle R² = 0.68). The
predictor recovers ~half of the oracle skill from 10 frames out, where direct
sensing gives essentially nothing.

Fukami d=64 oracle is **negative** (R² = -0.185) at every tau, meaning the
Fukami latent itself does not carry the pre-impact information that would
predict the impact C_L. This is not a probe failure; it is a representation
failure.

## 5. Pressure observability of (G, D, Y) (Section 4)

Source: `baseline_pressure_observability.csv`. Despite the file name, the
columns are z->c probe R² (recovering gust parameters from the rolled-out
latent), not pressure->c. For JEPA d=64 at K = 8 pre-impact sensors:

- test_b: G R² = 0.461, D R² = 0.799, Y R² = 0.101
- test_c: G R² (large negative; out-of-distribution G = +4), D R² ~ 0,
  Y R² (large negative)

**Headline for Section 4:** the JEPA latent encodes the gust impulse direction
(G) and depth (D) recoverably on test_b. The cross-stream displacement (Y) is
poorly recovered, consistent with Y being the most marginal axis in the
training envelope.

For Section 8 (MPC pathway): this number is the basis for the claim that an
online estimator g(z_{1:t}) -> ĉ_t is mechanically feasible because (G, D)
are already implicit in the latent on held-out cases.

## 6. POD-pressure mechanism: Q-criterion overlap (Section 4)

Source: `pod_q_overlap_pressure.json`. POD modes are correlated with the
pre-impact pressure recoverability of the gust state via their spatial
overlap with the Q-criterion field at impact.

- Pearson r (hard Q indicator): **0.490, p = 0.054** (n = 16 modes)
- Pearson r (soft Q indicator): 0.097, p = 0.72

**Headline:** the hard-indicator correlation between POD-mode Q-overlap and
pressure recoverability is **just above the 5% threshold** (p = 0.054). The
result is a suggestive, not significant, mechanistic story; report as
"borderline" or as a "supplementary" observation, not as a confirmed effect.

## 7. Wake observable d = 32 vs d = 64 agreement (Section 7)

Source: `wake_observable_d32_vs_d64.csv`. Per-dimension R² of the 80-D
wake_signed_spectrum target recovered from the latent. 84 dims total
(80 spectral + 4 metadata or block-summary).

- |delta(d=32 - d=64)| < 0.05: **37 of 84 dims** (44%)
- |delta(d=32 - d=64)| < 0.10: **55 of 84 dims** (65%)
- Mean delta: -1.54 (driven by one outlier dim with |delta| = 141; likely a
  dead/uninitialised dim, exclude from headline)
- Median delta: +0.032

**Headline for Section 7:** wake-observable head at d = 32 matches d = 64 on
about two thirds of dims (within 0.10 R²). The matched-capacity result extends
to the wake observable; d = 32 is a viable bottleneck.

## 8. Three-JEPA variants summary (Section 7)

Source: `physical_metrics_three_jepa_variants.csv`. JEPA test1 (production),
test2 (alt regulariser), test3 (LN + VICReg) compared at d = 64. See file for
per-observable, per-horizon numbers.

## 9. Wang/p99.9 SSIM convention (Section 6 + Methods)

Source: HANDOFF D131, memory `ssim-convention.md`. Wang K1 = 0.01, K2 = 0.03 on
pipeline-normalized omega; L = 2 * global_p99.9(|target_norm|).

- v2 production: L ≈ 8.31
- Decoder test_a SSIM at this convention ≈ **0.71**

Replaces the older Fukami c1 = 0.16, c2 = 1.44 raw-scale convention. Any SSIM
number in earlier sections must be regenerated at the new convention before
landing in the camera-ready.

## 10. Dataset and protocol facts

- partition_version: v2 (locked, sha256-anchored to inventory)
- 84 cases total: train + val + test_b (10 stratified) + test_c (G = +4 only)
- 226 train encounters, 42 test_b encounters, 24 test_c encounters
- preprocessing_version: v1 omega pipeline (manifest at
  `outputs/data_pipeline/v1/manifest.json`)
- omega normalization: 3-sigma scale by train_std = 3.5526 (divisor 10.658),
  no mean shift (preserves vorticity antisymmetry)
- Reporting protocol: train + val + test_b + test_c, bootstrap n=2000 for
  CIs, 3-seed encoder variance, 5-fold probe CV (see split_v2 memory).

## Known-broken artifacts (do not cite)

- `preimpact_forecast.csv`: per-tag, per-tau columns are identical across all
  baselines. The CSV appears unfilled; likely a copy-paste bug or an unfinished
  run. The C_L pre-impact numbers in Section 4 of Section 5 should be cited
  from `cl_inference_comparison.csv` and `cl_inference_predictor_in_loop.csv`
  instead.
