# Session 17 Plan: Physical closure, trajectory geometry, and closed-loop pressure observability

Date: 2026-05-27
Lead: Carlos Sanmiguel Vila (INTA, UC3M)
Hardware: 2x RTX 6000 Blackwell (sm_120), bf16 mixed precision
Target venue: JFM (primary, realistic for submission in 3-4 months)

---

## Strategic frame

Session 17 converts the Session 16 findings from latent-RMSE statements into fluid-mechanics statements. No new encoder retraining, no predictor retraining (the conditioning-dropout retrain is deferred to Session 18 if Experiment 5 motivates it). Five experiments. Each produces a figure for the JFM paper. Each closes one specific gap in the Session 16 framing.

The paper this session builds toward makes five claims:

**Claim 1.** The impact-frame latent is a privileged dynamical state where parameter information, surface pressure footprint, and near-body vorticity become co-observable. Parameter recoverability decays rapidly away from this frame.

**Claim 2.** The latent trajectories across encounters are organized in a low-dimensional space with physically interpretable arcs, paralleling the Fukami-Taira lift-augmented AE trajectories but predictive rather than reconstructive.

**Claim 3.** Conditional Markov closure holds in physical variables, not just latent RMSE: short-horizon force, vorticity impulse, enstrophy, and spectral content are preserved by Markov-only rollout out to H ≈ 16 frames in-distribution.

**Claim 4.** The pixel-level structures driving the encoded representation, identified by gradient-SHAP and validated by intervention, correspond to physically interpretable connected regions at the leading edge, wake roll-up, and wake centerline. The most localized structure is the leading-edge Y-signature, which sign-flips with vortex offset.

**Claim 5.** Sparse pressure (K = 2 to K = 16 sensors) can drive the Markov rollout with physical metrics within a defined tolerance of the true-DNS-conditioning rollout, closing the deployment loop.

---

## Theoretical anchors

- Fukami, Taira, PRF 10, 084703 (2025): lift-augmented autoencoder reduces vortex-gust flows to three latent variables with interpretable trajectory structure. Adopted as the trajectory-geometry baseline.
- Cremades, Hoyas, Vinuesa, Nat. Commun. 16, 10189 (2025): connected-component coherent structures from gradient-SHAP, with intervention validation.
- Finzi, Qiu, Jiang, Izmailov, Kolter, Wilson, arXiv:2601.03220v2 (2026): epiplexity decomposition of observables (already partially applied in Session 16 via D100, reused here for the trajectory-stage analysis).
- Sessions 11-16 D-entries: production stack, intrinsic dim consensus = 3, conditional Markov closure, pixel-level SHAP structures.

---

## What is and is not in scope

**In scope.** Trajectory geometry across encounters; physical Markov closure (C_L, vorticity impulse, enstrophy, spectrum); per-frame decay of Y-recoverability; SHAP-to-structures via connected components and Q-overlap; closed-loop sparse-pressure rollout; cross-seed agreement of the nonlinear extraction function (not just the score).

**Not in scope.** No retraining of the encoder. No retraining of the predictor (conditioning dropout deferred). No new sensor selection methods. No new architectural variants of the decoder. No extension to other Reynolds numbers (deferred to a future paper or to DICE). The Fukami head-to-head is one row of one table, not a full baseline sweep.

---

## Experiment 1: Trajectory geometry of the latent across encounters

### Hypothesis

The latent trajectories z(t) across encounters trace out organized arcs in a low-dimensional projection. In a properly chosen 3D projection (PCA on impact-frame latents, or supervised PLS against (G, D, Y) + phase), trajectories cluster by perturbation parameters before impact, converge to a parameter-dependent region around impact, and diverge into wake-shedding arcs afterward. The impact frame is a topologically distinct point on each trajectory.

### Inputs

- Production E d=64 encoder. Per-encounter latent trajectories z(t) for t in {0, ..., 87} frames, on Test B (56 encounters), Test C (24 encounters), and a sample of 30 train encounters for context.
- Three Thrust 6 seed retrains, same encounters, for cross-seed comparison.

### Procedure

**Part (a) Projection construction.**
Build three candidate 3D projections of the per-frame latent:
1. PCA on impact-frame latents only (the projection Session 16 used).
2. PCA on the pooled per-frame latents.
3. Supervised PLS against (G, D, Y, sin(2*pi*phase), cos(2*pi*phase)), where phase is the impact-relative frame index normalized to [-1, +1] over the encounter.

Report the variance explained by the first 3 components in each.

**Part (b) Trajectory visualization and classification.**
For each projection, plot the trajectories of 10 representative encounters (5 from Test B at G > 0, 5 at G < 0) in the 3D space, colored by impact-relative phase. Mark the impact frame on each. Compute:
- Pre-impact trajectory length L_pre = sum_{t=0}^{t_impact-1} ||z(t+1) - z(t)||
- Post-impact trajectory length L_post = sum_{t=t_impact}^{T-1} ||z(t+1) - z(t)||
- Pre-impact spatial extent (max distance from initial point)
- Post-impact spatial extent (max distance from impact point)
- Convergence-to-impact metric: ||z(t_impact) - z_center(G, D, Y)||, where z_center is the train-mean impact latent for the nearest (G, D, Y) bin

**Part (c) Topological signature of the impact frame.**
Compute, for each encounter, the local curvature of the trajectory in 3D as a function of t:
kappa(t) = ||z(t+1) - 2 z(t) + z(t-1)|| / (||z(t+1) - z(t-1)||/2)^2
Test the hypothesis that kappa(t) has a peak at t = t_impact across encounters. Report the median curvature profile aligned by impact frame across Test B and Test C, separately for high-|G| and low-|G| encounters.

**Part (d) Cross-seed trajectory agreement.**
For the same 10 representative encounters, compute the trajectory in each of the 4 seeds (production + Thrust 6 seed 0, 1, 2). Each seed's trajectory lives in its own 3D projection (different PLS/PCA basis per seed). To compare across seeds, use a basis-invariant trajectory descriptor: the pairwise distance matrix between consecutive frames within an encounter, normalized by the median distance. Compare these distance matrices across seeds using Spearman correlation.

### Outputs

```
outputs/session17/exp1/projections.npz                       three projection matrices
outputs/session17/exp1/trajectory_descriptors.csv            per-encounter lengths, extents, convergence
outputs/session17/exp1/curvature_profiles.npz                per-encounter kappa(t)
outputs/session17/exp1/cross_seed_distance_corr.json         Spearman across seeds
outputs/session17/figures/exp1_trajectory_panel.png          3D trajectories for 10 encounters
outputs/session17/figures/exp1_curvature_at_impact.png       median kappa(t) aligned by impact
outputs/session17/figures/exp1_cross_seed_distance.png       distance-matrix Spearman across seeds
```

### Compute estimate

Latents already extracted. PLS/PCA closed-form. Curvature is one-liner per encounter. Cross-seed comparison reuses the four seed encoders' impact-frame latents from Session 16. About 3-4 hours wall time.

### Acceptance gate

Claim 2 is supported if (i) trajectories cluster by (G, D, Y) in at least one projection, (ii) the median curvature kappa(t) shows a peak within ±3 frames of t_impact with the peak height at least 2x the off-peak baseline, (iii) cross-seed trajectory distance correlations exceed 0.7 on at least 7 of 10 test encounters. If only (i) and (ii) hold but not (iii), the trajectories are physically meaningful but seed-dependent; if only (i) holds, trajectories are organized but the impact frame is not topologically distinct.

---

## Experiment 2: Physical Markov closure

### Hypothesis

The conditional Markov closure observed in latent RMSE (D119) survives in physical observables: force, vorticity impulse, integrated wake enstrophy, and spectral lambda ratio. Markov-only rollout from z_impact tracks Full-context rollout in these physical variables out to H ≈ 16 frames in-distribution.

### Inputs

The Markov-only, AR-from-impact, and Full-context rollouts already computed in Session 16 (outputs/session16/exp4/markov_closure_per_encounter.npz). Decode every frame through the production SL decoder. The conditioning-true rollouts only (cond = 0 ablation is a separate diagnostic, see Diagnostic D below).

### Procedure

**Part (a) Decode and compute physical metrics.**
For each (rollout mode, encounter, horizon H), decode z(t) to omega(t) through the production SL decoder. Compute, per frame:

- C_L from the force integration over the airfoil surface (your existing force computation)
- C_D same
- Integrated wake enstrophy E_w over a fixed box [x/c ∈ [0.5, 4], y/c ∈ [-1, +1]]
- Positive circulation Gamma_pos = integral over Omega_+ of omega dA, where Omega_+ = {omega > omega_threshold}
- Negative circulation Gamma_neg analogously
- Vorticity impulse I_x = -integral y omega dA, I_y = integral x omega dA
- 2D spectral lambda ratio (the existing metric from your evaluation pipeline)
- Spectral centroid in the wake region

For each metric, compute the rollout error vs DNS ground truth at H ∈ {1, 4, 8, 16, 24, 32, 48, 64, 79}.

**Part (b) Markov-vs-Full-context delta in physical variables.**
For each metric and horizon, compute the per-encounter difference between Markov-only and Full-context. Report the bootstrap 95 percent CI over encounters. The headline number for the paper: the smallest H at which the bootstrap CI of (Markov - Full) excludes zero by more than 10 percent of the metric's standard deviation across encounters.

**Part (c) Vorticity impulse vs lift correlation.**
For each rollout mode, compute the Pearson correlation between dI_y/dt and C_L(t) across all rollout frames pooled. The DNS ground truth should give r > 0.95 (this is essentially Wu's impulse theorem). The Markov rollout's correlation indicates whether the predicted vorticity field is dynamically consistent. If r drops below 0.7, the rollout is producing fields that look right by metrics but are dynamically inconsistent.

**Part (d) Spectral fidelity at horizon.**
At H = 16 and H = 32, compute the radially-averaged power spectrum of the predicted wake region. Plot DNS, Markov-only, Full-context on the same axes. Report the L2 spectral error and the high-k band (k > k_cutoff) energy ratio.

### Outputs

```
outputs/session17/exp2/physical_metrics_by_rollout.csv       per encounter, mode, H, all metrics
outputs/session17/exp2/markov_vs_full_delta.json             bootstrap CIs
outputs/session17/exp2/impulse_lift_correlation.json         r(dI_y/dt, C_L) per mode
outputs/session17/exp2/spectra_at_horizon.npz                radial spectra at H=16, H=32
outputs/session17/figures/exp2_physical_closure_horizon.png  C_L, I_y, enstrophy vs H, three modes
outputs/session17/figures/exp2_impulse_lift_scatter.png      dI_y/dt vs C_L scatter, DNS vs Markov
outputs/session17/figures/exp2_spectrum_at_H16_H32.png       radial spectra comparison
```

### Compute estimate

Decode is the dominant cost: 80 encounters × 3 modes × 88 frames = 21,000 decoder passes. About 4 hours on RTX 6000. Force and impulse integration is fast. Total: 6-8 hours.

### Acceptance gate

Claim 3 is supported if (i) the bootstrap CI of (Markov - Full) for C_L at H = 16 includes zero or is within 10 percent of the C_L std, (ii) the same holds for I_y, (iii) the impulse-lift correlation r > 0.85 for Markov-only at H ≤ 16, (iv) the spectral L2 error of Markov-only is within 1.5x that of Full-context at H = 16. If only (i) and (ii) hold, the rollout preserves integrated forces but not the field's dynamical consistency; if (iii) fails, the predicted fields are statistically plausible but not physically consistent.

---

## Experiment 3: State-functional alignment at impact

### Hypothesis

Parameter recoverability from the latent is concentrated at the impact frame. Away from the impact frame, parameter information decays. The decay rate is faster for parameters that require nonlinear extraction (Y) than for parameters that have linear correlates (G).

### Inputs

Per-frame latents from production E d=64 on Test B and Test C. Existing kernel-ridge regressor recipe from D118-bis.

### Procedure

**Part (a) Per-frame parameter recovery.**
For each frame offset tau in {-20, -10, -5, -2, 0, +2, +5, +10, +20, +40} relative to impact, fit a kernel ridge regressor with CV bandwidth on train z(t_impact + tau) → (G, D, Y). Evaluate on Test B and Test C. Report R^2 per parameter as a function of tau.

**Part (b) Decay rate fit.**
Fit R^2(tau) for Y to a Gaussian decay model: R^2(tau) = R^2_peak * exp(-tau^2 / (2 * sigma_tau^2)). Report the characteristic time scale sigma_tau in frames and in convective time units (sigma_tau * dt * U_inf / c).

**Part (c) Cross-seed agreement of the recovery function itself.**
This is the test the GPT analysis correctly identified and Session 16 did not run. For each pair of seeds (s_i, s_j), fit a kernel ridge regressor on z_{s_i}(t_impact) → Y, then *apply this regressor to z_{s_j}(t_impact)* and measure R^2. If the function transfers across seeds (R^2 > 0.5), the nonlinear extraction is a property of the data, not the model. If it does not transfer (R^2 < 0.1), the same input data produces different latent encodings and the extraction function is seed-specific.

To make this test meaningful, the latents from different seeds must be in comparable normalization. Standardize each seed's train latents to zero mean, unit variance per dimension before fitting.

**Part (d) SHAP attribution decay (companion to D121-bis).**
For 5 representative Test B encounters, compute SHAP-on-Y at frame offsets tau in {-10, -5, 0, +5, +10}. Report the spatial concentration of the attribution (e.g., the fraction of total absolute attribution within a fixed disk around the LE) as a function of tau. The hypothesis is that the LE attribution is sharply peaked at tau = 0 and diffuses at |tau| > 5.

### Outputs

```
outputs/session17/exp3/per_frame_recovery.csv                R^2(tau) per parameter, per seed
outputs/session17/exp3/decay_fits.json                       Gaussian decay parameters
outputs/session17/exp3/cross_seed_function_transfer.json     R^2 matrix for kernel transfer
outputs/session17/exp3/shap_decay.npz                        attribution concentration vs tau
outputs/session17/figures/exp3_param_recovery_vs_tau.png     R^2(tau) for G, D, Y
outputs/session17/figures/exp3_shap_decay_panels.png         SHAP maps at tau = -10, -5, 0, +5, +10
outputs/session17/figures/exp3_function_transfer_heatmap.png cross-seed kernel transfer R^2
```

### Compute estimate

Per-frame KRR is fast (closed-form per tau). Cross-seed transfer requires fitting 4 × 4 = 16 regressors but each is fast. SHAP at 5 offsets × 5 encounters = 25 additional SHAP runs at 32 integration steps each. About 4-6 hours.

### Acceptance gate

Claim 1 is supported if (i) Y R^2 at tau = 0 exceeds Y R^2 at |tau| = 10 by at least 0.3, (ii) the Gaussian decay fit has sigma_tau < 15 frames, (iii) cross-seed kernel transfer R^2 > 0.5 for Y between at least 4 of 6 seed pairs, (iv) the SHAP attribution concentration peaks at tau = 0 and falls below 50 percent of peak by |tau| = 10. If (iii) fails, the impact-frame alignment exists but is seed-specific; this is a weaker but still publishable result.

---

## Experiment 4: From SHAP maps to coherent structures

### Hypothesis

The four Session 16 SHAP attribution maps (centroid_x, circulation_pos, peak_neg_omega, Y) admit connected-component extraction at stable thresholds. The extracted structures are physically interpretable, have nonzero circulation and vorticity-impulse contributions, and partially but not entirely overlap with the strongest Q-criterion vortices in the same frame.

### Inputs

The Session 16 attribution maps (outputs/session16/exp3/shap_attribution.npz, shap_Y_attribution.npz). Velocity fields (u, v, w) at the impact frame from the DNS, used to compute Q-criterion structures.

### Procedure

**Part (a) Connected-component extraction.**
For each (target, stable encounter), threshold the absolute SHAP map at the 98th percentile of magnitude. Exclude the airfoil mask. Extract connected components with minimum area 10 grid cells. For each component, compute:
- Centroid (x_c, y_c)
- Area A
- Orientation theta (principal axis of inertia)
- Mean absolute SHAP magnitude
- Mean vorticity inside the component (signed)
- Circulation inside the component (integral of omega dA)
- Vorticity impulse contribution (-y_c times circulation for I_x; x_c times circulation for I_y)
- Distance from LE (the airfoil's leading edge point)

**Part (b) Threshold sensitivity.**
Repeat extraction at percentiles {95, 97.5, 98, 99, 99.5}. A structure is "stable" if its centroid moves by less than 5 grid cells across these thresholds and its area changes by less than 50 percent. Report the fraction of stable structures per target.

**Part (c) Q-criterion comparison.**
Compute Q = 0.5 * (||Omega||^2 - ||S||^2) at the impact frame, where Omega is the antisymmetric and S is the symmetric part of the velocity gradient. Extract Q > 0 connected components. For each SHAP structure, compute:
- IoU with the nearest Q structure
- Whether the SHAP structure is inside, partially overlapping, or outside any Q structure
- Centroid distance to nearest Q structure
- Circulation ratio (SHAP-structure circulation divided by nearest Q-structure circulation)

**Part (d) Sign analysis for Y.**
For Y SHAP structures: separate stable encounters by sign of Y. Confirm the report's claim that the structure flips spatial position with Y sign. Report the centroid (x_c, y_c) of the strongest structure as a function of Y, with bootstrap error bars over the stable encounter set.

### Outputs

```
outputs/session17/exp4/structure_catalog.csv                 one row per (target, encounter, structure)
outputs/session17/exp4/threshold_sensitivity.json            fraction stable per target
outputs/session17/exp4/q_overlap.csv                         per-structure Q-IoU, centroid distance
outputs/session17/exp4/Y_sign_flip.json                      centroid vs Y sign with CIs
outputs/session17/figures/exp4_structures_4target_panel.png  the magazine figure: 4 targets x 3 rep encounters
outputs/session17/figures/exp4_q_overlap_summary.png         IoU and circulation ratio distributions
outputs/session17/figures/exp4_Y_sign_flip.png               structure centroid vs Y
```

### Compute estimate

Connected-component extraction is fast. Q-criterion requires velocity gradients (you have u, v, w from DNS). About 2-3 hours total.

### Acceptance gate

Claim 4 is supported if (i) at least 3 of 4 targets have > 50 percent stable structures across thresholds, (ii) the SHAP structures have circulation contributions in physically reasonable ranges (not zero, not dominated by single-pixel outliers), (iii) the Y sign-flip claim is confirmed with bootstrap-significant centroid displacement.

---

## Experiment 5: Closed-loop sparse pressure observability

### Hypothesis

Sparse surface pressure (K = 2 to K = 16 sensors) can recover z_impact and (G, D, Y) sufficiently well that driving the Markov rollout with these inferred quantities produces physical metrics (C_L, I_y, enstrophy) within a defined tolerance of the true-DNS-conditioning rollout.

### Inputs

- Existing pressure-to-z TCN proxy from Session 15 D115 at K = 2, 4, 8, 16 with the consensus sensor selection from D112.
- Existing pressure-to-C_L linear regressor from D111.
- The production predictor and SL decoder.

### Procedure

**Part (a) Pressure → z_impact estimator.**
For each K, retrain or evaluate the TCN proxy (already exists from Session 15) to predict z_impact from a 30-frame pre-impact pressure window. Report cross-pool Test B and Test C z R^2.

**Part (b) Pressure → (G, D, Y) estimator.**
Train a separate small MLP from the same K-sensor 30-frame pressure window to predict (G, D, Y). Report per-parameter R^2 on Test B and Test C. This is c_hat for the closed-loop rollout.

**Part (c) Three closed-loop rollouts at each K.**
For each Test B and Test C encounter, run the Markov rollout in three modes:
- Mode A: oracle z_impact, oracle c (the Session 16 baseline)
- Mode B: pressure-inferred z_hat_impact, oracle c (z observability only)
- Mode C: pressure-inferred z_hat_impact, pressure-inferred c_hat (full closed loop, the deployment story)

For each mode, compute the physical metrics from Experiment 2 (C_L, I_y, enstrophy, spectral lambda ratio) at H = 8, 16, 32. Bootstrap CIs over encounters.

**Part (d) Tolerance curves.**
Define a "deployment tolerance" for each metric: the value of the metric error above which the rollout is no longer useful. For C_L, suggest 10 percent of the encounter-mean |C_L|. For I_y, 15 percent of the encounter-mean |I_y|. For enstrophy, 25 percent. Plot, for each metric, the fraction of encounters within tolerance as a function of K, separately for Mode B and Mode C, on Test B and Test C.

**Part (e) Sensor-region attribution sanity check.**
For the K = 2 closed loop, run SHAP on the pressure-to-z TCN to confirm the sensors carrying most attribution are the same ones consensus-selected in Session 14 D112 (the leading-edge cluster plus mid-chord pressure side). If the SHAP-identified sensors differ from the consensus selection, this is a separate finding worth reporting.

### Outputs

```
outputs/session17/exp5/pressure_to_z_R2.csv                  K x split z R^2
outputs/session17/exp5/pressure_to_c_R2.csv                  K x split (G, D, Y) R^2
outputs/session17/exp5/closed_loop_physical_metrics.csv      Mode x K x H x metric, with CIs
outputs/session17/exp5/tolerance_curves.npz                  fraction in tolerance vs K
outputs/session17/exp5/sensor_shap_consistency.json          attribution vs consensus selection
outputs/session17/figures/exp5_K_curve_physical_metrics.png  C_L, I_y, enstrophy error vs K, 3 modes
outputs/session17/figures/exp5_tolerance_envelope.png        fraction-in-tolerance vs K
outputs/session17/figures/exp5_sensor_locations.png          consensus sensors + SHAP top sensors on airfoil
```

### Compute estimate

TCN exists; retraining at each K is fast (~30 min per K). MLP for c is one model. Closed-loop rollouts: 80 encounters × 3 modes × 4 K values × 88 frames = 84,000 passes. About 8-10 hours.

### Acceptance gate

Claim 5 is supported if (i) Mode C at K = 8 produces C_L errors within tolerance for > 80 percent of Test B encounters at H = 16, (ii) the same holds for I_y with > 70 percent, (iii) Mode B and Mode C diverge by less than 20 percent of Mode A's error at K = 8 (i.e., the c estimator does not dominate the error budget). If only (i) holds, the deployment story is restricted to forces; if (ii) and (iii) hold as well, the deployment story extends to the full wake structure.

---

## Diagnostic D: Long-horizon conditioning paradox

The cond = 0 vs cond = true result from D119-bis (cond = 0 beats cond = true at H ≥ 64) is a one-paragraph diagnostic, not a separate experiment.

Compute the histograms of predicted z norms at H = 32, 64, 79 for cond = true and cond = zero rollouts on Test B. If cond = true drifts outward (mean ||z|| growing with H) while cond = zero contracts (mean ||z|| stable or decreasing), the mechanism is clear: explicit AdaLN-Zero modulation applied repeatedly to a drifting state amplifies systematic prediction errors. Report as a single figure (norm histograms) and a single paragraph in the discussion.

Outputs:
```
outputs/session17/diagnostic_d/z_norm_histograms.png
outputs/session17/diagnostic_d/drift_summary.json
```

About 1 hour.

---

## Execution order

Day 1: Experiment 1 (trajectories) Parts (a), (b), (c). The decision point is whether trajectories cluster and whether kappa(t) peaks at impact. If yes, this becomes Figure 2 of the paper.

Day 2: Experiment 1 Part (d) (cross-seed) and Experiment 3 (state-functional alignment). These are linked because both test cross-seed agreement of physical functions.

Day 3-4: Experiment 2 (physical Markov closure). This is the most compute-heavy experiment but produces the centerpiece figure.

Day 5: Experiment 4 (structures from SHAP).

Day 6-7: Experiment 5 (closed-loop pressure). The most important experiment for the deployment story.

Day 8: Diagnostic D, D-entries D123-D127, paper outline lock, venue confirmation.

---

## D-entries to land in HANDOFF.md

- **D123:** Trajectory geometry of the latent across encounters (Experiment 1). Trajectories are organized, the impact frame is topologically distinct, cross-seed agreement at the trajectory level.
- **D124:** Physical Markov closure in force, impulse, enstrophy, spectrum (Experiment 2). The headline JFM figure.
- **D125:** State-functional alignment at impact: Y recovery decays away from impact; cross-seed function transfer; SHAP attribution decay (Experiment 3).
- **D126:** Coherent structures extracted from SHAP attribution: connected components, circulation, Q-overlap (Experiment 4).
- **D127:** Closed-loop sparse pressure observability: K-curves for physical metrics, tolerance envelopes, sensor-region SHAP sanity (Experiment 5).
- **D128:** Session 17 outcome decision and paper venue lock. JFM submission target with timeline.

---

## Paper outline (JFM, post-Session-17)

Working title: "A predictive low-dimensional state for vortex-airfoil impact: trajectory geometry, conditional Markov closure, and sparse-pressure observability"

**Section 1.** Introduction. Frame the problem: parametric vortex-gust interaction at Re = 5000. The questions: what is the dimensionality of the dynamical state, how does it close the dynamics, how does it manifest in surface pressure. Cite Fukami-Taira (low-dimensional manifold exists), Cremades-Hoyas-Vinuesa (response-relevant structures methodology), and recent latent-ROM work in JFM.

**Section 2.** Methodology. Joint embedding predictive architecture (compressed description; full details in Methods appendix and prior sessions). Trajectory projection and curvature analysis. Conditional Markov rollout. Gradient-SHAP with connected-component structure extraction. Closed-loop pressure observability protocol.

**Section 3.** Trajectory geometry. The 3D projection. Pre-impact and post-impact arc structure. The impact frame as a curvature maximum. Cross-seed agreement of the trajectory geometry.

**Section 4.** State-functional alignment at impact. Parameter recoverability concentrates at the impact frame. The decay timescale. The cross-seed transferability of the nonlinear extraction function.

**Section 5.** Physical Markov closure. C_L, vorticity impulse, enstrophy, spectrum versus horizon for Markov-only versus Full-context rollouts. The dI_y/dt versus C_L correlation as the dynamical-consistency check.

**Section 6.** Coherent structures from attribution. The four targets. Connected components. Circulation and Q-overlap. The Y sign-flip.

**Section 7.** Closed-loop pressure observability. K-curves. Tolerance envelopes. The deployment story.

**Section 8.** Discussion. The state-functional alignment as a fluid-mechanics statement (not just a learning result). The conditioning paradox at long horizons. Limitations: Re = 5000, 2D mid-plane, single airfoil. Future directions.

**Methods.** Full details of the JEPA architecture, training, evaluation pipeline.

---

## Reproducibility

All experiments use existing checkpoints:
- Production E d=64 at outputs/runs/session12/S12_E_d64/encoder/checkpoint_iter020000.pt
- SL decoder at outputs/runs/session12/S12_E_d64/encoder/decoder_specloss_recipe/decoder_iter012000.pt
- Thrust 6 seed retrains at outputs/runs/session14/thrust6/jepa_d64_seed{0,1,2}/

All experiments use configs/splits/split_v1p5.json.

No retraining of encoder, decoder, or predictor in this session.

---

## On readiness for publication

Honest assessment at the start of Session 17:

- JFM submission: realistic in 3-4 months after Session 17 completes. The fluid mechanics story is clean, the methodology is appropriate, the data is sufficient, the venue fit is strong. Five experiments × 1 figure each = 5 figures, plus Methods, plus 1-2 supporting figures from Sessions 11-16.
- Nat. Commun. submission: requires either (a) a second flow case for generalization, or (b) Session 17 producing a striking cross-domain claim. The state-functional alignment idea has cross-domain potential but is not yet validated outside Re = 5000 NACA 0012. Recommended path: submit to JFM first; if JFM accepts with strong reviews, the result is established and a follow-up Nat. Commun. paper can frame the cross-domain implications. Risk of starting at Nat. Commun.: 6-month delay for marginal upside, plus broader reviewer pool that may not appreciate fluid mechanics depth.

The recommendation is JFM, primary submission, Session 18 drafts the manuscript.

---

## What this session does not address

These items are deferred to Session 18 or to a follow-up paper:

- Conditioning-dropout retrain of the predictor (only run if Experiment 5 Mode C fails for the c-inference reason).
- Generalization to other Re or other airfoils (requires DNS data not in scope).
- Persistent homology of latent point clouds (interesting but not physics).
- Full baseline sweep across Fukami AE, POD, beta-VAE, PLDM (one row of one table in this paper, full sweep in a follow-up).
- Conditioning-dropout retrain to eliminate the AdaLN-Zero load-bearing dependence.

---

## References

1. Fukami, K., Taira, K. "Grasping extreme aerodynamics on a low-dimensional manifold." Phys. Rev. Fluids 10, 084703 (2025).
2. Fukami, K., Taira, K. "Compact reduced-order modeling of nonlinear vortical flows over time and parameter space." J. Fluid Mech. 1021, A39 (2025).
3. Cremades, A., Hoyas, S., Vinuesa, R. "Classically studied coherent structures only paint a partial picture of wall-bounded turbulence." Nat. Commun. 16, 10189 (2025).
4. Finzi, M., Qiu, S., Jiang, Y., Izmailov, P., Kolter, J. Z., Wilson, A. G. "From Entropy to Epiplexity: Rethinking Information for Computationally Bounded Intelligence." arXiv:2601.03220v2 (2026).
5. Solera-Rico, A., et al. "β-Variational autoencoders and transformers for reduced-order modelling of fluid flows." Nat. Commun. 15, 1361 (2024).
