# Session 16 Plan: Physics-First Analysis of the Vortex-JEPA Predictive State

Date: 2026-05-25
Lead: Carlos Sanmiguel Vila (INTA, UC3M)
Hardware: 2x RTX 6000 Blackwell (sm_120), bf16 mixed precision
Target venue: JFM (default); Nat. Commun. (if Experiment 3 produces a clean structures-discovery result)

-----

## Strategic frame

Session 16 closes the engineering-ablation phase of the project and opens the physics-analysis phase. No retraining of encoder or decoder. No latent-dimension sweeps. No new architectural variants. Every experiment runs on the production E d=64 + SL decoder pipeline plus the three Thrust 6 seed retrains (for variance estimates), with small probe models trained on top.

The paper this session is building toward makes four claims, each backed by one experiment:

**Claim A.** The impact-instant predictive state has intrinsic dimension 3, with coordinates causally interpretable as the perturbation parameters (G, D, Y).

**Claim B.** Wake observables decompose cleanly into structural information predictable from the impact state and chaotic time-bounded entropy that no learned representation can recover.

**Claim C.** Lift-relevant coherent structures, identified by event-conditioned gradient-SHAP and validated by physical-counterfactual intervention, are not the strongest vortices in the flow.

**Claim D.** The latent is approximately Markov-sufficient for the forced dynamics, distinguishing the JEPA predictive state from a Fukami-style reconstruction manifold.

-----

## Theoretical anchors

- Cremades, Hoyas, Vinuesa, Nat. Commun. 16, 10189 (2025): SHAP-based coherent structures methodology, transplanted from channel turbulence to vortex-airfoil interaction.
- Finzi, Qiu, Jiang, Izmailov, Kolter, Wilson, arXiv:2601.03220v2 (2026): epiplexity as the structural information content of data, prequential estimator (Eq. 8), separation from time-bounded entropy.
- Fukami and Taira, PRF 10, 084703 (2025) and JFM 1021, A39 (2025): low-dimensional aerodynamic response manifold (~3 variables) for vortex-gust airfoil interactions. Treated as honored baseline; the paper does not compete on latent dimensionality.

-----

## Experiment 1: Causal manifold test for (G, D, Y)

### Hypothesis

The 3-dimensional impact-frame manifold’s coordinates are the perturbation parameters (G, D, Y) up to a smooth, learnable transformation.

### Inputs

Existing impact-frame latents from production E d=64 on Test B (56 encounters at v1.5) and Test C (24 encounters). The three Thrust 6 seed retrains used for variance bands only.

### Procedure

**Part (a): Supervised manifold extraction.**
Fit a partial least squares regression from z (64D) onto (G, D, Y), retaining the first 3 PLS components. Call this projection P_base, and define:

```
z_base := P_base @ z
z_fiber := z - P_base.T @ P_base @ z
```

Train P_base on the train split only; evaluate on Test B and Test C. Report per-parameter R^2 (R^2_G, R^2_D, R^2_Y) on held-out data.

**Acceptance:** R^2 > 0.85 on all three parameters on Test B for at least 2 of 3 seeds.

**Part (b): Counterfactual decode.**
For 20 selected Test B encounters, compute the supervised direction in z corresponding to a unit shift in G (the first PLS axis). Apply a perturbation of magnitude matching the difference to the nearest training case with shifted G, decode, and compare to the actual nearest-G training case’s impact-frame field. Report cosine similarity and relative L2 in pixel space, separately for the wake region and the leading-edge region.

This is the metric-vs-direction question from Thrust 3 (D102), re-run in the supervised PLS basis rather than the averaged-difference concept vectors.

**Part (c): Pressure observability hierarchy.**
Reuse the Session 15 TCN proxy and the K=2, K=4, K=8 sensor sets from D115. Train the TCN to predict z_base (3D output) and z_fiber (61D output) separately, with the same K-sensor input. Report cross-pool Test B z_R^2 for each.

**Hypothesis:** at K=2, z_base R^2 > 0.95 and z_fiber R^2 < 0.4. At K=8, z_fiber R^2 climbs toward 0.7.

### Outputs

```
outputs/session16/exp1/pls_projection.npz
outputs/session16/exp1/manifold_recovery_metrics.json
outputs/session16/exp1/counterfactual_decode_metrics.json
outputs/session16/exp1/observability_hierarchy_K_curves.json
outputs/session16/figures/exp1_manifold_3d.png
outputs/session16/figures/exp1_observability_K_curves.png
```

### Compute estimate

Existing latents are extracted; PLS is closed-form; counterfactual decode is 20 forward passes through the decoder; TCN probe retraining at K=2/4/8 is the existing Session 15 pipeline. About 4 hours wall time including diagnostics.

### Acceptance gate

Claim A is supported if Part (a) R^2 holds AND Part (c) shows the predicted z_base vs z_fiber separation at K=2. If Part (a) succeeds but Part (c) fails, the manifold is real but not pressure-observable, and the paper reframes around the unsupervised manifold finding without the observability hierarchy.

-----

## Experiment 2: Epiplexity decomposition of wake observables

### Hypothesis

The wake-prediction task decomposes into structural information (epiplexity, captured by z_base or z_full from the impact state) and time-bounded entropy (chaotic, not capturable by any latent at any size). The decomposition transitions smoothly with observable fineness.

### Theoretical anchor

Finzi et al. 2026 prequential estimator (Eq. 8 in the paper, `src/evaluation/epiplexity.py` already implements it). Used here on small probe MLPs, not on the JEPA encoder itself, because IID prequential coding applies cleanly to the probe setting.

### Inputs

Existing impact-frame latents, existing impact-frame parameter triples (G, D, Y), existing pressure-history windows at the K=8 sensor set.

### Wake observable hierarchy

**Coarse observables (one scalar per encounter):**

- Shedding frequency f_s at x/c=3 in the wake (FFT of v at a probe point, post-impact window)
- Integrated wake enstrophy E_w over a fixed box, post-impact mean
- Mean wake-deficit centerline at x/c=3

**Medium observables (low-dim vector per encounter, ~10-30 dims):**

- Power spectral density in 5 wake-region wavenumber bands at frame H=32
- Vortex shedding wavelength estimate
- Lift impulse over [25, 55] frames, drag impulse over same window

**Fine observables (high-dim, frame-level):**

- Pixel-wise vorticity at H=8 (whole field)
- Pixel-wise vorticity at H=32
- Pixel-wise vorticity at H=88

### Inputs to probe

For each observable, train four separate probes:

- **Probe (i):** from (G, D, Y) only (3D input)
- **Probe (ii):** from z_base only (3D input, supervised projection from Exp 1)
- **Probe (iii):** from z_full (64D input)
- **Probe (iv):** from pressure history at K=8 sensors over a 30-frame window pre-impact

### Probe architecture

Small MLP: 3 hidden layers, width 256, ReLU, no batchnorm. Trained with MSE loss for scalar/vector observables, with a small ConvDecoder head for fine observables. The probe is intentionally small so the prequential epiplexity estimate is meaningful and not dominated by overparameterization.

### Prequential estimation

For each (observable, input) pair, train the probe on the train split with online IID sampling. Record loss curve. Compute epiplexity as area under the loss curve above the final loss (Finzi prequential, Eq. 8). Compute final entropy as the held-out test loss times test set size. Total information: epiplexity + entropy.

### Critical sanity checks

- **IID assumption:** encounters within a case are correlated. Mitigate by sampling at most one frame per (case, encounter) per epoch.
- **Probe overfit:** the prequential estimator approximates description length of the model only if the probe is in its compute-optimal regime. Run a small N-vs-D sweep on probe size and report the Pareto frontier following Finzi’s Section B.1.
- **Random baseline:** a probe trained on shuffled (input, output) pairs should give near-zero epiplexity. Run this for at least one observable as a control.

### Outputs

```
outputs/session16/exp2/probe_loss_curves/{observable}_{input}.npz
outputs/session16/exp2/epiplexity_decomposition.csv
outputs/session16/figures/exp2_stacked_bars.png
outputs/session16/figures/exp2_observable_fineness_curve.png
```

The headline figure (exp2_stacked_bars.png): per-observable stacked bars, epiplexity captured by each input + residual entropy.

The supporting figure (exp2_observable_fineness_curve.png): epiplexity fraction vs observable fineness; x-axis is a fineness index, y-axis is the epiplexity captured by z_base divided by total information.

### Compute estimate

4 inputs × 9 observables = 36 probe trainings. Each is small (a few minutes for scalar observables, ~30 min for fine pixel observables). Total: 8-12 hours.

### Acceptance gate

Claim B is supported if:

1. Coarse observables show epiplexity captured by (G, D, Y) at parity with epiplexity captured by z_full (within 10 percent).
1. Fine observables show a clear residual entropy floor that does not shrink with input richness.
1. The transition is monotonic with observable fineness.

This is the figure that buries the “you have 64 dimensions, Fukami has 3” comparison forever. It shows the question is not the latent size but the task fineness, and it shows where the information ceiling is.

-----

## Experiment 3: Lift-relevant coherent structures via event-conditioned SHAP

### Hypothesis

Lift-relevant coherent structures, identified by phase-locked gradient-SHAP attribution and validated by physical-counterfactual intervention, are physically distinct from the strongest vortices in the flow.

### Sample size honesty

Before any structure is reported, this experiment runs a bootstrap convergence diagnostic. Test B has 56 encounters; pre-impact, impact, and post-impact phase windows each pool around 56 × 10 frames = ~560 samples. This is enough for stable averaging on broad event sets. Test C has 24 encounters; conditional events (high G, high |C_L|) within Test C have ~6-8 encounters, likely too few. The experiment commits to dropping non-converging structures.

### Targets (two only, not ten)

- **Target T1:** future C_L at H=8 (short-horizon lift)
- **Target T2:** future C_L at H=32 (mid-horizon lift, beyond the H_roll=8 training horizon)

### Event sets (three only, all phase-locked)

- **E_pre:** frames 20-30 (pre-impact)
- **E_imp:** frames 35-45 (during impact)
- **E_post:** frames 50-60 (post-impact wake settling)

Six (target, event) maps total. Each requires bootstrap convergence to be reported.

### Procedure

**Step 1: Implement gradient-SHAP for the omega-to-C_L pipeline.**
The model is omega → encoder → z → predictor (with G,D,Y) → decoder → frame at delta H → C_L head. The differentiable chain ends at a scalar C_L target. Background: phase-matched mean field from the train split’s no-gust baseline encounters (encounters 0-3 of train cases).

**Step 2: Compute attribution maps for each (target, event) pair.**
For each of the 56 Test B encounters, compute the per-frame absolute SHAP map for each frame in the event window. Phase-align by impact frame. Average across encounters.

**Step 3: Bootstrap convergence diagnostic.**
For N in {8, 16, 32, 56}, compute the mean attribution map and measure:

```
epsilon_N := ||A_N - A_56||_2 / ||A_56||_2
```

Report a convergence curve per (target, event). Drop any pair with epsilon_16 > 0.3.

**Step 4: Extract structures.**
For each converged map, threshold at the 98th percentile of attribution magnitude (the top 2 percent area), exclude the airfoil mask, and extract connected components. Filter components by minimum area (10 grid cells, matching DNS scale).

**Step 5: Physical intervention validation.**
For each structure C and each of the original 56 Test B encounters, compute three counterfactuals:

- **Smooth-inpaint deletion:** replace C’s region with a Gaussian-blurred version of the same field at the same scale (sigma = 3 grid cells). Physically realizable in the sense that it preserves the field smoothness without injecting OOD content.
- **Random equal-area deletion:** same operation on a random equal-area patch elsewhere in the field.
- **Q-criterion equal-circulation deletion:** same operation on the strongest Q-criterion vortex of comparable circulation, if one exists.

For each, decode and predict C_L. Measure Delta C_L under each intervention.

**Step 6: Statistical test.**
For each structure, perform a Welch t-test on the Delta C_L under structure deletion versus random deletion (n=100 random samples per encounter). A structure is “lift-relevant” if p < 0.01 AND the median Delta C_L from structure deletion is at least 2x that of random deletion.

**Step 7: Classical-structure comparison.**
For each lift-relevant structure, compute IoU with the Q-criterion structures at the same phase. Report whether the SHAP structures are inside, partially inside, or outside the Q structures. This is the Cremades-style headline.

### Outputs

```
outputs/session16/exp3/shap_maps/{target}_{event}.h5
outputs/session16/exp3/bootstrap_convergence.csv
outputs/session16/exp3/structure_catalog.csv
outputs/session16/exp3/intervention_validation.csv
outputs/session16/figures/exp3_shap_maps_grid.png
outputs/session16/figures/exp3_structure_overlays.png
outputs/session16/figures/exp3_intervention_curves.png
```

### Compute estimate

Gradient-SHAP per encounter per frame is one forward + one backward pass per integration step (use 32 integration steps following Cremades). Total: 56 encounters × ~50 frames × 32 steps × 2 targets ≈ 180k passes. On 2x RTX 6000 with batching, about 12-16 hours. Intervention validation adds 56 encounters × 100 random samples × 6 structures ≈ 34k decoder forward passes, about 2-3 hours.

### Acceptance gate

Claim C requires at least 2 of 6 structures to pass intervention validation AND to be physically distinct from the strongest Q-criterion vortices at the same phase. If 0-1 pass, the structures result moves to the supplement, the paper does not headline it, and the Nat. Commun. target is dropped.

-----

## Experiment 4: Markov closure test (predictive sufficiency)

### Hypothesis

Given z_t and the static parameters (G, D, Y), the JEPA predictor can roll out z_{t+delta} without access to earlier latents. This distinguishes the predictive state from a Fukami reconstruction manifold and supports the “predictive sufficiency” framing.

### Procedure

For each Test B and Test C encounter, take the impact-frame latent z_imp and the parameters (G, D, Y). Run three rollouts forward to H=88:

- **Rollout (i) — Full history:** standard autoregressive predictor with all prior latents in context (as trained).
- **Rollout (ii) — Markov-only:** feed only z_imp as the latent context at each step; the predictor’s internal causal mask sees only the impact frame. At step t+1, the predictor takes z_t (the just-predicted latent) and (G, D, Y) and produces z_{t+1}. No earlier latents visible.
- **Rollout (iii) — Parameter-only:** predict z_{t+H} from (G, D, Y) and impact-relative phase only, no latent context. This is the maximum-compression baseline.

For each, decode every frame, and compute SSIM, mse_wake, lambda_ratio, predicted C_L, and predicted enstrophy at H ∈ {1, 8, 32, 88}. Bootstrap CIs over encounters.

### Statistical test

Hypothesis: (i) and (ii) are statistically indistinguishable at horizons up to H=32 on Test B (Welch t-test p > 0.05 for all metrics). Rollout (iii) is significantly worse than both (i) and (ii). On Test C, all degrade but (ii) tracks (i) within 1.5 sigma.

### Edge case

If the predictor’s architecture genuinely requires history context (e.g., the predictor itself uses RoPE positional encoding that breaks with truncated history), Rollout (ii) must be implemented carefully. The cleanest implementation is to mask all but the impact-frame latent in the predictor’s attention and verify the gradient pathways. The implementing agent should document the masking choice and validate it on a no-gust baseline encounter (where the answer should be trivially recoverable).

### Outputs

```
outputs/session16/exp4/rollout_metrics.csv
outputs/session16/exp4/markov_closure_test_results.json
outputs/session16/figures/exp4_rollout_comparison.png
```

The figure: SSIM, C_L_R^2, enstrophy_R^2 vs H, three lines for three rollouts, Test B and Test C subplots.

### Compute estimate

80 encounters × 3 rollouts × 88 horizons = 21k decoder + predictor passes. About 3-4 hours.

### Acceptance gate

Claim D is supported if Rollout (ii) tracks Rollout (i) within 1 sigma on Test B at H ≤ 32 for SSIM and C_L. Failure indicates the predictor uses history information beyond the impact frame, and the predictive-sufficiency story weakens to “the impact frame plus the first 4-8 frames are jointly Markov-sufficient.”

-----

## Execution order and dependencies

Experiment 1 must run first because z_base and z_fiber from its Part (a) feed into Experiments 2 and 3.

Experiments 2 and 4 can run in parallel once Experiment 1 Part (a) is done.

Experiment 3 should start after Experiments 1 and 4 have completed pilot results, so the team knows whether the predictive-state framing holds before investing in the structures analysis.

**Day-by-day:**

|Day|Activity                                                     |Decision point           |
|---|-------------------------------------------------------------|-------------------------|
|1  |Experiment 1 parts (a), (b), (c)                             |Does Claim A hold?       |
|2  |Experiment 4 (Markov closure). Start Experiment 2 scaffolding|Does Claim D hold?       |
|3-4|Experiment 2 full probe sweep                                |Does Claim B hold?       |
|5-7|Experiment 3 SHAP + intervention validation                  |Does Claim C hold? Venue?|
|8  |Draft figures, write D-entries D118-D122, paper outline      |Submit-ready outline     |

-----

## What to keep out of Session 16

- No new encoder retraining at any latent dimension.
- No new decoder variants.
- No new pressure-sensor selection methods.
- No predictor architecture changes.
- No data extension beyond v1.5.
- No expansion of the SHAP analysis to more than the 6 (target, event) pairs specified.

If during the session a result suggests one of these is needed, log it as a Session 17 follow-up; do not execute in this session.

-----

## D-entries to land in HANDOFF.md

- **D118**: Causal manifold test (Experiment 1) outcome.
- **D119**: Epiplexity decomposition of wake observables (Experiment 2).
- **D120**: Lift-relevant coherent structures via event-conditioned SHAP (Experiment 3).
- **D121**: Markov closure test for the predictive state (Experiment 4).
- **D122**: Session 16 outcome decision; paper outline; venue choice.

-----

## Paper outline (target)

Working title: “The predictive state of vortex-airfoil interaction: a low-dimensional causal manifold with pressure-observable parametric coordinates and a chaotic wake fiber.”

**Section 1:** Introduction. Frame around the gap between low-dimensional aerodynamic response manifolds (Fukami) and predictive, observable, scale-resolved digital twins. Cite Cremades for the explainability methodology.

**Section 2:** Methodology. JEPA architecture (compressed, point to prior sessions for full details). PLS supervised projection. Gradient-SHAP with phase-matched ergodic averaging. Prequential epiplexity estimation. Intervention validation.

**Section 3:** The impact-frame manifold has intrinsic dimension 3 with causally interpretable coordinates (Experiment 1).

**Section 4:** Wake observables decompose into structural information and time-bounded entropy (Experiment 2). This is the headline information-theoretic figure.

**Section 5:** Predictive sufficiency of the impact-frame state (Experiment 4). This is the wedge against Fukami.

**Section 6:** Lift-relevant coherent structures (Experiment 3). Promoted to Section 5 if intervention validation is clean and the venue is Nat. Commun.

**Section 7:** Discussion. Limitations (Re=5000, 2D mid-plane). Future directions (Re sweep, 3D). Connection to DICE (gust-airfoil ROM at higher Re).

-----

## Reproducibility

- All experiments use `configs/splits/split_v1p5.json`.
- All experiments use the production E d=64 checkpoint at `outputs/runs/session12/.../checkpoint_iter080000.pt` (decoder: SL recipe per D99).
- The three Thrust 6 seed retrains used only for variance bands: `outputs/runs/session14/thrust6/seed_{0,1,2}/`.
- Probe random seed: 42 for all Experiment 2 probes; report std over 3 reseeds for one observable as a sanity check.
- Bootstrap seeds for Experiment 3: 100 samples, seed grid {0..99}.

-----

## References

1. Cremades, A., Hoyas, S., Vinuesa, R. “Classically studied coherent structures only paint a partial picture of wall-bounded turbulence.” Nat. Commun. 16, 10189 (2025).
1. Finzi, M., Qiu, S., Jiang, Y., Izmailov, P., Kolter, J. Z., Wilson, A. G. “From Entropy to Epiplexity: Rethinking Information for Computationally Bounded Intelligence.” arXiv:2601.03220v2 (2026).
1. Fukami, K., Taira, K. “Grasping extreme aerodynamics on a low-dimensional manifold.” Phys. Rev. Fluids 10, 084703 (2025).
1. Fukami, K., Taira, K. “Compact reduced-order modeling of nonlinear vortical flows over time and parameter space.” J. Fluid Mech. 1021, A39 (2025).