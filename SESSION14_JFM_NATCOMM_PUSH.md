# SESSION14_JFM_NATCOMM_PUSH.md

Session 14 plan: push from “good engineering result” (Sessions 11-13) to
“high-impact journal contribution” (JFM or Nature Communications).

Last updated: 2026-05-24.

## Honest evaluation of current state

What Sessions 11-13 delivered, ranked by paper-impact strength:

1. **E d=64 + SL is the production winner** (D99). Test B SSIM mean 0.526 /
   median 0.522, λ-ratio 1.64 ✅ (PRF 2026 factor-2 criterion crossed),
   wake2D-IoU 0.397. Test C SSIM mean 0.303, λ-ratio 1.15 ✅.
1. **9 of 9 SL retrains cross the PRF criterion on Test C**; 6 of 9 on Test B.
   The PRF 2026 spectral loss is the load-bearing decoder modification.
1. **The lambda_wake monotonic scaling finding from Session 11** plus the
   Session 12 U-shape extension is a real mechanistic discovery: PR(z) scales
   with external wake pressure, not d budget.
1. **The PCA k=12 finding** (Session 11 7b): the JEPA latent has effective
   rank ~12, with the tail 20 PCs contributing ~10 SSIM on Test B and ~30 on
   Test C. This is the closest the project has come to an “intrinsic
   dimensionality” claim.
1. **The matched-d=32 Fukami AE comparison** (Session 11 D81): JEPA encodes
   GDY 2-4x better than Fukami at the same d. The reconstruction is tied;
   the latent physics is not.

What is missing for JFM or Nat. Commun.:

1. **No forecasting evaluation past H=8.** Solera-Rico Figure 2 rolls out
   to H=200; we have no equivalent. The paper’s “trajectory framing” (D2)
   is unjustified by data.
1. **No causal/structural latent claim.** AeroJEPA does concept vector
   arithmetic for (alpha, Re, Ma). We have not done it for (G, D, Y),
   despite the RBF probe R²=0.93/0.94/0.85 saying it should work linearly
   in the right basis.
1. **No quantitative intrinsic dimensionality.** PCA k=12 is suggestive,
   not definitive. The PR(z) of 11.66 at lambda_wake=1.0 is a snapshot,
   not a manifold-dimension claim. Need an Isomap residual + maximum
   likelihood dimension estimator + cross-validated probe argument.
1. **No methodological principle with a name.** “Wake observable head”
- “SL loss” + “d=64” + “TC penalty” is a recipe, not a principle.
1. **Visual Figure 3 is still BLURRY** at frame 55 post-impact. The
   λ-ratio = 1.64 says “within factor 2 of DNS” but the eye says “still
   smoothed.” No reviewer will be convinced by IMG_2201 as a Figure 3.
1. **Test C is fundamentally broken at G=+4.** The decoder saturates at
   the colorbar range because the raw |omega| > 100 outside training. SSIM
   0.338 is the best but the visual (IMG_2200) is poor. Either we frame
   this honestly as “OOD limitation” or we extend the training envelope.
1. **No paper-grade comparison with the canonical competitors.** Fukami
   PRF 2025 + JFM 2025 results were not benchmarked head-to-head with
   matched compute / matched data / matched d. AeroJEPA is concurrent
   and untouchable for direct comparison.

## The Nature Communications differentiator: epiplexity for fluid data

A genuinely novel idea, available to us via the Finzi, Qiu, Jiang, Izmailov,
Kolter, Wilson paper (arXiv:2601.03220, March 2026). “From Entropy to
Epiplexity: Rethinking Information for Computationally Bounded
Intelligence.” The paper introduces **epiplexity**, the structural
information a computationally bounded observer extracts from data.

Why this elevates our paper from “engineering result” to “Nat. Commun.
methodological contribution”:

1. **Nobody has measured epiplexity for a fluid dataset.** The original
   paper measures it on language, chess, ECA, images, and video. Fluid
   mechanics data is conspicuously absent. We can be first.
1. **Epiplexity quantifies what makes a fluid dataset learnable.** The
   PREVENT vortex-gust dataset has parameters (G, D, Y) and we can ask:
   what is the structural information content the JEPA encoder absorbs?
   This is exactly the prequential coding measurement: area under the
   training loss curve above the final loss (Equation 8 of the
   epiplexity paper).
1. **Different data interventions have different epiplexity.** Frame skip,
   sub-trajectory length L, impact-aware sampling weight, ordering
   (causal vs anti-causal): each is a deterministic transformation of the
   data that can change epiplexity. The epiplexity paper Section 5.2
   shows the **chess board→moves** ordering gives higher epiplexity and
   better OOD generalization than **moves→board**. The analogous test
   for us is **vorticity field → forces** vs **forces → vorticity field**.
1. **Epiplexity should predict OOD generalization.** The paper’s Section
   6.1 shows this empirically on chess. If we measure epiplexity for
   each Session 12 configuration (Direction A through F) and find it
   correlates with Test C SSIM, we have an external information-theoretic
   variable that explains the in-distribution / out-of-distribution
   tradeoff we see in the data.
1. **A clean methodological claim**: “We propose epiplexity, an
   information-theoretic measure of structural content in fluid datasets,
   as a quantitative tool for fluid ML data curation. We demonstrate it
   on the PREVENT parametric vortex-gust dataset and find that epiplexity
   correlates with out-of-distribution generalization in a JEPA encoder
   trained on it.” This is a method-with-a-name contribution that has
   no analog in the fluid ML literature.
1. **It directly addresses the Fukami / Solera-Rico comparison.** Both
   competitors operate without explicit data complexity measurement.
   Epiplexity gives us a vocabulary for saying “JEPA absorbs more
   structural information than Fukami AE at matched d=32” with units,
   not just R² numbers.

This is the single biggest methodological lever for the paper. It is
also low-risk technical work: the prequential coding measurement is
just area under the loss curve. We already have loss curves from
Sessions 5-13. The only new code is the integration step.

## Session 14 attack plan: six major thrusts

All thrusts run in parallel. GPU budget is not a constraint.
Implementation work is moderate; the heaviest item is Thrust 2
(forecasting rollout) and Thrust 6 (Fukami head-to-head).

### Thrust 1: Epiplexity measurement for the vortex-gust dataset

The biggest single novelty contribution. Implements the Finzi et al.
2026 prequential coding methodology and applies it to the PREVENT
dataset, the JEPA encoder, and competitor architectures.

**1a. Implement prequential coding estimator.** Module
`src/evaluation/epiplexity.py`. The core estimator is from epiplexity
paper Equation 8:

```
|P_preq| ≈ Σ_i [log 1/P_i(Z_i) - log 1/P_M(Z_i)]
        = area under (train loss - final loss) over training tokens
```

Two estimators: prequential (Section 4.1 of paper) and requential
(Section 4.2). Prequential is cheaper, requential is rigorous. Start
with prequential.

**1b. Measure epiplexity for the JEPA encoder.** For each Session 12
configuration (A default, A high, B GAN, C lam=200/300/500, D
coarse288/512, E d=64, F TC=0.01/0.03/0.10) compute:

- |P_preq| (encoder structural information)
- H_T (residual time-bounded entropy = final loss)
- (S_T, H_T) decomposition (Figure 8a of epiplexity paper)

Use the existing training logs; no retraining. ~200 lines of code.

**1c. Compare epiplexity across architectures at matched d=32.**
JEPA vs Fukami AE vs Solera-Rico β-VAE (if we can train one).
Same train data, same compute budget. The hypothesis: JEPA absorbs
more epiplexity (steeper loss decay sustained longer) than Fukami AE
even when their final losses match. This is the information-theoretic
restatement of the Session 11 D81 finding.

**1d. Correlation of epiplexity with OOD SSIM.** For each Session 12
configuration: scatter plot of |P_preq| (x-axis) vs Test C SSIM mean
(y-axis). The epiplexity paper Section 6.1 predicts positive
correlation. If we observe it on fluid data, that is a publishable
finding regardless of paper venue.

**1e. Epiplexity for different data interventions.** Hold the
architecture fixed (E d=64 + SL). Vary:

- Frame skip ∈ {1, 2, 5, 10}
- L (sub-trajectory length) ∈ {8, 16, 32, 64, 128}
- Impact-aware sampling weight ∈ {0%, 30%, 50%, 70%, 100%}

For each: measure |P_preq| and Test C SSIM. The epiplexity paper
Section 6.4 predicts that data orderings/interventions that produce
higher epiplexity should give better OOD performance. This is the
fluid-data analog of Adaptive Data Optimization (Jiang et al. 2025).

**Headline claim if it works**: “We introduce epiplexity, an
information-theoretic measure of structural content in fluid datasets,
as a quantitative tool for data curation in fluid machine learning.
Applied to the PREVENT parametric vortex-gust dataset, epiplexity
correlates with out-of-distribution generalization (Pearson r > 0.7
across Session 12 configurations) and identifies impact-aware sampling
as the highest-information data intervention.”

Expected wall time: ~6h (estimator implementation 3h + measurement 3h).
GPU time: minimal (re-uses existing training logs for prequential;
maybe one fresh training for a requential estimate as a sanity check).

### Thrust 2: ROM-style forecasting evaluation (Solera-Rico parity)

We have an autoregressive predictor. We have a JEPA encoder. We have
not measured forecast horizon. Solera-Rico’s Figure 2 is the paper’s
hero figure — they show latent trajectory rollout at H=200 (about 10
t/c). We need our equivalent.

**2a. Latent rollout RMSE vs DNS at H ∈ {1, 8, 16, 32, 64, 128, 256}.**
Take an encoder + predictor pair. Encode the first L=32 frames of a
Test B encounter. Roll out the predictor open-loop for H additional
frames. Decode each predicted latent with the LapFiLM + SL decoder.
Compute RMSE in raw omega vs DNS.

Plot RMSE(H) for each Session 12 configuration. The Solera-Rico
comparison anchor is “JEPA + LapFiLM + SL maintains RMSE below
[threshold] at H = [horizon] t/c, comparable to / better than
β-VAE + transformer at matched d.” We get to define what the
threshold and horizon are.

**2b. Predictor sensitivity to encoder.** All Session 12 encoders use
the same predictor architecture (6-layer transformer, hidden 384,
AdaLN-Zero conditioning). Retrain the predictor for the production
encoder (E d=64) since the latent dimension changed. Compare with
Session 9 predictor at d=32 for the W0_C_lam100 encoder.

**2c. Long-rollout decoded reconstruction.** Pick the canonical Test
B encounter G+1.00_D1.00_Y+0.10 encounter 00. Roll out the predictor
from frame 32. Render Figure 3-style decoded omega at H = 16, 32, 64,
128. Show that the wake structure remains spatially coherent at long
horizons. This is the JFM hero figure.

**2d. Forecast horizon as a function of (G, D, Y).** Heatmap of
forecast horizon (defined as H at which RMSE crosses a fixed threshold)
across the (G, D, Y) parameter cube. Identifies which regimes the
predictor handles cleanly vs which break down.

**Headline claim**: “JEPA + LapFiLM + SL produces accurate vorticity-
field rollouts at H ≥ X convective times on parametric vortex-gust
test encounters, comparable to β-VAE + transformer (Solera-Rico Nat.
Commun. 2024) at matched latent dimension. The active wake observable
head improves long-horizon spectral fidelity over passive Fukami AE

- LSTM at matched d=32 (Fukami JFM 2025).”

Expected wall time: ~10h (predictor retrain for d=64 6h + evaluation
3h + figure generation 1h). GPU time: ~8h.

### Thrust 3: Concept vector arithmetic in latent space (AeroJEPA-style)

AeroJEPA (Giral et al., May 2026, our D90) demonstrates concept vector
arithmetic in their d=128 token-wise latent for (alpha, Re, Ma). The
linear directions corresponding to (Δalpha, ΔRe, ΔMa) are recovered
by averaging encoded latent differences across cases. We can do the
same for (G, D, Y) at d=64.

**3a. Construct concept vectors.** From the train set, identify all
pairs of encounters that differ ONLY in G (D and Y matched). Average
the latent differences to get v_G. Repeat for v_D and v_Y. This is
AeroJEPA Equation 9.

**3b. Test linear extrapolation.** Take a train encounter at
(G_0, D_0, Y_0). Add k * v_G to its latent. Decode. Compare to a real
test encounter at (G_0 + k * Δ, D_0, Y_0). If the decoded field
matches, the latent supports linear (G, D, Y) arithmetic, which is
non-trivial given JEPA does not explicitly enforce it.

**3c. Linear Jacobian probe.** AeroJEPA Equation 11: train a 3D linear
probe on (G, D, Y), then take its closed-form Jacobian (linear
weights) as the local concept-vector estimate. Compare to the global
averaging estimate from 3a. Coincidence of the two estimators is
itself a result.

**3d. Decoded fields at synthetic (G, D, Y).** Generate decoded omega
for (G, D, Y) points OUTSIDE the training envelope (e.g. G = +6, D =
2.0). Visual check whether the synthesized fields are physically
plausible. This is a strong demonstration of latent disentanglement.

**Headline claim**: “The JEPA encoder produces a latent representation
in which (G, D, Y) act as approximately linear, separable concept
directions. Concept vector arithmetic recovers held-out vortex
parameters with relative error < X%, comparable to the closed-form
Jacobian of a linear probe. This linear accessibility is non-trivial:
JEPA does not explicitly enforce it. SIGReg’s effect on the encoded
distribution is the mechanistic explanation.”

Expected wall time: ~4h. GPU time: ~1h (evaluation only, no retraining).

### Thrust 4: Quantitative intrinsic dimensionality of the impact-instant manifold

Session 11 PCA k=12 was suggestive. We need a definitive number.

**4a. Three estimators, agreement test.**

- PCA spectrum: 94.3% variance at k=12 (Session 11). Already have.
- Isomap residual variance vs k (Session 11 partial). Extend to k up
  to 32 on the impact-frame latents.
- Maximum Likelihood Estimator of Levina-Bickel 2004 (the standard
  intrinsic-dim estimator). New implementation, ~80 lines.
- Two-NN estimator of Facco et al. 2017 (more robust to noise).
  ~80 lines.

The four estimators should agree on a number. If they all return
12 ± 2, we have the manifold dimension. If they disagree, the
disagreement itself is informative (about which subspace each
estimator captures).

**4b. Dimensionality vs (G, D, Y) region.** Compute intrinsic
dimension on subsets: high-|G| only, low-|G| only, post-impact only,
pre-impact only. If the dimensionality varies with regime, that is a
mechanistic discovery: the manifold has variable curvature.

**4c. Comparison to AE matched at d=12.** Train a Fukami AE at d=12.
If the manifold is really 12-D, a 12-dim AE should reach comparable
reconstruction. If it does, JEPA’s contribution is the geometry, not
the dimensionality. If it does not, JEPA is doing something AE can’t.

**4d. Total correlation as a sufficient statistic.** Session 12
Direction F shows TC penalty broadens PR. Does it also reduce the
estimated intrinsic dimension? Theoretical prediction: TC penalty
should not change the manifold dimension (the data is what it is) but
should make the latent more axis-aligned with the manifold axes,
reducing the gap between PR and intrinsic dim.

**Headline claim**: “The intrinsic dimensionality of the parametric
vortex-gust impact manifold at Re = 5000 is empirically 12 ± 2, as
measured by four independent estimators (PCA, Isomap, MLE, Two-NN).
This dimension exceeds the (G, D, Y) parameter count (3) by an order
of magnitude, reflecting the shedding phase and impact-time fine
structure that the encoded latent must represent. The JEPA encoder
at d=64 spans this manifold and contains substantial unused capacity
(PR(z) ≈ 12); the wake observable head and TC penalty allocate the
unused capacity to physically meaningful directions.”

Expected wall time: ~4h. GPU time: minimal.

### Thrust 5: Reverse-factorization test (epiplexity paper Section 5.2)

The most speculative thrust. Could be a clean win or a clean negative
result; either is publishable.

The epiplexity paper Section 5.2 shows that for chess data, training a
transformer in **reverse order** (board state → moves) gives higher
epiplexity AND better OOD transfer than the forward order
(moves → board state). The mechanism: the reverse task requires
inverting a non-trivial function, which forces the model to learn
richer board-state representations.

The fluid analog: predicting **vorticity field → integrated forces
(C_L, C_D)** is easy (a simple post-processing inversion). Predicting
**forces → vorticity field** is hard (an ill-posed inverse problem
with many vorticity fields consistent with the same forces).

**5a. Train a “reverse” model.** Predictor input: time series of
(C_L(t), C_D(t), phi_t, G, D, Y). Predictor output: encoded latent
z_t = E(omega_t). Train at d=64 with SL decoder on top.

**5b. Measure epiplexity of forward vs reverse training.** Apply
Thrust 1’s prequential estimator. Test the prediction that reverse
training has higher epiplexity.

**5c. Test OOD transfer of reverse model.** Evaluate on Test C
(G=+4). Does the reverse model generalize better than the forward
model? The epiplexity paper predicts yes.

This is a high-risk high-reward thrust. If the prediction holds, we
have a fluid demonstration of one of the most surprising findings in
the epiplexity paper, which would be a clean Nat. Commun. story. If
it fails, we have a clean negative result that adds rigor.

Expected wall time: ~10h. GPU time: ~8h (one fresh encoder + predictor
training pair).

### Thrust 6: Fukami head-to-head with confidence intervals

Sessions 1-13 have a single Fukami AE baseline at d=32 (D81). For a
JFM paper we need a more rigorous comparison.

**6a. Three Fukami AE seeds.** Same d=32, same data, three different
seeds. Mean ± std of SSIM, wake_enstrophy, GDY r2.

**6b. Three JEPA seeds.** Same E d=64 + SL configuration, three
different seeds. Mean ± std of the same metrics.

**6c. Statistical significance.** Welch t-test on every metric.
Confidence interval on the JEPA - Fukami delta. The paper needs to
say “JEPA beats Fukami at d=32 by Δ = X ± Y with p < 0.01” or
honestly admit “the gap is within seed variance.”

**6d. Matched-compute comparison.** Train Fukami AE for the same
total iterations (20k) as JEPA. Also train Fukami AE for the same
total epochs as JEPA. The PRF / JFM convention.

**6e. Forecast horizon comparison.** Train a Fukami AE + LSTM at
d=32 (their JFM 2025 recipe). Measure forecast horizon as in Thrust
2. JEPA + transformer at d=64 vs Fukami AE + LSTM at d=32: the
head-to-head benchmark the paper needs.

**Headline claim**: “JEPA + LapFiLM + SL at d=64 outperforms Fukami
AE + LSTM at d=32 on every Test B and Test C metric with statistical
significance (p < 0.01 on SSIM mean, wake_enstrophy, and forecast
horizon at H = X). The gain comes from active wake supervision in
the encoder, not the larger latent.”

Expected wall time: ~15h. GPU time: ~12h (six fresh training pairs).

## Session 14 sequencing and parallelism

With two cards plus optional cloud:

**Phase 1 (parallel, ~6h)**:

- cuda:2: Thrust 6a-6b (three Fukami + three JEPA seeds, parallel
  pairs)
- cuda:3: Thrust 2a-2b (predictor retrain at d=64 + forecast eval)
- Implementation work (Thrusts 1, 3, 4 estimators) in parallel with
  GPU work

**Phase 2 (parallel, ~6h)**:

- cuda:2: Thrust 6e (Fukami AE + LSTM training)
- cuda:3: Thrust 5a-5b (reverse-factorization training)
- Thrust 1 epiplexity measurement on all Session 12 logs
- Thrust 3 concept vector arithmetic
- Thrust 4 intrinsic dim estimators

**Phase 3 (parallel, ~4h)**:

- Thrust 2c-2d (long-rollout decoded reconstruction figure +
  forecast horizon heatmap)
- Thrust 1d-1e (epiplexity correlation with OOD + data intervention
  sweep)
- Thrust 5c (reverse-factorization OOD test)

**Phase 4 evaluation (~3h)**: assemble all results into a unified
table. Decide paper-headline metric. Generate hero figures.

Total session: ~20-25 hours including implementation work. With
cloud GPU acceleration on Thrusts 2 and 6 (the heaviest), can
compress to ~15h.

## Success criteria for Session 14

**Numerical (paper-grade thresholds)**:

1. Epiplexity (Thrust 1) is measured for at least 8 Session 12 configs,
   with Pearson correlation |r| > 0.5 between |P_preq| and Test C
   SSIM mean.
1. Forecast horizon (Thrust 2) at the production configuration
   maintains RMSE below 0.5 × DNS standard deviation at H ≥ 32
   (about 1.6 t/c).
1. Concept vector arithmetic (Thrust 3) recovers held-out (G, D, Y)
   with linear-extrapolation relative error < 30%.
1. Intrinsic dimensionality (Thrust 4) agreed within ± 3 across
   four independent estimators.
1. Reverse-factorization (Thrust 5) either confirms epiplexity-OOD
   relationship (positive result) or provides a clean negative
   refutation.
1. Fukami head-to-head (Thrust 6) produces a Welch t-test result
   with p < 0.05 on at least one paper-headline metric.

**Qualitative**:

1. JFM-grade Figure 3 with long-rollout decoded omega (H=64+).
1. Nat. Commun.-grade headline figure pairing epiplexity (x-axis)
   with Test C SSIM (y-axis), showing the data-intervention sweep
   from Thrust 1e.
1. Concept-vector visualization figure (Thrust 3d).

**Paper claim trajectory**:

After Session 14 the paper has three independent contributions:

1. **Methodological**: A JEPA + SL + wake-observable recipe for
   fluid representation learning with documented latent geometry
   and forecast horizon.
1. **Empirical**: Intrinsic dimensionality of vortex-gust impact at
   Re=5000 quantified at 12 ± 2.
1. **Information-theoretic (Nat. Commun. differentiator)**: Epiplexity
   measurement for fluid data, demonstrated to correlate with OOD
   generalization.

Any one of these is a paper. All three is a Nature Communications
candidate. The Section 5 narrative becomes:

- 5.1 Pipeline + decoder baseline.
- 5.2 Wake observable head sweep.
- 5.3 Session 12 / 13 multi-direction ablation, headlining E d=64 + SL.
- 5.4 Comparison vs Fukami AE at matched d (now with seeds + CI).
- 5.5 Disentanglement diagnostic + concept vector arithmetic.
- 5.6 Forecast horizon evaluation (vs Solera-Rico).
- **5.7 NEW: Epiplexity-based information theoretic analysis.**
- **5.8 NEW: Intrinsic dimensionality of the impact manifold.**

## Critical implementation order

1. **Prequential coding estimator** (Thrust 1a). 200 lines. Blocks all
   of Thrust 1. Highest leverage for the Nat. Commun. story.
1. **Predictor retrain at d=64** (Thrust 2a). 6h GPU. Blocks Thrust 2.
1. **Concept vector arithmetic** (Thrust 3a-3c). 100 lines. Blocks
   Thrust 3. Cheap, high paper value.
1. **MLE intrinsic dim estimator** (Thrust 4a). 80 lines. Blocks
   Thrust 4.
1. **Reverse-factorization training script** (Thrust 5a). New training
   recipe, ~300 lines + scripts. Blocks Thrust 5.
1. **Fukami AE + LSTM** (Thrust 6e). Training new recipe, ~400 lines
- Fukami JFM 2025 details.

Items 1, 3, 4 are critical path (block multiple downstream things and
are cheap to implement). Item 2 is the highest GPU-cost single item.

## Risk register

|Risk                                                  |Probability|Mitigation                                                                                                                                                                                 |If it fires                                                                                                                                             |
|------------------------------------------------------|-----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
|Epiplexity-OOD correlation is weak (|r| < 0.5)        |medium     |Try requential estimator as sanity check; epiplexity paper Section 6.1 chess example also showed weak correlation but the qualitative finding held                                         |Section 5.7 becomes “we measure epiplexity for the first time on a fluid dataset; the OOD correlation is inconclusive in our regime” — still publishable|
|Forecast horizon is short (H < 16)                    |medium-high|The Session 11 wake-probe finding that scalar wake info has r2=0.94 but spatial wake has r2=0.30 implies the latent does NOT carry enough spatial info for long rollout; honest expectation|Section 5.6 reports the measured horizon, however short, with comparison to Solera-Rico                                                                 |
|Concept vector arithmetic fails (relative error > 50%)|low        |The RBF probe R²=0.93 says nonlinear access works; if linear fails, the curvature is the story                                                                                             |Reframe as “the latent supports nonlinear but not linear (G,D,Y) access”                                                                                |
|Intrinsic dim estimators disagree wildly              |low-medium |Use the median of the four; report all four with error bars                                                                                                                                |Report the disagreement as the finding                                                                                                                  |
|Reverse-factorization training does not converge      |medium     |The inverse problem (forces → field) is genuinely hard; expect training instability                                                                                                        |Report negative result and use it to bound how much the epiplexity claim transfers from chess to fluids                                                 |
|Fukami head-to-head shows insignificant difference    |medium     |The Session 11 D81 finding (JEPA 2-4x better on every probe) suggests significance; if it doesn’t appear in SSIM, it should in GDY R²                                                      |Highlight the GDY R² gap, not SSIM, in the paper                                                                                                        |

## D-entries to record

- **D100**: Epiplexity measurement for the PREVENT vortex-gust dataset
  and the Session 12 configurations.
- **D101**: Forecast horizon evaluation results (Solera-Rico parity test).
- **D102**: Concept vector arithmetic results.
- **D103**: Intrinsic dimensionality estimators agreement test.
- **D104**: Reverse-factorization training result.
- **D105**: Fukami AE + LSTM head-to-head benchmark.
- **D106**: Session 14 outcome decision and paper Section 5 finalization.

## What this session DELIBERATELY does not do

- Image-resolution upgrade (192×96 → 384×192). The decoder is the
  bottleneck for visual blur, not the encoder. SL loss is helping;
  further sharpness requires diffusion refinement or perceptual loss,
  which is Session 15.
- Training-set expansion beyond v1.4 (65 cases). The data-shift D98
  finding says new data hurts spectral fidelity without SL retrain.
  Adding more data without rebuilding the recipe is dangerous.
- Diffusion model integration. The Session 11 plan flagged this as a
  Track 4 fallback; the SL + GAN combination has not yet been fully
  pushed.
- Higher Re. The PREVENT dataset is Re=5000. JFM and Nat. Commun.
  reviewers will want Re > 10⁴ for impact; we can defer this honestly
  as future work, since the methodology transfers.

## Headline figure design (Nat. Commun.)

If the session works as planned, the paper hero figure is:

**Figure 1 (2x2 panel)**:

- (a) Schematic: PREVENT vortex-gust setup + JEPA + wake observable
  head + LapFiLM + SL decoder.
- (b) Test B Figure 3 reconstruction (E d=64 + SL, frame 25/40/55)
  with wake spectral overlay showing |F(ω̂)| matches |F(ω)| at high
  wavenumbers.
- (c) Concept vector arithmetic: latent v_G direction interpreted via
  decoded fields at synthetic (G_0 + k * Δ).
- (d) Epiplexity vs Test C SSIM scatter across all Session 12 + 14
  configurations (the Nat. Commun. methodological differentiator).

**Figure 2 (1x3 panel)**:

- (a) Forecast horizon: RMSE(H) for JEPA + transformer at d=64 vs
  Fukami AE + LSTM at d=32.
- (b) Intrinsic dimensionality: four estimators agreeing at d ≈ 12.
- (c) Long-rollout decoded omega at H=64, side-by-side with DNS.

These two figures plus the existing wake-observable ablation table
plus the Fukami head-to-head table give the paper four
quantitative claims and four figures. That is a Nat. Commun.
candidate. Without Thrust 1 (epiplexity), it is a strong JFM paper.

## Verification before launch

1. The epiplexity paper (arXiv:2601.03220v2) is cited correctly. The
   prequential coding formula (Equation 8) and the requential coding
   formula (Equation 9) are implemented faithfully.
1. The AeroJEPA concept vector arithmetic (Equation 9) is implemented
   in our d=64 latent.
1. The Levina-Bickel 2004 MLE estimator is the standard intrinsic-dim
   estimator (DOI 10.1162/0899766054287873, NIPS 2004).
1. The Two-NN estimator is from Facco, d’Errico, Rodriguez, Laio,
   Sci. Rep. 7, 12140 (2017).
1. The Solera-Rico Nat. Commun. 15, 1361 (2024) forecast-horizon
   methodology is the comparison anchor for Thrust 2.

## Pre-registered predictions for accountability

1. Epiplexity-OOD Pearson correlation across Session 12 configs:
   credence 70% for |r| > 0.5. The epiplexity paper Section 6.1
   chess example showed positive correlation but with confounders;
   our fluid setup may have less noise OR more confounders.
1. Forecast horizon at the production config: credence 50% for
   H ≥ 32 (1.6 t/c). The Session 11 wake-probe finding bounds the
   spatial wake info available for rollout.
1. Concept vector linear extrapolation relative error: credence 60%
   for < 30%. AeroJEPA achieved similar gains at d=128 token-wise;
   our d=64 global-CLS may be worse but should be ballpark.
1. Intrinsic dimensionality agreement at 12 ± 2: credence 65%. PCA
   k=12 already established by Session 11.
1. Reverse-factorization OOD improvement: credence 35%. The chess
   analogy may not transfer to fluids cleanly; the (force → field)
   inverse problem may be too ill-posed.
1. Fukami significance: credence 80%. D81 already shows 2-4x
   probe gaps, which should be statistically significant.

Net credence Session 14 produces a Nat. Commun.-grade paper: 50%.
Net credence Session 14 produces a JFM-grade paper: 90%.

## Beyond Session 14: what would push to 90% Nat. Commun.

If Session 14 succeeds on Thrusts 1, 3, 6 but fails on 2, 4, 5:

- Session 15 launches diffusion refinement of the SL decoder
  (proven super-resolution upgrade that PRF paper explicitly
  recommended in their Conclusions).
- Session 16 builds a higher-Re dataset (Re=10000 or Re=20000) and
  trains the same recipe to validate transferability.

Both are major investments. The Nat. Commun. publishability does not
require them if Session 14 delivers Thrusts 1, 3, 6 + reasonable
Thrust 2.

## Reference verifications

- Finzi, Qiu, Jiang, Izmailov, Kolter, Wilson, “From Entropy to
  Epiplexity,” arXiv:2601.03220v2, 16 March 2026. Equation 8
  (prequential) and Equation 9 (requential).
- Balasubramanian, Cremades, Vinuesa, Tammisola, “Sharper
  Predictions,” PRF 11, 044907 (2026). DOI 10.1103/26js-tpg4.
- Giral, …, Vinuesa, “AeroJEPA,” arXiv:2605.05586, May 2026.
- Solera-Rico, …, Sanmiguel-Vila, …, Vinuesa, “beta-VAE +
  transformer,” Nat. Commun. 15, 1361 (2024).
- Fukami et al., Phys. Rev. Fluids 10, 084703 (2025); J. Fluid
  Mech. 1021, A39 (2025).
- Levina, Bickel, “Maximum Likelihood Estimation of Intrinsic
  Dimension,” NIPS 2004.
- Facco, d’Errico, Rodriguez, Laio, “Estimating the intrinsic
  dimension of datasets by a minimal neighborhood information,”
  Sci. Rep. 7, 12140 (2017).
- Isola, Zhu, Zhou, Efros, “Image-to-Image Translation with
  Conditional Adversarial Networks,” CVPR 2017,
  arXiv:1611.07004.