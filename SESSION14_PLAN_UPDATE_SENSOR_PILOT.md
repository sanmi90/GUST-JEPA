# SESSION14_PLAN_UPDATE_SENSOR_PILOT.md

Update to `SESSION14_JFM_NATCOMM_PUSH.md` after critical evaluation of the
GPT-authored sensor-selection proposal. Adds a scoped pilot of the GPT
track as a seventh thrust; defers the full sensor track to Session 15.

Date: 2026-05-24.

## Critical evaluation of the GPT proposal

The GPT proposal is “epiplexity-guided sparse pressure sensor selection.”
It is a serious, well-reasoned plan that uses target-conditioned
structural-information proxies (S_preq, H_res, G, Eff) to choose K of 192
surface pressure taps for force / phase / latent / wake estimation. The
plan includes four selection algorithms (greedy, beam, backward pruning,
ADO gates), seven baseline selectors (uniform, random, qDEIM, etc.), four
objective profiles (balanced, force_phase, field_physics,
latent_digital_twin), and a full robustness sweep (noise, dropout,
leave-one-out).

This is a significant chunk of work. ~30-40 hours including
implementation, training, evaluation, and figure generation.

### What the GPT proposal gets right

1. **Target-conditioned, not marginal.** GPT explicitly distinguishes
   S_T(Y | p_S) from raw epiplexity of pressure traces. This is the
   right reading of the Finzi et al. paper (Definition 11 of
   arXiv:2601.03220v2 defines conditional epiplexity exactly for this
   use). Raw epiplexity of a pressure trace could be high because the
   trace is chaotic noise, not because it carries reusable physics. The
   conditional form is the right primitive.
1. **The four-proxy decomposition (G, H_res, S_preq, Eff) is sound.**
   Structural gain G = N × max(0, L_null − L_star) is “useful captured.”
   Residual entropy H_res = N × L_star is “irreducible noise.”
   S_preq is “how hard the learning was.” Eff = G / S_preq is the
   ratio. The objective J(S) combining these with the right sign
   conventions matches the epiplexity paper’s intuition.
1. **The leakage rule is good practice.** Train and Test A may be used
   for selection; Test B and Test C are frozen. This is the discipline
   most fluid ML papers omit.
1. **The baseline list is comprehensive.** qDEIM (Manohar et al.),
   greedy_CL, correlation_CL, random_K with 50 seeds, uniform_K, and
   anatomical baselines (leading edge, trailing edge, suction side,
   pressure side). This is the right way to demonstrate the method
   beats both naive and sophisticated alternatives.
1. **The objective profiles map to real deployment.** force_phase,
   field_physics, latent_digital_twin are not academic categories;
   they correspond to “what is this sensor array for?” Different
   industrial applications care about different targets. Reporting
   selected sensors under each profile is genuinely useful.

### What is wrong, missing, or oversold

1. **“Epiplexity-guided” is overclaiming for what is implemented.** GPT
   acknowledges this in passing (“Do not call it exact epiplexity in
   the main paper unless the estimator is properly likelihood-
   calibrated”) but then names the entire track “epiplexity-guided”
   anyway. The Finzi et al. paper defines epiplexity in bits via a
   careful prequential coding scheme that REQUIRES the loss to be a
   negative log likelihood. Ridge regression and TCN regression with
   MSE loss do NOT give epiplexity in bits without calibration. They
   give something proportional to it for Gaussian residuals, with an
   unknown constant. Publishing as “epiplexity-guided” without that
   caveat front and centre will be rejected by a careful reviewer (and
   Vinuesa or coauthors are likely peer reviewers given the PRF and
   AeroJEPA citation chain). **The honest name is “conditional
   structural-information-guided sensor selection, inspired by the
   epiplexity framework.”**
1. **The proxy learner choice is underdetermined.** Different model
   classes give different time-bounded entropy and epiplexity (the
   whole observer-dependent framing of the original paper). If we
   screen with ridge and TCN, we are reporting “structural info at
   the ridge complexity level” or “at the TCN complexity level,” and
   these can differ substantially. The paper needs to commit to a
   model class for the headline measurement. **Recommendation: use
   the same architecture as the final pressure-to-field estimator
   for the headline measurement, and report ridge as the cheap
   screen.**
1. **The leakage rule has a subtle escape hatch.** Hyperparameter
   tuning of the proxy learner on Test A IS information leakage. The
   correct protocol is train-internal CV for hyperparameter tuning,
   Test A for sensor selection scoring, Test B/C for final
   evaluation. GPT’s language conflates these.
1. **The ADO-style sensor curriculum (C*.6) is over-engineered for
   the marginal value.** Differentiable gates with annealing are
   theoretically clean but fiddly in practice. Greedy + beam + backward
   pruning + bootstrap stability selection already cover the space.
   **Defer ADO; do not implement in Session 14 or 15 unless the
   simpler methods produce unstable selections.**
1. **The “coherent structures” target family requires vortex
   detection that does not exist.** Number of positive vortices,
   top-k centroids, top-k circulation: these need a Γ_1, Γ_2,
   Q-criterion, or λ_2 implementation that the project has not
   built. Adding it is ~300 lines plus a threshold calibration headache.
   Each vortex detection method gives different answers; the choice is
   itself a methodological commitment that needs justification.
   **Recommendation: drop the coherent-structure target family in
   Session 14 (defer to Session 15 or beyond); keep latent, force,
   phase, wake scalar, spectra/POD.** The spectra/POD family already
   carries most of what coherent-structure detection would tell us.
1. **The 1536 individual sensor option needs verification.** GPT
   writes “raw /sensors/p with 192 surface points times 8 spanwise
   stations.” Whether this is directly accessible in the project cache
   needs to be checked. Default to the (T, 192) p_wall and defer the
   1536 variant.
1. **The objective profile weights are unmotivated.** “latent 0.25,
   force 0.25, phase 0.20, wake scalar 0.10, coherent 0.10,
   spectra/POD 0.10” — where do these come from? They are guesses.
   The proper move is to report results across all four profiles and
   show the selected sensors are stable. Or treat the weights as
   themselves hyperparameters. GPT proposes the stability check, which
   is fine; but the language should not present the weights as
   “balanced” without evidence.
1. **The “field_SSIM” column conflates two things.** Field SSIM from
   sensors requires a pressure-to-field map (a separate model) OR
   inverting the pressure → JEPA latent → LapFiLM decoder pipeline. The
   choice matters; both should not be reported under the same column.
1. **The proposal does not address the central strategic question:
   does the sensor track belong in this paper, or in a follow-up?** It
   is a major scope addition. For JFM mainline, sensor selection is
   borderline (JFM cares about physics; sensor selection is methods).
   For Nature Communications, it strengthens the paper by adding a
   second methodological contribution and a deployment story. The plan
   should commit.

## Strategic decision

**For JFM submission**: do not include the sensor track in the paper.
The JFM paper is JEPA-for-vortex-gust with rigorous head-to-head vs
Fukami and Solera-Rico, plus the intrinsic dimensionality and concept
vector arithmetic findings. Add sensor selection as a follow-up paper
to PRF or J. Comput. Phys.

**For Nature Communications submission**: include sensor selection as a
second methodological contribution. The paper now has two named
methods (JEPA + wake observable head + SL decoder; and
target-conditioned structural-information sensor selection) plus the
intrinsic dimensionality finding. Three independent contributions in
one paper is Nat. Commun.-grade scope.

**The pragmatic path**: do a SMALL pilot of the sensor track in
Session 14 (4-6 hours of scoped work). The pilot tests feasibility,
produces one figure, and lets us decide whether Session 15 is the full
sensor track or whether we revert to diffusion refinement / higher Re.
**Option D**, in the language of the four options I considered.

## Thrust 7: sensor selection pilot (4-6 hours)

Scope: single objective profile (balanced), single proxy learner
(ridge for screening + TCN for confirmation), three target families
(latent, force, phase), three baselines (uniform_K, random_K_median,
qDEIM_pressure), three K values (8, 16, 32). One figure. One decision
gate.

### Thrust 7a. Implement the conditional structural-information proxies

Module `src/evaluation/conditional_structural_information.py`:

```python
def compute_proxies(pred_loss_curve, target, null_predictor):
    L_star = pred_loss_curve[-1]  # final loss
    L_null = null_predictor.loss(target)  # mean predictor
    N = len(target)
    S_preq = sum(max(0, L_i - L_star) * dN_i for L_i, dN_i in pred_loss_curve)
    H_res = N * L_star
    G = N * max(0, L_null - L_star)
    Eff = G / (S_preq + 1e-6)
    return dict(S_preq=S_preq, H_res=H_res, G=G, Eff=Eff)
```

~150 lines plus tests. The proxy returns the four scalar quantities;
the objective J(S) is a separate function that combines them.

### Thrust 7b. Individual sensor screening (192 ridge regressions)

For each of 192 sensors and three targets (z_oracle from E d=64 + SL
encoder, C_L(t), impact phase τ): train a ridge regression p_j[t-W+1:t]
to target, compute the four proxies. Save the per-sensor heatmap.

W = 17 only for the pilot (the GPT proposal swept 5 windows; defer).
Ridge regression is fast (<1 second per fit); 192 sensors × 3 targets
× 5 CV folds is ~3000 fits, fits in 10 minutes on a laptop.

Output: `outputs/session14/sensor_pilot/individual_screening.csv` and
a heatmap figure showing structural gain G across sensor index and
target family.

### Thrust 7c. Greedy forward selection at K=8, 16, 32

For each K and each of three target weighting schemes (force_heavy,
latent_heavy, balanced), run greedy forward selection using ridge as
the inner learner. Track the J(S) trace.

Output: `outputs/session14/sensor_pilot/greedy_K{8,16,32}.json` with
selected sensor indices.

### Thrust 7d. Compare against three baselines

uniform_K (every 192/K-th sensor), random_K with 50 seeds (report
median + 95% interval), qDEIM_pressure_K (need implementation; ~80
lines from Manohar et al. 2018 algorithm).

For each (selector, K) compute:

- z_R2 (E d=64 latent regression R²)
- C_L_present_R2
- impact_phase_RMSE_frames

Output: `outputs/session14/sensor_pilot/baseline_comparison.csv`.

### Thrust 7e. Pilot decision gate

Pass conditions (Session 15 launches full sensor track):

- Conditional-SI-guided K=16 beats uniform_K AND random_K_median on
  at least 2 of 3 target metrics, with a margin > 1 standard deviation
  of the random_K distribution.
- Conditional-SI-guided K=16 reaches z_R2 > 0.85 AND C_L_R2 > 0.95
  AND impact_phase_RMSE < 3 frames.
- At least one direction (k=16 OR k=32) is within 5% of all_192
  performance on z_R2.

Fail conditions (Session 15 reverts to diffusion refinement):

- The conditional-SI selection performs WORSE than random_K_median.
- z_R2 < 0.7 at K=16 (the latent is not recoverable from sparse
  pressure even with smart selection).
- qDEIM is comparable to conditional-SI (no methodological gain).

### Thrust 7f. Documentation

If the pilot passes: write `SESSION15_SENSOR_TRACK.md` integrating
the GPT proposal with the critical fixes (proper naming, model class
commitment, proper validation split, drop coherent-structure target
for now, drop ADO, drop 1536-sensor variant).

If the pilot fails: write the honest negative result in the
Session 14 report as one paragraph plus one figure showing the
failure mode.

### Total Thrust 7 budget

- 7a (proxies): 1.5 hours.
- 7b (192 ridge fits + heatmap): 1 hour wall-time, mostly compute
  parallelism.
- 7c (greedy K = 8, 16, 32, three profiles): 1.5 hours.
- 7d (three baselines): 1 hour.
- 7e (decision figure): 30 min.
- 7f (documentation): 30 min.

Total: ~6 hours. Fits within Session 14’s existing wall-clock
budget. No additional GPU time required (ridge fits on CPU).

## How Thrust 7 changes the Session 14 plan

Thrust 7 ADDS to the six existing thrusts, does not replace any.
The total Session 14 budget shifts from 20-25h to 26-31h. With
parallel execution this is still one long day plus one short day.

The headline figure design changes IF Thrust 7 passes:

**Figure 1 becomes 2×3** rather than 2×2:

- (a) Schematic.
- (b) Test B Figure 3 reconstruction with spectral overlay.
- (c) Concept vector arithmetic.
- (d) Epiplexity vs Test C SSIM scatter.
- (e) Intrinsic dimensionality agreement.
- (f) Sensor selection pilot result: airfoil with K=16 selected
  sensors + a small bar chart of z_R2 vs selector.

**Figure 2 stays 1×3** (forecast horizon, intrinsic dim, long rollout).

**New figure 3** (if Session 15 runs):

- Full sensor track results from the GPT proposal as the paper’s
  practical deployment demonstration.

## Renaming the contribution honestly

The paper-facing name for what GPT calls “epiplexity-guided sensor
selection” should be:

**“Target-conditioned structural-information sensor selection (TCSI),
inspired by the epiplexity framework of Finzi et al. 2026.”**

This is honest because:

- It IS target-conditioned (S_T(Y | X), not S_T(X) alone).
- It IS a structural-information criterion (the four proxies decompose
  loss-curve area, not Shannon entropy).
- The connection to epiplexity is by inspiration / motivation, not by
  formal equivalence (the loss is not log-likelihood-calibrated).
- TCSI gives the method a name without overclaiming.

The Session 14 report should adopt this naming consistently. The
relationship to epiplexity is stated explicitly in the methods section
and the discussion, with a paragraph explaining what would be needed
to upgrade TCSI to a formal epiplexity measurement (requential coding
of a likelihood-based estimator).

## Critical reference: the connection to JEPA encoder epiplexity (Thrust 1)

Thrust 1 of the original Session 14 plan measures JEPA encoder
epiplexity directly via prequential coding on the negative log
likelihood (or its surrogate, the SIGReg + L_pred loss). That IS a
direct application of the Finzi et al. estimator because the JEPA
loss is calibrated to a log-likelihood interpretation under standard
assumptions.

Thrust 7 (sensor selection) uses MSE-based proxy learners that are
NOT log-likelihood-calibrated. So:

- **Thrust 1**: “We measure epiplexity of the JEPA encoder via
  prequential coding (Finzi et al. 2026 Equation 8).”
- **Thrust 7**: “We propose TCSI, a target-conditioned structural-
  information criterion for sensor selection, inspired by the
  epiplexity framework.”

This distinction matters. The paper can use both honestly. The first
is a direct application of an information measure. The second is a
selection heuristic derived from the same intellectual framework.

## Final structural recommendation

Session 14 runs all seven thrusts. The seventh is the pilot. The
decision gate at the end of the pilot determines Session 15.

Pre-registered prediction for the pilot:

- Conditional-SI K=16 beats uniform_K and random_K_median: credence 70%.
  The pressure surface is small (192 chordwise) and the physics
  (impact dynamics, lift response) is concentrated on a few key
  locations (leading edge, suction-side separation, trailing edge).
  Smart selection should beat uniform spacing.
- z_R2 > 0.85 at K=16: credence 40%. This depends on whether the
  pressure surface contains enough information to recover a 64D
  latent. The Fukami JFM 2025 pressure-to-flow paper achieves
  reasonable reconstruction at K=20 pressure sensors for similar
  geometry, so it is plausible but not certain.
- qDEIM gap exists: credence 60%. qDEIM optimises for pressure
  reconstruction, not for downstream latent / force / phase
  prediction. The target-conditioned objective should beat it on
  non-pressure targets.

Net credence the pilot passes: 50%. If it does, Session 15 is the
full GPT sensor track (with the critical naming and scoping fixes).
If it does not, Session 15 reverts to diffusion refinement of the
SL decoder.

## Updated D-entries

- **D100**: Epiplexity measurement for the PREVENT vortex-gust dataset
  and the Session 12 configurations. **Direct application of Finzi
  et al. prequential coding.**
- **D101**: Forecast horizon evaluation.
- **D102**: Concept vector arithmetic.
- **D103**: Intrinsic dimensionality agreement test.
- **D104**: Reverse-factorization training.
- **D105**: Fukami head-to-head with confidence intervals.
- **D106**: Session 14 outcome decision.
- **D107**: TCSI sensor selection pilot. **Inspired by but not
  equivalent to Finzi et al. epiplexity. Names and protocol fixed per
  the critical evaluation in this document.**

## Why I am not adopting the full GPT proposal as-is

1. The naming (“epiplexity-guided”) would not survive critical peer
   review. Vinuesa is on the PRF 2026 SL paper that Direction A
   adopts; he or a coauthor is the most likely peer reviewer for our
   paper; the AeroJEPA team is in the citation neighbourhood. They
   know the epiplexity literature better than we do at this point. If
   our paper uses “epiplexity” loosely, they will catch it and reject.
1. The scope (30-40h, 6 scripts, full robustness sweep, 19 baselines)
   is too large to commit to before knowing whether the underlying
   pressure-to-latent map is solvable at all. The pilot tests the
   solvability first.
1. The vortex-detection-based “coherent structures” target adds
   methodological scope that does not align with the project’s
   existing infrastructure. Deferring it keeps the pilot clean.
1. The ADO gates are interesting research but not the right vehicle
   for a first-publication paper.

The GPT proposal will get a fair test in pilot form. If it works, the
full track gets a clean run in Session 15. If not, the paper still
has the six other Session 14 contributions and remains JFM/Nat. Commun.
viable.

## What the critical reading leaves us with

GPT’s instinct is right: there is a publishable contribution in the
intersection of (epiplexity framework) × (sensor selection) ×
(parametric vortex-gust). The criticism is about how to do it without
overclaiming and without exploding the scope.

The TCSI pilot in Thrust 7 captures the core idea:
target-conditioned structural-information proxies, the four-quantity
decomposition, the leakage discipline, and the comparison against
sophisticated baselines. It does so in a way that:

- Names the contribution honestly (TCSI vs claiming epiplexity).
- Commits to one proxy class (ridge for screening, TCN for
  confirmation).
- Tests feasibility before committing to the full plan.
- Keeps Session 14 finishable in a reasonable window.

If the pilot succeeds, the paper gets a second methodological
contribution that pushes it from “strong JFM” to “credible Nat.
Commun. candidate.” If it fails, the paper does not lose anything
because the pilot was a 6-hour scoped investment.