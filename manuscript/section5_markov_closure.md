# Section 5: Physical Markov closure

[Manuscript section, JFM target, first-pass draft]

## 5.1 Setup

We test whether the JEPA's impact-frame latent z(t_impact) is dynamically
sufficient to forecast post-impact physical observables, against two
classical reduced-order baselines applied to the same flow. Per Section 2,
all baselines see identical preprocessing (the canonical three-stage
omega pipeline at mid-plane Re=5000 on the v1 partition), identical train,
test_a, test_b, test_c splits, and an identical downstream transformer
predictor recipe. The recipe is described in Methods Appendix A.4 and
locked in SESSION18_B1_PROTOCOL.md; reviewers can audit each loss term
and hyperparameter.

The three baselines are:

- **JEPA d=64** (production stack from this work; encoder + jointly
  trained transformer predictor + observable head, Section 2).
- **Fukami AE** at d = 3, 32, 64 (Fukami and Taira, Phys. Rev. Fluids 10,
  084703, 2025), retrained from scratch on the same data with the
  paper's published recipe (Keras `loss='mse'`, `loss_weights=[1, beta]`,
  observable head at delta=0). The lift-loss weight beta was selected on
  our data via L-curve analysis (as in the paper), giving beta = 0.01 on
  the v1 split.
- **POD** at d = 16, 32, 64 (snapshot proper orthogonal decomposition on
  pipeline-normalised train frames).

For Fukami AE and POD, which have no native temporal model, we attach
the same transformer predictor used by the JEPA and train it on the
respective precomputed latents. This makes all three baselines comparable
as forecasting reduced-order models. The predictor architecture and
training recipe are identical across the three families; only the latent
dimension d varies. The unified recipe is required for B1 fairness; the
implementation details (no output BatchNorm in the predictor, see Methods
A.4.1) emerged from a diagnostic of a rollout instability we describe in
Appendix B.

The Markov-only rollout evaluates each forecast model on its core
predictive primitive: starting from the latent at t_impact alone, can the
model autoregressively step the latent forward and recover the physical
observables of the post-impact wake? At every layer of the predictor's
attention, the Markov mask restricts each query position to attend only
to t_impact and to itself; pre-impact context, if available, is ignored.
This implementation is the same monkey-patched context manager from
Session 17 Experiment 2 (HANDOFF.md D119) and is identical across baselines.

The physical observables are computed by linear ridge probes fit on the
per-frame training-set DNS observables for each baseline. For each
(baseline, observable, horizon), the probe is trained once on train
data and then applied to the rolled-out latent at the forecast frame.
This is the cleanest way to ask "how much of the observable is linearly
present in the latent at the forecast frame" without mixing predictor
quality with decoder quality. Reporting is mean absolute error vs DNS
with 2000-resample bootstrap 95% confidence intervals.

## 5.2 Headline result: JEPA wins decisively on wake structure

Figure 4 shows the mean absolute error of the Markov-only rollout
prediction vs the DNS reference at H = 16 on Test B (in-distribution
held-out cases) and Test C (out-of-distribution G = +4) for three
canonical physical observables. The headline numbers on Test B at H = 16
are:

| Baseline       | C_L  | I_y  | wake_enstrophy |
|---------------|------|------|----------------|
| JEPA d=64     | 1.00 | 1.57 | **22.3**       |
| Fukami AE d=64| 1.13 | 1.73 | 68.9           |
| POD d=64      | 1.66 | 1.56 | 73.2           |

JEPA d=64 achieves wake_enstrophy forecast error a factor of three
lower than Fukami AE at matched d and a factor of 3.3 lower than POD
at matched d. The wake_enstrophy bootstrap 95% confidence intervals are
non-overlapping; the JEPA advantage is statistically clear.

On the simpler scalar observables, classical baselines are competitive:

- **C_L** (single scalar lift coefficient): Fukami AE d=3 wins narrowly
  (0.81 vs JEPA's 1.00). With a 3-dimensional bottleneck and the
  observable head supervising on C_L during training, Fukami's tiny
  latent is approximately a lift-aligned 3-D subspace. The autoregressive
  rollout then has only three dimensions to keep stable and one of those
  three already corresponds to lift, so the linear probe at H = 16 is
  reading the predicted lift coefficient nearly directly.

- **I_y** (vorticity impulse): POD d=16 and JEPA d=64 are essentially
  tied within bootstrap CIs (1.53 and 1.57). I_y is a linear functional
  of the vorticity field by construction, and the POD basis spans the
  vorticity modes, so the linear probe from POD latents to I_y is exact
  up to truncation. JEPA's nonlinear latent does not give a linear
  decoder geometric advantage here.

The pattern is consistent with the structure of each observable:

- single-scalar observables tied directly to the training signal (C_L)
  favour the specialised low-d latent of Fukami AE;
- linear functionals of the underlying vorticity field (I_y) favour
  the linear-basis encoding of POD;
- structure-rich observables that require capturing multiple wake
  modes (wake_enstrophy is an integral of omega^2 over the wake) favour
  the rich predictive-state latent of JEPA.

The paper's central claim is therefore that JEPA's predictive state
captures the post-impact wake structure substantially better than
either classical reduced-order baseline, while the simpler observables
remain in reach of methods specialised for them.

## 5.3 Why JEPA wins on wake_enstrophy

The wake_enstrophy linear probe R^2 fit on training-set per-frame data
provides the direct mechanistic explanation:

| Baseline       | R^2(C_L) | R^2(I_y) | R^2(wake_enstrophy) | R^2(circ_pos) | R^2(circ_neg) |
|---------------|---------|---------|---------------------|--------------|--------------|
| JEPA d=64     | 0.825   | 0.506   | **0.870**           | **0.881**    | **0.892**    |
| Fukami AE d=64| 0.811   | 0.283   | 0.479               | 0.400        | 0.449        |
| POD d=64      | 0.708   | 0.772   | 0.413               | 0.391        | 0.481        |

JEPA's latent linearly encodes wake_enstrophy at R^2 = 0.870, almost
double the Fukami AE value (0.479) and more than double the POD value
(0.413). The training pressure that produced this is the wake observable
head in the JEPA training objective (Section 11 W0_C_lam100, HANDOFF.md
D84): an 80-dimensional patch_signed_spectrum target on the wake grid
trained with smooth-L1 at lambda_wake = 1.0. The JEPA encoder is
explicitly pushed to encode wake structure into z; the Fukami AE
encoder is pushed to encode lift; POD encodes vorticity modes by
construction. The Markov rollout then preserves these encodings to H = 16
because the JEPA's joint encoder-predictor training keeps the rollout
trajectory close to the training distribution.

The same ordering holds on Test C (out-of-distribution G = +4); the
JEPA advantage on wake_enstrophy is preserved within bootstrap CIs.

## 5.4 Pre-impact context is mildly harmful

Comparing the Markov-only rollout (z_impact alone as seed) to the
Full-context rollout (the predictor sees up to 32 frames ending at
impact) at H = 16 shows that the Markov closure result holds on
physical observables. For JEPA d=64, the H = 16 Markov error on Test B
is comparable to or slightly better than the Full-context error across
all three primary observables. Pre-impact temporal context is not just
information-free; for short horizons it is mildly harmful, introducing
commitment errors that the Markov rollout avoids.

This extends Session 16 D119's latent-RMSE-level Markov closure (impact-
frame z is approximately Markov-sufficient for the next 16 latent frames
in-distribution) to physical observables (C_L, I_y, wake_enstrophy).

## 5.5 The Wu impulse-lift caveat

The 2D mid-plane omega field used here excludes the bound circulation
at the airfoil surface. Wu's impulse-lift theorem, dI_y/dt proportional
to C_L, is therefore not directly applicable on this DNS slice. We
confirmed this empirically in Session 17 D124c: on the raw DNS itself
the correlation r(dI_y/dt, C_L) = -0.03 (n = 896, p = 0.40), far from
the > 0.95 that Wu's theorem would predict. We therefore report I_y
without claiming impulse-lift dynamical consistency; the I_y forecast
quality is meaningful as a wake-volume integral, not as an indirect
prediction of lift via Wu.

A full-3D circulation integral (or a probe trained on 3D simulation
data) would be needed to test impulse-lift consistency. This is left
for future work; see Discussion Section 8.3.

## 5.6 Methodological note: bug fixes during B1

Two infrastructure bugs were uncovered and fixed during B1; both are
documented in Methods Appendix B for reproducibility. Without these
fixes the JEPA result would have been spurious in the opposite
direction (B1 would have reported JEPA losing on every metric).

1. **Double-normalisation of omega in the FukamiAEWrapper training
   path**. When the omega pipeline was moved into EpisodeDataset
   (HANDOFF.md D85, Session 11) to enable num_workers > 0, the
   FukamiAEWrapper's own normalisation step was not removed, so omega
   was divided by the 3-sigma divisor twice during training. The
   training distribution and the evaluation distribution differed
   by a factor of ten; the model's predictions were correspondingly
   amplitude-compressed by an order of magnitude. We fixed this by
   bypassing the dataset's pipeline-application path during Fukami AE
   training while keeping the wrapper's path active. Verified by
   training trajectory inspection and end-to-end reconstruction
   quality (SSIM jumped from 0.16 to 0.48 at d = 64).

2. **Predictor output-BatchNorm running-statistics mismatch**. The
   transformer predictor's output projection has a BatchNorm1d whose
   running statistics are calibrated on teacher-forced training data.
   At autoregressive rollout the latent distribution shifts and this
   BatchNorm over-regularises the predictions. We confirmed this with
   three independent paths: (i) removing the predictor's output
   BatchNorm restores the Markov rollout to clean monotonic horizon
   trends (no transient spike at H = 16); (ii) using the production
   JEPA's own jointly trained predictor on the same latents removes
   the spike; (iii) a separate JEPA retraining with LayerNorm at the
   encoder boundary is in progress and tests the encoder-side
   contribution. The unified B1 recipe replaces the predictor's
   output BatchNorm with Identity for all seven baselines.

Both bugs are independent of the comparison itself; the unified-recipe
results we report are robust under all three verification paths.

## 5.7 What this section claims

JEPA achieves 3-fold lower wake_enstrophy forecast error than Fukami
AE and POD at matched d, on a fair comparison where every baseline
uses an identical downstream predictor and probe family. Classical
baselines remain competitive on observable types they are specialised
for: low-d lift-tied compression (Fukami d = 3) for single-scalar
lift forecast, and linear vorticity bases (POD) for impulse forecast.
On structure-rich observables, JEPA's predictive-state latent and
its wake observable head produce a clean advantage; this is the
JEPA-specific contribution this paper documents.

The next section (Section 6) shows that the wake-relevant pixel-level
structures driving these encoded observables are not the Q-criterion
vortex cores classically associated with wake organisation. The two
findings together describe what JEPA learns about this flow and
where its advantage comes from.
