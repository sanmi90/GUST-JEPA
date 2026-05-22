# SESSION12_CRISP_WAKE.md

Session 12 plan: push wake reconstruction past blurry to crisp.
Six parallel attack directions, all run because the budget is
irrelevant and the goal is publication-grade results.

Last updated: 2026-05-22.

## Framing

Session 11 cleared its numerical success criteria. The winning
configuration W0_C_lam100 + E1 decoder retrain produced Test B
SSIM 0.523 and wake_enstrophy_rel_err 0.431. Both Session 11
thresholds passed.

But the Figure 3 from W0_C_lam100 shows the wake is **present but
blurry**. The gust core is well reconstructed at all three frames.
The wake placement is correct. The wake amplitude is correct on
average. The wake’s fine-scale structure (the small-vortex texture
that makes a vortex street look like a vortex street rather than a
smooth lobe) is missing. The residual rows show this directly: the
high-frequency wake content is what is left in the difference field.

The user’s description: “We improve wake getting a bit blurry wake
in particular in lambda wake =1.”

This is the gap between “passes SSIM threshold” and “publication-
grade Figure 3 for a high impact journal.” Session 12 closes that
gap. Six attack directions, all run in parallel because we have two
RTX 6000 Blackwell cards plus the option to use cloud GPU if
necessary, and the only constraint is the quality of the final
result.

## Critical input: Balasubramanian et al. PRF 2026

A paper directly relevant to our exact failure mode was published
in April 2026 in Physical Review Fluids (arXiv companion not used
here; the DOI is 10.1103/26js-tpg4, PRF 11, 044907 (2026)).
Balasubramanian, Cremades, Vinuesa, Tammisola, “Sharper Predictions:
The role of loss functions for enhanced turbulent-flow sensing.”

The paper compares three losses on near-wall turbulence
reconstruction at Re_tau = 180 and 550:

- **MSL**: pure mean squared error (the conventional baseline).
- **CL** (composite loss): MSE + amplitude matching + correlation
  coefficient. Equivalent in spirit to our region + enstrophy +
  circulation construction.
- **SL** (spectral loss): CL + gradient consistency (Frobenius norm
  of gradient difference) + 2D Fourier amplitude difference (L1 norm
  of FFT magnitude difference).

Key findings directly applicable to our problem:

1. **The spectral loss recovers small-scale energy content that MSL
   underestimates.** Their Figures 5 and 6 show the 2D premultiplied
   power spectral densities of (u, v, w) at y+ = 30, 50, 100 for the
   three losses. The SL contours align with DNS at all wavelengths
   including the small-scale tail. MSL underestimates energy at
   high wavenumbers, producing reconstructions that look like
   ours: large-scale right, small-scale blurry.
1. **The spectral loss at Re_tau=550 gives a 2x improvement** in
   relative RMS error for streamwise component at y+=50 (7.31% MSL
   vs 3.71% SL). The improvements persist at all wall-normal
   positions and Reynolds numbers tested.
1. **The spectral loss is GLOBAL FFT amplitude difference**, not the
   per-patch focal-frequency loss (FFL) we tried in Session 10. We
   have never tested the global spectral amplitude formulation.
1. **The spectral loss is robust to input noise (up to 100 percent
   Gaussian) and to spatial coarsening (F = 2 to 8).** This is
   relevant because if it works for noisy wall measurements, it
   should work for our compressed-latent decoding.
1. **The paper explicitly recommends GAN or diffusion refinement as
   the natural next step** after spectral loss. Section IV’s closing
   paragraph and the Conclusions both flag generative models for
   further improvement.

The PRF paper’s loss formulation is the obvious thing to add to
our recipe. Its physical motivation (turbulence has a wide range
of scales and MSE-based losses smooth them out) matches our wake
reconstruction problem exactly.

## What this session does

Six attack directions, all run in parallel where possible. The
session ends when at least one configuration reaches **Test B SSIM

> = 0.60 AND wake_enstrophy_rel_err <= 0.40 AND visual Figure 3
> with crisp wake AND 2D premultiplied wake power spectrum within
> factor 2 of DNS at high wavenumbers**.

The last criterion is new for Session 12. Following Balasubramanian
et al. PRF 2026 Figure 5 methodology, we compute the 2D premultiplied
power spectral density kx*kz*phi_omega(lambda_x, lambda_z) for both
prediction and target, in the wake ROI, and require the 10%, 50%,
and 90% contours to align within a factor 2 in wavelength space.

If multiple configurations pass, the winner is whichever produces
the best wake spectrum agreement (the strongest paper claim).

## What Session 12 deliberately does NOT do

- Lower expected-impact ablations (E3 params_phase conditioning,
  bilinear_conv upsampling, etc).
- The full Section 7 evaluation matrix (separate Session 13 task).
- POD comparisons beyond what already exists in HANDOFF.
- Section 7c isomap diagnostic extensions.

These are valid but lower-priority work for a Session 13 paper
finalisation pass after Session 12’s results are in.

## Locked decisions carried forward

- Frame-skip 1, L=32 = 1.6 t/c (D34).
- Partition v1.2 (D35).
- Two RTX 6000 Blackwell cards on cuda:2 and cuda:3 (D38, D40).
- Omega pipeline v1; losses in normalized space; raw scale for
  metrics and figures (D71 bug-fix).
- The Session 11 W0_C_lam100 encoder is the starting point for
  decoder-only directions (A, B). The encoder is retrained from
  scratch for directions C, D, E, F.

## Reference architecture: PRF 2026 SL loss

Implement the Balasubramanian SL loss exactly as written. From
their Equations (6)-(8):

```
L_SL = alpha * L_MSE
     + beta  * (RMS_pred - RMS_target)^2
     + gamma * (1 - rho(pred, target))^2
     + delta * L_gradient
     + zeta  * L_spectral_amplitude

where

L_gradient(pred, target) = (1/N_b) * sum_b || grad(pred_b) - grad(target_b) ||_F^2

L_spectral_amplitude(pred, target) = (1/N_b) * sum_b || |F(pred_b)| - |F(target_b)| ||_1

F = 2D Fourier transform along the homogeneous (x, z) directions.
```

The paper uses alpha = beta = gamma = delta = zeta = 1.0 as defaults.
Their Appendix B sensitivity study shows performance is roughly
robust to coefficient choice within +/-0.3 in relative weights. We
use the default for the first run; sweep coefficients if needed.

Implementation: ~80 lines in
`src/models/decoder_losses.py:spectral_amplitude_loss()` and
`gradient_consistency_loss()`. The losses are computed on the
PREDICTED FIELD after decoder, comparing to the target field, in
the SAME normalized space as the existing pyramid loss (D71
discipline).

Verified citation: Balasubramanian, Cremades, Vinuesa, Tammisola,
“Sharper Predictions: The role of loss functions for enhanced
turbulent-flow sensing,” Physical Review Fluids 11, 044907 (2026).
Equations 6-8 of the paper.

## Direction A: PRF 2026 spectral loss on W0_C_lam100 decoder

**Run S12_A_specloss** on cuda:2.

Encoder: frozen W0_C_lam100 from Session 11.
Decoder: LapFiLM with the existing E1 recipe (region + pyramid +
enstrophy + circulation) PLUS the new SL spectral-amplitude term
and gradient-consistency term.

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/session9_train_decoder.py \
    --omega-pipeline-manifest outputs/data_pipeline/v1/manifest.json \
    --encoder-run outputs/runs/session11/W0_C_lam100 \
    --decoder-type lapfilm \
    --decoder-upsample pixelshuffle \
    --decoder-cond none \
    --decoder-loss region_pyr_specloss \
    --lambda-region 1.0 \
    --lambda-pyramid 0.4 \
    --lambda-gradient 1.0 \
    --lambda-spectral-amp 1.0 \
    --lambda-enstrophy 0.02 \
    --lambda-circulation 0.01 \
    --max-iters 30000 \
    --B 16 --T 32 --seed 42 \
    --output-dir outputs/runs/session12/S12_A_specloss_default
```

30k iters (not the standard 20k) because spectral losses are slower
to converge and the PRF paper used 75-250 epochs.

Additionally run two variants:

- **S12_A_specloss_low**: lambda_gradient = 0.3, lambda_spectral = 0.3 (lower).
- **S12_A_specloss_high**: lambda_gradient = 3.0, lambda_spectral = 3.0 (higher).

Sweep the relative weight of the new spectral terms. The PRF paper’s
Appendix B suggests the defaults are reasonable but our problem may
benefit from heavier spectral weight given the high-frequency wake
content.

Expected outcome: Test B SSIM > 0.58, wake spectrum agreement at high
wavenumbers improves to within factor 2 of DNS, residual reduces in
the wake region.

**Risk**: PRF paper found pure MSE could slightly increase at large
y+ when SL is applied (their Section III.A.2 discussion). The
SSIM/MSE tradeoff is real. Mitigation: report both metrics; the
paper claim shifts to “spectral fidelity at small wake scales”
rather than “best SSIM.”

## Direction B: GAN refinement of LapFiLM

**Run S12_B_gan_refine** on cuda:3.

Take the W0_C_lam100 + E1 decoder output. Train a small discriminator
to distinguish predicted omega fields from true omega fields in the
wake ROI. Use the discriminator as an adversarial loss on a generator
that takes the coarse LapFiLM prediction and refines wake details.

Architecture:

**Generator (refiner)**: takes the LapFiLM output (192x96) and the
target wake ROI mask. Produces a residual map (192x96) that is added
to the LapFiLM output. Architecture: 6-block ResNet with GroupNorm,
SiLU, 64 channels, ~200k parameters.

**Discriminator**: patchGAN style (Isola et al. 2017, pix2pix). 4-layer
CNN with leaky ReLU, instance norm. Takes (predicted field, true
field, wake_ROI_mask), outputs a 24x12 patch decision map. ~150k
parameters.

**Loss**:

- L_gen = L_recon(refined_pred, target) + 0.05 * L_adv(refined_pred)
- L_disc = standard hinge loss

L_recon is the existing region + pyramid + enstrophy + circulation
recipe. The adversarial weight 0.05 follows the pix2pix convention.

Refiner trains for 20k iters with the discriminator. The LapFiLM
weights are frozen during refiner training (this is a two-stage
training, not joint).

Implementation: ~400 lines new code in
`src/models/refiner.py` and `src/models/discriminator.py`.

Expected outcome: visually crisper wake, possibly at small SSIM cost
(GAN losses sometimes trade pixel accuracy for perceptual quality).
This is the proven super-resolution upgrade path.

**Citation**: Isola, Zhu, Zhou, Efros, “Image-to-Image Translation
with Conditional Adversarial Networks,” CVPR 2017, arXiv:1611.07004.

## Direction C: extended lambda_wake ladder

**Run S12_C_lam200, S12_C_lam300, S12_C_lam500** on whichever cards
are free.

Session 11 lambda ladder was monotonic in Test B SSIM: 0.358 (S9
baseline) -> 0.419 (lam=0.1) -> 0.451 (lam=0.1) -> 0.472 (lam=0.3)
-> 0.482 (lam=0.5) -> 0.523 (lam=1.0). The relationship has not
saturated. Extend the ladder upward.

Three runs at lambda_wake in {2.0, 3.0, 5.0}, encoder retrained from
scratch with Mode C wake target. After each encoder finishes, retrain
the LapFiLM decoder with the E1 recipe.

Each encoder run is 20k iters (matches Session 11 production budget).
Decoder retrain is 20k iters.

```bash
# Example for S12_C_lam200 encoder
CUDA_VISIBLE_DEVICES=2 python -m src.training.train_jepa \
    --partition v1 --all-train --max-iters 20000 --seed 42 \
    --observable-head cl_future \
    --observable-head-weight 0.01 \
    --wake-observables-path outputs/data_pipeline/v1/wake_observables_patch8x4_signed_spec16.h5 \
    --wake-observable-type patch_signed_spectrum \
    --wake-head-dim 80 \
    --lambda-wake 2.00 \
    --wake-head-deltas 0 8 16 24 \
    --projection-norm batchnorm --anticollapse sigreg \
    --lambda-sigreg 0.01 \
    --output-dir outputs/runs/session12/S12_C_lam200/encoder
```

Expected outcome: continued monotonic gain in SSIM and wake metrics
up to some saturation point. PR(z) was 11.66 at lambda=1.0; extending
may take it to 15-20. The encoder may eventually saturate at PR
roughly equal to d (32) at which point wake observable head pressure
is fully reflected. This is the natural ceiling of the lambda
direction.

**Risk**: too-high lambda_wake may finally damage GDY enough that
prediction quality (forecasting) suffers. Run the full Session 5-8
diagnostic suite on each new encoder: r2(z->c) on Test B for
G/D/Y/CL, plus r2(z->CL_future) at deltas {8, 16, 24}. If forecasting
degrades by more than 10 percent, the encoder is no longer
contribution-claim-compatible regardless of reconstruction gains.

## Direction D: higher-dimensional wake observable target

**Run S12_D_coarse288, S12_D_coarse512** on whichever cards are free.

Session 11 used patch_signed_spectrum (80D). The Track 2 plan was
wake_coarse_pool (288D = 24x12 average-pooled wake ROI vorticity)
but it never fired because Track 1 succeeded.

Now run wake_coarse_pool at lambda_wake = 1.0 (matching W0_C_lam100)
and a higher-resolution variant at 32x16 = 512D average-pooled.

The hypothesis: 80D is the bottleneck. Forcing the encoder to predict
a higher-D wake field gives it more spatial detail to encode.

```bash
# Example for S12_D_coarse288
CUDA_VISIBLE_DEVICES=3 python -m src.training.train_jepa \
    --partition v1 --all-train --max-iters 20000 --seed 42 \
    --observable-head cl_future \
    --observable-head-weight 0.01 \
    --wake-observables-path outputs/data_pipeline/v1/wake_observables_coarse_288.h5 \
    --wake-observable-type wake_coarse_pool \
    --wake-head-dim 288 \
    --lambda-wake 1.00 \
    --wake-head-deltas 0 8 16 24 \
    --output-dir outputs/runs/session12/S12_D_coarse288/encoder
```

Plus a 512D variant. Total: 2 encoder runs + 2 decoder retrains.

Expected outcome: SSIM in the same range as W0_C_lam100 (0.52) or
slightly higher (0.55) at the cost of some wake_enstrophy degradation.
The 80D target may have been the sweet spot or the bottleneck; this
test settles which.

## Direction E: d=64 latent (breaking the LeWM d=32 lock)

**Run S12_E_d64** on cuda:2 after Direction A finishes.

The Sessions 7-8 work locked d=32 per LeWM intrinsic-dim arguments.
Session 11 showed PR scales with lambda_wake (11.66 at lambda=1.0).
The bottleneck may be d itself, not the effective rank.

Retrain the JEPA encoder at d=64 with the W0_C_lam100 recipe
(SIGReg + OBS_cl + Mode C wake head at lambda_wake=1.00).

```bash
CUDA_VISIBLE_DEVICES=2 python -m src.training.train_jepa \
    --partition v1 --all-train --max-iters 20000 --seed 42 \
    --latent-dim 64 \
    --observable-head cl_future \
    --observable-head-weight 0.01 \
    --wake-observables-path outputs/data_pipeline/v1/wake_observables_patch8x4_signed_spec16.h5 \
    --wake-observable-type patch_signed_spectrum \
    --wake-head-dim 80 \
    --lambda-wake 1.00 \
    --wake-head-deltas 0 8 16 24 \
    --projection-norm batchnorm --anticollapse sigreg \
    --lambda-sigreg 0.01 \
    --output-dir outputs/runs/session12/S12_E_d64/encoder
```

Then retrain LapFiLM decoder.

Expected outcome: if d=32 was the bottleneck (and the omega_direct
4608D ceiling at 0.551 suggests there is room), d=64 should give a
substantial SSIM gain. If d=32 was not the bottleneck (which would
contradict the Session 11 PR analysis), d=64 might give marginal gain
or even hurt due to the harder SIGReg optimization in higher d.

**Risk**: changes the paper’s architectural anchor. d=32 is what
Sessions 7-8 locked. Need to test d=64 to know whether the paper
should claim d=32 or d=64 is optimal. This is a paper-architecture
decision worth making with data.

## Direction F: total-correlation disentanglement (Wang et al. 2026 motivated)

**Run S12_F_TCpenalty** on cuda:3 after Direction B finishes.

The Section 8 future direction from Session 11. Penalize cross-
dimensional covariance on the encoder output. Mechanism: in addition
to SIGReg, add a term L_TC = || off_diag(Cov(z)) ||_F^2 with small
weight (lambda_TC = 0.01).

This is a JEPA-native version of total-correlation, not the VAE
formulation. The encoder output z is projected through SIGReg’s
projection layer; we add the off-diagonal Frobenius norm of the
covariance matrix of the projected z over a batch.

```bash
CUDA_VISIBLE_DEVICES=3 python -m src.training.train_jepa \
    --partition v1 --all-train --max-iters 20000 --seed 42 \
    --observable-head cl_future \
    --observable-head-weight 0.01 \
    --wake-observables-path outputs/data_pipeline/v1/wake_observables_patch8x4_signed_spec16.h5 \
    --wake-observable-type patch_signed_spectrum \
    --wake-head-dim 80 \
    --lambda-wake 1.00 \
    --wake-head-deltas 0 8 16 24 \
    --total-correlation-weight 0.01 \
    --projection-norm batchnorm --anticollapse sigreg \
    --lambda-sigreg 0.01 \
    --output-dir outputs/runs/session12/S12_F_TC0p01/encoder
```

Plus a sweep at lambda_TC in {0.03, 0.10}. Three runs total.

Expected outcome:

- PR(z) climbs toward d=32 (decorrelation pressure broadens the latent).
- Reconstruction either holds or marginally improves (if the
  redundancy was actually noise) OR degrades (if the apparent
  redundancy was carrying tail-PC information that the decoder needs,
  consistent with the Section 7b PCA k=12 finding).

The publishable result either way: the first empirical measurement
of the intrinsic dimensionality of the parametric vortex-gust impact
manifold at Re=5000. If TC penalty hurts reconstruction at every
lambda, the answer is “the manifold is approximately 12-dim plus
non-negligible tail.” If TC penalty does not hurt at lambda=0.01 but
hurts at lambda=0.10, the threshold tells us how much extra
dimensionality is necessary.

**Citation context**: motivated by Wang, Tirelli, Discetti, Ianiro
arXiv:2604.18059 April 2026 (verify on arXiv MCP if used in paper
text). Same NACA 0012 + parametric vortex gust setting from a UC3M
group; their VAE-based total-correlation approach inspires our
JEPA-native version but is not directly ported.

## Sequencing and parallelism

Two RTX 6000 cards. With minimal queue management:

**Phase 1 (parallel, ~3-4h)**:

- cuda:2: Direction A spectral loss default (S12_A_specloss_default,
  30k iters, ~2.5h)
- cuda:3: Direction B GAN refinement (S12_B_gan_refine, 20k iters
  refiner training, ~2h)

**Phase 2 (parallel, ~3h)**:

- cuda:2: S12_A_specloss_low (1.5h) then S12_A_specloss_high (1.5h)
- cuda:3: S12_C_lam200 encoder (1.5h) + decoder (1h) = 2.5h

**Phase 3 (parallel, ~5h)**:

- cuda:2: S12_C_lam300 encoder + decoder = 2.5h, then S12_C_lam500
  encoder + decoder = 2.5h
- cuda:3: S12_D_coarse288 encoder + decoder (2.5h), then
  S12_D_coarse512 encoder + decoder (2.5h)

**Phase 4 (parallel, ~5h)**:

- cuda:2: S12_E_d64 encoder + decoder (3h)
- cuda:3: S12_F_TC0p01 encoder + decoder, then 0p03 and 0p10
  (~5h sequential)

**Phase 5 evaluation (~3h)**:

- Compute extended metrics across all configurations including
  2D premultiplied power spectrum agreement.
- Generate Figure 3 for each.
- Identify the winner.

Total session wall-clock: roughly 18-22 hours of GPU compute plus
evaluation. Plan for two consecutive days of execution.

Note: cloud GPU is available if needed for parallel acceleration.
The plan does not require cloud GPU but if Phase 3 or 4 is the
bottleneck and we want results faster, launch on a third or fourth
H100 instance.

## Success criteria for Session 12

**Numerical**: at least one configuration reaches all of:

- Test B SSIM median >= 0.60 (Session 11 was 0.523, target +15%).
- Test B wake_enstrophy_rel_err median <= 0.40 (Session 11 was 0.431).
- Test B radial spectrum L2 wake median <= 0.30 (Session 11 was 0.397).

**Qualitative**: Figure 3 wake shows visible vortex-street structure
at frame 55 that a fluid dynamicist would recognize as physically
correct.

**Spectral (new for Session 12)**: 2D premultiplied power spectrum
agreement in the wake ROI. Following Balasubramanian et al. PRF 2026
Figure 5/6 methodology, the 10%, 50%, 90% contours of
kx*kz*phi_omega(lambda_x, lambda_z) for prediction must align with
DNS within factor 2 in wavelength.

The winner is the configuration that passes the most criteria,
prioritizing the spectral agreement (the most novel paper claim).

## Paper claims after Session 12

If Direction A (spectral loss) wins: the paper says “adding the
Balasubramanian PRF 2026 spectral loss to JEPA + wake observable
recovers crisp wake structure that the wake observable alone cannot
achieve.” This is a clean, well-cited story.

If Direction B (GAN refinement) wins: the paper says “GAN
refinement of the JEPA + wake observable + LapFiLM output achieves
crisp wake reconstruction. The encoder provides the information; the
GAN extracts it.” Different story but also valid.

If Direction C (higher lambda_wake) wins: paper extends the
“lambda_wake monotonic scaling” finding from Session 11.

If Direction D (higher-D wake target) wins: paper claim shifts to
“the wake observable head needs target dimensionality matching the
intrinsic wake complexity.”

If Direction E (d=64) wins: paper architecture changes; need to
re-do Sessions 7-8 analyses at d=64.

If Direction F (TC penalty) wins: paper has an unexpectedly clean
interpretability story.

The expected paper section structure after Session 12:

- 5.1 Pipeline + decoder baseline (Session 9/10).
- 5.2 Wake observable head sweep (Session 11).
- 5.3 Whichever Session 12 direction wins.
- 5.4 Comparison vs Fukami AE at matched d.
- 5.5 Disentanglement diagnostic (Section 7c Session 11 + Direction F).
- 5.6 Downstream prediction (forecasting) skill.

## Risk register

|Risk                                                                                           |Probability                                                                                            |Mitigation                                                                                                                |If it fires                                    |
|-----------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------|
|Direction A spectral loss does not converge                                                    |low                                                                                                    |The PRF paper’s hyperparameters are well-validated; if it fails to converge, sweep alpha/beta/gamma/delta/zeta            |-1h diagnostic, no replan                      |
|Direction A produces SSIM gain but worse pointwise MSE (as PRF paper showed for large y+)      |medium                                                                                                 |This is the known tradeoff; report both and focus paper claim on spectral fidelity rather than SSIM alone                 |Paper Section 5.3 framing adjusts              |
|Direction B GAN training is unstable                                                           |medium-high                                                                                            |Use spectral normalization on discriminator, two-time-scale update rule, and reduce adversarial weight from 0.05 to 0.01  |+2h tuning, possibly drop Direction B          |
|Direction C reveals saturation (lambda > 1 stops helping)                                      |medium                                                                                                 |The Session 11 ladder was monotonic; if it saturates at lambda=2, Direction C is a clean negative result                  |No replan; report saturation as paper finding  |
|Direction D reveals wake_coarse_pool is worse than patch_signed_spectrum at matched lambda_wake|medium-high (the Session 11 wake_probe showed wake_coarse_pool r2 was lower than patch_signed_spectrum)|Drop Direction D as obviously inferior; record as ablation                                                                |-1h compute waste                              |
|Direction E d=64 SIGReg does not stabilize                                                     |medium                                                                                                 |LeWM d=32 was conservative; d=64 may need lambda_sigreg adjustment                                                        |+2h hyperparameter sweep                       |
|Direction F TC penalty improves disentanglement but degrades reconstruction                    |medium                                                                                                 |This is the “physics is ~12-dim” finding; document and report                                                             |Becomes the Section 5.5 / Section 8 paper story|
|All six directions hit roughly the same SSIM ceiling (~0.55)                                   |low-medium                                                                                             |The omega_direct ceiling at 0.55 may be a real decoder-architecture limit; would require Direction G (decoder family swap)|Session 13 launches ViT or diffusion decoder   |
|Cloud GPU unavailable when needed                                                              |low                                                                                                    |The plan does not require cloud GPU; just accept slower sequential execution                                              |+5-8h sequential time                          |

## D-entries to record

(Renumbered from the original draft D85-D91. Session 11 close-out
already used D85-D88 for the pipeline move, Fukami+wake broken,
PCA k=12 retrain, and the CV-honest probe correction. Session 12
starts at D89.)

**D89**: Direction A spectral loss results (3 runs: default, low, high).
The Balasubramanian PRF 2026 SL formulation applied to the
JEPA + wake observable decoder.

**D90**: Direction B GAN refinement results.

**D91**: Direction C extended lambda_wake ladder (lam=2, 3, 5).
Saturation point identified.

**D92**: Direction D higher-D wake target results (288D, 512D).

**D93**: Direction E d=64 latent results.

**D94**: Direction F total-correlation penalty results (3 lambda
values).

**D95**: Session 12 outcome decision. Which direction(s) won, what
paper Section 5 looks like after Session 12, what Session 13 does.

## Implementation order

1. Implement Balasubramanian SL loss components in
   `src/models/decoder_losses.py` (~80 lines + 4 unit tests). This
   blocks Direction A.
1. Implement GAN refiner + discriminator in `src/models/refiner.py`
   and `src/models/discriminator.py` (~400 lines + 6 unit tests).
   Blocks Direction B.
1. Implement total-correlation penalty in
   `src/models/total_correlation.py` (~50 lines + 2 unit tests).
   Blocks Direction F.
1. Implement wake_coarse_pool target precomputation (already exists
   from Session 11) and extend to 32x16=512D (~20 lines edit).
   Blocks Direction D.
1. Add `--latent-dim 64` to train_jepa.py (one-line argparse change,
   the rest already supports variable d). Blocks Direction E.
1. Implement the 2D premultiplied power spectrum metric
   in `src/evaluation/decoder_metrics.py` (~60 lines + 2 unit tests).
   Required for all Session 12 evaluations.

Items 1, 2, 6 are critical path. Items 3, 4, 5 can happen in
parallel with the Phase 1 runs.

## Pre-flight checks

1. Session 11 W0_C_lam100 checkpoint exists at
   `outputs/runs/session11/W0_C_lam100/checkpoint_iter020000.pt`.
1. All Sessions 2-11 unit tests pass.
1. Both RTX 6000 cards healthy.
1. Cloud GPU access verified (if planning to use).

## Predictions worth pre-registering

Three pre-registered credences.

**Direction A (spectral loss) reaches Test B SSIM > 0.58**: credence
70%. The PRF 2026 paper demonstrated a 2x RMS improvement on near-
wall turbulence; the mechanism is directly applicable to our wake
problem. The high credence is because the failure mode (blurry wake)
matches the failure mode the SL was designed to address. If this is
wrong, our wake problem differs from theirs in a way we have not
identified.

**Direction B (GAN refinement) reaches visually crisp wake but
possibly lower SSIM**: credence 60% on visual crispness, 30% on
SSIM improvement. GANs reliably improve visual quality but often at
small pixel cost.

**Direction C (extended lambda_wake) saturates at lambda=2 or 3**:
credence 65%. The Session 11 ladder was monotonic but the rate of
gain was decreasing (0.451 -> 0.472 -> 0.482 -> 0.523). Continued
monotonic gain would imply we can keep going indefinitely, which
contradicts the d=32 + intrinsic dim arguments. Most likely the
ladder reaches a saturation point.

**Direction E (d=64) gives meaningful gain**: credence 50%. The
omega_direct ceiling at 0.551 suggests there is room. But d=32 was
locked for principled intrinsic-dim reasons (LeWM); doubling d might
just add noise rather than capacity.

Net credence that some Session 12 direction produces a visually
crisp Figure 3 worth a high-impact-journal headline: 85%. The 15%
failure scenario is the one where no decoder-architectural or
encoder-training change moves the needle past the current 0.52 SSIM

- blurry wake state, in which case Session 13 considers a decoder
  family swap (Track 4 from Session 11 plan, now elevated to primary).

## Decision references

Carry forward: D34, D35, D38, D40, D44-D49, D50-D57, D60-D69,
D70-D77, D78-D88.

This session: D89-D95.

External: Balasubramanian et al. PRF 11 044907 (2026). Isola et al.
CVPR 2017 (pix2pix). Wang et al. arXiv:2604.18059 (motivation for
Direction F).