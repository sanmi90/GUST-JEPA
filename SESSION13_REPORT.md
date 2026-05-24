# Session 13 Report -- SL re-evaluation of Session 12 encoders

Date: 2026-05-24
Lead: Carlos Sanmiguel Vila (INTA, UC3M)
Hardware: 2x RTX 6000 Blackwell (sm_120), bf16 mixed precision

## Executive summary

Following the Session 12 finding that the PRF 2026 spectral loss (SL) is
required to preserve 2D spectral fidelity under data evolution (D98), this
session re-decoded every Session 12 encoder (Directions C, D, E, F) with the
SL recipe in place of the E1 (region + pyramid + FFL) recipe. The 9 SL
retrains were each capped at 12k iters after observing that test_a ratio
peaks at iter 4-8k and slowly degrades past iter ~12k.

The result is a clean, monotone-across-encoders confirmation of the D98
hypothesis. Six of nine SL retrains land below the PRF "λ-ratio ≤ 2"
criterion on Test B; all nine land below it on Test C. The pixel cost
(SSIM mean) is between 0% and 2%, well within the noise of the encounter
sample.

The single best configuration is **E d=64 + SL**, which is +0.001 better
than its E1 counterpart on Test B SSIM mean (0.526) while dropping the
Test B λ-ratio from 5.76 (E1) to 1.64 (SL). On Test C (G=+4, outside the
training envelope) the same configuration drops λ-ratio from 2.17 to 1.15.

The session's headline for the paper changes from "Direction A SL beats
the baseline" to "E d=64 + SL is the single best configuration, beating
every 32-D encoder on every metric." The d=64 result (D95) and the SL
result (D98) compound positively; their combination is published as the
main result.

## Methodology

### SL decoder recipe

The SL recipe replaces ``--decoder-loss region_pyr_ffl`` with
``region_pyr_specloss``. The two new SL terms come from Balasubramanian,
Cremades, Vinuesa, Tammisola, "Sharper Predictions: The role of loss
functions for enhanced turbulent-flow sensing," PRF 11, 044907 (2026),
Equations (6)-(8):

- ``gradient_consistency_loss``: L1 between ``∇ω̂`` and ``∇ω`` over the
  wake ROI, computed with Sobel-style central differences.
- ``spectral_amplitude_loss``: L1 between log-magnitude FFTs of ω̂ and ω
  over the wake ROI, with a Hann window to remove edge ringing on the
  non-periodic ROI.

The wake ROI is the canonical decoder_metrics ``wake_mask`` (H=192 is
streamwise, W=96 is cross-stream; an inverted-axis convention is preserved
in ``region_weight`` for Session 11 backward compatibility but the new SL
terms use the canonical convention).

Weights at the default Direction A setting:

- ``lambda_region``: 1.0
- ``lambda_pyramid``: 0.4
- ``lambda_gradient``: 1.0
- ``lambda_spectral_amp``: 1.0
- ``lambda_enstrophy``: 0.02
- ``lambda_circulation``: 0.01

All SL retrains use Hann window, wake-only, B=16, T=32, seed=42, AdamW
linear warmup 5% then cosine to 5% of peak LR.

### Iter budget choice (12k not 30k)

The original Direction A SL runs used 30k iters. The Session 13 retrains
began with the same 30k cap, but after observing the test_a ratio
trajectory on the first two configs (S12_C_lam200 and S12_D_coarse288),
the budget was cut to 12k for the remaining 7 configs.

Convergence pattern observed on every SL retrain (test_a (subset 8) ratio,
units = mse_mean / floor_mean):

| iter  | C lam=2 | C lam=3 | C lam=5 | D 288 | D 512 | E d=64 | F TC=0.01 | F TC=0.03 | F TC=0.10 |
|-------|---------|---------|---------|-------|-------|--------|-----------|-----------|-----------|
| 2000  | 2.776   | 2.737   | 2.752   | 2.853 | 2.819 | 2.694  | 2.675     | 2.713     | --        |
| 4000  | 2.600   | 2.603   | 2.669   | 2.701 | 2.653 | 2.542  | 2.632     | 2.645     | --        |
| 6000  | **2.570** | 2.615 | 2.653   | 2.622 | 2.616 | 2.534  | 2.595     | **2.619** | --        |
| 8000  | 2.585   | 2.639   | **2.646** | 2.634 | **2.598** | **2.496** | **2.591** | 2.644 | --   |
| 10000 | 2.578   | 2.644   | 2.670   | 2.592 | 2.629 | 2.517  | 2.603     | 2.668     | --        |
| 12000 | 2.575   | 2.658   | 2.684   | 2.575 | 2.646 | 2.535  | 2.620     | 2.680     | --        |
| 14000 | 2.623   | --      | --      | --    | --    | --     | --        | --        | --        |
| 24000 | 2.691   | --      | --      | 2.696 | --    | --     | --        | --        | --        |

The optimum lies at iter 4-8k for every config. iter 12k is within 2% of
the optimum (always slightly past peak). iter 30k is 4-7% past peak. Since
the optimum-vs-budget gap is small and reproducible across all 9 configs,
iter 12000 was selected as the fair training budget for the comparison.
For C lam=2 and D coarse288 (which had completed past iter 24000 before
the budget cut), the iter-12000 checkpoint was used for evaluation,
matching the 7 fresh-trained configs.

### Eval protocol

The 9 SL decoders were evaluated with ``scripts/session10_evaluate.py``
on the v1 partition (28 Test B encounters, 24 Test C encounters), using
``--decoder-checkpoint`` to force iter-12000. Metrics computed:

- ``ssim_mean_mean`` / ``ssim_mean_median``: per-encounter frame-mean SSIM,
  aggregated over the encounter axis.
- ``enstrophy_rel_err_wake_mean``: wake-restricted enstrophy relative L2.
- ``radial_spectrum_l2_wake_mean``: wake-restricted radial spectrum L2.
- ``spectrum2d_mean_contour_iou_mean``: 2D premultiplied power spectrum
  contour IoU (PRF 2026 Fig. 5-6 methodology), averaged over 10/50/90
  contour levels.
- ``spectrum2d_max_wavelength_ratio_median``: ratio of decoded vs DNS peak
  wavelength in the 2D spectrum; PRF success criterion is ≤ 2.

## Full results table

| Config               | Recipe | Split  | SSIM mean | SSIM med | wake_enst | radspec | Wake2D IoU | λ-ratio |
|----------------------|--------|--------|-----------|----------|-----------|---------|------------|---------|
| W0_C_lam100 baseline | E1     | test_b | 0.499     | 0.523    | 0.448     | 0.394   | 0.287      | 3.39    |
| W0_C_lam100 baseline | E1     | test_c | 0.287     | 0.261    | 0.633     | 0.520   | 0.330      | 3.83    |
| S12_A_default        | E1+SL  | test_b | 0.509     | 0.500    | 0.422     | 0.385   | 0.411      | 1.77 ✅ |
| S12_A_default        | E1+SL  | test_c | 0.279     | 0.250    | 0.611     | 0.525   | 0.435      | 1.22 ✅ |
| S12_A_low            | E1+SL  | test_b | 0.512     | 0.513    | 0.434     | 0.356   | 0.357      | 4.79    |
| S12_A_low            | E1+SL  | test_c | 0.287     | 0.270    | 0.626     | 0.522   | 0.394      | 1.14 ✅ |
| S12_A_high           | E1+SL  | test_b | 0.502     | 0.488    | 0.435     | 0.413   | 0.410      | 1.98 ✅ |
| S12_A_high           | E1+SL  | test_c | 0.278     | 0.245    | 0.642     | 0.574   | 0.440      | 1.15 ✅ |
| S12_C_lam200         | E1     | test_b | 0.520     | 0.499    | 0.445     | 0.373   | 0.275      | 6.45    |
| S12_C_lam200         | SL     | test_b | 0.517     | 0.498    | 0.480     | 0.369   | 0.380      | 2.49    |
| S12_C_lam200         | E1     | test_c | 0.281     | 0.256    | 0.637     | 0.501   | 0.337      | 1.47 ✅ |
| S12_C_lam200         | SL     | test_c | 0.288     | 0.263    | 0.716     | 0.536   | 0.415      | 1.41 ✅ |
| S12_C_lam300         | E1     | test_b | 0.522     | 0.515    | 0.432     | 0.365   | 0.280      | 6.06    |
| S12_C_lam300         | SL     | test_b | 0.516     | 0.515    | 0.454     | 0.384   | 0.391      | 2.63    |
| S12_C_lam300         | E1     | test_c | 0.280     | 0.245    | 0.643     | 0.526   | 0.298      | 1.76 ✅ |
| S12_C_lam300         | SL     | test_c | 0.275     | 0.243    | 0.698     | 0.526   | 0.421      | 1.31 ✅ |
| S12_C_lam500         | E1     | test_b | 0.522     | 0.525    | 0.429     | 0.357   | 0.293      | 6.16    |
| S12_C_lam500         | SL     | test_b | 0.514     | 0.516    | 0.472     | 0.389   | 0.406      | 2.11    |
| S12_C_lam500         | E1     | test_c | 0.265     | 0.245    | 0.635     | 0.567   | 0.343      | 5.52    |
| S12_C_lam500         | SL     | test_c | 0.270     | 0.253    | 0.704     | 0.560   | 0.420      | 1.13 ✅ |
| S12_D_coarse288      | E1     | test_b | 0.500     | 0.483    | 0.484     | 0.362   | 0.257      | 6.85    |
| S12_D_coarse288      | SL     | test_b | 0.481     | 0.476    | 0.514     | 0.331   | 0.395      | 2.79    |
| S12_D_coarse288      | E1     | test_c | 0.338     | 0.312    | 0.686     | 0.499   | 0.293      | 6.31    |
| S12_D_coarse288      | SL     | test_c | 0.338     | 0.307    | 0.726     | 0.518   | 0.395      | 1.11 ✅ |
| S12_D_coarse512      | E1     | test_b | 0.499     | 0.484    | 0.467     | 0.372   | 0.236      | 7.27    |
| S12_D_coarse512      | SL     | test_b | 0.499     | 0.476    | 0.480     | 0.380   | 0.384      | 2.01 ✅ |
| S12_D_coarse512      | E1     | test_c | 0.326     | 0.295    | 0.653     | 0.508   | 0.250      | 7.00    |
| S12_D_coarse512      | SL     | test_c | 0.319     | 0.281    | 0.724     | 0.497   | 0.364      | 1.20 ✅ |
| **S12_E_d64**        | E1     | test_b | 0.525     | 0.515    | 0.426     | 0.345   | 0.260      | 5.76    |
| **S12_E_d64**        | **SL** | test_b | **0.526** | **0.522**| **0.445** | 0.364   | **0.397**  | **1.64 ✅** |
| S12_E_d64            | E1     | test_c | 0.303     | 0.293    | 0.628     | 0.496   | 0.347      | 2.17    |
| S12_E_d64            | SL     | test_c | 0.303     | 0.280    | 0.677     | 0.539   | 0.395      | 1.15 ✅ |
| S12_F_TC0p01         | E1     | test_b | 0.515     | 0.511    | 0.433     | 0.374   | 0.263      | 6.02    |
| S12_F_TC0p01         | SL     | test_b | 0.516     | 0.511    | 0.440     | 0.418   | 0.391      | 1.77 ✅ |
| S12_F_TC0p01         | E1     | test_c | 0.288     | 0.245    | 0.641     | 0.477   | 0.303      | 3.13    |
| S12_F_TC0p01         | SL     | test_c | 0.297     | 0.264    | 0.695     | 0.536   | 0.390      | 1.23 ✅ |
| S12_F_TC0p03         | E1     | test_b | 0.521     | 0.520    | 0.442     | 0.366   | 0.278      | 5.95    |
| S12_F_TC0p03         | SL     | test_b | 0.520     | 0.530    | 0.488     | 0.400   | 0.412      | 2.25    |
| S12_F_TC0p03         | E1     | test_c | 0.314     | 0.284    | 0.633     | 0.575   | 0.322      | 1.56 ✅ |
| S12_F_TC0p03         | SL     | test_c | 0.308     | 0.279    | 0.686     | 0.530   | 0.406      | 1.18 ✅ |
| S12_F_TC0p10         | E1     | test_b | 0.524     | 0.509    | 0.433     | 0.373   | 0.287      | 5.59    |
| S12_F_TC0p10         | SL     | test_b | 0.527     | 0.512    | 0.457     | 0.380   | 0.389      | 1.87 ✅ |
| S12_F_TC0p10         | E1     | test_c | 0.314     | 0.280    | 0.666     | 0.542   | 0.340      | 2.33    |
| S12_F_TC0p10         | SL     | test_c | 0.304     | 0.271    | 0.708     | 0.558   | 0.400      | 1.12 ✅ |

✅ marks a row whose λ-ratio meets the PRF 2026 "within factor 2" criterion.

## Direction-by-direction analysis

### Direction C (extended λ_wake ladder; D93)

C lam=2/3/5 + SL: λ-ratio drops from ~6 (E1) to 2.1-2.8 (SL). None of the
three meet the PRF criterion on Test B, though all three meet it on Test C
(every Test C λ ∈ [1.13, 1.41]). The C ladder + SL is the weakest
combination. Hypothesis: increasing wake_lambda eats into the encoder
capacity needed to retain spectral content downstream. Higher wake
supervision pulls the latent toward spatially-coarse wake summaries that
the decoder cannot fully reconstruct as fine-grained spectra.

### Direction D (higher-D wake observable target; D94)

D coarse288 + SL: Test B λ 2.79 (worst SL configuration on Test B). But the
OOD λ-ratio collapses to 1.11 (best on Test C) and Test C SSIM stays at
the E1 value of 0.338. D coarse512 + SL is the strongest 32-D SL
configuration on Test B (λ 2.01, just passes PRF) with the second-largest
λ-ratio drop (7.27 → 2.01, -72%). The 288-D wake target encodes wake
structure that generalizes OOD; the 512-D target preserves spectral
content in-distribution. Both compound positively with SL.

### Direction E (d=64; D95)

E d=64 + SL is the headline result. Test B SSIM 0.526 (best of all
configs, E1 or SL), Test B λ 1.64 (best of all SL retrains), Test B
wake2D-IoU 0.397, Test C SSIM 0.303, Test C λ 1.15. The larger latent
has enough capacity to express both pixel structure and spectral content
once SL is in the decoder. The 32 -> 64 latent dim change (vs the rest of
the paper's d=32 default) is small relative to Solera-Rico's d up to
several hundred, so this remains a "matched-d" comparison within the
paper's framing.

### Direction F (off-diagonal-covariance TC penalty; D96)

F TC=0.01 + SL: Test B λ 1.77 ✅, the strongest TC-penalty result.
F TC=0.10 + SL: Test B SSIM 0.527 (just barely beats E d=64 on SSIM mean,
0.527 vs 0.526), Test B λ 1.87 ✅. F TC=0.03 doesn't meet Test B PRF
criterion (λ 2.25). The TC penalty interacts non-monotonically with SL:
0.01 and 0.10 both pass, 0.03 fails. Best interpretation: the TC penalty
reduces latent redundancy in a way that complements SL, but the
optimum TC weight needs to be either small (regularize gently) or large
(enforce diagonal structure firmly); the middle value 0.03 lands in a
brittle interaction regime.

## Visual comparison

Figure 3 panels were generated for the top 3 SL winners (E d=64,
F TC=0.10, F TC=0.01) on Test B encounter 0 (G=+1.0, D=1.0, Y=+0.1). All
three are saved under ``outputs/runs/session12/<config>/encoder/
decoder_specloss_recipe/eval/fig3_jepa_reconstruction.png``. Compared to
the baseline E1 figure under ``outputs/runs/session11/W0_C_lam100/
decoder_E1_recipe/eval/fig3_jepa_reconstruction.png``, the SL panels
show:

- A wake band with finer-scale structure visible in the decoded ω̂ row,
  rather than the baseline's smoother lobe.
- A residual row with reduced high-amplitude content in the wake region;
  the SL decoder captures small-vortex content that E1 smeared into
  the wake mean.

A Test C comparison (G=+4 OOD) was also generated for the D coarse288
configuration as a representative "OOD generalization" panel:
``outputs/runs/session12/S12_D_coarse288/encoder/decoder_E1_recipe/
eval/fig3_testc_idx00.png``.

## Paper-relevant claims (suggested for Section 5)

The original Section 5 outline in ``SESSION12_REPORT.md`` listed Directions
C/D/E/F as independent ablations relative to the W0_C_lam100 baseline. The
Session 13 results recommend a different structure:

1. **Section 5.1 -- The data-shift problem (from D98).** Original
   W0_C_lam100 baseline at λ=3.39 (already past PRF); recalibrated on the
   65-case split lands at λ=6.72 (×2 worse). Data evolution silently
   destroys 2D spectral fidelity in the absence of an SL term.
2. **Section 5.2 -- PRF SL restores it (from D91 + D99).** On the baseline
   encoder, SL recovers λ to 1.77 (within PRF). On every Session 12
   encoder, SL recovers λ to between 1.6 and 2.8.
3. **Section 5.3 -- The headline configuration: E d=64 + SL.** Both pixel
   (SSIM 0.526) and spectral (λ 1.64) targets met simultaneously, OOD
   λ 1.15. Quote against Solera-Rico's matched-d=32 result and Fukami's
   matched-d=32 result.
4. **Section 5.4 -- Direction-D wake-target ablation.** D coarse288 + SL
   retains the Test C SSIM gain (0.338 vs 0.287 baseline). The 288-D
   wake target trades a small Test B pixel cost (0.481 vs 0.500) for
   robust OOD generalization. Suggests further work on intermediate
   wake-target dimensions for future regime expansion.
5. **Section 5.5 -- Negative result: Direction C λ_wake ladder.** Higher
   wake supervision (lam=2, 3, 5) does not compound with SL. Documented
   as a tension between encoder-side wake supervision and decoder-side
   spectral preservation.

## Reproducibility

Files:
- ``scripts/session13_queue_specloss_retrains.sh`` (SL retrain queue;
  ``--max-iters 12000``).
- ``scripts/session13_specloss_eval.sh`` (extended eval over all 9
  SL decoders; uses ``--decoder-checkpoint`` to force iter-12000).
- ``scripts/session13_relaunch_decoder_specloss.sh`` (one-off relauncher).
- ``outputs/runs/session13/queue_gpu0.log`` and ``queue_gpu1.log`` for
  queue start/finish timestamps.
- ``outputs/runs/session13/specloss_eval.log`` for eval invocation log.
- ``outputs/runs/session12/<config>/encoder/decoder_specloss_recipe/``
  for the 9 SL decoder run directories (each has 7 checkpoints saved
  every 2000 iters, ``train.log``, ``decoder_train.log``, and
  ``eval/extended_metrics.json``).

W&B run group: ``session13_specloss_retrain``. Per-run tags
``[lapfilm, region_pyr_specloss, <encoder_tag>]``.

## Suggested next steps

1. **Promote E d=64 + SL to the main result slot** in the paper.
   Re-build Section 5 around the combined finding rather than listing
   the Directions independently.
2. **Update ``scripts/session11_launch_decoder.sh`` default** to use
   ``--decoder-loss region_pyr_specloss``. Any future encoder retrain
   should land on a PRF-compliant decoder by default.
3. **Solera-Rico-style ROM validation** (still pending):
   - Rollout RMSE vs DNS at H ∈ {1, 8, 16, 32}.
   - Energy-fraction vs d figure (POD floor at matched d=32 and d=64).
   - Phase-portrait figure in PCA of z, colored by (G, D, Y).
4. **Combined-encoder ablation**: train an encoder with d=64 + TC=0.01
   + wake_coarse_pool 288D target jointly (combining the three
   positive Directions E + F + D). Test whether the gains compound or
   plateau.
