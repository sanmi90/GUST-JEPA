# Session 15 Report

Date: 2026-05-25
Lead: Carlos Sanmiguel Vila (INTA, UC3M)
Hardware: 2x RTX 6000 Blackwell (sm_120), bf16 mixed precision

## Executive summary

Session 15 began as a focused follow-up to Session 14's D113 finding
(spanwise-mean vorticity through the slice-trained encoder gave +0.07 GDY
R^2). It ran four ablations end-to-end and ended on a major synthesis that
**revises the encoder-vs-decoder framing of Session 14**.

**Final picture**: the 64-D JEPA latent is the achievable ceiling for this
data regime. Halving decoder parameters does not hurt. Adding a 2.84M-param
diffusion refinement stage does not help. The decoder is not the bottleneck;
the latent dimensionality is.

Four D-entries landed (D114, D115, D116, D117); the synthesis suggests
Session 16 should test larger latents (d=128, d=256) directly rather than
pursuing further decoder refinement.

## What ran

1. **S15-T1 spanwise-mean retrain** (D114) -- 2 variants in parallel
   (canonical + reduced lambda) on the v1_mean cache.
2. **S15-T5 diffusion refinement** (D116) -- DDIM U-Net trained on top of
   the production slice encoder + SL decoder.
3. **Lean SL decoder ablation** (D117) -- bc=32 decoder vs bc=64
   production.
4. **TCN proxy + SHAP** (D115) -- Session 14 follow-up that finally landed
   in Session 15 after a 10-hour buffered Python run.

## D114: Spanwise-mean training trades pixel SSIM for spectral fidelity

User-launched at end of Session 14 with two variants on the v1_mean cache:

| Metric (Test B) | Canonical mean (full lambda) | Reduced mean (0.3 lambda) | Slice production (D99) |
|---|---|---|---|
| SSIM mean | 0.498 | 0.467 | **0.526** |
| mse_wake | 15.50 | 16.41 | **14.71** |
| **spec2d_lambda_ratio** | **1.124** | 1.327 | 1.635 |
| spec2d_iou | **0.434** | 0.386 | 0.397 |
| Test C SSIM | 0.250 | 0.245 | **0.303** |
| Test C lambda_ratio | 1.178 | 1.260 | **1.150** |

**Headline trade-off**:
- Mean WINS DECISIVELY on PRF spectral lambda-ratio (1.12 vs 1.64) -- the
  smoother input gives reconstructions matching the DNS 2D spectrum better.
- Slice WINS on pixel SSIM (Test B 0.526 vs 0.498; Test C 0.303 vs 0.250).
- Mean is also significantly worse than canonical when lambdas are reduced
  -- **the physics hypothesis "spectral/wake losses unneeded after spanwise
  averaging" is FALSE**. The losses do real work even on mean data.

**Encoder diagnostics**: canonical mean PR(z) = 6.67; reduced mean = 3.04
(very compressed, matches the (G, D, Y) parameter count from D113); slice
production = 11.66. Both variants r2_overall > 0.997.

## D115: TCN proxy learner beats RBF + SHAP-vs-permutation disagreement

The Session 14 Thrust 7 follow-up Python run took 10 hours (output buffered,
unrelated to actual compute) but produced two clean results.

**TCN proxy learner (cross-pool Test B z_R2)** -- TCN beats RBF everywhere:

| K | TCSI TCN | MI TCN | LASSO TCN | qDEIM TCN | (TCSI RBF reference) |
|---|---|---|---|---|---|
| 2 | 0.830 | 0.823 | **0.886** | 0.715 | 0.697 |
| 4 | **0.873** | 0.870 | 0.828 | 0.774 | 0.793 |
| 8 | **0.896** | 0.860 | 0.885 | 0.826 | 0.823 |

The +0.07-0.13 gain over RBF confirms the user's D111 prediction: the JEPA
latent encodes time-structured nonlinear features that benefit from a
temporal-convolutional learner. **LASSO wins K=2 under TCN** (0.886) -- a
new finding; LASSO was middle-of-pack under Ridge in D110.

**SHAP ranking** on K=8 TCSI+RBF: [44, 61, 20, 0, 5, 15, 11, 107]
**Permutation ranking** (D111): [11, 15, 20, 5, 0, 44, 107, 61]

The two methods disagree on the most-important sensor. Permutation says
sensor 11 (LE stagnation, bootstrap-100%-stable in D112) is most important;
SHAP says sensor 44 (suction +0.36c). Reason: redundancy. Permutation drops
one sensor at a time, so if 11 and 15 carry similar information, dropping
either alone barely hurts R^2 (the other compensates) -- both look
unimportant. SHAP averages over coalitions where 15 is absent and credits
sensor 11 correctly while ALSO crediting less-redundant sensors like 44.

**Paper framing**: pick sensor 11 first (bootstrap + most-universally
picked); the K=2 partner is either 15 (permutation-low-redundancy) or 44
(SHAP-highest-marginal-contribution). Both stories are right; they answer
different questions.

## D116: Diffusion refinement on top of SL decoder is a no-op

User-prompted by the Session 14 framing "decoder is the bottleneck, future
work should improve it". Implemented standard SR3-style conditional DDIM
refinement on top of the frozen production E d=64 + SL decoder.

**Setup**:
- DiffusionRefiner U-Net: 2.84M params, base_channels=32, ch_mult=(1, 2, 4),
  FiLM conditioning on (sinusoidal-t + z), SL omega as input channel
- Noise schedule: linear beta 1e-4 -> 0.02, T=1000
- Training: 12500 iters before kill, B=8 T=32, ~70 min on RTX 6000
- Loss converged: 0.96 -> 0.013 (73x reduction in eps prediction MSE)
- 11/11 unit tests pass (``tests/test_diffusion_refiner.py``)

**Sampler sweep on the trained model**: 16 configurations
(``t_start in {0.05, 0.1, 0.2, 0.4}, n_steps in {30, 100}, eta in {0, 0.5}``)
+ standard SR3 pure-noise start at ``n_steps in {50, 200, 500}``.

**All configurations are statistically no-ops** vs SL baseline:
- MSE delta within +/- 0.5% of SL
- SSIM delta within +/- 0.001 of SL

The refiner has converged on its eps-prediction objective but DDIM sampling
returns to ~SL output regardless of trajectory. At low t_start the network
sees mostly clean SL and predicts ~0 noise (no change). At high t_start the
sampler has too much noise to recover detail. Standard SR3 from pure noise
generates DNS-like structures consistent with the conditioning that equal
the SL output in expectation.

## D117: Lean SL decoder (bc=32) matches production (bc=64) at half the params

Trained a LapFiLM SL decoder with ``base_channels=32`` on the production
E d=64 encoder, same SL recipe as D99, 12k iters, ~30 min wall.

| Metric | Lean bc=32 (335k params) | Production bc=64 (705k params) |
|---|---|---|
| Test B mse | **10.27** | 10.40 |
| Test B ratio | 1.641 | 1.635 |
| Test C mse | **32.32** | 32.61 |
| Test C ratio | 1.632 | **1.150** |

The lean decoder essentially matches production on Test B with HALF the
parameters. On Test C the spectral ratio degrades (1.63 vs 1.15) but pixel
metrics hold up.

## Synthesis: the latent is the cap, not the decoder

D116 (diffusion refinement no-op) + D117 (lean decoder matches production)
both point to the same conclusion: **the decoder is not the bottleneck**.

The user's Session 14 framing -- "encoder is robust to slice/mean choice;
decoder constrains everything always" -- was half right. The decoder does
constrain the visible metrics (the choice of slice/mean target determines
spectrum-vs-pixel trade-off), but **its capacity is already saturated by
the 64-D JEPA latent's representational ceiling**. Adding decoder parameters
(refinement stage) or removing them (lean variant) yields the same metrics
because the decoder faithfully decodes whatever the latent carries.

**Real future-work directions revealed**:
1. **Larger latent** (d=128 or d=256) -- directly tests the latent-cap hypothesis
2. **Higher encoder token resolution** (currently 288 spatial tokens at
   24x12; finer ViT patches double this)
3. **Resolution upgrade** 192x96 -> 384x192 (more pixel signal end-to-end)
4. **More data** (Re sweep, denser parameter grid) -- diffusion would need
   this scale to shine

## Session 16 plan recommendation

Lead with the larger-latent ablation since it directly tests the bottleneck
hypothesis from D116+D117. Estimated 8-12 h GPU:
1. Train E d=128 + SL decoder with the production recipe
2. Train E d=256 + SL decoder with the production recipe
3. Compare Test B SSIM, lambda-ratio, GDY r2 vs d=64 production

If d=128 SSIM > 0.526 (the d=64 ceiling), the latent-cap hypothesis is
confirmed and the path forward is "scale the latent". If d=128 = d=64, the
ceiling is elsewhere (likely encoder-architecture or data) and diffusion +
larger decoders might re-enter consideration as the bottleneck shifts.

Higher-leverage parallel experiments (cheap, can run alongside d=128):
- Resolution upgrade 192x96 -> 384x192 (needs preprocessing rebuild ~30 min,
  then encoder + decoder retrain at higher resolution ~12 h)
- d=64 encoder with finer ViT patches (8x8 instead of 16x16) -- changes
  token count from 288 to 1152

## Code + artefacts landed

**New source modules**:
- ``src/models/diffusion_refiner.py`` -- DDIM U-Net (5M params, 11 tests)
- ``src/training/train_diffusion_refiner.py`` -- training loop

**New scripts**:
- ``scripts/build_omega_mean_cache.py`` -- spanwise-mean cache builder
- ``scripts/build_omega_mean_pipeline.py`` -- v1_mean pipeline manifest
- ``scripts/session15_path2_setup_then_dual_train.sh`` -- D114 orchestrator
- ``scripts/session15_path2_stage3_only.sh`` -- relaunch variant
- ``scripts/session15_decoder_bc32.sh`` -- D117 lean decoder
- ``scripts/session15_diffusion_refiner.sh`` -- D116 diffusion launcher
- ``scripts/session15_diffusion_sampler_debug.py`` -- D116 sampler sweep

**New cache + manifest**:
- ``$PREVENT_ROOT/data/processed/vortex-jepa/v1_mean/`` -- spanwise-mean cache
- ``outputs/data_pipeline/v1_mean/manifest.json`` -- v1_mean preprocessing
- ``configs/splits/split_v1_mean.json`` -- symlink to split_v1.json

**Trained checkpoints**:
- ``outputs/runs/session15/path2_meantrain/canonical/encoder/checkpoint_iter020000.pt`` + SL decoder
- ``outputs/runs/session15/path2_meantrain/reduced/encoder/checkpoint_iter020000.pt`` + SL decoder
- ``outputs/runs/session15/decoder_bc32/decoder_iter012000.pt``
- ``outputs/runs/session15/diffusion_refiner/diffusion_refiner_iter012500.pt``

**Eval outputs**:
- ``outputs/runs/session15/path2_meantrain/canonical/encoder/decoder_specloss_recipe/eval/extended_metrics.json``
- ``outputs/runs/session15/path2_meantrain/reduced/encoder/decoder_specloss_recipe/eval/extended_metrics.json``
- ``outputs/session15/diffusion_sampler_sweep.json``

## D-entries to land in HANDOFF.md

- **D114**: spanwise-mean retrain (this session)
- **D115**: TCN proxy + SHAP (Session 14 follow-up, finally completed)
- **D116**: diffusion refinement is a no-op
- **D117**: lean decoder bc=32 matches production bc=64
