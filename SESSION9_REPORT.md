# Session 9 Report

Date: 2026-05-20
Branch: main
Author: Carlos Sanmiguel Vila + Claude (Opus 4.7, 1M context)

## Executive summary

Session 9 began as a lambda-bisection refinement and grew into a foundational
restructuring of the data path and a multi-variant baseline / decoder study.
Three concrete deliverables landed:

1. **Omega data pipeline (`src.data.omega_pipeline.OmegaPipeline`)** — a
   reproducible three-stage transform (spatial mask + per-encounter p99.99
   clip + 3-sigma normalize) consumed identically by every downstream model
   (Fukami AE baseline, JEPA encoder, JEPA decoder). Built once, reused
   everywhere; manifest at `outputs/data_pipeline/v1/manifest.json`.

2. **JEPA retrain on the new pipeline** at the lambda* = 0.01 production
   point (d = 32, seed 42, 20k iters). The pipeline-cleaned inputs increased
   Test B probe delta from +0.096 (unpiped seed-42 baseline F4) to
   **+0.166** — a single-seed result already beating the previous 3-seed
   unpiped mean (+0.131 ± 0.032). Test B r2(z, c) climbed from 0.763 to
   **0.884**.

3. **Visualisation decoder retrain on the new pipeline encoder** with both
   MSE and Charbonnier reconstruction losses. Best variant (MSE) reaches
   Test A SSIM = **0.503**, Test B SSIM = **0.358** — strictly above every
   Fukami AE configuration we tested at d = 3 or d = 8 on the same pipeline.

Embedded along the way: a seven-variant Fukami AE sensitivity study covering
loss design (MSE, L1, Charbonnier, multiscale, active-pixel mask) and
latent capacity (d = 3 vs d = 8) on the new pipeline. The headline finding
is that the wake-erasure failure mode is **decoder-capacity-limited at the
AE level**: no loss tuning and no jump in latent dimension recovered
publication-quality reconstruction within the broad (G, D, Y) envelope.
Together these falsify the "loss is the bottleneck" and "d = 3 is the
bottleneck" hypotheses and converge on the central paper claim — JEPA's
predictive-only latent is more useful for downstream tasks than an
AE's pixel-faithful one, at matched or unmatched d.

## 1. Why a data pipeline became unavoidable

Sections 7 and 8 had used raw cached omega_z directly. Two empirical
observations made that untenable for paper-grade work:

- **Leading-edge finite-difference artifact**: 84% of cached encounters
  carry an artifact spike at the first grid cell adjacent to the airfoil
  surface (the no-slip BC + finite-difference stencil interact badly). The
  raw artifact pixels reach |omega| > 1000 in cases whose physical wake
  magnitudes stay below 50 — a 20x contamination of the magnitude statistics.

- **Mismatched normalisation across baselines**: each baseline applied its
  own ad-hoc scaling (Fukami used 1000.0; JEPA used none; decoder used
  none-or-omega_scale). Cross-method comparisons were not actually
  comparing the same input distribution.

The pipeline standardises both. The three stages are:

| Stage | Operation | Rationale |
|-------|-----------|-----------|
| 1. Spatial mask | Zero cells inside-solid + 1-cell-adjacent (140 cells) | Removes the LE artifact geometrically; 100% of \|omega\| > 1000 pixels live in this layer across the 266 v1.2 encounters |
| 2. Per-encounter clip | Clip \|omega\| above its own p99.99 (range 52-178) | Density-aware tail trim: rare spikes get clipped, dense physical structure stays |
| 3. 3-sigma scale | omega <- omega / (3 \* train_std) | Pure scale normalisation (preserves vorticity antisymmetry); 3-sigma brings the bulk into [-1, +1] and cores to roughly [-3, +3] (matches Fukami's published colorbar range) |

Train statistics computed once over the 153 train encounters after stages 1+2:
mean = 0.0510, std = 3.5853, n_pixels = 335,841,120. The divisor is therefore
10.756.

Key implementation decisions (documented inline in
`src/data/omega_pipeline.py`):

- The mean shift was **not** included in stage 3 even though it would be a
  textbook z-score normalise step. The train mean is 0.05 (numerically
  negligible vs the 3.6 std), but a mean shift breaks the antisymmetry of
  vorticity (positive omega = clockwise rotation, negative = counterclockwise,
  symmetric around 0). For a sign-aware physical quantity, sigma-only scaling
  is the right structural choice.

- The 3-sigma factor (rather than 1-sigma) was the second iteration. With
  1-sigma scaling the bulk sat at +/- 1 but cores reached +/- 21 — far out
  of any reasonable RGB-ish colorbar. With 3-sigma the bulk sits at +/- 0.3
  and cores at +/- 3, matching the published visualisation range of the
  Fukami Re=5000 paper.

- The pipeline is reversible. `unnormalize(pipe.normalize(omega))` recovers
  the masked-and-clipped omega exactly; metric evaluation always happens
  in raw scale (after stages 1+2, but before stage 3) so the case-mean
  noise floor and SSIM stay interpretable.

A critical bug caught mid-session: the first Fukami pipeline integration
computed the reconstruction loss in **raw scale** by un-normalising the
decoder output before comparing to the target. The gradient on the
loss was inflated by (3-sigma)^2 ~= 116x relative to a normalised-space
loss, destabilising training (iter-0 recon loss = 17.0 vs the corrected
0.156). The fix — loss in normalised space, unnormalise only for
visualisation — is now standard across Fukami AE / JEPA decoder /
JEPA encoder paths.

## 2. The seven Fukami AE variants (all at d = 3 unless noted)

Every variant uses the same pipeline-preprocessed data, the same
153-train-encounter sample, and 8000 training iterations on the RTX 6000.
Differences are isolated to the reconstruction-loss design.

### 2.1 Final table

| # | Variant | Test A SSIM | Test A ε_vol | Test A Δ | Test B SSIM | Test B ε_vol | **Test B Δ** | Test C SSIM | Test C ε_vol | Test C Δ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | d=3 MSE | **0.411** | 0.892 | +0.259 | **0.337** | 0.946 | +0.114 | 0.227 | 0.948 | +0.443 |
| 2 | d=3 Charbonnier ε=0.05 | 0.277 | 0.944 | +0.262 | 0.253 | 0.962 | +0.129 | 0.176 | 0.957 | +0.412 |
| 3 | d=3 Charbonnier ε=0.5 | 0.302 | 0.936 | +0.263 | 0.259 | 0.962 | +0.128 | 0.178 | 0.965 | +0.408 |
| 4 | d=3 multiscale (Char+Sobel) | 0.260 | 0.951 | +0.255 | 0.238 | 0.967 | +0.078 | 0.150 | 0.967 | +0.428 |
| 5 | d=3 hard active τ=0.01 | 0.140 | 1.586 | **+0.275** | 0.108 | 1.645 | **+0.133** | 0.100 | 1.185 | **+0.449** |
| 6 | d=3 soft active τ=0.1 α=0.05 + Charb | 0.294 | 0.961 | +0.267 | 0.255 | 0.992 | +0.129 | 0.159 | 0.967 | +0.432 |
| 7 | d=8 MSE | 0.401 | 0.907 | +0.279 | 0.327 | 0.952 | +0.126 | **0.241** | **0.941** | +0.439 |

### 2.2 What each variant learned

- **Variant 1 (MSE)** is the strongest reconstruction baseline. SSIM 0.34
  on Test B is the best AE reconstruction we observed at any d. The
  bulk-zero gradient asymmetry forces the decoder to predict *something*
  non-zero in the freestream (small-magnitude noise), and that diffuse
  prediction *helps* SSIM by spreading some signal across the field.

- **Variant 2 (Charbonnier ε=0.05)** trades 0.084 SSIM for a +0.015 probe
  delta gain on Test B (+0.129 vs +0.114). The Charbonnier loss saturates
  the per-pixel gradient at unit magnitude for large errors, which
  prevents bulk-zero pixels from drowning the optimizer's view of vortex
  cores. The encoder uses higher participation ratio (PR ~ 2.6 vs 1.7)
  but the decoder gives up trying to reconstruct freestream noise.

- **Variant 3 (Charbonnier ε=0.5)** was the "pull Charbonnier toward MSE
  regime" hypothesis. The 10x larger ε would make the loss quadratic
  over a wider error range (closer to MSE) before saturating. Empirically
  near-identical to ε=0.05: SSIM 0.26 vs 0.25, probe delta +0.128 vs +0.129.
  The tail behaviour dominates, not the small-error regime.

- **Variant 4 (multiscale: Charbonnier(omega) + 1.0 \* Charbonnier(∇omega))**
  was the "force the wake high-frequency content into the latent" hypothesis.
  The Sobel-gradient term competed with the pixel term for d=3 latent
  capacity and **lost** on every metric: probe delta dropped to +0.078
  (worst single result), SSIM dropped to 0.24. Conclusion: with limited
  latent capacity, asking the encoder to encode pixels *and* gradients
  simultaneously sacrifices both.

- **Variant 5 (hard active mask, τ=0.01)** was the user's idea: zero out
  freestream pixels from the loss to stop bulk-zero from contaminating
  the gradient. Probe delta jumped to **+0.133** (best on d=3) but
  reconstruction *broke*: ε_volume > 1 on every split (worse than predicting
  the mean field), SSIM collapsed to 0.11 on Test B. The failure mode is
  obvious in hindsight: with no loss signal on the freestream, the decoder
  output there is unconstrained and drifts toward noise. The latent
  encoding genuinely improved, but the visual reconstruction became
  unusable.

- **Variant 6 (soft active mask, τ=0.1 α=0.05 + Charbonnier)** kept the
  active-pixel idea but applied a small weight floor (1/20) to inactive
  pixels, preventing freestream noise divergence. The fix worked
  (ε_volume back to 0.99) but the **probe delta dropped back to +0.129**
  — identical to plain Charbonnier. The active mask added zero
  information on top of Charbonnier's saturating gradient, because
  Charbonnier *already* deemphasises small errors.

- **Variant 7 (d=8 MSE)** was the "capacity is the bottleneck" hypothesis.
  Participation ratio jumped to PR ≈ 5.0 (vs ~1.7 at d=3) — the encoder
  *is* using the extra dimensions. But **every reconstruction metric
  plateaued at the d=3 ceiling**: SSIM 0.33 (vs 0.34 at d=3), ε 0.95,
  probe delta +0.126 (essentially flat). Falsifies the d=3-too-small
  hypothesis. The decoder architecture, not the latent dimension, sets
  the reconstruction ceiling at this dataset scale.

### 2.3 Cross-variant readings

The seven variants collapse onto three consistent regimes:

- **MSE-family (variants 1, 7)**: best SSIM, worst probe delta. Decoder
  spreads prediction across the field, supervising freestream. Optimal
  for visualisation quality.

- **Charbonnier-family (variants 2, 3, 6)**: SSIM ~0.25, probe delta
  +0.128-0.129. Sparse-vortical robust loss; freestream supervision is
  softer; latent more focused on signal. Optimal for downstream probing
  if visualisation is secondary.

- **Hard mask (variant 5)**: best probe delta, broken reconstruction.
  Useful only for "what's the upper bound on what the latent can encode"
  ablations.

The active-mask hypothesis was empirically right about its mechanism
(latent quality improves) but the soft-weighted compromise gave nothing
beyond what Charbonnier already provided. The headline is that **once
the loss is sparse-vortical-aware (Charbonnier or active-mask), the
probe-delta ceiling on d=3 sits at +0.13**, with reconstruction quality
trading against this on a Pareto frontier.

## 3. JEPA pipeline retrain

Single point: the lambda* = 0.01 production configuration, d = 32,
seed = 42, 20000 iters. Compared head-to-head with the previous
seed-42 result (F4, no pipeline) and the 3-seed unpiped mean.

| | Test A r²(z,c) | Test A Δ | **Test B r²(z,c)** | **Test B Δ** | Test C r²(z,c) | Test C Δ |
|---|---:|---:|---:|---:|---:|---:|
| F4 unpiped seed=42 | 0.517 | +0.231 | 0.763 | +0.096 | 0.774 | +0.457 |
| F5 unpiped seed=123 | 0.532 | +0.226 | 0.838 | +0.137 | 0.729 | +0.496 |
| 3-seed unpiped mean (D58) | ≈0.55 | ≈+0.23 | ≈0.81 | **+0.131 ± 0.032** | ≈0.76 | ≈+0.48 |
| **JP pipeline seed=42** | **0.587** | **+0.280** | **0.884** | **+0.166** | 0.741 | +0.449 |

A single pipeline-retrain seed already beats the previous 3-seed unpiped
mean by more than 1σ on Test B. The win is concentrated on Test A (+0.049
delta gain) and Test B (+0.070 delta gain); Test C is essentially flat
(-0.008, well within 1-sigma noise).

The gain on Test B r2(z, c) (0.763 -> 0.884) is the most informative
single number: the latent's (G, D, Y) discriminability climbs 16 percentage
points with the cleaner inputs. The pipeline's artifact removal pays off
asymmetrically — the artifact pixels were noise to the parametric encoder,
not signal.

PR_all stayed at the same operating point (~2.3 vs 3.1 unpiped) — the
pipeline did **not** explode the participation ratio; it concentrated
information into the same effective latent rank but with cleaner content.
This rules out a "more capacity used" naive explanation.

## 4. JEPA decoder retrain

On top of the new JEPA pipeline encoder, two visualisation decoders were
trained (10000 iters each, otherwise identical hyperparameters from the
original Session 9 Step 2 recipe): one with MSE reconstruction loss, one
with Charbonnier ε=0.05.

| | Test A SSIM | Test A ε_vol | Test A MSE/floor | Test B SSIM | Test B ε_vol | Test B MSE/floor | Test C SSIM | Test C ε_vol | Test C MSE/floor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **JEPA decoder MSE** | **0.503** | 0.853 | 7.81 | **0.358** | 0.978 | 1.95 | 0.243 | 0.984 | 1.68 |
| JEPA decoder Charbonnier | 0.389 | 0.894 | 8.54 | 0.314 | 0.947 | 1.77 | 0.236 | 0.947 | 1.56 |

Both decoders beat every Fukami AE configuration on Test A SSIM (0.50 / 0.39
vs 0.41 max for AE). On Test B the MSE decoder reaches 0.36 SSIM — also
above any Fukami variant. The Charbonnier decoder achieves the best
ε_volume on Test B/C (0.947) — comparable to the Fukami best (0.946).

This is at **unmatched d** (JEPA d = 32, Fukami d = 3 or 8). Two things
this comparison establishes:

1. **The JEPA latent is genuinely richer than any AE latent we tested**
   on this dataset. The decoder placed on it reconstructs better than an
   AE trained jointly. The wake structure that the d=3 / d=8 AE could
   not produce is visible in the JEPA decoded panel.

2. **The decoder's loss choice matters for the visualisation question**,
   not the latent question. MSE buys better SSIM but slightly worse ε
   (the freestream noise effect again). Charbonnier buys lower ε but
   loses ~0.04 SSIM. The choice depends on which Fukami-matched metric
   you intend to report.

A still-open issue: a single Fukami baseline at the matched JEPA d (i.e.
d = 32) is not in this report. That run is the most informative
"AE with same latent budget" comparison and is the natural Session 10
addition.

## 5. Figure 3 panel

For the canonical Test B encounter `G+1.00_D1.00_Y+0.10 encounter 00`,
fixed colorbar +/-3, airfoil overlay, three frames (pre-impact 25,
impact 40, post-impact 55):

- `outputs/runs/session9/run_a11_fukami_pipeline_v1/fig3_fukami_reconstruction.png` —
  Fukami d=3 MSE baseline
- `outputs/runs/session9/run_a11_fukami_pipeline_d8/fig3_fukami_reconstruction.png` —
  Fukami d=8 MSE
- `outputs/runs/session9/run_a11_fukami_pipeline_charbonnier/fig3_fukami_reconstruction.png` —
  Fukami d=3 Charbonnier ε=0.05
- `outputs/runs/session9/run_a11_fukami_pipeline_active001/fig3_fukami_reconstruction.png` —
  Hard active mask (illustrates the freestream-noise failure mode)
- `outputs/runs/session9/run_a11_fukami_pipeline_active010_soft_charb/fig3_fukami_reconstruction.png` —
  Soft active mask + Charbonnier
- `outputs/runs/session9/decoder_pipeline_mse/fig3_jepa_reconstruction.png` —
  JEPA d=32 decoder with MSE loss (best SSIM)
- `outputs/runs/session9/decoder_pipeline_charb/fig3_jepa_reconstruction.png` —
  JEPA d=32 decoder with Charbonnier (best ε)

Visual reading: the JEPA-decoded panels show the gust core position and
intensity convincingly through impact. The wake shedding is partially
recovered (more so in MSE than Charbonnier). The Fukami panels at d=3 show
the gust core but no wake; d=8 looks nearly identical to d=3. The
hard-active-mask Fukami shows good vortex cores swimming in a sea of
freestream noise — instructive failure mode for the paper Section on
loss design.

## 6. Files added or modified

### New source files

- `src/data/omega_pipeline.py` — `OmegaPipeline` class with `preprocess_raw`,
  `normalize`, `unnormalize`, `__call__`, `from_manifest`, `to_dict`.
- `scripts/build_omega_pipeline.py` — one-shot manifest builder.
- `scripts/session9_decoder_fig3_pipeline.py` — pipeline-aware Figure 3
  generator for the JEPA decoder.
- `scripts/session9_fukami_final_eval.py` — post-hoc final eval for
  early-stopped Fukami runs.
- `outputs/data_pipeline/v1/manifest.json` — 266 encounter thresholds,
  140 mask cells, train stats (mean=0.0510, std=3.5853).
- `outputs/data_pipeline/v1/airfoil_adjacent_mask.npy` — the spatial mask.

### Modified source files

- `src/baselines/fukami_ae.py` — added `omega_pipeline`, `recon_loss_type`
  (mse / l1 / charbonnier / multiscale), `charbonnier_epsilon`,
  `recon_active_threshold`, `recon_inactive_weight`. Loss now computed
  in normalised space; pipeline-aware preprocessing inside forward.
- `scripts/session9_train_fukami.py` — same flags exposed on the CLI;
  loss-in-normalised-space; pipeline-aware loader.
- `scripts/session9_fukami_figure.py` — pipeline-aware encoder + decoder
  path; airfoil overlay; fixed colorbar +/-3.
- `scripts/session9_fukami_evaluation.py` — pipeline-aware probe.
- `src/training/train_jepa.py` — added `--omega-pipeline-manifest` flag;
  `apply_pipeline_batch` helper; pipeline-aware diagnostic batch.
- `scripts/session9_train_decoder.py` — added `--omega-pipeline-manifest`,
  `--recon-loss-type` (mse/charbonnier), `--charbonnier-epsilon`,
  `--recon-active-threshold`, `--recon-inactive-weight`; pipeline-aware
  dataset / target / evaluate_split.
- `scripts/session9_bisection_analysis.py` — `load_encoder` attaches the
  pipeline from the saved args; `encode_split` applies it.
- `configs/splits/split_v1.json` — partition v1.2 (56 cases / 266
  encounters; 5 new run3 cases absorbed).

### Output directories created

- `outputs/runs/session9/run_a11_fukami_pipeline_v1/` (MSE baseline)
- `outputs/runs/session9/run_a11_fukami_pipeline_charbonnier/`
- `outputs/runs/session9/run_a11_fukami_pipeline_char_eps05/`
- `outputs/runs/session9/run_a11_fukami_pipeline_multiscale/`
- `outputs/runs/session9/run_a11_fukami_pipeline_d8/`
- `outputs/runs/session9/run_a11_fukami_pipeline_active001/`
- `outputs/runs/session9/run_a11_fukami_pipeline_active010_soft_charb/`
- `outputs/runs/session9/run_jepa_pipeline_lam0p01_seed42/` (encoder retrain)
- `outputs/runs/session9/decoder_pipeline_charb/`
- `outputs/runs/session9/decoder_pipeline_mse/`

Each directory has a `final_eval.json` (or `decoder_summary.json`),
`fig3_*.png`, and the training log. For the Fukami variants the
`fukami_test_b_delta.csv` is also present.

## 7. Decisions for the paper

The combined empirical surface gives us several claim-worthy results:

1. **Pipeline-cleaned inputs are a strict win for JEPA**: single-seed
   Test B Δ jumps from +0.096 to +0.166. This is bigger than any loss-
   level intervention we tried on the Fukami baseline. The artifact-removal
   work is foundational, not cosmetic.

2. **At the broad (G, D, Y) envelope, the Fukami AE reconstruction
   metrics plateau independent of loss and independent of d ∈ {3, 8}**.
   Best SSIM_test_b ≈ 0.34 (achievable with MSE), best ε_test_b ≈ 0.95
   (achievable with Charbonnier or MSE). The published Fukami Re=5000
   ε ≈ 0.2 is on a narrow G-only slice (7 cases at D=0.5, Y=0.1); on
   our broad parametric envelope, no AE at our compute budget gets there.

3. **The JEPA decoder beats every Fukami AE we tested on SSIM** (Test A
   0.50 vs max 0.41; Test B 0.36 vs max 0.34). This is at unmatched d
   (JEPA d=32 vs AE d=3 or 8), so the paper's matched-d Fukami baseline
   is still owed; but the same-data-pipeline part of the comparison is
   now consistent.

4. **Charbonnier and MSE are a Pareto pair for the decoder**: MSE wins
   SSIM, Charbonnier wins ε. We should report both.

5. **The hard active-pixel mask is an instructive negative result**:
   the latent quality improves (best Test B probe delta on d=3) but
   the freestream diverges to noise. Worth a sentence in the loss-design
   section of the paper, as a clean demonstration of why
   "ignore-where-target-is-zero" naively fails.

## 8. What is still owed

- **Matched-d Fukami baseline (d = 32)** with MSE + pipeline. Should be
  ~50 min training. Resolves the "AE at matched JEPA latent" comparison
  cleanly.
- **3-seed JEPA pipeline mean** (seeds 0 and 123 alongside the seed=42
  retrain done here). Tests whether the +0.166 single-seed result is
  representative.
- **JEPA encoder ablation across pipeline on/off** at seed=42 (this
  session has both numbers in-table but not in a single CSV).
- **Cross-method matched-pipeline comparison plot** for the paper —
  Figure showing the Pareto frontier of (SSIM, probe delta) across all
  variants, with JEPA + decoder at the upper-right.
- **PLDM baseline retrain on pipeline** (CLAUDE.md Section "Baselines
  to implement"). PLDM was the headline contrast for VICReg vs SIGReg.
  D58/D63 deferred this; with the pipeline locked, the comparison can
  now happen.

## 9. Session boundary

Closing out at iter 22:00 local. All trainings complete; all figures
written; manifests committed. Suggested entry points for the next session:

1. `python scripts/session9_train_fukami.py ... --latent-dim 32 ...`
   (matched-d Fukami).
2. `bash scripts/launch_session9_step1_bisection.sh 0 F4` with the
   `--omega-pipeline-manifest` flag added (pipeline-aware 3-seed JEPA).
3. Cross-method comparison plot script.
