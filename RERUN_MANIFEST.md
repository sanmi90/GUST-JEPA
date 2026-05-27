# Rerun manifest for the paper

Comprehensive recipe to regenerate every paper-load-bearing artifact from
scratch when the final dataset lands. Stages are topologically ordered;
each stage's outputs feed the next. Compute estimates assume the two RTX
6000 Blackwell cards used in parallel.

Document compiled 2026-05-27 by surveying all session reports (Sessions
9-18), HANDOFF.md D-entries, src/, scripts/session{9-18}/, configs/,
and outputs/.

## Files you MUST NOT discard before rerun

These are pure inputs (code + docs); the rerun produces everything else.

- `CLAUDE.md`, `HANDOFF.md`
- `SESSION18 PLAN.md`, `SESSION18_B1_PROTOCOL.md`
- `configs/preprocessing.yaml`
- `build_split_manifest.py` (encodes the Test B selection)
- All of `src/`
- All of `scripts/session{9,10,11,12,13,14,15,16,17,18}/`
- `data_manifest/raw_cases_inventory.yaml` (will be overwritten by Stage 0)

Outputs safe to delete before rerun: everything under `outputs/runs/`,
`outputs/session{14,15,16,17,18}/`, `outputs/data_pipeline/v1/`. Keep
`outputs/schema_inspection/schema.yaml` for reference.

## Topological order

```
Stage 0  inventory  (data_manifest YAML, refreshed from PREVENT)
Stage 1  cache       (per-encounter .h5 files at $VORTEX_JEPA_CACHE)
Stage 2  split       (configs/splits/split_v1.json)
Stage 3  pipeline    (outputs/data_pipeline/v1/manifest.json)
              |
              +--> Stage 4  JEPA encoder + predictor + decoder + latents
              |               (4 encoders: production + 3 seeds; 1 SL decoder; latents NPZ)
              |
              +--> Stage 5  baselines (Fukami AE x 3 d; POD x 3 d) + DNS metrics
              |
              +--> Stage 6  Session 16 analyses (D118-D121)
              +--> Stage 7  Session 17 analyses (D123-D128)
              +--> Stage 8  Session 18 B1 (Case A locked) + Figures 4, 5, S
              +--> Stage 9  Paper-figure assembly (8 main figs, supplementary)
```

## Stage 0: Data inventory

| Artifact | Producer | Rerun trigger |
|---|---|---|
| `data_manifest/raw_cases_inventory.yaml` | PREVENT side (`/home/carlos/PREVENT/scripts/periodic_v2/100c_raw_cases_inventory.py`); copy here | New raw `.h5` files in `$PREVENT_ROOT/data/raw/periodic{,/run3}` |

After PREVENT regenerates the manifest:
```bash
cp $PREVENT_ROOT/data_manifest/raw_cases_inventory.yaml data_manifest/
```
Compute: seconds. Verify `manifest_version` bump and `summary.n_cases_total`.

## Stage 1: Preprocessing cache

| Artifact | Producer | Inputs |
|---|---|---|
| `$VORTEX_JEPA_CACHE/v1/{case_id}/encounter_{k:02d}.h5` | `scripts/preprocess.py` | inventory + `configs/preprocessing.yaml` |

Each encounter file: `omega_z (120, 192, 96)`, `p_wall (120, 192)`, `C_L (120,)`, `C_D (120,)` + 17 attrs.

```bash
python scripts/preprocess.py --partition v1
```
Compute: 1-2 h (raw H5 read bandwidth bound).

## Stage 2: Split manifest

| Artifact | Producer | Inputs |
|---|---|---|
| `configs/splits/split_v1.json` | `build_split_manifest.py` | inventory |

Split policy (hardcoded in `build_split_manifest.py`):
- G == +4 -> test_c; case_id in TEST_B_CASE_IDS (6 hand-picked interior cases) -> test_b; else train
- Train cases: periodic uses encounters [0,1,2,3] for train, [4,5] for test_a; run3 uses [0,1,2] for train, [3] for test_a
- Baseline.h5 stays in train + test_a with `is_calibration_reference: true`

```bash
python build_split_manifest.py
```
Compute: seconds. Verify `summary.n_cases_total` and the test_b case list still match policy.

## Stage 3: Omega pipeline manifest

| Artifact | Producer | Inputs |
|---|---|---|
| `outputs/data_pipeline/v1/manifest.json` | `scripts/build_omega_pipeline.py` | cache + split |
| `outputs/data_pipeline/v1/airfoil_adjacent_mask.npy` | same | same |

Three-stage pipeline (mask + per-encounter p99.99 clip + 3-sigma scale).
Manifest carries `train_stats.{mean, std}` (current std = 3.5526; divisor =
10.658) and `thresholds[case_id][encounter_idx]` for per-encounter clipping.

```bash
python scripts/build_omega_pipeline.py --partition v1 \
    --output-dir outputs/data_pipeline/v1
```
Compute: 10-20 min.

## Stage 4: JEPA encoder + decoder + latents

### 4a. Production JEPA d=64 (Session 12 Direction E, D99 winner)

| Artifact | Path | Compute |
|---|---|---|
| Production checkpoint | `outputs/runs/session12/S12_E_d64/encoder/checkpoint_iter020000.pt` | ~1.6 h on one card |

The locked training recipe (do not deviate without explicit user approval):
```bash
python -m src.training.train_jepa \
    --all-train --max-iters 20000 --seed 42 \
    --d 64 --B 16 --T 32 --H-roll 8 \
    --lambda-sigreg 0.01 \
    --lr-encoder 1.5e-4 --lr-predictor 5e-4 \
    --weight-decay 0.05 --warmup-frac 0.05 --num-workers 4 \
    --projection-norm batchnorm --anticollapse sigreg \
    --observable-head cl_future --observable-head-weight 0.01 \
    --observable-head-deltas 0 \
    --wake-observable-type patch_signed_spectrum --lambda-wake 1.00 \
    --wake-loss smooth_l1 --wake-loss-beta 0.5 --wake-head-hidden 128 \
    --omega-pipeline-manifest outputs/data_pipeline/v1/manifest.json \
    --gpu 0 --output-dir outputs/runs/session12/S12_E_d64/encoder
```
Wrapper script: `scripts/session12_launch_direction_e.sh`.

### 4b. Seed retrains (Thrust 6, three seeds)

| Artifact | Path | Compute |
|---|---|---|
| Seed 0/1/2 encoders | `outputs/runs/session14/thrust6/jepa_d64_seed{0,1,2}/encoder/checkpoint_iter020000.pt` | ~5 h sequential on one card, or ~3 h with parallel |

Identical recipe, only `--seed` varies (0, 1, 2). Required for cross-seed
analyses (Session 16 D118, Session 17 D123, D125).

```bash
bash scripts/session14_thrust6_jepa_seeds.sh 0   # GPU 0
bash scripts/session14_thrust6_jepa_seeds.sh 1   # GPU 1 in parallel
```

### 4c. SL decoder on frozen production encoder

| Artifact | Path | Compute |
|---|---|---|
| SL decoder | `outputs/runs/session12/S12_E_d64/encoder/decoder_specloss_recipe/decoder_iter012000.pt` | ~3-4 h |

LapFiLM decoder + spectral-amp + gradient loss (PRF 2026 recipe).
Operating point is iter 12000 per `extended_metrics.json` eval. Pass
criterion D99: Test A SSIM_mean >= 0.60.

```bash
bash scripts/session13_relaunch_decoder_specloss.sh \
    outputs/runs/session12/S12_E_d64/encoder 0
```

(Optional) Also train SL decoders on the 3 seed retrains if the
cross-seed visualisation appendix is needed.

### 4d. Pre-extract latents (consumed by Sessions 16, 17, 18)

| Artifact | Path | Compute |
|---|---|---|
| Production latents | `outputs/session14/latents/S12_E_d64/{train,test_a,test_b,test_c}.npz` | ~10 min |
| Per-seed latents | `outputs/session17/seed_latents/{production,seed0,seed1,seed2}/{train,test_b,test_c}.npz` | ~15 min total |

NPZ keys: `z, z_full, G, D, Y, case_id, encounter_index, impact_frame, split`.

```bash
python scripts/session14_encode_latents.py
python scripts/session17/encode_seed_latents.py
```

## Stage 5: Baselines + DNS metrics + flow descriptors

All Stage 5 substages are INDEPENDENT and can run in any order; they all
need Stage 1-3 but NOT Stage 4 outputs.

### 5a. Fukami AE x 3 d (Session 18 B1 Part a, paper-faithful per arXiv:2305.08024)

| Artifact | Path | Compute |
|---|---|---|
| d=3 checkpoint | `outputs/session18/exp_b1/fukami_ae_d3/checkpoint_iter020000.pt` | ~3 h per d |
| d=32 checkpoint | `outputs/session18/exp_b1/fukami_ae_d32/checkpoint_iter020000.pt` | (or 4-6 h total parallel) |
| d=64 checkpoint | `outputs/session18/exp_b1/fukami_ae_d64/checkpoint_iter020000.pt` | |
| L-curve sweep | `outputs/session18/exp_b1/lcurve_sweep/d3_beta{...}/` | ~1 h total |

**Critical**: the SESSION18_B1_PROTOCOL.md-locked recipe is `--recon-loss-type
mse --observable-head-deltas 0 --lambda-lift 0.01` (β=0.01 from L-curve
elbow on our data; the paper's β=0.05 came from L-curve on Fukami's
narrower data; methodology preserved, value re-derived). See
`scripts/session18/README.md` for the per-d invocation.

```bash
# Sweep first to verify the L-curve elbow on the new data:
bash scripts/session18/lcurve_sweep.sh   # or expand manually per README
python scripts/session18/lcurve_analysis.py
# Then train d=3, 32, 64 at the elbow beta (likely still 0.01):
for d in 3 32 64; do
  bash scripts/session18/train_fukami_baselines.sh "$d" 0  # or 1
done
```

### 5b. POD bases x 3 d (B1 Part b)

| Artifact | Path | Compute |
|---|---|---|
| d=16, 32, 64 | `outputs/session18/exp_b1/pod_d{16,32,64}/pod_basis.npz` + `pod_summary.json` | ~15-30 min CPU |

Closed-form snapshot SVD on pipeline-normalised train frames; no training.

```bash
bash scripts/session18/compute_pod_baselines.sh "16 32 64"
```

### 5c. DNS physical metrics (consumed by S17 Exp 2 + S18 B1 Part d)

| Artifact | Path | Compute |
|---|---|---|
| Per-frame DNS metrics | `outputs/session17/exp2/dns_physical_metrics.npz` | ~15 min |

Per-frame `C_L, C_D, I_y, I_x, wake_enstrophy, circulation_pos/neg` for every encounter in every split.

```bash
python scripts/session17/exp2_dns_physical_metrics.py
```

### 5d. Per-frame flow descriptors (consumed by S16 Exp 2)

| Artifact | Path | Compute |
|---|---|---|
| Targets | `outputs/session16/exp2/per_frame_targets/{train,test_a,test_b,test_c}.npz` | ~20 min |

14 scalar columns per (encounter, frame) plus `z_full` mirrored from
Stage 4d.

```bash
python scripts/session16/exp2_build_targets.py
```

## Stage 6: Session 16 analyses (D118-D121)

All scripts in `scripts/session16/`. Outputs under `outputs/session16/`.
Depends on Stage 4d (latents) + Stage 5d (flow descriptors).

Headline experiments (rerun ALL of these; paper Section 4 + Section 6 sources):

| Experiment / D-entry | Script(s) | Output | Figure |
|---|---|---|---|
| Exp 1 manifold geometry (D118) | `exp1a_pls_base.py`, `exp1a_pca_base.py`, `exp1a_diagnostics.py`, `exp1a_bis_nonlinear.py`, `exp1a_ter_followups.py`, `exp1b_decode_axes.py`, `exp1b_axis_summary.py`, `exp1c_seed_variance.py`, `exp1c_pairwise.py` | `outputs/session16/exp1/*` | `outputs/session16/figures/exp1b_axis_decoded_panel.png` |
| Exp 4 Markov closure on latent (D119) | `exp4_markov_closure.py`, `exp4_cond_ablation.py`, `exp4_figure.py` | `outputs/session16/exp4/*` | `outputs/session16/figures/exp4_markov_closure.png` |
| Exp 2 state>parameter content (D120) | `exp2_probe_sweep.py`, `exp2_redo_probes.py`, `exp2_figure.py` | `outputs/session16/exp2/*` | `outputs/session16/figures/exp2_probe_sweep.png` |
| Exp 3 SHAP attribution (D121) + Y SHAP (D121-bis) | `exp3_shap.py`, `exp3_bootstrap.py`, `exp3_intervention.py`, `exp3_figure.py`, `exp3_figure_v2.py`, `exp3_shap_Y.py`, `exp3_shap_Y_figure.py` | `outputs/session16/exp3/*` | `outputs/session16/figures/exp3_shap_{mean,hero_test_b,hero_test_c,Y_mean,Y_hero_test_b,Y_hero_test_c}.png` |

Compute: ~half-day total once latents + targets exist.

## Stage 7: Session 17 analyses (D123-D128)

All scripts in `scripts/session17/`. Outputs under `outputs/session17/`.
Depends on Stage 4 (latents + encoder) + Stage 5c (DNS metrics).

Headline experiments:

| Experiment / D-entry | Script(s) | Output | Figure |
|---|---|---|---|
| Exp 1 trajectory geometry (D123) | `exp1a_projections.py`, `exp1b_trajectory_panel.py`, `exp1c_curvature.py`, `exp1c_extra_signatures.py`, `exp1d_cross_seed.py`, `exp1_day1_summary.py` | `outputs/session17/exp1/*` | `outputs/session17/figures/{exp1_trajectory_panel, exp1_curvature_at_impact, exp1_signatures_at_impact, exp1_cross_seed_distance}.png` |
| Exp 2 physical Markov closure (D124) | `exp2_rollouts_and_probes.py`, `exp2_aggregate.py` | `outputs/session17/exp2/*` (rollout_metrics, horizon_summary, markov_vs_full_delta, etc.) | `outputs/session17/figures/{exp2_physical_closure_horizon, exp2_impulse_lift_scatter}.png` |
| Exp 3 state-functional alignment (D125) | `exp3a_param_recovery.py`, `exp3b_decay_fit.py`, `exp3c_cross_seed_transfer.py`, `exp3d_shap_decay.py` | `outputs/session17/exp3/*` | `outputs/session17/figures/{exp3_param_recovery_vs_tau, exp3_function_transfer_heatmap, exp3_shap_decay_panels}.png` |
| Exp 4 structures from SHAP (D126) | `exp4_structures_shap.py` (reads S16 exp3 SHAP) | `outputs/session17/exp4/*` | `outputs/session17/figures/{exp4_structures_4target_panel, exp4_q_overlap_summary, exp4_Y_sign_flip}.png` |
| Exp 5 closed-loop pressure (D127) | `exp5_nonlinear.py` (canonical; `exp5_closed_loop.py` is the negative linear-ridge variant kept for reproducibility) | `outputs/session17/exp5/*` | `outputs/session17/figures/{exp5_nonlinear_K_curve, exp5_nonlinear_tolerance}.png` |
| Diagnostic D z-norm drift | `diagnostic_d_znorm.py` | `outputs/session17/diagnostic_d/*` | `outputs/session17/diagnostic_d/z_norm_histograms.png` |

Compute: ~half-day to one day. Exp 2 rollouts are the heaviest single step (~1 h).

## Stage 8: Session 18 B1 (paper Section 5 + Figures 4, 5, S)

All scripts in `scripts/session18/`. Outputs under `outputs/session18/`
and `outputs/session18/exp_b1_test3/`. Depends on Stage 4a (production
JEPA encoder + jointly-trained predictor + observable_head), Stage 4d
(latents), Stage 5a-c (Fukami AE checkpoints + POD bases + DNS metrics).

Dependency chain inside B1:
```
Fukami AE + POD checkpoints      -+
JEPA d=64 latents (Stage 4d)     -+--> encode_baseline_latents.py
                                          |
                                          v
                                  train_baseline_predictor.py   --no-output-bn
                                          |
                                          v
                                  eval_baseline_rollouts.py
                                          |
   DNS metrics (Stage 5c) --------+-------+
                                  v
                       physical_metrics_from_rollouts.py
                                  |
                                  v
                      Figure 4 + Figure 5 + Supplementary Figure S
```

**Critical**: the headline JEPA row uses `--no-output-bn` (B1 bug fix; see
D129 and `SESSION18_B1_PROTOCOL.md` for the rationale). The `exp_b1_test3/`
suffix marks the no-BN variant; `exp_b1/` is the original buggy variant
kept for reproducibility.

Launch sequence (per `scripts/session18/README.md`):
```bash
# Encode latents for all 7 (3 Fukami, 3 POD, 1 JEPA)
for d in 3 32 64; do
  python scripts/session18/encode_baseline_latents.py \
      --baseline fukami --d "$d" \
      --checkpoint outputs/session18/exp_b1/fukami_ae_d"$d"/checkpoint_iter006000.pt
done
for d in 16 32 64; do
  python scripts/session18/encode_baseline_latents.py \
      --baseline pod --d "$d" \
      --basis outputs/session18/exp_b1/pod_d"$d"/pod_basis.npz
done
ln -sfn ../../session14/latents/S12_E_d64 \
   outputs/session18/exp_b1/latents_jepa_d64   # symlink the JEPA latents

# Train predictors (no output BN); JEPA uses Test1 path
for tag in fukami_d3 fukami_d32 fukami_d64 pod_d16 pod_d32 pod_d64; do
  python scripts/session18/train_baseline_predictor.py \
      --latents-dir outputs/session18/exp_b1/latents_"$tag" \
      --tag "$tag"_noBN --no-output-bn \
      --output-dir outputs/session18/exp_b1_test3/predictor_"$tag"_noBN
done
python scripts/session18/train_baseline_predictor.py \
    --latents-dir outputs/session18/exp_b1/latents_jepa_d64 \
    --tag jepa_d64_test1_noBN --no-output-bn \
    --output-dir outputs/session18/exp_b1_test3/predictor_jepa_d64_test1_noBN

# Rollouts on test_b and test_c
for tag in fukami_d3 fukami_d32 fukami_d64 pod_d16 pod_d32 pod_d64; do
  python scripts/session18/eval_baseline_rollouts.py \
      --latents-dir outputs/session18/exp_b1/latents_"$tag" \
      --predictor outputs/session18/exp_b1_test3/predictor_"$tag"_noBN/checkpoint_iter006000.pt \
      --tag "$tag"_noBN \
      --output-dir outputs/session18/exp_b1_test3/rollouts_"$tag"_noBN
done
python scripts/session18/eval_baseline_rollouts.py \
    --latents-dir outputs/session18/exp_b1/latents_jepa_d64 \
    --predictor outputs/session18/exp_b1_test3/predictor_jepa_d64_test1_noBN/checkpoint_iter006000.pt \
    --tag jepa_d64_test1_noBN \
    --output-dir outputs/session18/exp_b1_test3/rollouts_jepa_d64_test1_noBN

# Symlink so physical_metrics_from_rollouts finds them
for tag in fukami_d3 fukami_d32 fukami_d64 pod_d16 pod_d32 pod_d64 jepa_d64; do
  ln -sfn ../../exp_b1/latents_"$tag" outputs/session18/exp_b1/latents_"$tag"_noBN
done
for tag in fukami_d3 fukami_d32 fukami_d64 pod_d16 pod_d32 pod_d64; do
  ln -sfn ../exp_b1_test3/rollouts_"$tag"_noBN outputs/session18/exp_b1/rollouts_"$tag"_noBN
done
ln -sfn ../exp_b1_test3/rollouts_jepa_d64_test1_noBN \
    outputs/session18/exp_b1/rollouts_jepa_d64_test1_noBN

# Physical metrics (3 probe families: linear ridge, KRR-RBF, MLP-reg)
python scripts/session18/physical_metrics_from_rollouts.py \
    --baselines jepa_d64_test1_noBN fukami_d3_noBN fukami_d32_noBN fukami_d64_noBN pod_d16_noBN pod_d32_noBN pod_d64_noBN \
    --d-per-baseline 64 3 32 64 16 32 64 \
    --baseline-kind jepa fukami fukami fukami pod pod pod \
    --probe-kind ridge \
    --output-csv outputs/session18/exp_b1_test3/physical_closure_noBN_unified.csv

python scripts/session18/physical_metrics_from_rollouts.py \
    --baselines jepa_d64_test1_noBN fukami_d3_noBN fukami_d32_noBN fukami_d64_noBN pod_d16_noBN pod_d32_noBN pod_d64_noBN \
    --d-per-baseline 64 3 32 64 16 32 64 \
    --baseline-kind jepa fukami fukami fukami pod pod pod \
    --probe-kind krr_rbf \
    --output-csv outputs/session18/exp_b1_test3/physical_closure_noBN_krr.csv

# Figures
python scripts/session18/build_figure4_centerpiece.py  # paper Fig 4 (linear ridge)
python scripts/session18/build_figureS_krr.py          # supplementary S (KRR-RBF)
python scripts/session18/build_comparison_figure.py    # candidate Fig 5
python scripts/session18/plot_lift_recon.py            # Fukami lift figure
python scripts/session18/plot_lift_recon_jepa.py       # JEPA lift figure
python scripts/session18/plot_lift_predictive_horizon.py
python scripts/session18/plot_recon_test_a.py
```

Compute: ~9-12 h with both cards in parallel per `scripts/session18/README.md`.

## Stage 9: Paper figures (canonical paths)

Eight main-text figures + supplementary. After Stages 4-8, the paths
below are the canonical source files for the manuscript.

| Paper figure | Path | Producer |
|---|---|---|
| Fig 1 (problem + JEPA schematic) | manual / illustrator | n/a |
| Fig 2 (trajectory geometry, 3 panels) | `outputs/session17/figures/{exp1_trajectory_panel, exp1_curvature_at_impact, exp1_cross_seed_distance}.png` | S17 Exp 1 scripts |
| Fig 3 (state-functional alignment, 2 panels) | `outputs/session17/figures/{exp3_param_recovery_vs_tau, exp3_function_transfer_heatmap}.png` | S17 Exp 3 scripts |
| Fig 4 (centerpiece, Markov closure) | `outputs/session18/figures/figure4_markov_closure_centerpiece.{png,pdf}` | S18 `build_figure4_centerpiece.py` |
| Fig 5 (baseline bars at H=16) | `outputs/session18/figures/{exp_b1_markov_closure_baselines, exp_b1_markov_closure_noBN_unified}.png` | S18 `build_comparison_figure.py` |
| Fig 6 (magazine cover, SHAP 4-target) | `outputs/session17/figures/exp4_structures_4target_panel.png` + S16 `outputs/session16/figures/exp3_shap_*` | S17 `exp4_structures_shap.py` + S16 `exp3_figure_v2.py` |
| Fig 7 (Q-overlap + Y sign-flip) | `outputs/session17/figures/{exp4_q_overlap_summary, exp4_Y_sign_flip}.png` | S17 `exp4_structures_shap.py` |
| Fig 8 (closed-loop pressure) | `outputs/session17/figures/{exp5_nonlinear_K_curve, exp5_nonlinear_tolerance}.png` | S17 `exp5_nonlinear.py` |
| Supplementary S1 z-norm drift | `outputs/session17/diagnostic_d/z_norm_histograms.png` | S17 `diagnostic_d_znorm.py` |
| Supplementary S2 speed/bend signatures | `outputs/session17/figures/exp1_signatures_at_impact.png` | S17 `exp1c_extra_signatures.py` |
| Supplementary S3 SHAP attribution decay | `outputs/session17/figures/exp3_shap_decay_panels.png` | S17 `exp3d_shap_decay.py` |
| Supplementary S6 probe-family sensitivity | `outputs/session18/figures/figureS_markov_closure_krr.{png,pdf}` | S18 `build_figureS_krr.py` |

Decoder Fig 3 (hero reconstruction, methods appendix):
- `outputs/runs/session12/S12_E_d64/encoder/decoder_specloss_recipe/eval/fig3_jepa_reconstruction.png` (produced as part of Stage 4c eval; see `scripts/session9_decoder_fig3_pipeline.py`).

## Optional appendix material (Session 14)

Not required for paper headline; rerun only if reviewer pushes back.

| Topic | Script | Output |
|---|---|---|
| Epiplexity (prequential coding) | `scripts/session14_tcsi_followups.py` etc. (calls `src/evaluation/epiplexity.py`) | `outputs/session14/epiplexity/*` + `outputs/session14/figures/thrust1_*.png` |
| Intrinsic dimension (3 estimators) | `scripts/session14_make_figures.py` (calls `src/evaluation/intrinsic_dim.py`) | `outputs/session14/intrinsic_dim/*` + figs |
| Concept vectors (AeroJEPA-style) | `scripts/session14_make_figures.py` | `outputs/session14/concept_vectors/*` |
| TCSI sensor selection | `scripts/session14_tcsi_pilot.py` then `_followups.py` | `outputs/session14/tcsi_pilot/*` + `sensor_regions_consensus.png` |
| Long-rollout hero | `scripts/session14_rollout_rmse.py` | `outputs/session14/rollout/S12_E_d64/*` |

## Single-shot rerun recipe

```bash
# Shell setup
cd /home/carlos/GUST-JEPA
source .venv/bin/activate
export PREVENT_ROOT=$HOME/PREVENT WANDB_PROJECT=vortex-jepa

# Stage 0-3 (data + pipeline; serial; ~1-2 h)
cp $PREVENT_ROOT/data_manifest/raw_cases_inventory.yaml data_manifest/
python scripts/preprocess.py --partition v1
python build_split_manifest.py
python scripts/build_omega_pipeline.py --partition v1 \
    --output-dir outputs/data_pipeline/v1

# Stage 4 (JEPA encoder + decoder + latents; parallel-cards; ~8-10 h)
bash scripts/session12_launch_direction_e.sh 0  &           # production on GPU 0
bash scripts/session14_thrust6_jepa_seeds.sh 1  &           # 3 seeds on GPU 1
wait
bash scripts/session13_relaunch_decoder_specloss.sh \
    outputs/runs/session12/S12_E_d64/encoder 0              # SL decoder
python scripts/session14_encode_latents.py
python scripts/session17/encode_seed_latents.py

# Stage 5 (baselines + DNS metrics + flow descriptors; ~6 h)
bash scripts/session18/train_fukami_baselines.sh "3 32 64" 0  &
bash scripts/session18/compute_pod_baselines.sh "16 32 64"    &
python scripts/session17/exp2_dns_physical_metrics.py
python scripts/session16/exp2_build_targets.py
wait

# Stage 6 (Session 16; ~half day)
bash scripts/session16/run_all.sh   # or per-experiment per the table above

# Stage 7 (Session 17; ~half day)
bash scripts/session17/run_all.sh

# Stage 8 (Session 18 B1; ~9-12 h)
bash scripts/session18/run_b1_full.sh   # implements the Stage 8 chain above

# Stage 9 (paper figure assembly + manuscript copy update)
```

Realistic end-to-end wall time with two RTX 6000 cards parallel and a
~60-case dataset scale: **~3 days of compute**.

## Findings / gotchas worth surfacing for the rerun

1. **Production JEPA checkpoint = `outputs/runs/session12/S12_E_d64/encoder/checkpoint_iter020000.pt`.** It contains both encoder AND predictor + observable_head + wake_observable_head state. Train command in `scripts/session12_launch_direction_e.sh` is the canonical Direction-E recipe.

2. **Sessions 16/17 do not retrain models.** They read pre-extracted latents from Stage 4d. Once latents exist, both sessions rerun without GPU training (except S17 Exp 2 rollouts which use the encoder).

3. **B1 headline JEPA row uses no-output-BN predictor on JEPA latents** (`exp_b1_test3/`). The output-BN bug fix from D129 is critical; `physical_closure_noBN_unified.csv` is what Figure 4 reads.

4. **Fukami AE β = 0.01** (not the paper's 0.05) on our data, selected via L-curve sweep at d=3 (Session 18). Methodology preserved; value re-derived.

5. **DNS physical metrics** at `outputs/session17/exp2/dns_physical_metrics.npz` are reused by S17 Exp 2 AND S18 B1 Part d. Single 15-min script.

6. **Test C is for final reporting only**; do not use for model selection. Per CLAUDE.md "Things to NOT do".

7. **16 run3 cases are currently unused** (`Gust_048-066` excluding ones already in the split). They are available for a Test D extension or for absorbing into training — decision deferred.

8. **Predictor output BatchNorm is removed** (`--no-output-bn` flag in `train_baseline_predictor.py`) for the B1 fairness comparison. JEPA's own internal predictor BN inside the joint training is preserved — only the generic predictor-on-top architecture has it removed.

9. **The transformer predictor recipe for B1 is identical across all 7 baselines** (AdaLN-Zero on (G,D,Y), hidden=384, depth=6, heads=16, dropout=0.1, max_seq_len=32, RoPE on Q/K, AdamW lr=5e-4 wd=0.05, 6000 iters, no output BN). Locked in `SESSION18_B1_PROTOCOL.md`; do not deviate.

10. **Three probe families** (ridge, KRR-RBF, MLP-reg) for the B1 physical-metric step. Linear ridge is the main paper figure (Fig 4); KRR-RBF is supplementary; MLP-reg is reported in the table only for transparency.

## Sanity checks after rerun

Before declaring the rerun complete, verify:

- [ ] `configs/splits/split_v1.json` has the expected `n_cases_total`, train/test_b/test_c counts match the new dataset
- [ ] `outputs/data_pipeline/v1/manifest.json` has new `train_stats.std` reasonable for the new dataset
- [ ] Production JEPA SL decoder Test A SSIM_mean >= 0.50 (D99-equivalent gate)
- [ ] B1 Fukami AE gate passes for each d (Test A SSIM_mean >= 0.60 OR ratio_mean < 2.0; `scripts/session18/verify_fukami_gate.py`)
- [ ] Session 17 Exp 2 Markov closure replicates the qualitative direction (Markov-only matches or beats Full-context at H <= 16 on physical observables)
- [ ] Session 18 Figure 4 + Figure 5 + Supplementary S6 render without errors
- [ ] All D-entries in `HANDOFF.md` that reference checkpoints still point to existing files
