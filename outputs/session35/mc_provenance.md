# Methods completeness contract provenance (MC-1..MC-12)

Session 35, started 2026-07-06. Rule (SESSION_35.md section 4): every constant
extracted from code or config with file/line provenance; nothing from memory or
from planning documents; missing constants marked [PENDING: author]. Entry
format: `symbol = value | file:line | "quoted code"`.

Status legend: [DONE] extracted and verified; [PARTIAL] some entries pending;
[PROSE] no extraction needed, written directly in Methods from cited decisions.

## MC-1 Problem statement [PROSE]

The state-space pair per estimator family (z^f_{t+1} = F(z^a_t) + Q-noise,
y_t = h(z_t) + eps) is prose assembled from the MC-2/MC-3/MC-5 extractions
below. Time base: dt_tc = 0.05 c/u_inf (cache attr `dt_tc`, set by
`configs/preprocessing.yaml`; every encounter file carries it). Assimilation
window [t_impact - 24, t_impact + 48] frames: PRE = 24, POST = 48 in
scripts/session32/track_b_pilot.py:58-59 ("PRE = 24  # assimilate from
t_imp - PRE"; "POST = 48"). The rex_filter-frame protocols (ladder, Session 34
D260 and the T5 two-stage run) instead run the FULL trajectory with
t_init = DELAY - 1 = 9 (scripts/session34/rex_filter.py:153) and score on the
cache impact window mask; both frames are stated in Methods.

## MC-2 Base stochastic EnKF (frozen D220 filter) [DONE]

- N = 64 | scripts/session32/track_b_pilot.py:376; mode stochastic | :375 and src/estimation/enkf.py:280
- inflation rho = 1.0 frozen | outputs/session32/filter_tuning_frozen.json:5; pilot sweep {1.00, 1.02, 1.05} recorded in the same file. Per-method inflation (fukami 1.05, jepa/pod 1.0, D252) [PENDING: confirm exact source in session33 filter artifacts at P3]
- Q = cov of one-step teacher-forced predictor residuals, n_samples 256, length 32, + 1e-6 I | track_b_pilot.py:143, 167 (D222)
- H = linear ridge z -> p_K, ridge 1.0 | track_b_pilot.py:218, src/estimation/obs_operator.py:276, form `zz @ weight.T + bias` :244
- R = E_obs-side validation residual variance per tap, encounter-grouped 1/5 holdout, floor 1e-6 | track_b_pilot.py:212, obs_operator.py:340-346
- field-free init: windowed (INIT_WINDOW = 12 | track_b_pilot.py:57) StandardScaler -> Nystroem RBF (n_components 300) -> Ridge, sampled with its residual covariance | track_b_pilot.py:233-240, pressure_infer.py:313-318, enkf.py:25-29
- assimilation window PRE = 24 / POST = 48, cadence every_frame | track_b_pilot.py:58-59, :445
- perturbed-obs update: Pxy = Xf^T Yf/(n-1); gain = Pxy Pyy^-1; d_pert = y + N(0,R); innov = d_pert - Hx; analysis = prior + innov gain^T | src/estimation/enkf.py:330-337
- leakage guard: "Only p_K ever enters the innovation" | enkf.py:269; probes NEVER in innovation | obs_operator.py:15-18
- transformer forecast max_context = 32 (max_seq_len), member-own analysis buffers | enkf.py:13-20, :109, :117-118

## MC-3 LAE-EnKF [DONE]

- A = ridge fit of z_{t+1} ~ A z_t, alpha 1.0, singular values projected <= 1 | scripts/session34/lae_enkf_pilot.py:88, :91, :117
- gates: one-step R2 on test_b | :155; free rollout 30 steps from frame 25 | :157-168
- E_obs: causal edge-padded delay embed L = 10 | :97-104, :114; ridge alpha 1.0 | :118; W solve | :187-188
- H = I; R_z = Gamma = cov(train E_obs residuals) + 1e-8 I | :14, :189-190
- Q = cov(A-fit residuals) + 1e-8 I | :92-93; rho default 1.0, multiplicative on deviations | :116, :220-221; N = 64 | :115
- DIVERGENCE (single definition everywhere): NIS > chi2_{0.99}(dof) for >= 5 consecutive frames OR analysis Mahalanobis ratio > 3.0; dof = obs dimension | src/estimation/metrics.py:145-156
- obs-rate protocol: --obs-every m = assimilate every m-th frame, pressure recorded at full rate | lae_enkf_pilot.py:119-120, :222
- cite arXiv 2603.06752 (Tong, Wang & Yan) | lae_enkf_pilot.py docstring

## MC-4 REX forecaster card [DONE]

- instance norm: mean centering + `sd = ctx.std(dim=1, keepdim=True).clamp_min(1e-3)` | scripts/session34/latent_rex.py:59
- arcsinh transform: `x = torch.asinh((ctx - mu) / sd)` | latent_rex.py:60; inverted `torch.sinh(out) * sd + mu` | latent_rex.py:63
- LSTM depth = 2 | latent_rex.py:50; tuned hidden = 512 | outputs/session34/rex_tune.json winner
- train context range 16..30 (`rng.integers(16, 31)`) | rex_tune.py:154; eval CTX = 25 | latent_rex.py:133
- horizon H = 40 | latent_rex.py:82
- quantiles q9 = (0.1, ..., 0.9) | rex_tune.py:47 `Q9 = tuple(np.round(np.arange(0.1, 0.91, 0.1), 1))`
- pinball rho_q(u) = max(q u, (q-1) u): `torch.maximum(q * e, (q - 1) * e).mean()` | latent_rex.py:66-72
- AdamW lr 1e-3, wd 1e-4 | latent_rex.py:102; CosineAnnealingLR | :103; batch 64 | :79; iters 6000 (tune) / 4000 (default) | rex_tune.py:173; grad clip 1.0 | rex_tune.py:163
- winner selection: grid {lstm, xlstm} x {256, 512} x {3, 9}q | rex_tune.py:205-207; val_frac 0.15 held-out cases | :175; metric val decoded C_L R2 | :244; winner lstm h512 q9 | rex_tune.json
- ORIGINAL band c* = 1.7675 selected as 80% one-step 10-90 coverage quantile on validation: `c_star = float(np.quantile(np.concatenate(res_ratio), 0.80))` | rex_tune.py:222-232 (NOT NIS; the T5 NIS selection is the Session 35 addition)
- sLSTM: SLSTMCell (Beck et al. 2024 xLSTM Eq. 12-19), exponential gating, log-domain stabilizer m_t, normalizer n_t: `log_f = -softplus(-f_)` | rex_tune.py:54-92
- TiRex citation arXiv 2607.01204 [PENDING: verify against the actual reference at P3]

## MC-5 REX-EnKF [DONE]

- BAND_TO_SIGMA = 2.5631 (q90-q10 of a Gaussian = 2 x 1.28155 sigma) | scripts/session34/rex_filter.py:43
- state-dependent Q: sigma_t = clip((q90_t - q10_t)/BAND_TO_SIGMA * band_scale, 1e-4, inf), member noise zf = med + N(0,1)*sigma | rex_filter.py:141-142, 160; identical in scripts/session35/two_stage_envelope.py:183
- N = 64 members | rex_filter.py:58; delay m = 10; K = 8 taps; E_obs ridge alpha = 1.0 | two_stage_envelope.py:67, rex_filter.py:62
- Gamma_global = cov(E_obs train residuals) + 1e-8 I | two_stage_envelope.py:140; phase variants imp/rel | rex_filter.py:117-120
- perturbed-obs update: `innov = z_obs[t] + N @ chol(Gam).T - zf; za = zf + innov @ K_g.T` | rex_filter.py:176-177
- NIS (latent obs): nu = z_obs[t] - mean(zf); S = P_f + Gamma; NIS = nu S^-1 nu / d | two_stage_envelope.py:226-227; tap-space variant dof = K | :238-239
- c* provenance: rex_filter_tuned.json band_scale 1.77; rex_tune.json band_c_star 1.7674535751342775
- T5 pre-registered grid {1.0, 1.4, 1.77, 2.5, 3.5, 4.5, 6.0} | two_stage_envelope.py:59; selection argmin |pooled impact NIS - 1| | :321-322
- two-stage schedule: inside impact window (cache window_mask) REX forecast + E_obs update (H=I, Gamma); outside AR-transformer forecast + Q_tf noise + tap-space obs via C_map/R_taps | two_stage_envelope.py:213-244

## MC-6 Fixed-lag RTS smoother [DONE]

- smoother gain C_s = P^a_s A^T (P^f_{s+1})^-1; backward update x_s = x^a_s + C_s (x_{s+1} - x^f_{s+1}) | scripts/session34/da_smoother.py:73-77 (full RTS), :82-84 (fixed-lag)
- lags tested {5, 10} frames; lag 5 = 0.25 t/c | da_smoother.py:12, :195-196; fixed-lag loop e = min(t + lag, T-1) | :79-80
- stack: linear A (spectrally projected), xf = A xa, Pf = A Pa A^T + Q | :65-66, :129
- EnKS-on-REX: lagged cross-covariance C = dPast^T dZ/(N-1), gain Kp = C (P + Gamma)^-1, history buffer L_enks default 10, N = 64 | :215-226, :100, :178
- no explicit inflation in the smoother pass [confirmed absent in code]

## MC-7 Sensor placement [DONE]

- TCSI greedy objective: score = 1.0 G - 0.5 S_preq - 0.5 H_res + 0.5 Eff, with G = n max(0, L_null - L_final), S_preq = max(0, L_null - L_final), H_res = n L_final, Eff = G/(S_preq + 1e-6); ridge on concatenated W-vectors per candidate tap, argmax selection | scripts/session14_tcsi_pilot.py:358, scripts/session32/osp_select.py
- window W = 30 causal impact-centred | scripts/session32/track_o1_recovery.py:482, osp_select.py:122
- TCSI target = PC-1 of the model's pooled latent at t_impact | osp_select.py:138-142
- nesting: staircase KS = (2, 4, 8, 16), greedy warm-start initial=sel so K=16 contains K=8 | osp_select.py:31, :91
- qDEIM shared array: column-pivoted QR of leading-k mode block, taps = sort(piv[:k]) | src/estimation/obs_operator.py:73-74; N_TAPS = 192 | osp_select.py:32

## MC-8 Phase windows and metrics [DONE]

- phase windows (half-open, clamped): lead_in [t_imp - 8, t_imp), impact [t_imp, t_imp + 16), relaxation [t_imp + 16, t_imp + 48); W_IN 8, W_IMP 16, W_RELAX 48 | src/evaluation/windows.py:38-40, :105-107
- R2 pooled convention: 1 - SSE/max(SST, 1e-12) with phase-frame aggregation before division | src/estimation/metrics.py:29-39; RMSE :42-47; MAE :50-56; per-encounter medians in da_phase summaries | scripts/session34/da_phase_eval.py:271
- peak value error = cl_hat[pp] - cl_true[tp] (pp/tp = argmax |C_L| in impact) | da_phase_eval.py:257; relative peak error = 100 |...|/max(|cl_true[tp]|, 1e-6) | :258-259; timing error = (pp - tp) * 0.05 t/c | :260
- peak-region R2: half_width 8 frames = 0.4 t/c, SSE/SST pooled across encounters | src/evaluation/lift_metrics.py:29-85; persistence floor frames [0, 25) held constant (trackc_lift_eval PRE_IMPACT_FRAMES = 25)
- phase lag: normalized cross-correlation, max_lag 20 frames, parabolic sub-frame refinement, dt 0.05; positive = prediction trails truth | lift_metrics.py:88-136
- SSIM: Wang K1 = 0.01, K2 = 0.03, L = 8.487 for split_v2p2 (2 x global p99.9 |target_norm| over val) | configs/ssim_data_range.json:3,7, src/data/omega_pipeline.py:247-285
- SSIM masks: near-body = feathered EDT band clip(1 - dist/0.3c, 0, 1) | src/data/lift_element.py:256-280; wake = (x/c in (0.0, 4.5)) and (|y/c| < 1.25) | src/evaluation/decoder_metrics.py:27-28, :51; full = ones | da_phase_eval.py:152-154.
  DISCREPANCY NOTE: the spec text says "wake window x/c in [0.5, 4], |y/c| <= 1"; the SSIM wake mask in code is (0, 4.5) x 1.25. The [0.5, 4] window may be a DIFFERENT object (e.g. the wake-enstrophy E_w region or the wake-observable ROI bbox). P3 must verify which window each Methods sentence describes and quote the code values.
- NIS: nis_coverage E[NIS] = dof, 95 percent chi2 band, chi2 0.99 upper tail | src/estimation/metrics.py:86-106; divergence = NIS > chi2_{0.99}(dof) for >= 5 consecutive frames OR Mahalanobis ratio > 3.0 | :140-180
- linear-probe R2 is a readability metric, not an information metric (probe-dilution control R7): state once in Methods

## MC-9 Training numerics [DONE]

- encoder LR 1.5e-4, predictor LR 5.0e-4, weight decay 0.05 | configs/_kit.yaml:60-62
- warmup_frac 0.05, grad clip 1.0 | src/training/train_canonical.py:146-147; cosine schedule floor 0.05 | :204-215
- B = 16, T = 32 | _kit.yaml:63, train_canonical.py:130; pooled-generation iters = 10000 (kit default 80000 overridden per run; convergence rule per feedback memory); bf16 autocast | train_canonical.py:523
- SIGReg: lambda_S = 0.02, M = 256 projections, 17 Epps-Pulley knots in [0.2, 4.0] | _kit.yaml:27-28, src/models/sigreg.py:34-39; held ON for the matched AE | _kit.yaml:24
- lift head: Linear(32->64) GELU Linear(64->1), smooth_l1 beta 0.5, weight 1.0, current-frame delta (0,) | src/models/observable_head.py:27-49, _kit.yaml:37, train_canonical.py:235
- wake head: WakeObservableHead LayerNorm-Linear(32->128)-SiLU-Linear(128->128)-SiLU-Linear(128->80), smooth_l1 beta 0.5, weight 1.0 | observable_head.py:101-124, _kit.yaml:42-43
- nearbody head: same WakeObservableHead class at 80-D, weight 1.0 | src/training/canonical_model.py:538-541, _kit.yaml:48-49
- PR monitor cadence: diagnostic_every 1000 | train_canonical.py:157; auto-fallback threshold_iter 20000, PR floor 0.3 d, probe R2 0.7 | src/training/auto_fallback.py:43, 79, 81 (could not fire at 10k iters; D258)
- predictor (kit ResUNet) context_length 2, H_roll 8 | canonical_model.py:509, _kit.yaml:17; vec/rex predictor classes per D250 + rexpred runs (train_canonical.py:188-200)
- AE anchors: same HybridCNNViTEncoder + SpatialLatentFieldDecoder recon MSE, SIGReg ON | canonical_model.py:459-476, 511-513
- Fukami reference: train_reference.py max_iters 10000, enc LR 1.5e-4 / dec LR 5e-4, wd 0.05, warmup 0.05, clip 1.0 | train_reference.py:83-97; ReLU + GroupNorm | src/baselines/fukami_ae.py:55, 73; lift delta (0,) | train_reference.py:145 [NOTE for P3: the tuned-default provenance paragraph (M2) should state the v1-era future-C_L deltas {8,16,24} finding and that the pooled-generation reference uses delta 0; verify which delta the flagship-era comparisons used before writing]
- POD: pipeline-normalised train omega, mean-centred, randomized SVD, lam = S^2/(n-1), coords A/sqrt(lam) truncated | scripts/session34/rom_pod_basis.py:108-112, scripts/session34/da_dims2.py:15-16

## MC-10 Chang head derivation [DONE]

- solver: sparse direct (spsolve) on the 192x96 grid | src/data/lift_element.py:162, :51-52
- staircase immersed Neumann: rhs[k] -= sign * gcomp / h per solid-adjacent face | lift_element.py:145-151; far-field Dirichlet phi_L = 0 | :115-117
- residual linf = 6.386e-13 | lift_element.py:174-178 + nearbody cache _manifest.json "phi_L_residual_linf"
- e_L = (-sin 14deg, cos 14deg) = (-0.24192, 0.97030) | lift_element.py:56-58 (ALPHA_DEG = 14.0)
- OMEGA_STORED_SIGN = -1.0, verified empirically (stored omega positive in the upper-surface boundary layer where standard omega_z is negative) | lift_element.py:67-72; applied in precompute :197-198
- e = omega_z (-v dphi/dx + u dphi/dy) | lift_element.py:230-253
- E_SCALE = 25.0 (|band e| p99 ~= 26) | lift_element.py:61-65
- band = clip(1 - dist/0.3c, 0, 1), EDT of solid+adjacent | lift_element.py:256-283, delta_n 0.3 | nearbody_observables.py:73
- 80-D observable: 64 = 8x4 patches x {relu(+x)^2, relu(-x)^2} log1p adaptive-avg-pooled + 16-bin Hann-windowed rfft2 radial spectrum log1p | nearbody_observables.py:110-131, :135-157, concat :160-164 (byte-matched to wake Mode C)
- QC gate: median |lagged corr| >= 0.4, |lag| <= 25 frames, gust train only (in_train_pool and |G| > 0); achieved 0.7355 | precompute_nearbody_observables.py:94-96, :258-263 + cache manifest
- proxy comparison: per-encounter 64-D patch-block cosine ~0.70-0.78 on the D254 sample (manifest records). DISCREPANCY NOTE: the results register carries "cosine 0.68"; the manuscript must quote the manifest values, not the remembered number
- cite Chang (Proc. R. Soc. Lond. A 437, 1992), Menon & Mittal (J. Fluid Mech. 918, 2021)

## MC-11 Runs and uncertainty accounting [DONE]

- bootstrap: N_BOOT_ENC = 2000, N_BOOT_CASE = 10000 | scripts/session28/stats_lib.py:25-26
- case-clustered resampling: cases resampled with replacement, encounter-mean and case-mean CIs | stats_lib.py:39-81, resample loop :62-66
- Holm-Bonferroni step-down | stats_lib.py:147-158; applied per question family (Q2_D255 cw_vs_cl, Q2alt_D256 cn_vs_cl, Q1_D257 cwn_vs_cw) | scripts/session34/trackc_gates.py:305-317
- tau_thresh = 0.1 t/c | trackc_gates.py:84; PR floor 9.6 = 0.3 d | :85; SEEDS = (0, 1, 2) | :87
- pre-registration citations: trackc_gates.py:1-62 docstring (Track C, written before results); paired_stats_v3.py:4 (HANDOFF D239); outputs/session35/p1_gates.md (Session 35 P1, commit 439d319 BEFORE launch). The M5 fix (cite the archived plan commit for the 0.2 strong-effect bar) [PENDING: locate the exact archived commit hash for the v3-era 0.2 bar at P3, or drop the bar]
- seed accounting for the manuscript: 3 encoder seeds (cube cells, d=4 bands, rexpred band T1, fk16 band T4); filter member-noise seeds s0-s4 (T3); single-seed grid cells disclosed per caption

## MC-12 Reproducibility trail [DONE]

SHA256 hashes at Session 35 P1 launch (commit 6942261, branch
session35-manuscript-v4):

- split manifest v2.2 `configs/splits/split_v2p2.json` =
  6ee8145cc77f40a3fcc54d6ac2f88742d7193db2660c9be0e8125d46c02b73a6
- kit config `configs/_kit.yaml` =
  c40109c04afcee0278346d82a3052e36058762c4bd8e27e2d4437b261a6118a9
- CLN cell config `configs/ablation/jepa_pool_ln.yaml` =
  b658111fffad70a43eedfdedc465bdff6153161c941d8a179312c48032f4abec
- flagship cell config `configs/ablation/jepa_pool.yaml` =
  34fcd57bbb9c91c2d385382c8b318633ff2454b42db485d4b9b1297de928adbf
- Fukami reference config `configs/reference/fukami_wake.yaml` =
  aaf4f55c8e87482b0b382d2a73141f7327c49c67bbe7343651903c6fb8836ace
- omega pipeline manifest `outputs/data_pipeline/v2p2/manifest.json` =
  6d8572a7b1c902369fcccdddbb5d32670b3c3d647655c0bdc69989085983f770

Numbers pipeline: parts JSONs in outputs/session33/numbers_parts/ ->
scripts/session33/eval_all_v3.py (ALLOWED_KEYS closed schema, line 30:
macro/value/fmt/ci_lo/ci_hi/seed_mean/seed_sd/n/split/endpoint/probe/
observable/horizon/run_tags/source/note/unit) -> numbers.json ->
scripts/session33/emit_macros_v3.py -> paper/macros_v3.tex. No hand-typed
numerals; the tracer (P4, scripts/session35/trace_numbers.py) enforces this at
build time.

Data availability additions for the Zenodo package (Session 34/35 artefacts):
outputs/session34/{trackc_gates,trackc_lift,trackc_region_ssim,lowd_race,
lowd_d4_seedband,probe_dilution_test,lift_dimension_ladder,da_phase_eval,
da_relative_errors,da_smoother,da_dims_grid,rex_tune,rex2_cov}.json,
outputs/session34/da_grid/*.json, outputs/session35/{p1_gates.md,
rexpred_d32_band,fk16_seed_band,nis_band_tuning,two_stage_envelope,
rex_filter_tuned_s*,rex_stream_*,rex2_cov_s*}.json. [PENDING: author] real
Zenodo DOI.
