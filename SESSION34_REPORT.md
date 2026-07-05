# Session 34 report (2026-07-04 / 2026-07-05)

Branch `session34-trackc` (off `session33-manuscript-v3`). Two-day campaign:
day 1 executed Carlos's uploaded Track C spec (conditioning / observable-supervision
ablation with a new Chang lift-element head); day 2 ran the post-gates program
(filter and forecast improvements, low-dimension exploitation, phase-resolved data
assimilation, manuscript v4 master plan). HANDOFF entries D253-D261.

## 1. Track C: the conditioning cube (D253-D258)

New machinery, all committed and tested:

- `src/data/lift_element.py`: Chang (1992) force-element potential phi_L solved on
  the 192x96 mid-plane grid with a staircase immersed Neumann BC (residual 6e-13),
  lift direction e_L rotated by alpha = 14 deg, lift-element field
  e = omega_z (-v dphi/dx + u dphi/dy) with the stored-omega sign convention
  verified (stored curlU is du/dy - dv/dx, so OMEGA_STORED_SIGN = -1).
- `src/data/nearbody_observables.py`: 80-D near-body lift-element observable
  (64 sign-preserving patch energies + 16-bin radial spectrum inside a
  delta_n = 0.3c EDT band), byte-for-byte mirroring the wake Mode C observable.
- Precompute over all 450 encounters from raw /u + /curlU; QC gate PASS (median
  lagged correlation of band-integrated e against stored C_L = 0.736 over the
  264 gust train encounters). D254 evidence: |omega|-proxy vs Chang cosine 0.68,
  so the principled target is not interchangeable with the proxy.
- Kit extension `supervision.nearbody_head` (default off; every existing config
  resolves to unchanged active_terms; old checkpoints load strict).

25 new runs (8 JEPA cells x 3 seeds on the pooled d=32 vector-predictor pipeline,
plus spec-exact AE anchors) trained overnight on the two RTX 6000s with a
work-stealing queue; zero failures; the full eval chain (frozen probes, decoded
peak-region C_L R2, phase lag, region SSIM, head closure, per-cell OSP + tuned
filter envelopes) ran unattended. Pre-registered gates (tau_thresh = 0.1 t/c,
case-clustered bootstrap, Holm, PR >= 9.6 guard) were written before any result
was read.

Gate outcomes:

- Q2 MIXED, decisive core: the wake head CANNOT replace the scalar lift head as
  the conditioning anchor. On top of L, W buys wake-geometry SSIM (+0.017) but
  no lift accuracy.
- Q2-alt WEAK-complementary: the Chang head alone cannot replace L either, but
  CLN beats CL on peak-region R2 (+2.09 points, CI [+0.03, +5.59]) and is the
  best cell overall (0.862 +/- 0.003, phase lag 0.019 t/c).
- Q1 STRONG with a caveat: N adds on top of W, but the comparison lives in the
  collapsed-PR regime.
- Central mechanism (D258): under the predictive objective the scalar lift head
  is the load-bearing anti-collapse anchor. Every cell without L (C0, CW, CN,
  CWN) collapses to PR 1.3-5 across all seeds; the reconstruction objective
  anchors without L (AE-W keeps PR 21.3 yet only reaches peak R2 0.471).

Wake-state check: no flagship swap. CLW owns the wake state (E_w probe 0.73 /
0.86 vs CLN 0.33 / 0.63); CLN owns lift readability. Two-model story: CLW is the
wake-bearing flagship, CLN the lift-critical specialist.

## 2. No-lift side arms (constructive nulls)

- Carlos's SIGReg-JEPA-ROM skeleton, adapted only to the data: PR 6.9, peak-region
  C_L R2 0.67 linear but 0.93 MLP. Emergent (unsupervised) lift readability
  survives nonlinearly at kit strength; it is not linearly accessible.
- AeroJEPA-style full-encoder arm (recon-on-predicted, arXiv 2605.05586): the only
  healthy no-lift PREDICTIVE model (PR 14.0) but peak R2 0.18. With a lift head
  added it reaches peak 0.952, above CLN, at the price of PR 3.2 (the lift
  gradient concentrates the code); kit-strength SIGReg on it is the scoped
  follow-up.

## 3. Forecast operator: latent-REX family (TiRex lessons)

DIRECT multi-horizon quantile forecaster (instance-norm + arcsinh, LSTM backbone,
pinball q0.1-0.9, no autoregressive compounding) replaces the transformer rollout
where it wins. Ledger closed against the TiRex-2 paper: adopted (direct
multi-horizon, quantile head, instance normalization, arcsinh), refuted for this
data (oracle covariate conditioning HURTS), rejected (capped-weight robustness,
tanh saturation exploded then weight-cap form still down-weighted gust
transients), deferred (variate mixer, TTA). Backbone bake-off at matched budget:
hand-coded sLSTM (exponential gating, log-domain stabilizer) ties LSTM within
seed noise; tuned LSTM h512 q9 = 0.701 val C_L R2. CLN x rexpred reaches 0.903
peak R2 single-seed at d=32 and 0.893-0.908 across 3 seeds at d=4.

## 4. Data assimilation program

- LAE-EnKF retrofit (arXiv 2603.06752): linear-A + delay-embedded pressure-to-latent
  E_obs with H = I. Zero filter divergences (transformer stack had 4); best at
  impact (+0.135 paired median, 29/42 wins); degrades gracefully with sparser
  observations (0.72 / 0.62 / 0.46 / 0.25 at every 1/2/4/8 frames) where the
  hybrid nonlinear filter collapses already at every-2.
- REX-EnKF: REX median forecast per member with quantile-band state-dependent Q;
  the impact-phase champion. Deployment-clean global Gamma; the test-peeked
  band = 4 variant (0.840) is flagged and NOT used; val-calibrated c* = 1.77
  gives 0.749 protocol-clean. test_a NIS band tuning is the legitimate path back
  toward 0.84 (Session 35 P1 decides the F20 headline).
- RTS fixed-lag smoother (lag 5 = 0.25 t/c): rescues the linear stack to
  best-overall (impact RMSE 0.286, peak error 10.3 percent median vs 11.5
  filtered). EnKS on REX degrades it (honest negative, reported).
- Phase-resolved protocol (user-directed): pre / impact / relax x {R2, RMSE, MAE,
  peak value / timing / percent error} x decoded-field SSIM (near-body, wake,
  full masks). Two corrections the physical units forced: the relax-phase R2
  "failure" is a variance artifact (relax RMSE 0.13-0.18 is the BEST phase), and
  relative peak accuracy is scale-invariant (~12-14 percent at every gust
  intensity).
- Own-stack family comparison (every family gets its own OSP staircase, own
  E_obs, own REX, own decoder): JEPA halves the AE lift error in every phase;
  JEPA exploits sensors (0.58 -> 0.16 impact RMSE from K=2 to K=16) where the AE
  saturates at K=8; JEPA at 20 percent sensor noise still beats the AE clean.

## 5. Low-dimension program

- d=4 seed bands (3 seeds each, peak-region R2): jepa 0.903 +/- 0.032,
  aerojepa+lift 0.910 +/- 0.012 vs fukami 0.796 +/- 0.045. Non-overlapping;
  the predictive families beat the AE hardest exactly where the code is small.
- Probe-dilution control (user-raised, core for Session 35): lift INFORMATION is
  d-invariant (MLP probe 0.88-0.90 at every d in {4, 8, 16, 32}); the d=4
  advantage is LINEAR ACCESSIBILITY. Best-4-coordinate probes at d >= 8 reach
  only 0.55-0.66, so the code is distributed rather than diluted by distractors.
- SSIM is dimension-driven the other way: CLN-rexpred full-frame SSIM rises
  0.734 -> 0.758 -> 0.768 -> 0.781 from d=4 to d=32 (near-body 0.554 -> 0.677).
  Lift saturates by d ~ 4-8; the wake needs d >= 16. This is the two-tier ROM
  argument (compact lift-critical code + wake-bearing flagship).

## 6. DA-vs-dimension grid (POD vs Fukami vs JEPA, closing deliverable)

Every cell own-stack (own OSP K=8 staircase, own E_obs, own latent-REX, own
decode-floor decoder), test_b, phase-resolved. POD coordinates are exact
truncations of the ordered basis; Fukami and JEPA columns use the retrained
d = 4 / 8 / 16 encoders plus the d=32 anchors.

Best assimilating recipe per cell (min impact-phase C_L RMSE over
rex_enkf / linear_lae / eobs); test_b, K=8, every-frame pressure, no noise:

| family | d | recipe | impact RMSE | relax RMSE | peak err % | SSIM nb (imp) | SSIM full (imp) |
|---|---|---|---|---|---|---|---|
| POD | 4 | eobs | 0.596 | 0.237 | 24.6 | 0.383 | 0.769 |
| POD | 8 | rex_enkf | 0.458 | 0.180 | 10.5 | 0.423 | 0.744 |
| POD | 16 | eobs | 0.503 | 0.167 | 16.1 | 0.397 | 0.745 |
| POD | 32 | eobs | 0.346 | 0.174 | 13.6 | 0.438 | 0.776 |
| Fukami AE | 4 | rex_enkf | 1.620 | 1.642 | 165.7 | 0.190 | 0.511 |
| Fukami AE | 8 | eobs | 0.734 | 0.691 | 24.2 | 0.307 | 0.638 |
| Fukami AE | 16 | rex_enkf | 0.180 | 0.168 | 4.9 | 0.367 | 0.676 |
| Fukami AE | 32 | linear_lae | 2.247 | 2.229 | 194.2 | 0.314 | 0.326 |
| JEPA CLW | 4 | eobs | 0.298 | 0.214 | 14.8 | 0.424 | 0.687 |
| JEPA CLW | 8 | eobs | 0.359 | 0.317 | 11.3 | 0.286 | 0.684 |
| JEPA CLW | 16 | eobs | 0.301 | 0.186 | 10.3 | 0.335 | 0.715 |
| JEPA CLW | 32 | eobs | 0.265 | 0.145 | 12.9 | 0.423 | 0.763 |
| JEPA CLN-rex | 4 | eobs | 0.445 | 0.384 | 16.7 | 0.194 | 0.718 |
| JEPA CLN-rex | 32 | eobs | 0.337 | 0.226 | 11.7 | 0.307 | 0.757 |
| kit AE-LW | 32 | eobs | 0.411 | 0.374 | 9.6 | 0.424 | 0.765 |

Readings (single seed per cell except where noted; all under one identical
protocol):

- JEPA CLW is UNIFORM across the dimension axis: impact RMSE 0.27-0.36 and peak
  error 10-15 percent at every d, monotone improving to d=32. Even d=4 (0.298)
  beats POD d=32 (0.346).
- POD is stable but a step worse everywhere (0.35-0.60), the expected linear
  floor.
- The Fukami AE is ERRATIC: catastrophic at d=4 and d=32 (peak error 166 / 194
  percent, assimilation at d=32 makes the open-loop forecast WORSE, 2.25 vs
  1.29), yet its d=16 cell is the single best in the table (0.180, peak error
  4.9 percent). Diagnosis (verified, not a pipeline bug): the linear probes on
  TRUE latents are fine at every d (R2 0.77-0.82), but the pressure-to-latent
  observation map E_obs fails to recover the C_L-relevant latent directions at
  d=4/32 while succeeding at d=16; the failure is insensitive to the E_obs ridge
  strength (alpha 1 to 3000), and the identical protocol succeeds for POD and
  JEPA at every d. The d=32 anchor check on JEPA reproduces the original filter
  numbers to machine precision.
- Honest framing for the paper: pressure-estimatability of the Fukami latent
  space is fragile and unpredictable across the design axis, whereas the
  kit-anchored families (JEPA, and the kit AE-LW) are uniformly estimatable.
  The Fukami d=16 cell is real and must be shown; the claim is robustness of
  the family, not a sweep of every cell. A Fukami d=16 seed band is owed before
  any manuscript claim touches that cell (Session 35 P1).

Assembler: `scripts/session34/assemble_da_grid.py` ->
`outputs/session34/da_dims_grid.json`; pipelines
`scripts/session34/da_dims.py` (JEPA lineage, GPU0) and
`scripts/session34/da_dims2.py` (POD + Fukami, GPU1).

## 7. Manuscript v4 master plan

`SESSION_35_MANUSCRIPT_V4.md`: honest review of the v3 manuscript (4 critical
issues, including the operator confound and the head asymmetry that Track C
resolves), a four-part restructure (A construct with lift-based heads / B
reconstructions / C temporal prediction / D data assimilation), a ~23-figure
inventory with the phase-resolved DA centerpiece, and phases P1-P4 to
submission.

## 8. Lessons (operational)

- bash `A && B & C` backgrounds the whole chain; never mix `&&` with a trailing
  ampersand in launches.
- Background Bash tool tasks die with their wrapper; long jobs need
  nohup + disown.
- `pkill -f <script>` from inside a compound Bash command matches the wrapper's
  own command line and kills the compound itself (exit 144).
- eval_all_v3 ALLOWED_KEYS is a closed schema; use seed_mean / seed_sd / n /
  note for band-like values.

## 9. What Session 35 owes (P1 of the v4 plan)

CLN-rexpred d=32 seeds s1/s2; filter and conditioning-null seed replicates;
two-stage filter wired into envelope_by_gust plus test_a NIS band tuning (decides
the F20 headline); then P2 figures, P3 prose, P4 review-fix. Author-owned as
before: DNS Table 1, Zenodo DOI, license / CRediT / funding.
