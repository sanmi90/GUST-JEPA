# Session 35 P1 pre-registration (written BEFORE reading any P1 result)

Date: 2026-07-06. Author: Claude (Session 35), per SESSION_35.md section 3.
This file is committed before any P1 run is launched. No P1 result has been
read at commit time. Existing Session 34 numbers cited below (0.903, 0.862,
0.180, 0.749, etc.) were read in Session 34 and appear here only to define
the acceptance bands.

Global rules (apply to every task):
- Tuning split: test_a (val) ONLY, defined as the held-out encounters of
  training cases in split v2.2. test_b is one-shot reporting; test_c is
  reporting-only and never used for any selection.
- No knob is adjusted after a test_b or test_c number is seen. One frozen
  run per configuration.
- Every stochastic number lands with its n (seed count) through
  eval_all_v3 ALLOWED_KEYS (seed_mean, seed_sd, n, note).
- Protocols are frozen to the Session 34 conventions: peak-region pooled
  R2 with the frozen linear probe (trackc_lift_eval convention, half_width
  8, persistence floor frames [0, 25)); phase-resolved DA per da_phase_eval;
  DA grid per da_dims2 (own OSP K=8 W=30 staircase, own E_obs delay 10,
  own latent-REX, own decode-floor decoder, test_b, every-frame, no noise).

## T1 CLN-rexpred d=32 seeds s1, s2

Runs: train_canonical, configs/ablation/jepa_pool_ln.yaml,
--predictor-class rex, --d 32, seeds 1 and 2, 10k iters, v2p2 pipeline
(byte-identical launch to the s0 run per its checkpoint args).
Eval: frozen linear probe peak-region pooled R2 on test_b, identical to the
s0 protocol that produced 0.903.

Acceptance: the 3-seed band (s0 = 0.903 plus s1, s2) is reported wherever
0.903 currently appears; band = mean +/- sd, n = 3.
Failure action: if the band MEAN drops below the CLN probe headline
0.862 (the CLN 3-seed mean +/- 0.003), the rexpred result moves to an
appendix note and F16 is dropped from the main text. No retuning, no seed
cherry-picking; the band is what it is.

## T2 Conditioning-null seed replicates

Runs: rex2_cov (arms none / phase / phase_gdy in one invocation), seeds 1
and 2 on the frozen jepa_pool_vec latents; s0 values (none 0.701 /
phase 0.713 / oracle 0.492 decoded C_L R2) already exist.
Acceptance: this is confirmation, not discovery. The refuted-leg claim
(oracle conditioning HURTS) survives if oracle < none in ALL three seeds
and the seed bands of oracle and none do not overlap.
Failure action: if any seed shows oracle >= none, the R12 refuted leg is
downgraded from "hurts" to "does not help" (band language), and the
F15 inset carries the full spread. The negative stays reported either way.

## T3 REX-EnKF filter-seed replicates + streaming/noise seed adds

Runs: rex_filter --tuned (band_scale 1.77, gamma_mode global, obs_mode
eobs, K=8, delay 10, members 64), member-noise seeds 1-4 (s0 = 0.749
exists); rex_stream at noise {0, 0.05, 0.1, 0.2}, seeds 1-2 each
(s0 exists: 0.824 / 0.808 / 0.723 / 0.548).
Acceptance: filter-seed sd is reported as the error bar on every Part D
REX-EnKF number; expected to be small (member noise only). Streaming
noise-sweep bands attach to R19.
Failure action: none possible in the gate sense; whatever the spread is,
it is reported. If the filter-seed sd exceeds 0.05 median-R2 units the
Part D prose must say the filter is seed-sensitive and headline the band,
not the s0 point.

## T4 Fukami d=16 seed band

Runs: train_reference, configs/reference/fukami_wake.yaml, --d 16, seeds
1 and 2, 10k iters (matching the s0 = fukami_wake_d16 launch); then encode
+ own-stack DA re-eval through the IDENTICAL da_dims2 grid protocol (own
OSP staircase, own E_obs, own latent-REX, own decoder, phase-resolved,
test_b, K=8, every-frame, no noise). Best-recipe cell = min impact-phase
C_L RMSE over {rex_enkf, linear_lae, eobs}, same as the grid.
Acceptance (decides FK16-A vs FK16-B, spec section 7):
- FK16-A (cell is real): all 3 seeds give impact RMSE <= 0.36 (the top of
  the JEPA uniform band) AND peak error <= 15 percent. Then the d=16 cell
  is cited as real and reproducible, band macro attached, and the
  fragility claim is restricted to the dimension axis.
- FK16-B (cell does not survive): any seed with impact RMSE > 0.60 (worse
  than the POD floor at its best) or peak error > 25 percent, or a
  bimodal band (max/min ratio > 2 on impact RMSE). Then the single-seed
  value is retained in the table with its band and the fragility reading
  extends to the training seed.
- Intermediate outcomes (band entirely between 0.36 and 0.60, unimodal):
  the cell is reported with its band and the text states it is
  intermediate between the JEPA band and the POD floor; the
  catastrophic-at-neighbouring-d contrast stands unchanged.
Either way the cell is SHOWN (locked decision D5); the branch only decides
the framing sentence.

## T5 Two-stage filter in envelope_by_gust + test_a NIS band tuning

Engineering: integrate the two-stage schedule (E_obs latent-encoded update
inside the impact window, classic envelope observation elsewhere, per the
D259(2) recommendation) into the envelope_by_gust protocol; add the NIS
statistic (nu_t^T S_t^{-1} nu_t / dim) to the filter diagnostics.
Tuning: the band scale (and any two-stage window boundary adjustment) is
selected on test_a ONLY, by NIS-matching toward E[NIS] = 1 through the
impact phase. The candidate grid for the band scale is fixed here, before
any test_a NIS value is seen: {1.0, 1.4, 1.77, 2.5, 3.5, 4.5, 6.0}.
Window boundaries stay at the frozen da_phase definitions unless the
test_a NIS forces a change, in which case the change is recorded in
mc_provenance and applied once.
Freeze rule: after selection on test_a, ONE frozen run over test_b and
test_c; no second look, no post-hoc grid extension.
Acceptance (decides F20-A vs F20-B, spec section 7):
- F20-A: the frozen test_b run improves the impact-phase median C_L R2
  over the protocol-clean 0.749 by at least +0.03 AND has no more
  catastrophic encounters (<= 2 at R2 < -1). The two-stage + calibrated
  band becomes the F20 headline with the full calibration disclosure.
- F20-B: anything else (including engineering not landing this session).
  Report only 0.749 protocol-clean; the ladder appears as a diagnostic;
  the 0.840 test-peeked value stays confined to the calibration
  disclosure appendix.

## T6 P0 d=4 band verification (no new tuning)

Check: lowd_d4_seedband.json (3 seeds x 3 families), the d=4 filter
band artifacts, and eval_all_v3 --check exit 0.
Acceptance: artifacts exist, n recorded, eval_all_v3 green.
Failure action: relaunch the missing runs with the identical Session 34
launch commands before P2. No protocol change permitted.

## T7 aerojepa_lift d32 seeds

SKIPPED by default per locked decision D4. Only Carlos flips this in
writing.
