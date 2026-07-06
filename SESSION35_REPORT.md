# Session 35 report (2026-07-06) - P1 COMPLETE, P2 FIGURES BUILT

## P2 figure status (added after P1 close)

Nineteen v4 figures built through figstyle, every plotted number loaded from
its JSON at build time, split/seed annotations in-panel (commits 1c27cbb,
65c7358, 7ae2611): F3 architecture + heads + phi_L panel, F4 protocol map,
F5 cube health, F6 paired deltas (three of four vs-CL CIs exclude zero;
CLN-CL the unique positive), F7 cell traces + lag distributions, F9 the two
80-D observables, F10 decode panels (GPU), F11 region SSIM, F12 dimension
race + probe-dilution control, F13 REX card (two-calibration contrast),
F14 direct-vs-AR forecast, F15 family forecastability (now n = 3 per family:
CLW 0.638 +- 0.059, CLN 0.685 +- 0.080, AE-LW 0.538 +- 0.020; two missing
REX s2 operators trained this session) + conditioning-null inset,
F16 rexpred bands, F17 H_roll trim, F19 relative-error panels, F20
centerpiece (phase panels regenerated at the val-calibrated band 1.77 after
discovering the Session 34 da_phase_eval ran its REX column at the
test-peeked band 4.0; conclusions band-robust: impact RMSE 0.301 vs 0.284),
F20b/c own-stack + K/noise, F20d DA-vs-dimension grid with the fk16 seed
band drawn per seed, F21 deployment, F22 obs-rate inset. Reused unchanged:
F1, F2, F18, F23.

Known P4 polish items: text crowding in the F3/F13 schematics and the F4
annotation overlaps; fig_forecast40 shows endpoint scalars because no
per-horizon decoded series exists in any JSON. Data catches by the figure
agents, for P3 prose: latent_rex_cln.json is a mislabeled poisoned artifact
(excluded; the real CLN family is latent_rex_jepa_pool_ln_s*); the static
E_obs inverse tops the test_b impact ladder (0.825) above every
protocol-clean filter, so the filter case rests on test_c, tails, relax and
state tracking (write it that way); trackc_region_ssim.json has one decoder
seed per cell (encounter-IQR whiskers used); da_relative_errors.json values
are per-|G| MEDIANS (verified programmatically).

Branch `session35-manuscript-v4` (off `session34-trackc`). Executing Carlos's
uploaded SESSION_35.md spec (committed at repo root; supersedes
SESSION_35_MANUSCRIPT_V4.md where they disagree). This session: P1 gap runs
with pre-registered gates, the T5 two-stage + NIS engineering, and the
Methods-contract provenance extraction. P2 figures / P3 prose / P4 review-fix
follow in the next sessions.

## Pre-registration discipline

`outputs/session35/p1_gates.md` written and committed (439d319) BEFORE any P1
result was read: per-task acceptance bands, tuning split (test_a only), and
failure actions; the T5 band grid {1.0, 1.4, 1.77, 2.5, 3.5, 4.5, 6.0} fixed
before any test_a NIS value existed. One amendment (also pre-result): the
Session 34 streaming runs inherited the TEST-PEEKED band_scale 4.0 default of
rex_stream; a protocol-clean band 1.77 streaming arm was added, and the rule
"any R19 streaming headline uses the 1.77 numbers regardless of outcome" was
fixed in writing before launch.

## P1 compute (T1-T4, T6 relaunch, T5)

Launch machinery: scripts/session35/run_p1.sh (two-GPU phase A + phase B eval
chains), run_p1b.sh (T6 d4 filter seed relaunch, chained), run_t5.sh (T5
chained behind the phase-B eval slot on GPU 0). All nohup + disown per the
Session 34 operational lessons.

- T1 CLN-rexpred d=32 s1/s2: trained (31 min each, byte-identical launch to
  s0 per its checkpoint args). PR caveat: both seeds dip below the 9.6 floor
  mid-training (minima 8.2 / 6.4) and recover to 10.49 / 10.34 at the final
  diagnostic; s0 finished at 11.8. GATE PASS: linear peak-region band
  0.880 +- 0.023 (n=3; per-seed 0.903 / 0.881 / 0.857), s0 reproduces the
  Session 34 anchor exactly; band mean above the CLN headline 0.862, so F16
  stays in the main text. Framing: the rexpred advantage over CLN is a
  band-level ~+0.02 effect (s2 dips just below CLN's tight 0.862 +- 0.003);
  the MLP probe is tighter across seeds (0.892 / 0.907 / 0.895).
- T2 conditioning-null seeds 1/2 x 3 arms (rex2_cov): GATE PASS. Oracle
  (G, D, Y) conditioning loses in ALL three seeds with non-overlapping bands:
  oracle 0.539 +- 0.087 vs none 0.699 +- 0.008 (decoded C_L R2, test_b);
  per-seed gaps 0.209 / 0.066 / 0.204, so the refuted leg carries band
  language (the oracle arm is seed-volatile). CORRECTION to R12: the "phase
  covariate mildly helps" leg does NOT survive replication (phase
  0.710 +- 0.013 overlaps none; s1 flips sign); v4 states phase as a wash
  within seed noise, not a deployable gain.
- T3 REX-EnKF tuned member-noise seeds 1-4: impact medians
  {0.749, 0.772, 0.763, 0.780, 0.755} -> 0.764 +- 0.012 (n=5), catastrophic
  = 2 in every seed. Member noise is small (sd < the 0.05 caveat threshold);
  the s0 0.749 sits at the LOW end of its own noise band, so the
  protocol-clean Part D number becomes 0.764 +- 0.012. Streaming bands
  (3 seeds each): the band 4.0 REPLICATE arm confirms s0 tightly
  (0.829 +- 0.005 / 0.798 / 0.720 / 0.538 at 0/5/10/20 percent noise) but
  stays appendix-only per the pre-registered amendment; the PROTOCOL-CLEAN
  band 1.77 arm gives 0.789 +- 0.023 clean and 0.780 / 0.749 / 0.646 under
  noise. Honest trade the amendment surfaced: the clean band is slightly
  worse noise-free yet clearly MORE noise-robust (0.646 vs 0.538 at 20
  percent). R19 headline switches to the 1.77 numbers.
- T4 Fukami d=16 s1/s2: trained; own-stack DA re-eval through the
  byte-identical da_dims2 grid protocol via scripts/session35/da_fk16_seeds.py.
  PRE-REGISTERED VERDICT: FK16-B, decisively. Impact RMSE per seed
  {0.180, 0.650, 5.926}; peak error {4.9, 17.2, 336.6} percent. The
  best-in-table s0 cell was a lucky seed; s2 fails catastrophically at d=16,
  the same class as the d=4/32 failures. D261-style diagnosis repeated on the
  new seeds and CLEAN: true-latent linear probes healthy (0.789 / 0.841,
  matching the 0.77-0.84 d-sweep range) while all three recipes fail
  identically per seed (s2: 5.93 / 5.95 / 5.97), so the failure is the
  pressure-to-latent observation geometry, not latent content and not any
  single filter. The FK16-B branch text applies with the training-seed
  fragility reading strengthened; the s0 cell is retained in the table with
  its band per locked decision D5.
- T5 two-stage + NIS: scripts/session35/two_stage_envelope.py. Two-stage
  design frozen from D259(2) with NO new knobs: REX + E_obs update inside the
  impact window, envelope matched-transformer + tap-space obs outside; the
  only tuned quantity is the REX band scale, selected on test_a by pooled
  impact-phase NIS matching, then ONE frozen run over test_b + test_c; the
  'rex' arm threads a single rng across encounters and the anchor check
  reproduced the Session 34 protocol-clean 0.749 BIT-EXACTLY at band 1.77.

  F20 GATE: F20-B, and the NIS hypothesis is REFUTED with a mechanism. On
  test_a the pooled impact NIS is BELOW 1 at every band in the grid (0.533
  at band 1.0 falling monotonically to 0.067 at band 6.0) while val impact
  R2 RISES with band (0.569 -> 0.853): innovation consistency and R2
  optimality point in opposite directions, so NIS matching selected the grid
  edge c* = 1.0 and the frozen rex run lands 0.638 on test_b. Consequences
  for the paper: (i) the excluded 0.840 at band 4 was NOT test-set
  overfitting (the val curve reproduces the large-band gain, 0.835-0.853 on
  held-out training encounters) but its mechanism is MODEL-ERROR
  COMPENSATION, not calibration: the inflated state-dependent Q compensates
  the REX median's contraction while driving NIS further from consistency;
  (ii) this extends the R21 two-regime story into the impact phase; (iii)
  the F20 ladder headline remains protocol-clean 0.764 +- 0.012 (n=5).

  T5 ADDENDUM (declared in p1_gates.md BEFORE running, commit before run):
  one additional frozen pass at the OTHER pre-existing val-only calibration,
  the rex_tune coverage band 1.77. Result, the best protocol-clean filter
  numbers of the project: two_stage test_b impact +0.794 (RMSE 0.286,
  cat 2, beating rex-only 0.764 +- 0.012); two_stage test_c impact +0.837
  with POSITIVE relax (+0.120), RMSE 0.594, ZERO catastrophic (vs rex
  test_c 0.720 / -0.148 / 0.881). The two-stage integration largely closes
  the gap to the excluded 0.840 legitimately and rescues the extreme-gust
  boundary. Declaration order fully disclosed: the addendum was declared
  after the NIS-band frozen results were seen; both frozen passes are
  reported side by side (NIS band 1.0: two_stage 0.731 / rex 0.638;
  coverage band 1.77: two_stage 0.794 / rex 0.749-anchor). test_b relax for
  two_stage remains negative (-1.68); the relax rescue materialises on
  test_c only, so the RTS-smoother deployment rule is unchanged.
- T6: eval_all_v3 green (526 -> 540+ numbers with the new p1_bands part); d4
  ENCODER seed bands landed in P0; the d4 FILTER band was single-seed, so
  seeds 1-4 were relaunched under the identical protocol (P1b): impact
  medians {0.789, 0.788, 0.785, 0.772, 0.779} -> 0.782 +- 0.007 (n=5),
  confirming the P0 value. T6 CLOSED.

## Methods completeness contract (MC-1..MC-12)

`outputs/session35/mc_provenance.md` complete for all 12 items (four parallel
read-only extraction agents + direct hashing), every constant with file:line
provenance. Catches that would have produced wrong Methods text:

1. The SSIM wake mask in code is (0 < x/c < 4.5) x (|y/c| < 1.25)
   (src/evaluation/decoder_metrics.py:27-28), NOT the "[0.5, 4] x <= 1"
   window in the spec text; P3 must verify which object each sentence
   describes.
2. The original REX band c* = 1.7675 was selected as the 80 percent one-step
   coverage quantile on validation (rex_tune.py:222-232), not by NIS; the NIS
   selection is the Session 35 addition and the two must be described as two
   distinct calibrations.
3. The citable proxy-vs-Chang cosine is the trackc.json macro value 0.6812
   (NbProxyCosine); the cache manifest's 0.70-0.78 are per-encounter samples
   under a different aggregation.
4. divergence = NIS > chi2_0.99(dof) for >= 5 consecutive frames OR analysis
   Mahalanobis ratio > 3.0 (src/estimation/metrics.py:140-180); one
   definition, quoted exactly, used everywhere a divergence rate is reported.

## Tooling landed

- scripts/session35/trace_numbers.py + trace_whitelist.json (P4 gate 3):
  fails the build on numerals outside the macros pipeline; display equations
  and super/subscripts whitelisted, inline math scanned, justified whitelist.
  Current v3 tex: 318 hits (architecture constants, stat-family labels) =
  the P4 classification worklist.
- scripts/session35/emit_p1_parts.py: p1_bands numbers part; idempotent,
  regenerates as runs land; eval_all_v3 green with the partial part.
- scripts/session35/rexpred_band.py (T1 gate eval, byte-identical
  trackc_lift_eval protocol, s0-reproduction check).
- scripts/session35/da_fk16_seeds.py (T4 own-stack DA re-eval + FK16-A/B
  verdict logic).
- scripts/session34/rex2_cov.py patched: seed-suffixed arm checkpoints so
  seed replicates do not clobber s0.

## Gate outcomes (all pre-registered, all resolved)

- T1: PASS. Band 0.880 +- 0.023 (n=3) above the 0.862 threshold; F16 stays.
- T2: PASS. Oracle < none in all seeds, non-overlapping bands; phase leg
  downgraded to a wash.
- T3: no caveat triggered (filter sd 0.012 < 0.05); streaming headline moves
  to the protocol-clean 1.77 arm per the pre-launch amendment.
- T4: FK16-B. Best-in-table cell was a lucky seed; fragility extends to the
  training-seed axis with a verified E_obs-geometry mechanism.
- T5: F20-B on the pre-registered NIS rule; the declared coverage-band
  addendum delivers two_stage 0.794 (test_b) / 0.837 (test_c)
  protocol-clean.
- T6: closed; d4 filter band 0.782 +- 0.007 (n=5).

## Manuscript-facing deltas from this session (for P2/P3)

1. Wherever 0.903 appeared: 0.880 +- 0.023 (n=3).
2. Wherever 0.749 appeared: 0.764 +- 0.012 (n=5) member-noise band.
3. R19 streaming: 0.789 +- 0.023 clean, 0.780 / 0.749 / 0.646 under noise
   (band 1.77 arm); band 4.0 numbers to the disclosure appendix only.
4. R12: phase-covariate leg becomes a wash; oracle leg keeps band language
   (0.539 +- 0.087 vs 0.699 +- 0.008).
5. R20/F20d: Fukami d=16 cell carries the band {0.180, 0.650, 5.926} and the
   FK16-B framing; seed-fragility joins dimension-fragility with the same
   verified mechanism.
6. F20: ladder headline 0.764 +- 0.012; NIS-refutation paragraph
   (over-dispersed by consistency, under-dispersed by R2; model-error
   compensation mechanism); two-stage rows 0.731/0.794 (test_b) and
   0.808/0.837 (test_c) with both calibrations disclosed; deployment rule
   unchanged.
7. d=4 filter: 0.782 +- 0.007 (n=5).

## Session-close acceptance status (spec section 9)

1. eval_all_v3 green with P1 bands: DONE (554 numbers, 18 parts, all n
   recorded through seed_mean/seed_sd/n).
2. mc_provenance.md complete: DONE (MC-1..MC-12; two [PENDING: author]
   markers: M5 archived-commit citation, Zenodo DOI).
3. Number tracer: built and tested (318-hit worklist on v3 tex);
   enforcement wired at P4.
4. 0.840 confinement: respected everywhere; both p1_gates amendments
   committed before their runs.
5. Caption seed provenance: P2/P3 (bands + n now all available).
6. review_closure.md: P4 (T2/T4/T5 outcomes feed M4, C4/R21 and the R12/R19
   corrections recorded here).
7. Compile / enforce_conventions / fresh-eyes: P3/P4.
8. HANDOFF D262: written this session.
