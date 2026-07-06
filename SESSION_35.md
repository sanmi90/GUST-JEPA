# SESSION 35 - Manuscript v4 execution spec (for Claude Code)

Written 2026-07-06. This file operationalizes `SESSION_35_MANUSCRIPT_V4.md` (the v4
master plan written in Session 34) into a runnable session. It adds three things the
master plan does not contain: (1) a METHODS COMPLETENESS CONTRACT benchmarked on
Gupta, Chen & Wan, "Full field reconstruction of turbulent systems from sparse
observations", J. Fluid Mech. 1036 (2026) A24, doi:10.1017/jfm.2026.11611, which is
the level of methods detail the final manuscript must reach; (2) locked decision
defaults so the session does not stall on author calls; (3) pre-drafted contingent
text branches and session-close acceptance gates. Where this file and the v4 plan
disagree, THIS FILE WINS. Inputs assumed in context or repo: SESSION34_REPORT.md,
SESSION_35_MANUSCRIPT_V4.md, main_23.pdf (the v3 manuscript), HANDOFF D253-D261.

Branching: continue from `session34-trackc`; create `session35-manuscript-v4`.

---

## 0. Mission and definition of done

Rewrite the manuscript to the four-part v4 narrative (A construct the state with
lift-based heads, B what the states reconstruct, C advance the state, D estimate the
state from the wall) using the Session 34 results, with Methods at Gupta et al.
(2026) completeness: every estimator specified by numbered equations with every
parameter reported, every stochastic number carrying its run count, and every
failure mode explained mechanistically in an appendix.

DONE means all of:
1. P1 runs landed and banded through `eval_all_v3` (ALLOWED_KEYS: seed_mean,
   seed_sd, n, note).
2. Figures built through `figstyle.py` with Gupta-style captions (parameters, seed
   counts, split named per panel).
3. Prose complete for all sections, `enforce_conventions` and compile gates green.
4. Methods contract MC-1..MC-12 (section 4 below) checked off, each with file/line
   provenance recorded in `outputs/session35/mc_provenance.md`.
5. Both contingent branches (section 7) resolved; the losing branch deleted from
   the tex, not commented out.
6. v4-plan Part 1 findings C1-C4 and M1-M7 each answered by a pointer into the new
   text (`outputs/session35/review_closure.md`).
7. Author-owned placeholders preserved and clearly marked: DNS Table 1 (from
   `paper/dns_metadata.yaml`), Zenodo DOI, license, CRediT, funding. Everything
   else final.

---

## 1. Locked decisions (defaults; only Carlos overrides, in writing)

D1. FLAGSHIP RULE. CLW is the estimation-thesis flagship (wake-bearing state, the
    Part D grid and envelope run on it). CLN is the conditioning result and the
    lift-critical specialist (Part A headline, CLN x rexpred accuracy ceiling in
    Parts B/C). State this division ONCE, in one declarative sentence in the
    Results preamble, and never re-litigate it. A paper that appears to sell two
    models sells none; the rule prevents that reading.

D2. TITLE. The v3 title "Wake-supervised coefficient states..." is superseded by
    the Track C attribution (the lift head, not the wake head, anchors the state).
    AUTHOR DECISION with default (a):
    (a) "Lift-anchored predictive coefficient states for wall-pressure estimation
        of extreme vortex-gust airfoil encounters"  [default]
    (b) "Constructing, advancing and estimating reduced coefficient states for
        extreme vortex-gust encounters from wall pressure"
    (c) Keep v3 structure, replace "Wake-supervised" with "Observable-supervised".
    Do not change the title silently; put the chosen option in the commit message.

D3. F20 HEADLINE POLICY. The Part D calibrated-filter headline is gated on the
    test_a NIS band tuning (P1 task T5). Both prose branches are pre-drafted in
    section 7; Results D prose is written LAST in P3 so it lands after the gate.
    The test-peeked band=4 value 0.840 appears NOWHERE except the calibration
    disclosure appendix, explicitly flagged as excluded.

D4. EXPLORATORY ARMS. The AeroJEPA no-lift arm and the SIGReg-JEPA-ROM skeleton
    stay out of v4 (HANDOFF-only). At most one discussion sentence, WITHOUT the
    0.952 number, framing the recon-on-predicted line as scoped follow-up. Default
    per the v4 data audit: exclude entirely. Do not run the optional aerojepa_lift
    d32 seeds unless Carlos flips this in writing.

D5. FUKAMI d=16. The single best cell of the DA-vs-dimension grid (impact RMSE
    0.180, peak error 4.9 percent) belongs to the baseline. The seed band (P1 task
    T4) is MANDATORY before that cell is citable, and the cell is shown
    prominently either way: the family-robustness claim is credible only if the
    baseline's best case is displayed, not buried. Framing branches in section 7.

D6. WAKE-ENSTROPHY FILTER NULL. Keep the v3 honest null (no family's filter tracks
    E_w) and now EXPLAIN it (v4 review M7): the wall constrains the lift-carrying
    near-body directions (the Chang visibility analysis) while E_w lives in wake
    structure the taps reach only through the delay window; connect explicitly to
    the K x W trade so the null stops being a loose end.

D7. NUMBER PROVENANCE AND REGENERATION. Every number in the manuscript comes from
    a JSON through the macros pipeline; nothing hand-typed from SESSION34_REPORT.md
    or the v4 plan (the two documents already disagree on the RTS peak error, 10.3
    vs 10.7 percent; the JSON `da_smoother.json` decides). Parts A/B/C tables are
    regenerated from the current pooled-pipeline caches. Carried v3 frozen numbers
    (D220 envelope, delay trade, preprocessing robustness) keep an explicit
    generation tag; never mix generations inside one table without a column note.

---

## 2. Canonical results register (cite by key; re-verify every value against its JSON before use)

Part A, construction (Track C, D253-D258; 8 JEPA cells x 3 seeds + spec-exact AE
anchors, pooled d=32 vector-predictor pipeline; pre-registered gates written before
any result was read: tau_thresh = 0.1 t/c, case-clustered bootstrap, Holm, PR >= 9.6
guard):

- R1  Lift anchor. Under the predictive objective every no-L cell (C0, CW, CN,
      CWN) collapses to PR 1.3-5 across all seeds; the reconstruction objective
      anchors without L (AE-W keeps PR 21.3 but peak R2 0.471). Source:
      `trackc_gates.json`, runs `metrics.jsonl`.
- R2  Nothing replaces L; N adds to it. CLN beats CL on peak-region C_L R2 by
      +2.09 points, CI [+0.03, +5.59]; CLN best overall 0.862 +/- 0.003, phase lag
      0.019 t/c. W on top of L buys wake-geometry SSIM (+0.017) but no lift
      accuracy. Source: `trackc_lift.json`, `trackc_gates.json`.
- R3  Division of labour v4: L anchors, N sharpens the load, W carries the wake.
      CLW owns the wake state (E_w probe 0.73 / 0.86 vs CLN 0.33 / 0.63); CLN owns
      lift readability. Two-model reading resolved by D1.
- R4  Chang head machinery: phi_L solved on the 192x96 mid-plane with staircase
      immersed Neumann BC (residual 6e-13), e_L rotated by alpha = 14 deg,
      lift-element field e = omega_z(-v dphi/dx + u dphi/dy), stored-omega sign
      convention verified (curlU stored as du/dy - dv/dx, OMEGA_STORED_SIGN = -1);
      80-D near-body observable = 64 sign-preserving patch energies + 16-bin
      radial spectrum inside a delta_n = 0.3c EDT band, byte-matched to the wake
      Mode C observable; QC gate PASS, median lagged correlation of band-integrated
      e against stored C_L = 0.736 over 264 gust-train encounters; |omega|-proxy vs
      Chang cosine 0.68 (the principled target is not the proxy). Source:
      `src/data/lift_element.py`, `src/data/nearbody_observables.py`,
      `outputs/data_pipeline/v2p2/phi_L.npz`.

Part B, reconstruction and dimension:

- R5  Region SSIM per cell: with L present, N buys the same near-body structural
      fidelity as W (0.709 = 0.709); collapsed cells 0.47-0.57; AE parity. Source:
      `trackc_region_ssim.json`, decoders in `outputs/session34/trackc_decoders`.
- R6  d=4 seed bands (peak-region R2, 3 seeds): jepa 0.903 +/- 0.032, aerojepa+lift
      0.910 +/- 0.012, fukami 0.796 +/- 0.045; non-overlapping. Source:
      `lowd_race.json` plus the P0 d4 band JSONs.
- R7  Probe-dilution control (CORE, user-flagged): lift INFORMATION is d-invariant
      (MLP probe 0.88-0.90 at every d in {4,8,16,32}); best-4-coordinate probes at
      d >= 8 reach only 0.55-0.66; the d=4 advantage is LINEAR ACCESSIBILITY of a
      distributed code, not more information. Must accompany EVERY dimension
      figure; linear-probe R2 is a readability metric, stated as such in Methods.
      Source: `probe_dilution_test.json`.
- R8  Dimension ladder: CLN x rexpred near-dimension-insensitive on peak lift
      (0.875-0.913 linear / 0.89-0.92 MLP across d; single-seed points carry about
      +/- 0.03), and 5x more seed-stable at d=4 than the flagship lineage
      (0.900 +/- 0.006 vs 0.903 +/- 0.032); flagship lineage declines with d
      (0.90 -> 0.84). Source: `lift_dimension_ladder.json`.
- R9  SSIM is dimension-driven the other way: CLN-rexpred full-frame SSIM
      0.734 -> 0.758 -> 0.768 -> 0.781 from d=4 to d=32 (near-body 0.554 -> 0.677).
      Lift saturates by d ~ 4-8; the wake needs d >= 16. Two-tier ROM statement.

Part C, temporal prediction:

- R10 Shared latent-REX protocol resolves the v3 operator confound (C1): trained
      identically per family, stable on every latent; JEPA > AE under BOTH the
      suited-operator and shared-REX protocols (0.665/0.571 vs 0.516/0.543 across
      seeds); CLN latents the most forecastable (0.69 +/- 0.07). Source:
      `latent_rex*.json`, `trackc_forecast.json`.
- R11 Direct vs autoregressive: rollout compounding is the failure mode; 40 steps
      through impact, -0.62 (AR) vs +0.70 (direct) decoded C_L on the identical
      protocol. Median contraction framed honestly (conditional-median forecast,
      not a sharp trajectory).
- R12 TiRex ledger: adopted (direct multi-horizon, quantile head, instance
      normalization, arcsinh); refuted for this data (oracle covariate
      conditioning HURTS: 0.492 vs 0.701, overfits held-out parameter combinations
      at 84-case scale; phase covariate mildly helps and is deployable); rejected
      (capped-weight robustness: tanh saturation exploded, weight-cap form still
      down-weights gust transients); deferred (variate mixer, TTA). Backbone:
      tuned LSTM h512 q9 = 0.701 val C_L R2; hand-coded sLSTM (exponential gating,
      log-domain stabilizer) ties within seed noise; mLSTM behind; 3 seeds each.
- R13 CLN x rexpred: 0.903 peak R2 single seed at d=32 (s1/s2 owed, T1);
      0.893-0.908 across 3 seeds at d=4.

Part D, assimilation:

- R14 Phase-resolved protocol: pre / impact / relax x {R2, RMSE, MAE, peak value,
      peak timing, percent-of-peak error} x decoded-field SSIM (near-body, wake,
      full masks). Two unit-forced corrections that MUST be stated: the relax
      "R2 failure" is a variance artifact (relax RMSE 0.13-0.18 is the BEST
      phase); relative peak accuracy is scale-invariant, ~12-14 percent at every
      gust intensity. Source: `da_phase_eval.json`, `da_relative_errors.json`.
- R15 LAE-EnKF retrofit (arXiv 2603.06752): linear-A dynamics + delay-embedded
      pressure-to-latent E_obs with H = I; ZERO filter divergences (transformer
      stack had 4); best at impact (+0.135 paired median, 29/42 wins); graceful
      degradation with sparser observations, 0.72 / 0.62 / 0.46 / 0.25 at
      every-1/2/4/8 frames, where the hybrid nonlinear filter collapses already at
      every-2. Source: `lae_*obs*.json`.
- R16 REX-EnKF: REX median forecast per member, quantile-band state-dependent Q,
      deployment-clean global Gamma; impact-phase champion; val-calibrated
      c* = 1.77 gives 0.749 protocol-clean; the test-peeked band=4 variant (0.840)
      is flagged and NOT used; test_a NIS band tuning is the legitimate path
      toward ~0.83-0.84 (T5 decides the F20 headline).
- R17 Fixed-lag RTS smoother (lag 5 = 0.25 t/c) rescues the linear stack to
      best-overall (impact RMSE 0.286; relax and peak numbers from
      `da_smoother.json`, see D7); EnKS on the REX filter DEGRADES it (lagged
      cross-covariance sampling noise; honest negative). Deployment rule: online
      -> nonlinear filter; a 0.25 t/c latency budget -> closed-form linear
      smoother.
- R18 Own-stack family comparison (every family with its own OSP taps, own E_obs,
      own REX, own decoder): JEPA-CLW halves AE-LW's assimilated load error in
      every phase (impact 0.27 vs 0.41, relax 0.15 vs 0.37); CLN in between (probe
      readability does not survive assimilation); field fidelity
      family-insensitive; AE nuance reported honestly (better relative PEAK VALUE,
      8.5-9.6 percent vs ~12, worse trace). JEPA exploits sensors (0.58 -> 0.16
      impact RMSE from K=2 to K=16) where the AE saturates at K=8; JEPA at 20
      percent tap noise still beats the AE clean. Source:
      `da_phase_{ae_wake_pool,jepa_pool_ln_s0}.json`,
      `outputs/session34/da_grid/*.json`.
- R19 Deployment realism: streaming multi-encounter, one pressure-only init, no
      oracle anywhere: 0.824; noise sweep 0.81 / 0.72 / 0.55 at 5/10/20 percent
      with principled Gamma inflation; every-frame pressure is load-bearing for
      the nonlinear filter, the linear filter degrades gracefully; d=4 filter
      0.79 impact (verify the P0 seed bands landed). Source:
      `rex_stream_noise*.json`.
- R20 DA-vs-dimension grid (D261, closing study; 15 own-stack cells, test_b, K=8,
      every-frame, no noise): JEPA CLW UNIFORM across d (impact RMSE 0.27-0.36,
      peak 10-15 percent everywhere, monotone improving to d=32; d=4 at 0.298
      beats POD d=32 at 0.346); POD stable linear floor (0.35-0.60); Fukami
      ERRATIC: catastrophic at d=4 and d=32 (peak error 165.7 / 194.2 percent;
      at d=32 assimilation makes the open-loop forecast WORSE, 2.25 vs 1.29) yet
      the single best cell in the table at d=16 (0.180, 4.9 percent). Verified
      diagnosis, not a pipeline bug: linear probes on TRUE latents fine at every d
      (0.77-0.82); E_obs fails to recover the C_L-relevant directions at d=4/32,
      insensitive to ridge alpha from 1 to 3000; identical protocol succeeds for
      POD and JEPA at every d; the d=32 JEPA anchor reproduces the original filter
      numbers to machine precision. Claim: robustness of the kit-anchored families
      (JEPA and kit AE-LW), fragility of the Fukami geometry across the design
      axis; NOT a sweep of every cell. Source:
      `outputs/session34/da_dims_grid.json`, `da_phase_dim_*.json`,
      `scripts/session34/{da_dims.py,da_dims2.py,assemble_da_grid.py}`.
- R21 C4 disambiguation (under-dispersion two-regime story): impact-side
      under-dispersion WAS calibration (state-dependent Q removes the impact
      divergences entirely, 0 vs 4 catastrophic); relax-side is model error
      (conditional-median contraction). v3's "calibrated noise model" outlook
      becomes a demonstrated partial fix. Write the two regimes separately.

Frozen v3 numbers carried with generation tags (D7): D220 envelope and Tables 8-10,
the delay-embedding K x W trade, preprocessing robustness, topology appendix,
parameter floor, DMD shedding clock (recompute for CLW if the pooled generation
changed the checkpoint; verify before carrying).

---

## 3. Phase P1 - runs and engineering (land BEFORE any dependent prose)

Pre-registration first: write `outputs/session35/p1_gates.md` BEFORE reading any P1
result. It must state, per task: the acceptance band, the tuning split (test_a
only), and the action on failure. Commit it before launching.

- T1  CLN-rexpred d=32 seeds s1, s2 (2 x ~40 min GPU). Gate: the 3-seed band is
      reported wherever 0.903 currently appears; if the band mean drops below the
      CLN probe headline (0.862 +/- 0.003) the rexpred result moves to an appendix
      note. Feeds F16.
- T2  Conditioning-null seed replicates: +2 training seeds x 3 arms (none / phase /
      oracle GDY). The effect is 0.2 R2-scale so this is confirmation, not
      discovery. Feeds the F15 inset; enables band language for R12's refuted leg.
- T3  REX-EnKF filter-seed replicates (member-noise seeds; cheap). Error bars for
      the Part D tables; also the streaming/noise deployment seed adds (CPU) from
      the v4 audit.
- T4  Fukami d=16 seed band: 2 additional encoder seeds (~40 min GPU each) +
      encode + own-stack DA re-eval through the identical grid protocol.
      MANDATORY before the F20d d=16 cell is citable (D5). Single seed everywhere
      else in the grid is disclosed in the caption.
- T5  Two-stage filter integrated into `envelope_by_gust` + test_a NIS band tuning
      (0.5-1 day engineering). Freeze rule: the band is selected on test_a ONLY
      (test_a = held-out encounters of training cases; say this in Methods), then
      ONE frozen run over test_b and test_c; no second look. Decides the F20
      headline (section 7 branches). If the engineering does not land, report only
      the 0.749 protocol-clean value and the ladder as a diagnostic (v4 gate).
- T6  Verify the P0 d=4 seed bands and the d=4 filter bands actually landed and
      are green in `eval_all_v3`; if not, relaunch before P2.
- T7  (decision-gated by D4, default SKIP) aerojepa_lift d32 seeds.

Operational discipline (Session 34, section 8; violations cost hours):
- Long jobs: `nohup ... & disown`; background Bash-tool tasks die with their
  wrapper.
- Never mix `&&` chains with a trailing `&`; `A && B & C` backgrounds the whole
  chain.
- Never `pkill -f <script>` from inside a compound command; it matches the
  wrapper's own command line and kills the compound (exit 144).
- Reuse the two-GPU work-stealing queue from Session 34 for T1/T2/T4.
- `eval_all_v3` ALLOWED_KEYS is a closed schema; band-like values go through
  seed_mean / seed_sd / n / note.

---

## 4. Methods completeness contract (Gupta parity), MC-1..MC-12

The benchmark. What makes Gupta, Chen & Wan (JFM 1036, A24) reproducible from the
text alone: (i) the system and the observation model are numbered equations, their
(2.1a,b); (ii) every DA method is written as update equations with every parameter,
their (2.3)-(2.7) for DS and stochastic EnKF including the sample covariance, gain,
perturbed observations and inflation, and (2.8)-(2.10) for 4D-Var including the
Tikhonov step; (iii) the parameters of every experiment are listed per figure
(N_ens and gamma per panel, Table 1 per case); (iv) every stochastic result states
its number of independent runs (100, 50, 10, 5); (v) failure modes get mechanistic
appendices (ensemble collapse, local minima, incremental-vs-standard, alpha
sweeps); (vi) method-selection guidance is an explicit discussion subsection
(their 5.3). The v3 manuscript is protocol-honest but equation-thin: the EnKF
update is never written, Q and R construction is verbal, and REX, LAE and RTS do
not exist in it. The contract below closes that gap. Rules: extract every constant
from code or config, never from memory or from the planning documents; if a
constant is not in the repo, write [PENDING: author] rather than invent; log each
item's source file and line in `outputs/session35/mc_provenance.md`.

Mapping (Gupta element -> our counterpart):
- eqs (2.1a,b) system/observation model      -> MC-1 state-space statement, s3.4
- eqs (2.5)-(2.7) EnKF + inflation           -> MC-2 base filter equations
- eqs (2.8)-(2.10) variational machinery     -> MC-5/MC-6 REX-EnKF Q_t, NIS
                                                calibration, RTS recursion
- Table 1 + per-panel N_ens, gamma           -> MC-2/MC-11 per-filter parameter
                                                table + caption parameter listing
- App A initial-ensemble generation          -> MC-2 field-free initialization
- App B alpha-sensitivity                    -> MC-3 E_obs ridge sweep, MC-5 band
                                                selection protocol
- App C collapse/local-minima mechanisms     -> MC-3/MC-5/MC-6 divergence
                                                definition, Fukami E_obs failure,
                                                EnKS and oracle-covariate negatives
- s5.3 choosing 4D-Var or EnKF               -> Discussion: choosing filter vs
                                                smoother; deployment rule (R17)

MC-1  Problem statement. One numbered pair of equations per estimator family:
      z^f_{t+1} = F(z^a_t) (+ member noise from Q) and y_t = h(z_t) + eps,
      eps ~ N(0, R), with the state, observation, time base (Delta t = 0.05 c/u_inf),
      assimilation window [t_impact - 24, t_impact + 48] frames, and what F, h, Q,
      R are for each filter. Define every symbol at first use.

MC-2  Base stochastic EnKF (the frozen D220 filter). Write the perturbed-
      observation ensemble, the sample forecast covariance, the Kalman gain and
      the analysis update (Gupta (2.5)-(2.6) form); N = 64; Q from the empirical
      covariance of the predictor's one-step residuals on train (state the
      estimator); R from held-out pressure residuals (state which); inflation
      selected per family on validation only, values quoted (1.00 predictive and
      linear, 1.05 reconstruction); field-free initialization from the windowed
      pressure-to-latent regressor and its residual covariance (Gupta App A
      analogue); the leakage guard (probes never in the innovation, unit-tested).
      Extend v3 Table 3 into a per-filter configuration table with one row per
      estimator in the ladder.

MC-3  LAE-EnKF. Three numbered ingredients: (i) the linear transition A fit on the
      frozen latent trajectories (estimator, regularization, fitting split);
      (ii) the observation-side encoder E_obs mapping the delay-embedded pressure
      window p_{t-m+1:t} in R^{K m} to a latent estimate (window m, estimator
      class, ridge alpha and its selection protocol); (iii) the filter run with
      H = I on the encoded observation, R_z from the E_obs residual covariance on
      validation. Cite arXiv 2603.06752. Define DIVERGENCE exactly as the code
      does (extract the criterion; do not paraphrase) and use that single
      definition everywhere a divergence rate is reported. Report the
      observation-rate protocol (every-1/2/4/8 frames) here.

MC-4  REX forecaster card. Per-window instance normalization (formula), the
      arcsinh transform, LSTM depth and width (h = 512), context length, the
      direct multi-horizon head with its horizon set, the nine quantiles
      q in {0.1,...,0.9}, and the pinball loss written out,
      rho_q(u) = max(q u, (q - 1) u); the median (q = 0.5) as the point forecast;
      optimizer, schedule, batch, epochs, seeds. Cite TiRex (verify the arXiv id
      2607.01204 against the actual reference) and the sLSTM lineage for the
      backbone note. The adopted/refuted/rejected/deferred ledger (R12) goes to an
      appendix, Gupta-App-B style, including the two negatives (tanh saturation,
      weight-cap down-weighting gust transients) and the oracle-covariate result.

MC-5  REX-EnKF. Member propagation by the REX median; the state-dependent process
      noise written as an equation, Q_t built from the per-coordinate quantile
      band width (q0.9 - q0.1) at forecast time scaled by the calibration factor
      (extract the exact formula and the global Gamma from code); the NIS
      statistic defined (nu_t^T S_t^{-1} nu_t / dim, expectation one for a
      consistent filter) and the calibration protocol: c* selected by NIS matching
      on validation only, c* = 1.77; if T5 lands, the two-stage schedule and the
      test_a tuning set described with the freeze rule (one frozen run on
      test_b/test_c). The excluded test-peeked band=4 variant is disclosed in the
      calibration appendix and nowhere else (D3).

MC-6  Fixed-lag RTS smoother. The backward recursion on the linear-A stack with
      the smoother gain C_k = P^a_k A^T (P^f_{k+1})^{-1}, lag L = 5 frames =
      0.25 t/c, and the latency interpretation (reanalysis with a 0.25 t/c delay
      budget). The EnKS-on-REX negative with its mechanism (lagged
      cross-covariance sampling noise) in the failure-mode appendix.

MC-7  Sensor placement. The per-family greedy OSP objective spelled out as an
      equation (extract the criterion the staircase optimizes and the nesting
      property), and the shared target-blind QR-pivot array of the delay-trade
      grid (already in v3 App B); state per figure which placement is used.

MC-8  Phase windows and metrics. Exact frame boundaries of pre / impact /
      relaxation extracted from the da_phase code and frozen as numbers in
      Methods; then formula-level definitions of: per-encounter R2 and the median
      convention, RMSE in C_L units, MAE, peak value error, peak timing error (in
      t/c), relative peak error |Delta C_L,peak| / |C_L,peak,true|; SSIM with
      K1 = 0.01, K2 = 0.03, L = 8.487 and the GEOMETRY of the three masks
      (near-body band definition, wake window Omega_w with x/c in [0.5,4] and
      |y/c| <= 1, full frame); NIS; divergence rate. State once that
      linear-probe R2 is a readability metric, not an information metric (R7).

MC-9  Training numerics for every learned component. Encoder and training
      predictor (complete v3 App A with learning rate, schedule, batch, steps,
      wall-clock, hardware: the two RTX 6000s); the three heads' architectures and
      loss weights; the SIGReg regularizer written as the Epps-Pulley
      characteristic-function statistic over M random projections with M and
      lambda_S = 0.02 stated, the PR monitor cadence and the variance-covariance
      fallback; the AE kit anchors' configuration; the tuned-baseline provenance
      paragraph (ReLU + GroupNorm Fukami is tuned, the kit freeze D206 is the
      symmetric answer for the flagship; fixes v4 review M2); POD construction
      (snapshot set, centering, ordering).

MC-10 Chang head derivation (new Methods subsection under states-and-heads).
      The auxiliary potential BVP: Laplace equation for phi_L exterior to the
      airfoil, Neumann condition on the body tied to the lift direction
      e_L = (-sin alpha, cos alpha) with alpha = 14 deg (take the sign from
      `lift_element.py`, not from memory), far-field decay; the staircase immersed
      discretization on the 192x96 grid, the solver and the residual (6e-13); the
      force-element density e = omega_z(-v dphi_L/dx + u dphi_L/dy) and the
      stored-vorticity sign verification (curlU stored as du/dy - dv/dx,
      OMEGA_STORED_SIGN = -1, and HOW it was verified); the delta_n = 0.3c band by
      Euclidean distance transform; the 80-D observable construction (64
      sign-preserving patch energies, patch grid geometry, 16 radial spectral
      bins) byte-matched to the wake observable; the QC gate (median lagged
      correlation 0.736 of band-integrated e against C_L over the 264 gust-train
      encounters, with the lag search range); the proxy comparison (cosine 0.68).
      Cite Chang (Proc. R. Soc. Lond. A 437, 1992) and Menon & Mittal (J. Fluid
      Mech. 918, 2021).

MC-11 Runs and uncertainty accounting, Gupta-style. Every stochastic number
      carries n: 3 encoder seeds for the cube cells and the d=4 bands; single-seed
      grid cells disclosed in captions and a seed-provenance column or caption
      clause in every table; filter member-noise seeds (T3); the 2000-resample
      case-clustered bootstrap; the Holm families listed WITH a citation to the
      archived pre-registration commit (fixes v4 review M5: cite the plan commit
      for the 0.2 strong-effect bar or drop the bar); one appendix table unifying
      the two pre-registration generations (v3 endpoints + Track C gates).

MC-12 Reproducibility trail. The v2.1 split manifest SHA256, kit config hashes and
      checkpoint identifiers per figure, the numbers pipeline (JSON -> macros, no
      hand-typed numerals), and the data availability statement updated to list
      the Session 34 artefacts that ship in the Zenodo package.

---

## 5. Phase P2 - figures

Adopt the v4 plan's F1-F23 inventory as written, with these deltas:

- Merges: F20 + F20b + F20c compose into at most two composite Part D figures;
  F20d (the DA-vs-dimension grid) stays STANDALONE as the closing study, it does
  not fold into F12. Target stays near 23.
- Caption contract (Gupta parity): every caption names the split, the filter and
  its parameters (K, assimilation rate, noise, inflation or c*), and the seed
  count per panel; single-seed cells say so explicitly.
- Gated figures: F12 (needs T6 verification), F16 (T1), F20 ladder panel (T5),
  F20d d=16 cell (T4). Build ungated figures first.
- F19 gains the relative-error panel (peak percent error and impact NRMSE as
  percent of peak vs |G| and D): the scale-invariance result R14.
- F20 carries the filter-vs-smoother columns (R17) and the deployment rule in the
  caption.
- Every figure through `figstyle.py`; every number in a figure regenerated from
  its JSON at build time, never transcribed.

---

## 6. Phase P3 - prose

Order of writing (dependencies force it):
1. Methods 3.1-3.5 rebuilt to the contract of section 4 (3.1 states and heads
   including the Chang subsection MC-10; 3.2 objectives, AE/JEPA kit, SIGReg
   MC-9; 3.3 forecast operators, as-built + shared REX MC-4, retiring the
   transformer/U-Net split to a legacy-protocol note; 3.4 estimator suite MC-1,
   MC-2, MC-3, MC-5, MC-6, MC-7 with the per-filter parameter table; 3.5 protocol,
   metrics and phases MC-8, MC-11).
2. Results A -> B -> C, each opening with the result and closing with the
   mechanism, register keys R1-R13.
3. Results D LAST, after the T5 gate resolves (D3), register keys R14-R21;
   includes the D6 explanation of the wake null and the two-regime
   under-dispersion story R21.
4. Abstract, intro, discussion, conclusions per the v4 section mapping; the
   discussion gains a "choosing the estimator" subsection (filter vs smoother vs
   static inverse, mirroring Gupta 5.3) and the two-tier dimension statement with
   the probe-dilution qualifier R7.
5. Appendices: architecture + cube table + backbone/capping ledger; sensing +
   obs-rate sweep + E_obs construction; TiRex ledger + REX details + calibration
   disclosure; failure modes (divergence mechanics, Fukami E_obs diagnosis R20,
   EnKS negative, oracle-covariate negative); paired tests; preprocessing
   robustness; topology.

Style contract for the manuscript (distinct from this working doc): JFM voice,
British spelling, flowing prose with NO bullet lists and NO em-dashes anywhere in
the tex, every number a macro, seed counts in table captions, results stated
before interpretation, "single seed" written wherever true, and the v4 honesty
gates carried over verbatim (no test-selected knobs; single-seed numbers never in
headlines; test_c reporting-only). Run `enforce_conventions` and the
`jfm_project_writing_style` fresh-eyes pass at the end of P3, not only in P4.

Abstract skeleton (four-part arc, keep the estimation-thesis close): the encounter
as a state-estimation problem; construct (lift-anchored supervision, the cube, L/N/W
division of labour); reconstruct and compress (two-tier dimension with the linear-
accessibility qualifier); advance (direct quantile forecasting beats rollout
compounding; conditioning null); estimate (phase-resolved filter family, calibrated
Q as a delivered partial fix, robustness across dimension where the reconstruction
geometry is fragile); the honest boundaries (wake not filterable, relax model
error, 3D observability limit).

---

## 7. Contingent text branches (pre-drafted; delete the loser in P4)

F20-A (test_a NIS tuning succeeds, T5): "Calibrating the band scale of the
state-dependent process noise on the validation encounters alone raises the
impact-phase analysis closure to [macro], with the normalized innovation squared
held near unity through impact; the calibration set never touches the test
encounters, and the single frozen run over test_b and test_c is reported without
further adjustment. The impact-side under-dispersion of the base filter was
therefore a calibration deficit, not a model deficit; the relaxation phase remains
governed by the forecast median's contraction, a model-error regime the smoother,
not the noise model, addresses."

F20-B (tuning does not land or fails its gate): "With the band scale calibrated on
the validation split alone the filter reaches [0.749 macro] at impact,
protocol-clean; a wider tuning of the band against the innovation statistics of
the held-out training encounters is identified as the legitimate route toward the
diagnostic ceiling and is reported here only as a diagnostic, since it was not
frozen before the test encounters were seen." (The 0.840 number stays in the
disclosure appendix only.)

FK16-A (Fukami d=16 band confirms the strong cell): "The reconstruction lineage's
best cell is real and reproducible across seeds [band macro], and it is the best
cell in the table; the claim is therefore not that the lineage cannot be
estimated, but that its pressure-estimatability is unpredictable across the design
axis, failing catastrophically at the neighbouring dimensions for a verified
observation-map reason, where the supervised families are uniform."

FK16-B (band collapses or is bimodal): "The d=16 cell does not survive seed
replication [band macro]; the single-seed value is retained in the table with its
band, and the fragility reading strengthens: the lineage's estimatability depends
on the training seed as well as the dimension." (Either way the cell is shown; D5.)

---

## 8. Phase P4 - review-fix, referee anticipation, mechanics

- Sweep the v4 plan Part 1 list (C1-C4, M1-M7, moderate items) and record each
  closure with a pointer in `outputs/session35/review_closure.md`.
- Number tracer: add `scripts/session35/trace_numbers.py` that scans the tex for
  numerals not produced by macros and fails the build on any hit outside
  whitelisted contexts (equation constants, dates, section numbers).
- Referee-anticipation list, each with the planted answer in the text:
  1. Single-seed grid cells -> caption disclosure + inferential language
     restricted to seeded comparisons (MC-11).
  2. Why REX and not one transformer for all -> C1 history stated, both protocols
     reported, ordering operator-robust (R10).
  3. The baseline owns the best grid cell -> shown prominently, seed-banded,
     mechanism verified (R20, D5).
  4. Why CLW is the flagship and not the more accurate CLN -> D1 sentence + the
     wake-state probe numbers (R3).
  5. Calibration audit trail -> MC-5 protocol, test_a defined, excluded variant
     disclosed.
  6. R2 flatters strong gusts -> physical-units and percent-of-peak tables
     everywhere (R14), carried from v3's own caveat.
  7. Wake enstrophy not filterable -> D6 explanation tied to the Chang visibility
     and the delay trade.
  8. Is the lift-anchor claim circular (supervise on lift, win on lift)? -> the
     anchor claim is about LATENT HEALTH (PR collapse without L, R1), not
     accuracy; the accuracy claim is the N-increment over CL under pre-registered
     gates (R2); say this explicitly in Results A.
- Author-owned, do not fabricate: DNS Table 1 from `paper/dns_metadata.yaml`;
  Zenodo DOI; license; CRediT; funding. Leave the v3 [PENDING] markers intact.

---

## 9. Session-close acceptance gates

1. `eval_all_v3` green; all P1 bands present with n recorded.
2. `outputs/session35/mc_provenance.md` complete: MC-1..MC-12 each with source
   file and line for every constant, or a [PENDING: author] entry.
3. No numeral in the tex outside the macros pipeline (tracer passes).
4. 0.840 appears only in the calibration disclosure appendix; test_c never used
   for selection; both section 7 branches resolved, loser deleted.
5. Every table caption states seed provenance; every figure caption states split,
   filter parameters and n.
6. `review_closure.md` maps C1-C4 and M1-M7 to text.
7. Compile clean; `enforce_conventions` clean; fresh-eyes style pass done.
8. HANDOFF entry written (D262+) with what moved, what is pending, and the exact
   state of the two contingent branches.
