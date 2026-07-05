# SESSION 35 — Manuscript v4: honest review of v3 + restructure master plan

Written 2026-07-05 (Session 34, post-Track-C). Basis: two full-manuscript reviews
(front half: abstract/intro/flow/methods; back half: results/discussion/conclusions/
appendices), the Track C verdicts (HANDOFF D253-D259), and the session-34 forecasting
and assimilation program. Target: Journal of Fluid Mechanics, up to ~23 figures.

User-directed narrative for v4 (Carlos, 2026-07-05): establish the new ROM with the
new lift-based heads using AE and JEPA approaches, then compare reconstructions, then
temporal prediction, and last the data assimilation part.

---

## PART 1 — HONEST REVIEW OF THE v3 MANUSCRIPT

### Verdict in one paragraph

The v3 manuscript is scientifically sound, honestly written, and already carries the
estimation thesis with unusual candor (nulls stated plainly, filter framed as
feasibility, seeds and CIs everywhere). Its weaknesses are of three kinds: (i) two
genuine methodological confounds a JFM referee will find (the transformer/U-Net
operator split and the wake-supervision asymmetry relegated to an appendix); (ii)
prose debt from three generations of rewrites (v2.1 spatial-era remnants, forward
references that hide trades, "design rule" language stronger than the ablation
coverage); and (iii) missing quantifications (dimension plateau spread, pooling cost
numbers, pre-registered 0.2 threshold unjustified, DNS table still author-pending).
None of it is fatal; most of it is now FIXABLE WITH DATA THAT ALREADY EXISTS from
Track C and Session 34, which is the central argument for the v4 restructure below.

### Critical (referee-blocking) findings

C1. Operator confound. The forecast-merit comparison runs the transformer on
    JEPA/controls and a residual U-Net on references because the transformer
    diverges on reference latents, undiagnosed. "Same operator for all" is claimed
    but not true. FIX AVAILABLE: the shared latent-REX operator (Session 34) trains
    identically per family and is stable on every latent; JEPA > AE holds under it
    (0.665/0.571 vs 0.516/0.543 seeds), making the ordering operator-robust. v4
    reports BOTH the suited-operator protocol AND the shared-REX protocol.

C2. Head/supervision asymmetry buried. The flagship carries a wake head the AE
    baseline lacks; the ablation lives in an appendix. FIX AVAILABLE: Track C's
    2x2x2 {L, W, N} cube at 3 seeds IS the systematic ablation, with pre-registered
    gates (D255-D257). v4 promotes it from appendix ablation to the construction
    narrative (Results part A), which is exactly the user-directed story.

C3. DNS metadata table still an author-fill checklist (unchanged; author-owned).

C4. Filter root-cause conflation: under-dispersion attributed both to "missing
    noise model" (4.5) and "unlearned null-space gain" (5.2); these are different
    claims (calibration vs model error). FIX PARTIALLY AVAILABLE: the session-34
    filter ladder disambiguates: a state-dependent forecast-derived Q (REX band)
    plus a full-rank latent observation removes the impact-phase divergences
    entirely (0 vs 4 catastrophic), showing the impact-side problem WAS
    calibration; the relax side is model-error (conditional-median contraction).
    v4 states the two regimes separately.

### Major findings

M1. Headline metric is representational closure; forecast closure demoted to
    "shows the latent stays usable". For a deployment paper both must be
    co-headline. FIX: v4 part C (temporal prediction) elevates forecasting with
    the shared-operator protocol and the as-built rollout, resolving also the
    "fitted-merit sleight-of-hand" (4.1 reads as a tie then pivots to native).
M2. Tuned-baseline vs published-flagship provenance unstated (ReLU+GN Fukami is
    tuned; was JEPA tuned?). FIX: one Methods paragraph; the kit freeze (D206)
    is the answer and is already documented in HANDOFF.
M3. Inflation tuned once ("no inflation") on the flagship's validation; D252
    already made it per-method; prose still reads single-policy in places.
M4. Parameter floor asymmetry (params-only at H=16 vs latent-only) framed as a
    bound without the combined variant. NEW ANSWER: the conditioning-null
    (D259: oracle (G,D,Y) covariates HURT the forecast) closes this loop far
    more decisively; v4 reports it.
M5. "Pre-registered strong-effect bar of 0.2" invoked without citing the plan;
    cite the archived analysis plan commit or drop the bar.
M6. Plateau/pooling-cost numbers referenced but not quoted (\PlateauSpread,
    \PfourSsimDeltaJepa); quote them.
M7. Wake-enstrophy filter null is honest but unexplained; v4 can now say more:
    the wall constrains the lift-generating near-body directions (Chang analysis)
    while E_w lives in wake structure the taps see only via delay; connect to the
    K x W trade rather than leaving a loose end.

### Moderate / prose findings (selected)

- Abstract: \VfiltCLGOne..\VfiltCLGFour opaque; "static inverse" undefined; the
  three-family comparison names no architecture.
- Intro: POD-vs-AE tension left hanging; "leakage-free" undefined at first use;
  three contributions at mixed abstraction levels.
- Section 2: sign-convention footnote buried; 120-frame window unjustified;
  case-clustering rationale deferred.
- Section 4.6 forward-references the decodability trade instead of quantifying it.
- Conclusions open stronger than the caveats support; "forward-usable" jargon.
- Appendix B admits the v2.1/v2.2 recovery-ordering split-brain; 4.3 should state
  the revalidation explicitly.

### Strengths to preserve (do not regress)

Honest nulls (wake not filterable; merit tie seed-fragile), the division-of-labour
attribution, triple uncertainty protocol, the delay-embedding organizing principle,
the |G|=4 observability-boundary framing, JFM voice of the abstract.

---

## PART 2 — v4 RESTRUCTURE MASTER PLAN

### The new narrative (user-directed) and how it maps

One-sentence thesis (unchanged, sharpened): a reduced gust-encounter state is
useful only if it can be built (supervised), decoded, advanced, and estimated from
wall pressure; we construct such states with lift-based supervision under both
reconstructive and predictive objectives, and show which ingredient buys which
property.

The four results parts:

A. CONSTRUCTING THE STATE: lift-based heads under AE and JEPA objectives
   (new; absorbs v3's 4.1 + 4.2 and the Track C campaign)
   - The head set: scalar lift head L (Fukami lineage), wake-spectrum head W
     (v3's), and the NEW Chang lift-element head N: phi_L auxiliary potential,
     band-restricted force-element density, 80-D sign-preserving observable
     (byte-matched capacity to W). Cite Chang (Proc. R. Soc. A 437, 1992) and
     Menon and Mittal (J. Fluid Mech. 918, 2021).
   - The 2x2x2 {L, W, N} cube x {JEPA, AE} anchors, 3 seeds, pre-registered gates:
     (i) under the predictive objective the scalar lift anchor is load-bearing for
     latent health (all no-L cells collapse, PR 1.3-5 vs floor 9.6); reconstruction
     anchors without it (AE-W PR 21).
     (ii) nothing replaces L; N ADDS to it: CLN beats CL, CLW and CLWN on peak
     lift tracking (0.862 +- 0.003, lag 0.019 t/c); W adds the wake-bearing state
     (E_w 0.73/0.86 vs CLN 0.33/0.63).
     (iii) division of labour v4: L anchors, N sharpens the load, W carries the
     wake; the flagship remains CLW for the estimation thesis, CLN is the
     lift-critical conditioning.
   - Optional (needs seeds): training through the direct predictor
     (predictor-class rex) lifts CLN to 0.903 peak (single seed, D259).

B. WHAT THE STATES RECONSTRUCT (absorbs v3's decode material + 4.3's
   energy-vs-information split)
   - Decode-floor panels (truth / CLW / CLN / AE-LW + error rows) on the
     representative case; region SSIM (nearbody / wake / full) per cell:
     with L present, N buys the same near-body structural fidelity as W
     (0.709 = 0.709); collapsed cells 0.47-0.57; AE parity.
   - Dimension study d in {4, 8, 16, 32}: lift is readable at d=4 in the
     predictive families (seed-banded: jepa 0.903+-0.032 / aero 0.910+-0.012
     vs fukami 0.796+-0.045, non-overlapping) where the wake-bearing state
     needs d >= 16 in EVERY family. Two-tier statement: a d=4 lift-critical
     state is viable; the wake state is not compressible to it.
   - BEST-RECIPE DIMENSION LADDER (lift_dimension_ladder.json): CLN x rexpred
     is near-DIMENSION-INSENSITIVE on peak lift (0.875-0.913 lin / 0.89-0.92 mlp across d = 4/8/16/32; single-seed points carry ~+-0.03) and
     5x more seed-stable at d=4 than the flagship lineage (0.900+-0.006 vs
     0.903+-0.032); the flagship lineage declines with d (0.90 -> 0.84). F12
     carries both lineages + Fukami.
   - CORE (user-flagged, 2026-07-05): THE PROBE-DILUTION CONTROL
     (probe_dilution_test.json). The linear-probe d-comparison conflates
     information with readout format: the MLP probe nearly equalizes all
     dimensions (0.88-0.90 at every d; lift INFORMATION is d-invariant),
     while the best-4-coordinate probe at d >= 8 recovers only 0.55-0.66
     (the code is DISTRIBUTED, not diluted-by-distractors; the same
     phenomenon as the wake distributed code). The d=4 advantage is therefore
     LINEAR ACCESSIBILITY: four coordinates a linear readout, filter probe or
     controller gain uses directly. This is the honest form of the two-tier
     claim and MUST accompany every dimension figure: F12 carries linear AND
     MLP curves plus a best-k-subset panel; prose states "compact and linear
     at d=4, distributed and partly nonlinear above" and never "small models
     know more". Also fixes review item M-class: R2-with-linear-probe is a
     readability metric, not an information metric; say so in Methods 3.
   - Energy-vs-information split (v3 4.3) folds here as "what reconstruction
     optimizes vs what estimation needs".

C. ADVANCING THE STATE: temporal prediction (absorbs v3's 4.2 rollout material,
   elevates forecasting to co-headline; all-new operators)
   - As-built rollouts (kit semantics, H=8-16) + the shared-operator protocol:
     latent-REX (direct multi-horizon quantile forecaster; TiRex lessons,
     arXiv 2607.01204, with the adopted/rejected ledger in an appendix) trained
     identically per family: JEPA > AE at matched heads under BOTH protocols
     (fixes C1); CLN latents the most forecastable (0.69 +- 0.07 across seeds).
   - Direct-vs-autoregressive result: rollout compounding is the failure mode
     (40-step through impact: -0.62 vs +0.70 decoded C_L on identical protocol);
     honest framing of median contraction (conditional-mean forecast, not sharp
     trajectory).
   - The conditioning null: oracle (G, D, Y) covariates DEGRADE the forecast
     (0.492 vs 0.701; overfits held-out parameter combinations at 84-case scale);
     phase covariate mildly helps (deployable). Third leg of the estimation
     thesis: forecast-only fails, parameter-oracle fails, model+pressure works.
   - Backbone note (appendix): LSTM ~ hand-coded sLSTM > mLSTM at this scale,
     3 seeds each; capping rejected (down-weights the gust transients).

D. ESTIMATING THE STATE FROM THE WALL: data assimilation (absorbs v3's 4.3.1,
   4.4, 4.5 + the session-34 filter program)
   - PHASE-RESOLVED EVALUATION (user-directed, the section's centerpiece):
     pre-impact / impact / relaxation, on load AND field, in R2 AND physical
     units AND relative (%%-of-peak) terms. Key findings already measured:
     relax R2 pessimism is a variance artifact (relax RMSE 0.13-0.18 is the
     best phase); assimilation buys 5x at impact over open-loop; peak timing
     within one frame; relative accuracy is SCALE-INVARIANT across the
     training envelope (~12-14%% peak error at every |G|); assimilated fields
     decode near the ceiling except the near-body region at impact.
   - PER-FAMILY OWN-STACK COMPARISON (user-directed): what the objective buys
     AT THE ASSIMILATION STAGE, every family with its own OSP taps, obs
     encoder, forecast operator, probe, decoder: JEPA-CLW halves AE-LW's
     assimilated load error in every phase (impact 0.27 vs 0.41, relax 0.15
     vs 0.37); CLN in between (probe readability does not survive
     assimilation); field fidelity family-insensitive. AE nuance reported
     honestly: better relative PEAK VALUE (8.5-9.6%% vs ~12%%), worse trace.
   - SENSOR-BUDGET AND NOISE STUDIES per family (user-directed): K sweep on
     own staircases + tap-noise sweep with induced Gamma inflation.
   - Keep v3's frozen D220 filter + envelope as the base protocol (unchanged
     numbers, unchanged honest nulls: wake not filterable, error doubles with |G|).
   - The filter ladder (new subsection): structural-consistency retrofit
     (LAE-EnKF, arXiv 2603.06752: linear-A filter, ZERO divergences, graceful
     obs-rate degradation) -> latent-encoded taps (E_obs, H=I; full-rank
     observation from delay-embedded taps) -> REX-EnKF (forecast-derived,
     state-dependent Q): impact-phase tracking 0.75 protocol-clean
     (val-calibrated band), ~0.83-0.84 pending test_a NIS band tuning; relax
     phase remains the transformer filter's regime (two-regime structure of the
     encounter: exogenous information at impact, endogenous dynamics in relax).
     [ENGINEERING GATE: two-stage integration inside envelope_by_gust + test_a
     band tuning BEFORE these numbers are citable; else report only the
     protocol-clean 0.749 and the ladder as a diagnostic.]
   - Deployment realism (new subsection): streaming multi-encounter, one
     pressure-only init, no oracle anywhere in the loop: 0.824; sensor-noise
     sweep 0.81/0.72/0.55 at 5/10/20% with principled Gamma inflation;
     assimilation-rate sweep (every-frame pressure is load-bearing for the
     nonlinear filter; the linear filter degrades gracefully). d=4 filter
     (over-determined observation): 0.79 impact with best relax [seed bands
     tonight].
   - Under-dispersion story rewritten per C4: impact-side = calibration
     (fixed by state-dependent Q), relax-side = model error (median
     contraction); the "calibrated noise model" outlook of v3 becomes a
     demonstrated partial fix.

Physics material (v3's 4.6 DMD/atlas/min-d) redistributes: shedding-clock DMD
into part A (what supervision preserves), min-d/plateau into part B (dimension
study), atlas stays discussion or appendix.

### Figure plan (23; R = reuse/adapt, N = new, * = gated on pending runs)

Setup
 F1  R  fig_staging_v2p1: the gust as wake-reorganisation event.
 F2  R  fig_paramspace_v3: cases, splits, envelope.
 F3  N  Architecture + heads: encoder/predictor schematic extended with the
        three heads; subpanel phi_L contours + band + lift-element snapshot
        (from outputs/data_pipeline/v2p2/phi_L.npz + lift_element.py).
 F4  R  fig_eval_protocol (update: heads matrix, shared-REX operator).
Part A (construction)
 F5  N  Conditioning-cube latent health: final PR per cell x 3 seeds vs floor;
        collapse trajectories inset (runs metrics.jsonl).
 F6  N  Peak-lift closure per cell, paired deltas vs CL with case-clustered CIs
        (trackc_lift.json + trackc_gates.json).
 F7  N  Decoded-C_L three-curve + phase-lag panel, representative encounter
        (trackc_lift records; CLN vs CLW vs CL vs collapsed CW).
 F8  N  Task-dependent readability matrix: {C_L peak, E_w, C_D, circulation}
        x cells incl. AE anchors (extends fig_readability_matrix_v3; recompute
        including CLN/CWN/N cells from existing caches).
 F9  N  The two 80-D observables: wake target vs near-body lift-element target
        construction (masks, patches, spectra) on one frame.
Part B (reconstruction)
 F10 N  Decode panels truth/CLW/CLN/AE-LW + error rows, 3 phases (decoders
        saved in outputs/session34/trackc_decoders).
 F11 N  Region SSIM bars (nearbody/wake/full) per cell (trackc_region_ssim.json).
 F12 N* Dimension race: peak-lift and E_w vs d in {4,8,16,32} x 3 families,
        3-seed bands at d=4 (lowd_race.json + tonight's d4 bands).
Part C (prediction)
 F13 N  latent-REX architecture card (instance-norm/arcsinh, LSTM, direct
        multi-horizon quantile head) + kit rex-predictor inset.
 F14 N  40-step forecast comparison: decoded C_L per horizon; REX vs own
        transformers vs persistence (trackc_forecast.json, latent_rex*.json).
 F15 N  Family forecastability under the shared operator (3-seed bars:
        CLW/CLN/AE-LW) + conditioning-null inset (none/phase/oracle GDY).
 F16 N* Training through the direct predictor: CLN-rexpred vs CLN (peak, PR)
        [gated on 3 seeds next session].
 F17 R  fig_mechanism_hroll_v3 (trim to the multi-step-stability panel).
Part D (assimilation)
 F18 R  fig_hero_traces_v3 + representative phase-resolved traces (da_phase
        representative_traces: truth vs analysis C_L values, pre/impact/relax
        shading).
 F19 R  fig_envelope_v3 (unchanged frozen envelope) + relative-error panel:
        peak %% error and impact NRMSE (%% of peak) vs |G| and D -- the
        scale-invariance result (da_relative_errors.json).
 F20 N  PHASE-RESOLVED DA CENTERPIECE (user-directed): per phase (pre/impact/
        relax) x estimator (eobs / linear-LAE / REX-EnKF / open-loop) x
        {C_L R2, RMSE, MAE, peak value+timing error, %%-of-peak} + decoded-field
        SSIM (nearbody/wake/full) with the encoded-truth decode ceiling row
        (da_phase_eval.json). Ladder variant [*gated on test_a band tuning]
        appears as a second panel or appendix. FILTER-VS-SMOOTHER columns
        (da_smoother.json): fixed-lag RTS (lag 5 = 0.25 t/c delay) rescues the
        linear stack to best-overall (impact 0.286 ties REX-EnKF, relax 0.149
        and peak 10.7%% beat it, = full-interval reanalysis); EnKS on the REX
        filter degrades (lag cross-covariance sampling noise, honest negative).
        Deployment rule: online -> nonlinear filter; 0.25 t/c delay budget ->
        closed-form linear smoother.
 F20b N PER-FAMILY OWN-STACK DA COMPARISON (user-directed): JEPA-CLW vs AE-LW
        vs CLN, each with its OWN OSP taps, obs encoder, forecast operator,
        probe and decoder, identical protocol; phase-resolved RMSE + SSIM
        (da_phase_{ae_wake_pool,jepa_pool_ln_s0}.json). Headline: the
        predictive wake-bearing state halves the assimilated load error in
        every phase at family-insensitive field fidelity.
 F20c N SENSOR-BUDGET AND NOISE EFFECT per family (user-directed): impact
        RMSE / peak %% error vs K in {2,4,8,16} (own OSP staircase per family
        and per K) and vs tap noise in {0,5,10,20}%% at K=8 with induced
        Gamma inflation (outputs/session34/da_grid/*.json).
 F21 N  Deployment: streaming vs reset + noise sweep (rex_stream_noise*.json).
 F22 R  fig_t_trade + N inset: assimilation-rate sweep (lae_*obs*.json).
 F23 R  fig_atlas_dmd_v3 (or appendix if length forces).

Count: 10 reuse/adapt + 15 new (F20/F20b/F20c may merge into two composite
figures to stay near 23); 3 gated (*) on pending runs.

### Data audit: citable now / needs runs / excluded

CITABLE NOW (multi-seed or frozen-protocol): Track C cube incl. CLN headline
(3 seeds, pre-registered gates); E_w/C_D/circulation probes per cell; region
SSIM (decode-floor, s0 per cell -- convention matches v3's decode floor);
shared-REX family ordering (2-3 seeds per family); conditioning null (single
training seed x 3 arms but the effect is 0.2 R2-scale; add 2 seeds cheaply);
backbone tie (3 seeds); D220 envelope + all v3 frozen numbers; streaming/noise
deployment (protocol-defined, single filter seed -- add seeds cheaply, CPU).

NEEDS RUNS BEFORE CITABLE:
 1. d=4 seed bands (RUNNING tonight) -> F12.
 2. CLN-rexpred s1/s2 (2 x 40 min GPU) -> F16.
 3. Two-stage filter in envelope_by_gust + test_a NIS band tuning
    (0.5-1 day engineering) -> F20 headline; else 0.749 protocol-clean only.
 4. Filter-seed replicates for REX-EnKF (cheap, seeds of member noise).
 5. Optional: aerojepa_lift d32 seeds if the exploratory 0.952 is to appear
    even as an appendix note (else exclude entirely).

EXCLUDED FROM v4 (exploratory, wrong scope): AeroJEPA no-lift arms and ROM
skeleton results (D259 stays HANDOFF-only or one discussion sentence); TiRex
zero-shot (dropped); LAE obs-rate hybrid variants beyond the ladder summary.

### Section mapping v3 -> v4

 abstract        REWRITE (four-part arc; keep estimation-thesis close).
 s1 intro        REVISE: add construction/conditioning contribution; fix POD
                 tension, leakage-free definition, contribution levels.
 s2 flow/data    LIGHT REVISE (sign convention up front, 120-frame note,
                 case-clustering summary); DNS table author-owned.
 s3 methods      RESTRUCTURE: 3.1 states & heads (add Chang subsection),
                 3.2 objectives (AE/JEPA kit), 3.3 forecast operators
                 (as-built + shared REX; retire the U-Net split or keep as
                 legacy protocol note), 3.4 filter (base + ladder methods),
                 3.5 protocol (unchanged core).
 s4 results     -> parts A-D as above.
 s5 discussion   REWRITE around: division of labour v4 (L anchors, N sharpens,
                 W carries wake), two-regime encounter structure, two-tier
                 dimension statement, deployment envelope; limitations updated
                 (median contraction, relax model error, 3D boundary).
 s6 conclusions  REWRITE (soften opener per review; quote error + R2).
 app A           keep + add: kit cube table, backbone/capping ledger.
 app B           keep + add: obs-rate sweep, E_obs construction.
 app C           keep + add: TiRex-2 lesson ledger, REX details.

### Execution phases (next sessions)

 P0 (tonight, automated): d4 seed bands land -> score F12 data; D260 closeout.
 P1 (0.5 day): gap runs (rexpred seeds, filter seeds, conditioning-null seeds);
    two-stage filter integration + test_a band tuning [decides F20 scope].
 P2 (1 day): figures F3, F5-F16, F20-F22 through figstyle.py; extend the
    trackc numbers part -> macros (eval_all_v3 green required).
 P3 (1-1.5 days): prose: methods restructure, results A-D, abstract/intro/
    discussion/conclusions rewrites; enforce_conventions + compile gates.
 P4 (0.5 day): review-fix sweep against PART 1 list; fresh-eyes
    jfm_project_writing_style pass; freeze.

Honesty gates carried over: no test-selected knobs (band-scale, delay L via
val/test_a only); single-seed numbers never in headlines; Test C reporting-only;
every figure through figstyle; every number through the numbers pipeline.
