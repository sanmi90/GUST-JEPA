# CHANGELOG.md (JFM rewrite program; structural moves, wording changes, open author decisions)

## Session 39 continued: figure-engineering pass (post-commit 6da0f8e)

- fig:centerpiece SPLIT (D-B, the biggest §4.6 lever): fig_da_centerpiece_v4.py
  refactored into two figures -- fig_da_tracking_v4 (a,b, main §4.6, the phase
  tracking result) and fig_da_calibration_v4 (c ladder, d smoother, e NIS,
  appendix C). Panel code unchanged; only the figure/gridspec setup, two footers
  and the save changed. Paper: fig:centerpiece -> fig:tracking in main; new
  fig:calibration in appendix C; the three panel references redirected.
- fig:relerr (peak-error scaling) -> appendix B (like fig:deploy earlier); the
  §4.6 scale-invariance sentence auto-resolves to it.
- Result: the main-text estimation figure shrinks from a 5-panel centerpiece to a
  2-panel tracking figure; the calibration machinery (ladder/NIS/smoother) is in
  the appendix. Main §4+§5 figure labels now 19 (target ~14).
- REMAINING figure merges: physics (fig:t1_spectra + fig:t3_portraits + fig:atlas
  -> one), forecast (fig:mechanism_hroll + fig:forecast + fig:phasesplit -> two),
  estimation tracking condense (fig:hero + fig:cl_envelope_traces -> one),
  fig:ownstack -> appendix. Gate: main 54pp rc=0, tracer PASS.

## Session 39 continued: the deployment ladder (Carlos: not just observables -- the chain)

- NEW fig:deployment_ladder (scripts/session39/fig_deployment_ladder.py): the money
  exhibit tying Part II together. Rows = the 5 rungs of the deployment chain (decode
  field / read wake / forecast / recover from wall / assimilate), cols = predictive
  (JEPA) / reconstruction (Fukami) / linear (POD), colour = holds/partial/fails, each
  cell a macro-bound number from its own table. Shows the COMPOUNDING separation:
  field decode a wash (0.81/0.72/0.82), then JEPA green all the way down
  (0.75/0.63/0.59/0.84) while POD fails the wake (0.19) and DA (0.25) and Fukami
  collapses at forecast (0.00) and diverges under assimilation. Placed as the §5.1
  synthesis opener with a paragraph: "field fidelity is a wash; the predictive state
  is the only family that holds every rung to deployment -- readable, forecastable,
  wall-recoverable and assimilable at once." Reframes §4.2->4.6 from a list of
  comparisons into one argument with a direction.
- Numbers sourced: field CritJepaImp/CritFukImp/CritPodImp; wake Xwake*; forecast T5
  through-impact h8; wall Wobs*; DA FeCLrtwo*/FeDiv* (Fukami 95% divergence -> "diverges").
- Gate: main 54pp rc=0, tracer PASS.

## Session 39 continued: multi-seed handling + all critical-instant assets in main

- MULTI-SEED (Carlos): the critical-instant metrics split by whether they touch
  the decoder. LATENT-only metrics (tab:obs_critical readability) use 3 encoder
  seeds for JEPA and AE (latents_*_s1/s2 exist at d=32) -> per-cell seed MEAN with
  encoder-seed SD (<= 0.09 in R2, macro \CritSeedSd); Fukami single seed at d=32;
  POD deterministic. DECODE metrics (tab:critical_ssim, fig:critical_ssim_dim, T4)
  are single-seed by necessity (only seed-0 decoders exist at d=32) -- stated in
  the captions. observable_critical.py rewritten to loop seeds (eval_seed()).
  Seed-mean shifts small (JEPA C_L impact 0.62->0.70); story unchanged.
- ALL IN MAIN (Carlos): tab:critical_ssim (decode control), fig:critical_ssim_dim
  (decode vs d control), tab:obs_critical (observable discriminator) all in s4.2.
- Gates: main 54pp rc=0, tracer PASS, macros 950.

## Session 39 continued: observable-at-critical-instants table (the discriminating one)

- NEW tab:obs_critical (scripts/session39/observable_critical.py): held-out
  linear-probe R2 for C_L and E_w at pre-impact / impact / peak-lift (+-2 frames),
  4 families, d=32. This is where the families SEPARATE at the critical moments
  (unlike field SSIM). POD LOSES the lift at the impact instant (R2 = -0.05 vs
  JEPA 0.62) and never reads the wake enstrophy (0.31 through impact vs JEPA
  0.66); the nonlinear supervised states carry both through the transient.
  Windowed sanity: POD E_w 0.21 ~= tab:closure 0.19 (probe validated). Landed in
  s4.2 after tab:critical_ssim with the contrast sentence ("the observables tell
  the story the field pixels hide"). Macros \CritCl*/\CritEw*.
- DEFINITION LOCKED (Carlos): observable readability at critical instants = a
  linear ridge probe fit on the TRAIN impact window, evaluated held-out on test_b
  at pre/impact/peak (+-2 frames), reported as R^2 (baseline = instant mean) AND
  physical error (RMSE, C_L in lift units, E_w in enstrophy units, window std 92),
  with a case-clustered bootstrap CI. Methods sentence added to s3.5. tab:obs_critical
  now shows R^2 (RMSE) per cell. Key: RMSE exposes POD worst at EVERY instant even
  where the peak's high variance flatters its R^2 (peak C_L R^2 0.74 but RMSE 1.80,
  worst in column). Macros \CritClRmse*/\CritEwRmse*.
- DECISION (Carlos): drop the redundant forecast-SSIM critical table (mirrors the
  decode one, POD competitive on field SSIM); keep tab:critical_ssim as the honest
  control; the observable table is the discriminator. forecast_critical_ssim.py +
  data retained for the record.

## Session 39 continued: POD latent/decoder mismatch fixed (Carlos "more d" caught it)

BUG: two POD latents in trackc_latents -- latents_pod (M1 family, ~4x scaled, used
for T5's scale-invariant forecast merit) and latents_pod_d32 (matched to
decoder_pod_d32). critical_ssim.py and t4_forecast_ssim.py paired latents_pod with
decoder_pod_d32 -> wrong-scale decode -> garbage POD SSIM (0.13-0.41). Fixed all
scripts to use pod_d32 (matched). Correct POD decode: 0.82 full / 0.67 near-body at
impact -- competitive with JEPA (0.81/0.63), consistent with the paper's own POD
SSIM~0.775 / best-VRMSE finding.
- CORRECTED STORY (decode floor AND forecast SSIM): field decode is NOT the
  discriminator. JEPA ~= AE ~= POD render the field ~equally (~0.6-0.8), only the
  published-recipe lineage a step behind (~0.45 nb). POD is energy-optimal. The
  families separate on the wake OBSERVABLES (POD readability 0.19) and observable
  forecast merit (T5: JEPA 0.78 vs POD 0.55 through-impact), not on field SSIM.
- FIXES landed: tab:critical_ssim POD row + prose + caption; §4.4 T4 forecast-SSIM
  paragraph rewritten (all competitive, JEPA modestly ahead near-body, field decode
  not the discriminator); multi_d_critical_ssim.py + fig_critical_ssim_dim.pdf
  (decode SSIM vs d: JEPA~=POD, Fukami-lineage lower). T4 re-run + forecast_critical
  re-run with matched POD. NOTE: T5 latent merit uses latents_pod (scale-invariant
  R2), unaffected and correct.
- Gates: main 52pp rc=0, tracer PASS.

## Session 39 continued: critical-instant decode SSIM table (Carlos request)

- NEW tab:critical_ssim (scripts/session39/critical_ssim.py): decode-floor SSIM
  (true latent -> field, no forecast) at the three CRITICAL INSTANTS -- pre-impact
  (frame ~30), impact (frame 40), peak lift (argmax |C_L|) -- instead of the window
  average, full field and near-body band, 4 families, 42 test_b encounters.
  Result (full / near-body at impact): predictive 0.81/0.63, AE-wake 0.81/0.65,
  Fukami-wake 0.72/0.45, POD 0.40/0.13. The nonlinear supervised states hold
  fidelity through the load event; POD at d=32 cannot resolve the near-body
  structures. Landed in s4.2 with a sentence; macros \Crit*/\CritNb* (901 total).

## Session 39 continued: storyline alignment (Carlos's 10-beat narrative)

Carlos restated the storyline (AE nice-structure-but-freq-mixing-chaotic-hard-forecast;
POD forecastable-but-many-coords; JEPA = nonlinear + efficient-vs-POD + predictable-vs-AE;
collapse needs an observable head; fair d=4->32 comparison; physics; central
predictability question pre/post impact latent+SSIM; observability; DA deployment).

- COLLAPSE DIRECTION LOCKED (Carlos confirmed): lift anchors (cell CL PR 13.7 healthy),
  wake-field-observable-alone does NOT (cell CW PR 2.5 collapsed). The clean two-beat
  (health: CL suffices; content: CL alone reads wake ~0.31, needs wake head) is the
  paper's stated finding, matching paper_redesign.md 2.1. Carlos's "CL alone is not
  enough" is true for CONTENT, false for COLLAPSE.
- INTRO sharpened to the exact framing: AE "mix high and low frequencies ... evolve
  irregularly ... predictors diverge"; POD "forecast more accurately ... but pay for it
  in compactness, needing many more coordinates"; central question stated verbatim
  ("at once more compact than the linear basis and more forecastable than a
  reconstruction-trained latent").
- FAIRNESS paragraph consolidated into one visible block in s3.2 (the 5-point protocol:
  byte-matched 80-D heads, shared network class/weight, matched AE-L/W/LW anchors, fixed
  untuned wake descriptor, pre-registered gates) + the POD-has-no-supervision answer.
- COMPUTE TRACK T4 (GPU, scripts/session39/t4_forecast_ssim.py): decoded-forecast SSIM
  by phase, 3 central families, decoders from trackc_decoders, Wang v2p2 range, near-body/
  wake/full masks. Result: JEPA >= AE >> POD in EVERY mask and phase, ordering stable
  across masks (through-impact h8 full 0.78/0.76/0.41; near-body 0.54/0.52/0.14 -> POD
  cannot resolve load-bearing structures at d=32). Landed in s4.4 next to T5, completing
  the "latent AND SSIM predicted" central question. Macros \SsimFc*.
- D-B (partial, cont.): streaming deployment-realism + d=4 filter (fig:deploy) moved to
  appendix B with its prose; s4.6 pointer left. (Fuller ladder/NIS/smoother demotion still
  needs the fig:centerpiece split -> M3 figure pass.)
- Gates: main 52pp rc=0, refs 0, tracer PASS, no em-dashes, macros 877.

## Session 39 (2026-07-11), comparison-led reframe (branch session39-comparison-lead)

Executes paper_redesign.md + lineage_style_notes.md: the estimation-thesis lead
becomes the three-family comparison lead (AE compact-but-irregular, POD
forecastable-but-linear, predictive state the third option). Off frozen
jfm-rewrite-v2; frozen Session 38 state recoverable.

- FRONT MATTER. Title -> D-D option 1 "Predictive latent states for extreme
  vortex gust--airfoil interactions" (drops wall-pressure estimation per D-B;
  options 2/3 flagged CARLOS-DECIDES in main.tex). Abstract rewritten to the
  redesign S1-S8 skeleton (comparison-led tension, 2 numbers, near-body dropped
  per D-A, 235 texcount words). Introduction rebuilt as the 8-paragraph lineage
  funnel (P1-P8) with the CORRECTED collapse direction (scalar lift anchors, the
  field observable does not) and enumerated findings (i)-(iv). "broadband"
  wording deferred to the T1 gate per D-E (clock/divergence evidence only).
- T6.1 BUG FIX (real, not wording). \TcRho* macros bound the EnKF inflation
  factor (~1.0), not PR(z); emit_trackc_parts.py fixed to read final diag/pr
  from run metrics (matching fig_cube_health_v4.py). AeW 1.00 -> 21.94 (healthy),
  collapsed no-L 2.5-4.3, healthy L 13.7-18.3. Pipeline regenerated
  (735 numbers / 835 macros, collision check passed); the three s4_a usages now
  consistent.
- RESULTS restructured into the six redesign beats via
  scripts/session39/restructure_s4.py (byte-exact line-range reassembly;
  token-set diff: zero macros removed, only the 4 VRMSE/SSIM caveat macros
  added): 4.1 a non-collapsed predictive state (cube + carries); 4.2
  representation across dimension (dimension + decode floor + POD VRMSE-vs-SSIM
  caveat, precision fix 2.3 + distributed code); 4.3 the physics the latent
  holds (atlas/DMD + anisotropic subspace + tab:mechanism), placed BEFORE
  forecast per D-G; 4.4 forecastability (s4_c + multi-step rollout para +
  fig:mechanism_hroll); 4.5 what wall pressure observes; 4.6 estimating from the
  wall. sec:res_forecast moved from s4_c to the 4.4 subsection (no duplicate
  label). Physics opener reworded off the estimation-thesis framing.
- DISCUSSION: capacity caveat (reviewer risk R3) added to 5.1 (shared operator
  capacity-selected incl sLSTM; matched transformer diverges at equal budget;
  tuned recovery raises all families, ordering unchanged -> floors not ceilings).
  "Choosing the estimator" ladder subsection flagged D-B demotion-bound.
- CONCLUSIONS: near-body number demoted (D-A) to a pointer; trimmed to the two
  R2 endpoints (redesign <=2 numbers); D310-consistent forecast framing; single
  design principle added.
- D-B PART 1: the "Choosing the estimator" discussion subsection moved to
  appendix_b_sensing.tex (figure-independent); s5 now has the redesign's
  four-subsection shape.
- COMPUTE TRACK T1 (CPU, cached latents; scripts/session39/t1_spectral_flatness.py):
  per-coordinate Welch spectral flatness at d=32. Median flatness Fukami 0.068,
  Pod 0.030, AeWake 0.024, JepaWake 0.012; case-clustered CI on
  (Fukami - JepaWake) = 0.056 [0.048, 0.065] EXCLUDES ZERO. GATE (D-E) PASSED:
  the "broadband" wording is now supported and USED (intro P4 + s4.3), scoped to
  the published-recipe reconstruction (the matched-supervision AE is NOT
  broadband). Macros FlatFukami/FlatPod/FlatAeWake/FlatJepaWake/FlatFukamiDelta*
  (842 macros); fig:t1_spectra added to s4.3.
- COMPUTE TRACK T3 (CPU; scripts/session39/t3_portraits.py): PC1-PC2 latent
  trajectory portraits per family; predictive/linear/matched-supervision trace
  coherent cycles, published-recipe wanders. fig:t3_portraits added to s4.3.
  (Both T1 and T3 figures carry FIGURE-TODO to merge into the physics figure in
  Phase 5, redesign fig 8.)
- COMPUTE TRACK T5 (GPU, background on RTX-6000 card 1;
  scripts/session39/t5_phase_split.py): the shared operator RE-RUN with each
  forecast sample tagged by phase (pre-impact / through-impact / post-impact
  relative to impact frame 40), 5 headline families x 3 seeds, operator/probe/merit
  identical to M1. Preliminary (JepaWake, AeWake): through-impact forecast is
  harder than pre-impact (JepaWake h8 pre 0.78 / through 0.63; AeWake 0.78 / 0.70),
  making the error-accumulation thesis phase-explicit. Integrates into s4.4 on
  completion. T4 (decoded-forecast SSIM) still needs the frozen decoders (Phase-5).
- D-A DONE: fig:cube trimmed to the L and W axes (fig_cube_health_v4.py CELL_ORDER
  -> c0/cw/cl/clw, regenerated); s4_a para condensed; the near-body N cells, the
  CLN increment and fig_cube_deltas moved to appendix A (app:nearbody). CLW is the
  sole predictive representative in the main text.
- T6 item 2 DONE: s4_a now disambiguates the E_w enstrophy probe (0.73) from the
  tab:closure pooled wake-descriptor readability (0.751), so the two close numbers
  do not read as an inconsistency.
- LIFT REPORTING (Carlos, 2026-07-11): the JEPA lift was under-reported. Fixes:
  (a) the per-cube-cell peak-lift 3-seed band is now emitted (\TcPeakRTwo*Sd) and
  the predictive CLW peak-lift is quoted as $0.84 \pm 0.015$ (s4_a) alongside the
  Chang CLN $0.86 \pm 0.003$ (appendix), instead of CLW as a delta-only tie;
  (b) tab:closure gains a matched LIFT READABILITY column (\XclRead*, windowed
  linear_r2 from q1_vec/q1_pool/q1_reference, the SAME probe/window as the wake
  column) for all ten families. Result: JepaWake lift 0.813 vs wake 0.751 (lift
  >= wake), every anchored state reads lift well, RegAE 0.206 the only failure.
  Caption + s4_a body updated; the paper's "lift is universal, wake is the
  discriminator" logic now shown, not asserted. NB the \Xcl* macros (0.586) are
  the shared-operator FORECAST lift (h8), a different quantity, unchanged.
- GATES at this checkpoint: main 50pp rc=0, zero undefined/multiply-defined
  refs, number tracer PASS, no em-dashes, no banned prose language. (50pp is
  temporary: the T1/T3 figures merge into fig 8 and the D-A/D-B figure-coupled
  demotions cut length in Phase 5.)
- REMAINING (Phase 5, figure+GPU): D-A N-cube -> appendix and D-B ladder/NIS/
  smoother/streaming (s4_d + its 5 figures) -> appendix (the length levers,
  figure-coupled); precision fix 2.2 further scoping now backed by T1/T3;
  figures 21->13 with the T1/T3 merges into fig 8; GPU tracks T4 (decoded-forecast
  SSIM) and T5 (phase-split forecast, rex_families_m1 re-run) + T2 (POD decode
  row); remaining T6 audits (item 2 wake 0.73-vs-0.751 + earlier-memo items);
  ledger updates (CLAIM_MAP/PROVENANCE); D302/D305 layout calls.

## Session 36 (2026-07-10/11), numbers-frozen gate

- M5 strong-effect bar dropped (cite-or-drop; % REVIEW-CLAIM at the Gate O
  paragraph); tracer to zero hits.
- Nomenclature migrated to paper/nomenclature.tex; archive split names only
  at the s2.2 definitional site; JEPA confined to s1; leakage-free reduced
  to one definitional sentence.
- tab:closure merit column: suited-operator (Xmerit*, h8) replaced by
  shared-operator (XmeritSh*, pre-registered h16; D310 null branch); caption
  rebuilt horizon-truthful; s4 merit paragraph and s5.5 seed-variance
  passage reworded (% REVIEW-CLAIM).
- Parameter-only floor re-run on v2p2 (M2b), Methods sentences confirmed.

## Session 37 Stage 3 (2026-07-11), structural moves

- Results reassembled into four subsections:
  4.1 Constructing a physically useful state (s4_a construction + s4_b
      decode floor + "What the coefficient state carries" + the physics
      subsubsection: DMD/shedding-clock and atlas paragraphs + fig:atlas);
  4.2 Compression and forecastability (new v4/s4_b2_dimension.tex, split
      out of s4_b: dimension tiers + probe-dilution + fig:dimrace; the
      distributed-code and spatial-trade paragraphs; s4_c forecasting;
      "Why the state stays usable under rollout");
  4.3 What wall pressure observes (retitled "What the wall can see",
      labels preserved, sec:res_wallobs added);
  4.4 Sequential estimation and operating limits (tracking + envelope +
      s4_d estimator ladder).
  All absorbed \subsection headers demoted to \subsubsection; every label
  preserved; sec:res_code alias moved with the distributed-code content
  and two references re-pointed (s4_a, s5).
- Appendices: A (architecture/regularisation/UQ) and B (sensing) unchanged;
  NEW in-paper appendix C = calibration disclosure (was D.3, kept in-paper
  per the memo/D305 recommendation); appendix D dissolved.
- NEW paper/supplementary.tex (JFM class, xr cross-refs to main, S-numbered):
  S1 forecaster ledger (was D.1), S2 failure modes (was D.2), S3
  supplementary figures (was appendix C content: decode gallery fig:recon,
  pooling-cost fig:pooling_cost), S4 suited-operator merit table (NEW,
  documents the superseded protocol; tab:suited_merit from the retained
  Xmerit* macros). In-paper references to moved content replaced by plain
  "supplementary material" mentions (5 sites).
- main.tex section retitles: 2 "Flow configuration, data and endpoints";
  3 "Reduced states, forecasting and wall-pressure estimation".
- Both targets compile: main 49pp rc=0, supplementary 3pp rc=0, zero
  undefined references in either log.

## Open author decisions

- D302 fig:atlas slim placement (Stage 5; memo recommends keep slim in 4.1).
- D305 calibration-audit placement: implemented as in-paper appendix C per
  the recommendation; confirm or move to supplementary at the Stage 5 STOP.
- Carlos-owned: session35 branch merge, DNS Table 1 (tab:dns_pending, 7
  \pending{} rows), Zenodo DOI, license/CRediT/funding.

## Session 37 Stage 4 (2026-07-11, overnight autonomous run)

- Front matter bound (D303/D306): new title in main.tex; abstract, intro
  requirements-and-roadmap paragraph, Concluding remarks replaced by the
  approved drafts, numbers macro-bound; roadmap bound to subsection labels;
  "Concluding remarks" section title.
- Mandatory claim repairs applied (memo catches 1, 2, 3/D304, 4 as
  FIGURE-TODO, 5 via the new tab:envelope RMSE column, 7, 12; 6 and 13
  resolved by the draft binding). All % REVIEW-CLAIM-marked.
- Language table swept to ZERO banned hits (was 85 at Stage 0, 66 after the
  front matter). Divergent criterion (impact R2 below -1) defined once.
- s3.5 sign-convention sentence flagged: inventory convention vs the
  physical-G rule; audit due at the Stage 5 figure pass.
- Word budgets: s1 1235/1300 OK, s6 376/450 OK, abstract 264/250 (macro-token
  counting artifact on the approved draft; Carlos to accept or trim).
- COMPRESSION DELIBERATELY STOPPED, Carlos decision needed: the Methods
  budget (2600; currently 4641) collides with the Session 35 Gupta MC
  completeness contract (numbered equations for every filter in s3.4,
  MC-1..MC-12 mapped in outputs/session35/mc_provenance.md). Proposal on the
  table: move the placement (OSP) subsubsection, the smoother configuration
  detail and eq-free configuration prose to appendix B (~-700 words), keep
  the filter-defining equations in s3.4; the remaining ~-1300 needs either a
  relaxed budget or moving equations to the appendix (undoing MC). Same
  spirit for s4 (-1290: prune multiply-stated claims per CLAIM_MAP), s5
  (-605: three-mechanism reorganisation), s2 (-357). These are
  author-judgment cuts, deferred to the next working session.
