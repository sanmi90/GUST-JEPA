# REVISION_PLAN.md — structural surgery pass (Carlos's referee-grade assessment)

Status after the scientific-consistency + prose sweeps (commits `d87bc89`..`3c4d8b0`,
all pushed). This plan covers ONLY the structural surgery Carlos asked me to plan
before executing: the section restructure (§9), Methods-to-appendix compression
(§4), figure surgery (§5), and the base-vs-two-stage envelope reconciliation (§1.3).

Current main text: **21 figures, 13 tables, 55 pp.** Target per the assessment:
**~10-12 figures, 5-7 tables.** So ~9 figures and ~7 tables must leave the main text.

---

## Part A — target section structure (assessment §9)

Proposed reorganisation (present order in parentheses):

1. **Introduction** (~1200-1500 w) — done bar a final read.
2. **Flow configuration and data** (present §2) — physical problem, DNS, splits,
   observables. Table 1 DNS rows are Carlos-owned.
3. **Reduced states and estimation framework** (present §3), three subsections only:
   3.1 reduced states and supervision; 3.2 common forward model; 3.3 pressure-only
   estimator. Everything else in present §3 -> appendix (Part C below).
4. **Results**, four subsections:
   4.1 physical content and compression (cube, readability, dimension);
   4.2 forecastability (common forecaster + multi-step mechanism);
   4.3 observability from wall pressure (state recovery + sensor-delay trade);
   4.4 sequential load estimation and operating limits (traces, physical-unit
   error, final two-stage estimator, strong-gust limit).
5. **Discussion** — three mechanisms (supervision / prediction / sensing geometry).
6. **Conclusions** (~300-400 w) — done bar a read.

This is close to the present layout; the real work is moving Methods detail out
and cutting figures, not renumbering sections.

---

## Part B — figure disposition (my proposal; Carlos confirms)

Current main-text figures 1-21. Proposed action per figure. "supp" = supplementary,
"app" = an in-paper appendix.

| # | label | proposed action |
|---|-------|-----------------|
| 1 | fig:staging | **keep**. Regen: add a `t_imp` marker on the lift trace, trim caption ~30%, state the shared baseline phase. |
| 2 | fig:paramspace | **keep**. Rename split labels physically: training / in-distribution test / edge-of-training test / \|G\|=4 extrapolation test. |
| 3 | fig:phi (Chang potential) | **move to app A**; one sentence in Methods. |
| 4 | fig:method | **combine with 5** into one 2-panel workflow (ω→z→ẑ ; p_wall→z^a). |
| 5 | fig:estimation_loop | combined into 4. |
| 6 | fig:cube | **keep, simplify**: PR-by-supervision + paired N-over-L increment; training-history inset -> supp. |
| 7 | fig:dimrace | **keep** as the compression figure (tighten the "not compressible" claim; add replicate seeds at d=4/16/32). |
| 8 | fig:decode | **redesign OR move to supp** (decision B1). Currently 9 rows, pale at page scale. Options in the open-decisions list. |
| 9 | fig:critical_ssim_dim | **move to supp** (it supports tab:critical_ssim; folds under fig 7's compression story). |
| 10 | fig:t1_spectra | **combine with 11** into one physics figure. |
| 11 | fig:atlas | combined into 10 (physics: spectra + latent atlas [+ portraits from app]). |
| 12 | fig:forecast | **keep** the common-forecaster (direct-vs-AR) panel; oracle panel -> supp. |
| 13 | fig:phasesplit | **move to supp** (phase-split forecast is a robustness cut). |
| 14 | fig:mechanism_hroll | **keep** (the multi-step mechanism). |
| 15 | fig:trade | **keep** (sensor-delay trade; claim already softened to "approaches"). |
| 16 | fig:hero | **combine with 17** into one condensed tracking figure. |
| 17 | fig:cl_envelope_traces | combined into 16. |
| 18 | fig:envelope | **combine with 19/relerr** into one operating-envelope figure: physical-unit lift RMSE vs \|G\|, consistency-failure rate vs \|G\|, sign asymmetry inset. R² panel demoted (misleading at high \|G\|). |
| 19 | fig:tracking | keep the phase-tracking panels; fold envelope into 18. |
| 20 | fig:dimsgrid | **keep** as the single distilled robustness figure (impact RMSE vs d for the three families). |
| 21 | fig:predictive_vs_reconstructive | **combine into 14** (both are the predictive mechanism) or **move to app**. |

Net: main-text figures ~21 -> ~11 (1, 2, 4+5, 6, 7, 10+11, 12, 14, 15, 16+17, 18, 20).
fig:decode either redesigned-in-main or in supp per decision B1.

Appendix figures (22-29) stay in the appendices/supp; the atlas slim and the
physics merge pull two of them (portraits) into the merged main physics figure or
leave them in app A.

---

## Part C — Methods -> appendix compression (assessment §4)

Move OUT of main Methods (present §3), into app A / supp:
- full architecture widths, layers, parameter counts (tab:architecture -> app);
- characteristic-function regulariser fallback details;
- TiRex ingredient lineage and the sLSTM alternative (present in §3.3);
- optimiser settings;
- the sensor-selection objective equation (unless we elevate placement as novelty);
- pre-registration history;
- the full estimator-variant catalogue (keep only the two-stage production filter
  + the static inverse anchor in main; ladder detail -> app / supp S1).

Keep in main Methods: flow/dataset, the three state families, predictive objective
+ supervision targets, the common direct forecaster, the pressure observation +
final two-stage estimator, metrics and splits.

Target: the ~15 pp before Results drops toward ~9-10.

---

## Part D — base-vs-two-stage envelope reconciliation (assessment §1.3)

Present state after batch 2: tab:envelope is flagged as the base frozen-filter
diagnostic and points to the two-stage production filter. Remaining structural work:
1. **Base-filter diagnosis** (one figure/table): tracks the mean but becomes
   under-dispersed from ~\|G\|=3 (consistency-failure rate, NIS).
2. **Final two-stage estimator** as THE production system for the performance
   envelope (impact R² 0.794 in-dist / 0.837 boundary; no large-error encounter).
3. **Remaining limitation**: the mid-plane state and wall observations stay
   incomplete at \|G\|=4.
Do NOT mix base-filter R², two-stage trajectories and base-filter failure rates in
one figure/table. This likely means one clearly-labelled base-diagnostic panel and
one two-stage-production panel, not the current interleave.

---

## Part E — open decisions for Carlos

- **B1. fig:decode**: redesign in main (crop to near-body + early wake; DNS +
  predictive + AE + POD only, drop CLN and Fukami rows; absolute-error rows; two
  colour ranges) **or** move the full grid to supp and keep a one-line pointer?
  My lean: **move to supp** (the field decode is a wash and is not the
  discriminator; the near-body detail lives better as a supp exhibit).
- **B2. Compression figure**: keep fig:dimrace (readability-vs-d) as the main
  compression figure and send fig:critical_ssim_dim (decode-SSIM-vs-d) to supp, or
  merge both into one two-panel? My lean: **fig:dimrace in main, SSIM-vs-d to supp.**
- **B3. Physics merge**: fold fig:t3_portraits (app A) into the merged main physics
  figure (10+11), or leave portraits in app A? My lean: **leave portraits in app A**;
  merge only spectra + atlas in main.
- **B4. tab budget**: which of tab:enkf (3), tab:filter_params (4), tab:baselines
  (5), tab:mechanism (9), tab:recovery (10) move to supp? My lean: enkf,
  filter_params, baselines, mechanism -> supp; keep closure (6), critical_ssim (7),
  obs_critical (8), family_filter (11), envelope (12), filter_error (13) + DNS (1)
  and a slim architecture table.
- **B5. Sensor-placement novelty**: is the OSP/greedy sensor-selection objective a
  claimed contribution (stays in main Methods with its equation) or a detail
  (equation -> app)? My lean: **detail -> app.**

Once B1-B5 are set I execute in this order, compiling + committing per group:
(1) Methods compression, (2) figure combines/moves + regens, (3) envelope
reconciliation, (4) table moves, (5) final consistency + gate pass.
