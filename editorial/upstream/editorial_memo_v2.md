# Editorial memo v2: main_25.pdf to JFM

Prepared against: the GPT-5.6 editorial verdict, `JFM_rewrite_session.md`, the full manuscript (main_25.pdf, ~29k words, 21 main figures, 13 tables), visual inspection of the flagged figure pages (6, 16, 17-19), and style calibration against the lineage corpus in the project (Fukami & Taira JFM 2025 R4; Fukami, Nakao & Taira JFM 2024; Fukami, Smith & Taira PRF 2025; Smith et al. JFM 2024; Odaka et al. JFM 2026).

## 0. Verdict on the previous verdict

The GPT plan is structurally right and should be kept as the skeleton: one question, four Results subsections, the language de-escalation, the title de-methodising, and the submission blockers. Its four primary claims are the correct four. What follows is (1) corrections where the previous verdict is wrong or reproduces an error of the manuscript, (2) catches its audit missed, and (3) extensions it could not know about (data-split provenance, the shared-operator gap in table 6, nomenclature mechanics, house-style specifics from the lineage corpus).

## 1. Corrections to the GPT verdict

**1.1 Its rewritten abstract reproduces a numerical error.** "A two-stage forecast-analysis scheme gives impact-phase R2 = 0.794 in distribution and R2 = 0.837 on the held-out |G| = 4 cases, without diagnosed divergences." Figure 16(c) reports the two-stage filter at 0.794 (2/42) on test_b and 0.837 (0/40) on test_c. There are two flagged encounters in distribution. "No divergences" is true only at the boundary set. Either scope the clause ("with no divergences at the boundary set") or, better, drop divergence counts from the abstract entirely; my draft abstract does the latter.

**1.2 "knob-free" is not just tone, it is inaccurate.** The GPT rule replaces "knob-free" with "without additional tuning". But c = 1.77 is a tuned scalar, calibrated on validation coverage (section 3.3.3). The correct replacement is "with a single validation-calibrated noise scale and no test-set tuning". The virtue being claimed is protocol integrity, not the absence of a parameter.

**1.3 The GPT abstract drops the paper's central attribution.** The load-bearing intellectual result (the C2 attribution) is that observable supervision supplies latent readability and the protected anisotropic geometry, while the multi-step predictive objective supplies rollout stability, not instantaneous readability. The GPT abstract states what each head does but not what the predictive objective does, which leaves the reader to assume the predictive training explains the readability, exactly the misattribution the manuscript is at pains to avoid. Restore one sentence. My draft: "the multi-step predictive objective contributes stability under rollout rather than instantaneous readability."

**1.4 No physical units in the GPT abstract.** The rewrite plan's own acceptance checklist requires physical-unit errors alongside variance-normalised scores. Add the RMSE: "impact-phase R2 = 0.794 (root-mean-square error 0.286 lift units)".

**1.5 The oracle result was dropped.** "Knowing the true gust parameters does not help the forecast" is the single most physical, memorable, referee-friendly result in the forecasting section and directly supports the paper's thesis (the release parameters do not fix the post-impact wake). Keep one scoped clause in the abstract: "supplying the true gust parameters provides no benefit". Do not say "degrades" in the abstract; the degradation is attributed in appendix D.1 to overfitting at the present case count, so the safe abstract-level claim is "no benefit".

**1.6 The proposed introduction ending lacks the organisation sentence and roadmap.** Every full-length JFM paper in the lineage ends the introduction with a section roadmap, and the house style uses enumerated (i)/(ii)/(iii) lists for the requirements (compare Fukami, Nakao & Taira 2024, which enumerates its three-step programme exactly this way). My draft in `front_matter_rewrite.tex` restores both.

**1.7 Figure 21 should not default to supplementary.** The DMD panel (the predictive latent holds the shedding clock at St = 0.666, modulus 0.993, where the Fukami-lineage autoencoders are damped and off-frequency) is the most JFM-native physics in the whole paper. It belongs, slimmed to the DMD spectrum plus at most one atlas panel, in section 4.1 as evidence that the constructed state is physically meaningful, not as an afterthought. The atlas PCA-variance panel and the remaining projections go to supplementary. (Decision stub D302.)

**1.8 Figure 10 should be dropped, not moved.** It restates table 6 as bars. Cut it outright and save the figure budget.

## 2. New catches from the number-and-claim audit

These are concrete errors or fragilities in the current manuscript that neither the GPT verdict nor the session plan lists. Each becomes a `% REVIEW-NUMBER` or `% REVIEW-CLAIM` item in the Claude Code pass.

1. **Section 4.7, false superlative.** "the predictive state alone combines the highest boundary closure (0.84 against 0.66 to 0.85 for the alternatives)". Table 9 gives Fukami (wake) 0.85 at |G| = 4, which exceeds 0.84. The claim as worded is false. The defensible statement is the combination: closure among the highest, by far the lowest divergence rate (0.72 against 0.93 to 0.97), consistency across gust strengths (Fukami-wake posts -0.39 at |G| = 1), and the least-degraded wake readout.
2. **Section 4.8, mismatched glosses.** "the mean normalised innovation squared from near unity in the undisturbed limit (0.6) to nearly twenty at |G| = 4 (15.2)". 0.6 is not near unity and 15.2 is not nearly twenty. State the numbers plainly: "from 0.6 in the undisturbed limit to 15.2 at |G| = 4".
3. **Section 4.8, duplicated value.** "crossing the half-divergence mark near a ratio of 3.0 for the compact cores and near 3.0 for the widest". The same value appears twice for what is framed as a contrast between core diameters. Almost certainly a macro or copy bug. Verify the intended per-D values against the envelope data (stub D304).
4. **Static-inverse median inconsistency.** Section 4.9 prose says 0.83; figure 16(c) says 0.825. Bind both to one macro.
5. **Section 4.8 vs table 11.** Prose quotes median C_L RMSE 0.68 at a gust ratio of three; table 11 has columns |G| = 1, 2, 4 only (0.31, 0.63, 0.72). Either add the |G| = 3 column or quote the table's own values.
6. **Duplicated interval.** The recoverability advantage 0.120 [0.096, 0.145] appears verbatim in section 4.6 and section 5.2. Keep it once, in Results; the Discussion references it without re-quoting.
7. **Split naming drift.** test_a is called "validation" in section 2.2 but "test A" in figure 16(e) ("NIS band calibration (test A, n = 100)"). Fix aliases once: validation (test_a), in-distribution test (test_b), boundary test (test_c); prose and captions use the aliases only, with the archive names defined once in section 2.2.
8. **Sign convention leakage into captions.** Figure 14's column headers carry archive case IDs (G+1.00_D0.50_Y+0.40) whose sign is opposite the physical G defined in section 2.1. Report physical G everywhere in the main text; archive identifiers appear only in the data-availability appendix with the s = -G rule restated there.
9. **"dimension-invariant"** (abstract, sections 4.2, 6): the evidence is an MLP probe moving 0.89 to 0.88 over d in {4, ..., 32}. Correct wording: "approximately unchanged over the tested dimensions under a nonlinear probe, and linearly accessible at d = 4". Two claims, two probes; never merge them.
10. **"beyond any wall-limited filter"** (abstract, sections 4.9, 6): narrow to "not recovered by the tap counts, delay windows and estimators considered here", with the delay-for-sensors trade of section 4.6 cited as what bounds the tested configurations. No formal observability bound is proved.
11. **"three-dimensional observability boundary"**: the evidence is four |G| = 4 cases and a single mid-plane enstrophy-fraction diagnostic (0.20 to 0.56). Wording everywhere: "an indication of a three-dimensional observability limitation", with n = 4 stated.
12. **Envelope narrative tension to reconcile explicitly.** Section 4.8 places the boundary at a gust ratio of three (calibration, divergence rate) while the abstract celebrates tracking at |G| = 4 (0.837 median). Both are true and must be stated as one two-part sentence: median tracking holds through |G| = 4 while the filter's uncertainty consistency degrades from |G| of about 3 (divergence rate 0.68 to 0.72, under-dispersion). Otherwise a referee will read the abstract and section 4.8 as contradicting each other.
13. **The static-inverse honesty clause must survive compression.** The best filter's in-distribution impact median (0.794) does not beat the static delay-embedded inverse (0.825). This is currently stated (section 4.9, conclusions) and must remain in the conclusions after the rewrite; the filter's case rests on tails, the boundary set, and state tracking.
14. **"pre-registered"** appears throughout. JFM readers will find the term foreign. Say once in section 3.5 that gates and calibrations were "fixed in advance of evaluation and archived with the data record", then stop repeating it.
15. **Table 1 framing.** "author-fill checklist ... owned by the simulation collaborators" cannot appear in a submission. The table simply presents the values. All PENDING entries are blockers (already flagged; repeated here because it gates everything).

## 3. Data-split provenance gate (new, blocks submission)

Several analyses in the manuscript originated on the previous-generation v2.1 models and the 85-case split, and the manuscript's own table 13 shows the correct disclosure pattern for one of them. Before submission, every main-text number must be tagged v2.2 (102 cases, 450 encounters, symmetric test_c) or explicitly disclosed as v2.1 with a justification, table-13 style. Audit and confirm, do not assume, the provenance of:

- the near-null / Mahalanobis departure mechanism (table 7, figure 12a);
- the parameter-only floor (sections 3.4, 4.3);
- the latent atlas and DMD spectrum (figure 21, section 4.10);
- the distributed-code gap (section 4.10);
- the topology check (appendix A);
- the pressure-recovery pillar (table 8, figure 13, appendix B);
- the preprocessing pillar (table 13; already disclosed, keep as the model).

The Claude Code session (Stage 1) builds a `PROVENANCE.md` ledger; anything untraceable gets a `% PROVENANCE-TODO` and is a compute task for the server session (`SESSION_MS_manuscript_compute.md`, Track M2).

## 4. The shared-operator gap in table 6 (new)

The rewrite plan correctly makes the shared direct forecaster the only primary cross-family forecast comparison. But table 6's "merit" column is computed under the suited operators (transformer on pooled states, U-Net on reference latents), so after the restructure the paper's central table would still carry the confounded comparison. Two options:

- **(a) Recompute** the five-observable merit at H = 16 under the shared direct forecaster for all ten table-6 rows (three operator seeds each). This is a cheap fit on frozen latents (Track M1 in the compute session). Preferred.
- **(b) Restructure** table 6 to representation-tier quantities only (wake R2, VRMSE, SSIM) and let all forecast merit live in the section 4.2 figure under the shared operator, with the suited-operator table in an appendix.

Decision stub D301. If (a), the acceptance gate is that the family ordering matches the suited-operator ordering within case-clustered confidence intervals; if it does not, that is itself a reportable result and the prose changes accordingly.

## 5. Nomenclature and acronym reduction

A referee currently has to hold CL/CLN/CLW/CLWN, JEPA, REX, LAE, EnKF variants, two smoothers, two operators and three split names. Rename map (implemented as text macros in `macros.tex` so consistency is mechanical, e.g. `\PredState`, `\LiftState`):

| Current | Main-text name |
|---|---|
| CLW, "flagship" | the wake-supervised predictive state (the primary state) |
| CLN, "specialist" | the lift-focused predictive state |
| CL / CN / CW / cube codes | lift-only, near-body-only, wake-only cells; codes survive only on figure 6/7 axes with a legend |
| JEPA | predictive (joint-embedding) model; the acronym appears in sections 1 and 3 lineage only, never in title, abstract, section headers or conclusions |
| REX / TiRex | the direct multi-horizon forecaster; the ingredient ledger goes to supplementary |
| REX-EnKF | the forecast-noise filter |
| two-stage REX | the two-stage filter |
| linear LAE-KF | the linear latent filter |
| static E_obs | the static delay-embedded inverse |
| "leakage-free" | defined once in section 3.3 ("the estimator senses wall pressure only; the lift and wake probes used for scoring never enter the innovation, enforced by construction and by a unit test"), thereafter "the pressure-only estimator" |
| test_a / test_b / test_c | validation / in-distribution test / boundary test |

## 6. Structure, mapping and word budget

Adopt the session plan's skeleton. Concrete mapping of the current ten Results subsections and budgets (main text target about 12.5k words, 11-12 figures, 5 tables):

| New | Absorbs (current) | Budget |
|---|---|---|
| Abstract | - | 250 |
| 1 Introduction | 1 | 1,300 |
| 2 Flow configuration, data and endpoints | 2.1, 2.2 | 1,600 |
| 3 Reduced states, forecasting and estimation | 3.1-3.5 compressed; 3.2.1 details, estimator sub-variants, placement to appendices | 2,600 |
| 4.1 Constructing a physically useful state | 4.1, 4.2 (decode part), 4.3, 4.10 (DMD slim) | 1,300 |
| 4.2 Compression and forecastability | 4.2 (dimension), 4.4, 4.5, 4.10 (distributed-code gap) | 1,200 |
| 4.3 What wall pressure observes | 4.6, 4.6.1 | 1,000 |
| 4.4 Sequential estimation and operating limits | 4.7, 4.8, 4.9 | 1,800 |
| 5 Discussion | 5.1-5.5 compressed around three mechanisms | 1,200 |
| 6 Concluding remarks | 6 | 450 |

Appendices (in-paper): A architecture, regularisation, UQ (compressed); B estimator configurations, placement, sensing; C calibration audit (current D.3 stays in the paper for integrity, relocated). Supplementary material: forecaster ledger (D.1), failure modes (D.2), topology, preprocessing robustness, streaming/noise replications, spatial-latent trade figures, suited-operator comparison, atlas panels.

Discussion is organised around exactly three mechanisms, one paragraph each plus a selection-guidance paragraph and a limitations paragraph: (i) supervision aligns the latent with observables and builds the protected geometry; (ii) multi-step training stabilises forward use; (iii) sensing geometry limits what the wall recovers. No number re-quoted in Discussion that already appears in Results, except by reference.

## 7. Figure plan v2

Verified by page inspection: figure 6 nests duplicated (a)/(b) labels inside outer (a)/(b) panels; figure 16 carries five arguments with in-panel prose including the test-peek annotation drawn inside the plot; page 36 stacks figures 17-19 at small label sizes. Plan (target 11-12 main figures):

| # | Content | Source | Action |
|---|---|---|---|
| 1 | The encounter as wake reorganisation | fig 1 | keep |
| 2 | Parameter-space sampling and splits | fig 2 | keep; alias split names |
| 3 | Pipeline schematic: represent, forecast, estimate | figs 4+5 merged | redraw as one figure; fig 22 to appendix |
| 4 | Supervision cube: PR per cell + paired forest | fig 6 | redraw; flatten to two panels, single (a)/(b) labelling |
| 5 | Task-dependent readability matrix | fig 7 | keep |
| 6 | Dimension axis: two tiers + probe-dilution control | fig 9 | keep |
| 7 | Forecasting: shared operator, direct vs AR, oracle null | figs 11+12(b,c) merged | redraw; fig 12(a) joins fig 4 or appendix |
| 8 | Physics of the state: DMD shedding clock (+1 atlas panel) | fig 21 slim | redraw, placed in 4.1 (D302) |
| 9 | Sensors traded for delays | fig 13 | keep |
| 10 | Tracking the encounter from the wall | fig 14 | keep |
| 11 | Operating envelope + phase errors in physical units | figs 15 + 16(a,b) + 17(a) distilled | redraw as one two-row figure |
| 12 | Estimator ladder + assimilation-vs-dimension grid | figs 16(c) + 20 | redraw; remove in-panel prose; test-peek note moves to appendix C text |

To appendix or supplementary: figs 3 (Chang potential construction), 8 (decode floor), 10 (drop entirely, duplicates table 6), 16(d,e), 17(b), 18, 19, 22-25. Caption template for every kept figure: what is shown; split and n; uncertainty convention; the single inference the reader should draw.

## 8. Language rules (extended)

All of the GPT list, plus (with replacements): "settle/settled" (establish, examine), "buys" (provides, confers), "earns its keep" (state the numbers), "honest/honesty/honest surprise/we say so plainly" (delete; state the result), "carries particular force" (delete), "celebrated" (delete), "erratic" (non-monotonic in dimension and irreproducible across seeds), "as-built model" (the co-trained predictor), "own-stack" (per-family end-to-end), "kit strength" (default configuration), "load-bearing" (necessary under the present objective), "pre-registered" (fixed in advance of evaluation; once). House rules from the author conventions: British spelling throughout; no em-dashes anywhere (commas, colons, parentheses); at most two headline numbers per paragraph; physical intuition before formalism; plain first-person goal statements.

## 9. House-style calibration notes (from the project corpus)

- Lineage abstracts are physics-led and carry at most one or two numbers; the present abstract should keep exactly three (0.794, 0.286, 0.837) plus the two dimension tiers.
- Conclusions are titled "Concluding remarks" in this lineage and run two to three paragraphs of prose, ending on the limitation and the outlook, never on a component list.
- Results headers are physical questions or physical statements, which the new 4.1-4.4 titles already satisfy.
- The lineage motivates method choices with prior physical knowledge explicitly (compare the Re_tau-selection paragraph of the R4 paper); the Chang-head and wake-descriptor design paragraphs already do this and should be preserved through compression.
- Enumerated (i)/(ii)/(iii) programme statements are standard in this lineage; use one in the introduction's final paragraph.

## 10. What runs where

- **Prose, structure, figures-as-plans, audits**: the Claude Code editorial session (`claude_code_jfm_rewrite_v2.md`). It never computes science; it flags.
- **Numbers that must be produced or confirmed**: the server compute session (`SESSION_MS_manuscript_compute.md`): Track M1 shared-operator merit for table 6; Track M2 v2.2 provenance re-runs; Track M3 figure regeneration through `figstyle.py`.
- **Author decisions**: stubs D301-D306 listed at the end of the Claude Code file, to be appended to HANDOFF.md at the next free numbers.
