<!-- GENERATED SKELETON, then hand-classified. Regenerate the ID, file,
line and type columns with
`python scripts/review/annotate_relevance.py --inventory`, which preserves
the class and rationale of every ID that still exists. Insert the markers
with --apply, remove every marker with --strip. -->

# Relevance map

One row per annotatable block reachable from `paper/main.tex`: every prose
paragraph and every figure or table caption the document typesets. The
overlay that renders it is `paper/reviewmarks.tex`.

| | class | colour | meaning |
|---|---|---|---|
| **K** | keep | green | stays in the body as it is |
| **T** | trim | orange | could be removed; the paper survives without it |
| **A** | annex | purple | belongs in an appendix, not the body |
| **S** | supp | brown | belongs in the supplementary material |
| **D** | delete | red | can be deleted outright, nothing is lost |

`note` is the rationale printed in the margin beside the block. It is
LaTeX, so escape `_`, `%`, `&` and `#`, and it must not contain `|`.

## Caveats

Two paragraphs separated only by a `%` comment parse as one block,
because a comment swallows its own line ending and the two really are one
paragraph in the PDF.

Generated, so re-running its generator drops the marker and --apply
has to be re-run: `sections/tables/table_dns_metadata.tex`.

## Tally

| class | blocks |
|---|---|
| K | 142 |
| T | 15 |
| A | 4 |
| S | 4 |
| D | 2 |
| **total** | **167** |

Blocks not marked keep, by file:

| file | T | A | S | D |
|---|---|---|---|---|
| `sections/appendix_a_regularisation.tex` | 3 |  | 3 |  |
| `sections/appendix_b_sensing.tex` | 2 |  |  |  |
| `sections/section_1_introduction.tex` | 2 |  |  |  |
| `sections/section_2_flow_and_data.tex` |  |  | 1 |  |
| `sections/section_3_methods.tex` |  |  |  | 1 |
| `sections/section_4_results.tex` | 4 | 1 |  | 1 |
| `sections/section_5_discussion.tex` | 2 |  |  |  |
| `sections/v4/s3_4_estimators.tex` |  | 2 |  |  |
| `sections/v4/s4_b_reconstruction.tex` | 1 |  |  |  |
| `sections/v4/s4_d_assimilation.tex` | 1 | 1 |  |  |

| ID | file | lines | type | class | note |
|---|---|---|---|---|---|
| AB-01 | sections/abstract.tex | 17-26 | para | K | The paper's contract with the reader, now carrying the in-distribution pair. Every clause maps to a Results exhibit. |
| S1-01 | sections/section_1_introduction.tex | 18-18 | para | T | Generic small-UAV motivation that the extreme-aerodynamics literature already establishes. S1-02 restarts with the same framing, so the section survives without it. |
| S1-02 | sections/section_1_introduction.tex | 20-20 | para | K | The empirical premise the paper builds on: these encounters admit low-dimensional descriptions. Names the lineage the comparison is against. |
| S1-03 | sections/section_1_introduction.tex | 22-22 | para | K | States the gap: compactness is not a plant state. Motivates wall pressure as the only sensor a wing carries. |
| S1-04 | sections/section_1_introduction.tex | 24-24 | para | K | The pivot. Compactness against forecastability, and nonlinearity against wake reach, is the tension the whole paper resolves. |
| S1-05 | sections/section_1_introduction.tex | 26-26 | para | K | Introduces the predictive objective as a diagnostic instrument rather than a proposal. This framing is what keeps the paper from reading as an architecture paper. |
| S1-06 | sections/section_1_introduction.tex | 28-28 | para | K | Previews the collapse finding, which turns the observable heads from an assumption into a result. Load-bearing for the contribution claim. |
| S1-07 | sections/section_1_introduction.tex | 30-30 | para | K | The three requirements, the design of the comparison, and the four claims. This is the contribution list a referee will check the Results against. |
| S1-08 | sections/section_1_introduction.tex | 32-32 | para | T | Pure signposting, and \S4's own subsection openings already announce the order. Removing it costs the reader nothing that the section headings do not supply. |
| S2-01 | sections/section_2_flow_and_data.tex | 13-45 | para | K | Configuration, solver provenance and the gust parametrisation. The DNS-not-LES and Fukami-as-reference-not-source distinctions live here. |
| S2-C1 | sections/section_2_flow_and_data.tex | 50-59 | caption | K | Defines the free-stream-aligned axes in which $(G,D,Y)$ are read. Without it the sign conventions of the envelope result are ambiguous. |
| S2-02 | sections/section_2_flow_and_data.tex | 63-77 | para | K | The staged reading of the encounter that the five observables summarise. Physics framing the Results repeatedly refer back to. |
| S2-C2 | sections/section_2_flow_and_data.tex | 82-94 | caption | K | The gust-strength sweep that shows the wake reorganisation the paper is about. Also states why the wake, not only $C_L$, is the endpoint. |
| S2-03 | sections/section_2_flow_and_data.tex | 98-118 | para | K | The physical reason $\lvert G\rvert = 4$ is a regime change and not one parameter step. This paragraph is what licenses the boundary reading in \S4.6, \S5.4 and \S6. |
| S2-04 | sections/section_2_flow_and_data.tex | 123-158 | para | K | Partition, encounter counts and the non-independence caveat behind the case-clustered statistics. Reproducibility rests on it. |
| S2-05 | sections/section_2_flow_and_data.tex | 163-172 | para | S | A data-integrity check on the run4 cases, not a result: the mirror-image lift transients confirm the file metadata. The one load-bearing sentence, that the boundary is sampled at both signs, is already in S2-04 and the fig:paramspace caption. |
| S2-06 | sections/section_2_flow_and_data.tex | 180-197 | para | K | What the cache stores and how the vorticity is normalised. Trim internally: the SSIM constants are restated verbatim in S35-01. |
| S2-07 | sections/section_2_flow_and_data.tex | 199-215 | para | K | Pre-registers the two co-primary endpoints. The multiplicity correction in appendix A is applied against exactly this declaration. |
| S2-08 | sections/section_2_flow_and_data.tex | 217-254 | para | K | Defines $E_w$ and $\Gamma^{\pm}$, the endpoints of the whole comparison. The threshold-collinearity check inside it could move to the data record. |
| S2-09 | sections/section_2_flow_and_data.tex | 256-269 | para | K | Bounds how the closure may be read: mid-plane wake diagnostics against three-dimensional forces. The PIV and load-cell realisability sentence is the removable part. |
| S2-C3 | sections/section_2_flow_and_data.tex | 274-283 | caption | K | Shows the sampling and the symmetric boundary, and names the $Y$ under-sampling that \S5.4 lists as a limitation. |
| TDNS-C1 | sections/tables/table_dns_metadata.tex | 20-23 | caption | K | The DNS reproducibility record. Generated from the DNS metadata YAML, so any wording change goes through that file and the renderer, never this one. |
| S3-01 | sections/section_3_methods.tex | 13-13 | para | K | The encoder and the justification of $d = 32$ as a deployment criterion rather than a tuned choice. Referees will look for exactly this. |
| S3-02 | sections/section_3_methods.tex | 15-21 | para | K | Carries equation (1), the one statement of the forecasting problem. The two-network description duplicates APA-01, which is where the detail belongs. |
| S3-03 | sections/section_3_methods.tex | 23-23 | para | K | The training objective and the anti-collapse term. Names the direct-forecaster-coupled lineage used at low dimension. |
| S3-04 | sections/section_3_methods.tex | 25-40 | para | K | Defines the three supervision heads and the wake descriptor. The five-citation provenance of the descriptor design could move to appendix A. |
| S3-05 | sections/section_3_methods.tex | 42-42 | para | K | The one asymmetry in the head comparison that is a design choice rather than an oversight: the linear basis carries no supervision channel. Cut from a five-point enumeration whose other four points are stated where they bind. |
| S3-C1 | sections/section_3_methods.tex | 51-69 | caption | K | The training-time architecture, with the stage attribution for every module that no other text states. |
| S3-06 | sections/section_3_methods.tex | 80-80 | para | K | The single principle the comparison rests on: same pipeline, only the encoder differs. Also names the shared operator and the retired suited-operator protocol. |
| S3-07 | sections/section_3_methods.tex | 84-84 | para | K | Defines representational closure, the headline figure of merit, and scopes the rollout as secondary. The whole Results section is read through this definition. |
| S3-08 | sections/section_3_methods.tex | 88-88 | para | D | A two-sentence analogy between the filter's mixed timescales and the training objective. It carries no evidence and nothing downstream refers to it. |
| S3-09 | sections/section_3_methods.tex | 97-97 | para | K | Defines the latent-drift diagnostic used in table 8 and \S4.4. Without it the near-null decomposition cannot be read. |
| S3-10 | sections/section_3_methods.tex | 99-99 | para | K | Defines the parameter-only floor that makes the wake result non-trivial. \S4.1 and \S5.1 both depend on it. |
| S3-11 | sections/section_3_methods.tex | 101-101 | para | K | The one-factor-at-a-time controls and the objective-free supervised control. This is the paragraph that answers the confound objection. |
| S31-01 | sections/v4/s3_1_chang_head.tex | 11-11 | para | K | The near-body head at the same depth as the other two: what the observable is, why it is not the wake target, and where the construction lives. Full construction moved to appendix A. |
| S31-C1 | sections/v4/s3_1_chang_head.tex | 19-48 | caption | K | The near-body head pipeline. Makes the geometry-only status of $\phi_L$ and the shared $64+16$ construction visible, which the prose can only assert. |
| S33-01 | sections/v4/s3_3_rex.tex | 7-46 | para | K | Defines the one operator under which every family is compared. The forecast result and the filter's process noise both depend on this specification. |
| S34-C1 | sections/v4/s3_4_estimators.tex | 18-46 | caption | K | The deployment loop on its own, with the two slots that distinguish the four estimator configurations. The leakage guarantee sits here because the innovation path is what carries it. |
| S34-01 | sections/v4/s3_4_estimators.tex | 50-78 | para | K | Why estimation must be sequential, and the delay-embedding principle honestly scoped as an organising idea rather than a guarantee. \S4.5 is read against it. |
| S34-02 | sections/v4/s3_4_estimators.tex | 80-121 | para | K | The shared state-space problem, equations (5) and (6), plus the leakage guarantee. Every filter in the suite is defined by naming four ingredients of this pair. |
| S34-03 | sections/v4/s3_4_estimators.tex | 126-165 | para | K | The base filter with its update equations. It is a diagnostic rather than the production estimator, but the equations are shared by the whole suite. |
| S34-04 | sections/v4/s3_4_estimators.tex | 170-201 | para | K | Carries equation (9), the encoded observation used by three of the four configurations, and the one fixed definition of consistency failure. Load-bearing well beyond its own subsection. |
| S34-05 | sections/v4/s3_4_estimators.tex | 206-241 | para | K | The production route: state-dependent noise, its two independent calibrations, and the two-stage schedule. This is the estimator the headline number comes from. |
| S34-06 | sections/v4/s3_4_estimators.tex | 247-264 | para | A | A textbook fixed-lag recursion whose result is three sentences in \S4.6.3 and one clause of appendix B guidance. The recursion itself can sit in appendix A with the other configuration detail. |
| S34-07 | sections/v4/s3_4_estimators.tex | 269-291 | para | A | The greedy placement criterion, whose own text notes that two of its four terms dominate. Placement robustness is already argued in appendix B, so the criterion belongs beside it. |
| S35-01 | sections/v4/s3_5_protocol.tex | 7-42 | para | K | Defines the three phases and every load metric used in the paper. Trim internally: the SSIM convention is already given in S2-06. |
| S35-02 | sections/v4/s3_5_protocol.tex | 44-62 | para | K | Points at the one uncertainty accounting and closes on the no-boundary-set-selection guarantee. The four-level enumeration it used to carry is now table 16; the sign-convention clause is still an open REVIEW-CLAIM. |
| S4-01 | sections/section_4_results.tex | 13-13 | para | K | Names the three families and fixes the reporting policy that keeps the boundary set out of every selection. The section is unreadable without it. |
| S4-02 | sections/section_4_results.tex | 24-24 | para | K | Establishes that the wake is a discriminating endpoint and reads table 5. The central representational result. |
| S4-03 | sections/section_4_results.tex | 26-34 | para | K | The objective-free control and the shared-operator tie, which together attribute readability and forecastability to supervision rather than to the objective. The paper's most self-critical result. |
| S4-04 | sections/section_4_results.tex | 36-36 | para | K | The inversion that fixes the coordinate-level reading: best lift forecast, worst wake readability. Sets up the distributed-code argument. |
| S4-05 | sections/section_4_results.tex | 52-52 | para | K | Concedes the linear basis its field-reconstruction win before the wake result is read. Removing it would make the comparison look stacked. |
| S4-06 | sections/section_4_results.tex | 54-54 | para | K | Resolves the decode floor to the critical instants, which is where the linear readout actually breaks. Carries table 6. |
| S4-07 | sections/section_4_results.tex | 58-58 | para | T | A single sentence pointing at an appendix figure to say, for the third time, that field decode does not separate the families. S4-05 and S4-06 have already made the point with a table. |
| S4-08 | sections/section_4_results.tex | 60-60 | para | K | The discriminating result the field pixels hide: supervision decides what a state carries at the critical instants. Carries table 7. |
| S4-09 | sections/section_4_results.tex | 67-67 | para | K | The distributed wake code and the readability plateau at $d = 32$. Both are cited in \S5 and \S6. |
| S4-10 | sections/section_4_results.tex | 69-69 | para | D | A one-line forward pointer to \S5.3 and nothing else. The reader reaches \S5.3 without it. |
| S4-11 | sections/section_4_results.tex | 77-77 | para | K | The shedding clock read by dynamic mode decomposition, and the scoping to the Fukami lineage rather than reconstruction in general. A precision the paper needs. |
| S4-12 | sections/section_4_results.tex | 79-79 | para | K | The coordinate spectra that locate the reconstruction failure in the frequency domain. The flatness gate behind the broadband wording passed here. |
| S4-13 | sections/section_4_results.tex | 81-81 | para | K | The parametric manifold, honestly scoped on the weakly readable offset. \S5.4 cites it as a limitation. |
| S4-14 | sections/section_4_results.tex | 83-83 | para | K | The near-null decomposition showing supervision, not anti-collapse, builds the protected subspace. The mechanism claim of the paper. |
| S4-C1 | sections/section_4_results.tex | 91-103 | caption | K | Covers the atlas, the DMD spectrum and the per-coordinate spectra in one caption. Panel (f) is lettered at float level. |
| S4-15 | sections/section_4_results.tex | 113-113 | para | A | A number-dense narration of a figure that already lives in appendix A. Paragraph and figure should sit together, wherever they end up. |
| S4-16 | sections/section_4_results.tex | 115-115 | para | T | The fourth statement that field decode does not separate the families, this time for the forecast. One sentence in S4-05 would carry it. |
| S4-17 | sections/section_4_results.tex | 123-123 | para | K | What the multi-step objective adds: long-horizon stability and lower drift. This is the only place the predictive objective earns its own credit. |
| S4-C2 | sections/section_4_results.tex | 128-133 | caption | K | The rollout mechanism figure: protected subspace, merit and drift against horizon. Self-contained. |
| S4-18 | sections/section_4_results.tex | 142-142 | para | K | The physics-versus-variance ordering, the wall observability result. \S5.1 and \S6 both rest on it. |
| S4-19 | sections/section_4_results.tex | 149-149 | para | K | Concedes that the margin over a field latent is modest and declines the strong-form claim. Exactly the honesty a referee rewards. |
| S4-20 | sections/section_4_results.tex | 154-154 | para | K | The delay window recovers the coordinate the instantaneous pressure misses. Confirms the delay-embedding reading empirically. |
| S4-21 | sections/section_4_results.tex | 156-156 | para | K | The tap-count against window-length trade, which is the deployment knob \S5.2 names. Carries figure 10. |
| S4-22 | sections/section_4_results.tex | 158-158 | para | T | Reconciliation arithmetic plus a single-tap filter control, and the reconciliation number is quoted a third time in APB-02. The filter-cannot-spare-taps finding is the part worth keeping. |
| S4-C3 | sections/section_4_results.tex | 165-171 | caption | K | Defines both panels of the trade figure and states which placement each uses. Self-contained. |
| S4-23 | sections/section_4_results.tex | 181-181 | para | K | Runs the estimator and reads the hero figure. Also states the control-literature pathway this closes. |
| S4-24 | sections/section_4_results.tex | 183-187 | para | K | The two boundaries: in-range tracking is not unique to this state, and no filter tracks the wake. Scoping without which the estimation claim would be oversold. |
| S4-C4 | sections/section_4_results.tex | 198-213 | caption | K | Covers both the hero traces and the envelope traces, with the production filter named. Panels lettered at float level. |
| S4-25 | sections/section_4_results.tex | 221-221 | para | K | Separates the base diagnostic filter from the production estimator, and gives the $R^2$-flatters-the-boundary warning the abstract now respects. The most important honesty paragraph in \S4. |
| S4-26 | sections/section_4_results.tex | 225-225 | para | T | A one-sentence pointer to a panel whose caption already says the same thing. Nothing depends on it. |
| S4-27 | sections/section_4_results.tex | 227-227 | para | K | The per-family filter comparison at the boundary, framed as a division of degradation rather than a win. Carries table 12. |
| S4-28 | sections/section_4_results.tex | 231-237 | para | K | Where the filter degrades: under-dispersion from a gust ratio of about three, with the innovation statistic. The missing-noise-model diagnosis follows from it. |
| S4-29 | sections/section_4_results.tex | 239-239 | para | K | The sign asymmetry of the envelope, measurable only because the boundary was sampled at both signs. Pays off S2-05. |
| S4-30 | sections/section_4_results.tex | 241-241 | para | K | Places the limit at $\lvert G\rvert = 3$ and splits the envelope into two claims with different owners. The paragraph that keeps the estimation result correctly attributed. |
| S4-C5 | sections/section_4_results.tex | 246-256 | caption | K | Names the diagnostic filter explicitly and points to the production tracking figure. The disambiguation lives in the caption as well as the text. |
| S4A-01 | sections/v4/s4_a_construction.tex | 8-29 | para | K | The conditioning cube and the collapse without the lift head. The construction finding the abstract leads with. |
| S4A-02 | sections/v4/s4_a_construction.tex | 31-46 | para | K | Nothing substitutes for the scalar anchor, with the paired differences. Also fixes the wake-headed state as the representative carried downstream. |
| S4A-03 | sections/v4/s4_a_construction.tex | 48-69 | para | K | The division of labour between the three heads, with the enstrophy probe. The closing caveat about which comparison supports which claim is worth keeping. |
| S4A-C1 | sections/v4/s4_a_construction.tex | 74-81 | caption | K | The collapse against the participation-ratio floor, with the demoted panels named. Self-contained. |
| TCLO-C1 | sections/tables/table_closure.tex | 18-38 | caption | K | The representational closure table, the headline exhibit of \S4.1. Reports readability and shared-operator merit side by side. |
| S4B2-01 | sections/v4/s4_b2_dimension.tex | 9-23 | para | K | The two-tier compression statement, which is the paper's answer on how small the state can be. \S6 quotes it directly. |
| S4B2-02 | sections/v4/s4_b2_dimension.tex | 25-37 | para | K | The probe-dilution control that disciplines the $d = 4$ claim. Without it the compression result would be over-read. |
| S4B2-C1 | sections/v4/s4_b2_dimension.tex | 42-49 | caption | K | Three panels including the control, with the seed provenance stated. Self-contained. |
| S4B-01 | sections/v4/s4_b_reconstruction.tex | 8-25 | para | T | A whole subsection whose conclusion is that field decode is a wash, with its figure already in the appendix. S4-05 and S4-06 state the same result from the tables. |
| TSSI-C1 | sections/tables/table_critical_ssim.tex | 7-21 | caption | K | Decode fidelity at the critical instants rather than window-averaged. The exhibit that shows where the linear readout survives. |
| TOBS-C1 | sections/tables/table_obs_critical.tex | 10-36 | caption | K | Readability of lift and wake enstrophy at the three critical instants. The discriminating exhibit of \S4.2. |
| S4C-01 | sections/v4/s4_c_prediction.tex | 5-27 | para | K | The operator-robust protocol that removes the operator confound, and the lift-forecastability result scoped to the lift readout. Carefully reconciled with the five-observable tie and should stay that way. |
| S4C-02 | sections/v4/s4_c_prediction.tex | 29-42 | para | K | Direct against autoregressive: the mechanism behind the forecasting result. Also flags the median-contraction behaviour the filter must correct. |
| S4C-03 | sections/v4/s4_c_prediction.tex | 44-52 | para | K | The oracle-conditioning negative, which closes the deployment argument from the third side. A genuine negative result worth its space. |
| S4C-C1 | sections/v4/s4_c_prediction.tex | 67-75 | caption | K | Both forecast panels with the seed provenance corrected to operator seeds. Self-contained. |
| TREC-C1 | sections/tables/table_recovery.tex | 5-16 | caption | K | What eight taps recover of each family's state, with the per-family estimator selection disclosed. The exhibit behind the physics-against-variance ordering. |
| TFAM-C1 | sections/tables/table_family_filter.tex | 5-13 | caption | K | The frozen filter run identically on every family. The exhibit behind the in-range tracking is not unique claim. |
| TENV-C1 | sections/tables/table_envelope.tex | 5-14 | caption | K | The operating envelope in both $R^2$ and lift units, under the diagnostic filter as the caption states. The RMSE column is what keeps the boundary claim honest. |
| TFIL-C1 | sections/tables/table_filter_error.tex | 7-23 | caption | K | What the filter costs in real lift units, per family and per gust strength. Decision D252's exhibit. |
| S4D-01 | sections/v4/s4_d_assimilation.tex | 10-22 | para | K | Physical units reorganise the phase story and expose the relaxation $R^2$ as a variance artefact. Decision D252 in one paragraph. |
| S4D-02 | sections/v4/s4_d_assimilation.tex | 24-40 | para | K | The estimator ladder with the static inverse on top, the qualifier that scopes the entire estimation result. The abstract depends on this paragraph. |
| S4D-03 | sections/v4/s4_d_assimilation.tex | 42-66 | para | A | The full calibration narrative, including the excluded test-selected variant, told at length in the body. Appendix C already exists for exactly this audit trail; the body needs the one-sentence outcome. |
| S4D-04 | sections/v4/s4_d_assimilation.tex | 68-77 | para | K | The smoother result and the deployment rule it changes. Short, concrete and cited in appendix B guidance. |
| S4D-05 | sections/v4/s4_d_assimilation.tex | 79-95 | para | K | The per-family end-to-end comparison, which is the fair version of the estimator claim. Also the sensor-budget and noise-robustness separation. |
| S4D-06 | sections/v4/s4_d_assimilation.tex | 97-120 | para | K | The dimension-by-family grid and the reconstructive seed fragility, with the observation-side mechanism verified. This is the deployment-robustness argument of the paper. |
| S4D-07 | sections/v4/s4_d_assimilation.tex | 122-133 | para | T | Two loose ends that restate \S4.6.2's under-dispersion and \S4.5's sensing geometry. Its consistency-failures-removed claim is also the one number in the section with no bound macro. |
| S4D-C1 | sections/v4/s4_d_assimilation.tex | 138-146 | caption | K | The phase-resolved tracking figure with the production band stated. Self-contained. |
| S4D-C2 | sections/v4/s4_d_assimilation.tex | 156-162 | caption | K | The dimension grid, with the single-seed disclosure per cell. Self-contained. |
| S5-01 | sections/section_5_discussion.tex | 17-17 | para | K | Reads the comparison as a chain of tasks and says where each family drops out. The clearest single statement of the paper's result. |
| S5-02 | sections/section_5_discussion.tex | 19-19 | para | K | The design rule and the geometric mechanism, with the earlier anti-collapse reading explicitly corrected. The contribution paragraph. |
| S5-03 | sections/section_5_discussion.tex | 21-21 | para | K | Why the wake and not the forces separates the families, and what the linear basis remains good for. Keeps the POD comparison fair. |
| S5-04 | sections/section_5_discussion.tex | 23-23 | para | K | The capacity caveat and the three controls that answer it. Pre-empts the most likely referee objection. |
| S5-05 | sections/section_5_discussion.tex | 28-28 | para | K | The estimation demonstration with its limits stated in the same breath. The feasibility-demonstration framing lives here. |
| S5-06 | sections/section_5_discussion.tex | 30-30 | para | K | The delay-embedding view of the deployment knob and the reason the envelope closes where it does. Ties the sensing and envelope results together. |
| S5-07 | sections/section_5_discussion.tex | 35-35 | para | T | A one-paragraph subsection whose two numbers are both in the supplementary material. It could become two sentences inside \S5.1 without loss. |
| S5-08 | sections/section_5_discussion.tex | 42-47 | para | K | Five limitations, including the seed-fragility disclosure and the calibration priority. The rehearsal of the superseded U-Net seed study inside it is the removable part. |
| S5-09 | sections/section_5_discussion.tex | 49-49 | para | K | Scopes the work against closed-loop control and states what the evidence does not cover. The narrowness is deliberate and should be visible. |
| S5-10 | sections/section_5_discussion.tex | 51-51 | para | T | A speculative alternative wake target with a citation and no test. Interesting, but it is future work stated at paragraph length. |
| S6-01 | sections/section_6_conclusions.tex | 11-11 | para | K | Answers the construction question and separates the collapse claim from the peak-load claim. Both rest on different comparisons and the text says so. |
| S6-02 | sections/section_6_conclusions.tex | 13-13 | para | K | The compression and forecasting statements, both two-tier and both scoped. Matches \S4.2 and \S4.4 exactly. |
| S6-03 | sections/section_6_conclusions.tex | 15-15 | para | K | The estimation statement, now leading with the in-distribution pair rather than the flattered boundary score. This is the F01 repair. |
| S6-04 | sections/section_6_conclusions.tex | 17-17 | para | K | The design principle and the three results that bound it. The static-inverse qualifier here is what appendix C must support. |
| APA-01 | sections/appendix_a_regularisation.tex | 8-33 | para | T | Opens by restating the architecture already given in \S3.1 and table 14, then spends twenty lines narrating a suited-operator protocol the paper no longer uses. The superseded narrative belongs with the suited table in the supplementary material. |
| APA-C1 | sections/appendix_a_regularisation.tex | 38-45 | caption | K | The predictive-against-reconstructive schematic, which is the clearest statement of what the objective changes. Referenced from \S3.1. |
| APA-02 | sections/appendix_a_regularisation.tex | 50-65 | para | K | The anti-collapse regulariser, its weight and its safeguard. Needed to reproduce the training. |
| APA-03 | sections/appendix_a_regularisation.tex | 68-80 | para | K | What the accounting table cannot say: which source dominates at this sample size, and why the plain encounter bootstrap overstates it. No longer re-enumerates the protocol. |
| APA-04 | sections/appendix_a_regularisation.tex | 84-88 | para | K | Explains why the estimation endpoints are tested paired and case-clustered. Two sentences that justify table 15. |
| APA-C2 | sections/appendix_a_regularisation.tex | 92-106 | caption | K | The paired-test table, including the null on the wake enstrophy. The pre-declared family structure is stated in the caption. |
| APA-05 | sections/appendix_a_regularisation.tex | 129-143 | para | S | A coordinate-free topology check that no claim in the body depends on. Genuine supporting evidence, which is what the supplementary material is for. |
| APA-06 | sections/appendix_a_regularisation.tex | 146-157 | para | S | A robustness check run on the previous-generation $d = 64$ encoders, honestly disclosed as such. The body sentence it supports can point at the supplementary material instead. |
| APA-C3 | sections/appendix_a_regularisation.tex | 161-169 | caption | S | The preprocessing-robustness table, on the same superseded encoders as APA-06. Moves with it. |
| APA-07 | sections/appendix_a_regularisation.tex | 199-241 | para | K | The full construction of the near-body head, moved out of Methods so the three heads sit at one depth in the body. Two equations, the discretisation and the pre-committed gate. |
| APA-08 | sections/appendix_a_regularisation.tex | 243-254 | para | K | The near-body conditioning increment, demoted here from \S4.1 and referenced from it. The strongest cell in the cube deserves this much. |
| APA-C4 | sections/appendix_a_regularisation.tex | 259-264 | caption | K | The paired cube deltas with case-clustered intervals. Self-contained. |
| APA-C5 | sections/appendix_a_regularisation.tex | 273-277 | caption | K | The latent portraits that corroborate the spectral result qualitatively. Referenced from \S4.3. |
| APA-C6 | sections/appendix_a_regularisation.tex | 286-290 | caption | K | The full task-dependent readability matrix behind the division-of-labour claim. Referenced from \S4.1. |
| APA-C7 | sections/appendix_a_regularisation.tex | 297-302 | caption | K | The conditioning null, demoted from the forecast figure and referenced from \S4.4. Self-contained. |
| APA-C8 | sections/appendix_a_regularisation.tex | 309-313 | caption | K | The collapse training history, demoted from the cube figure. Shows the collapse is flat from the first diagnostic. |
| APA-C9 | sections/appendix_a_regularisation.tex | 320-325 | caption | K | The evaluation protocol applied identically to every family. Referenced from \S3.2 and the framework caption. |
| APA-09 | sections/appendix_a_regularisation.tex | 337-341 | para | T | A list of what the following figures show, immediately before figures whose captions each say it. Pure connective tissue. |
| APA-C10 | sections/appendix_a_regularisation.tex | 346-350 | caption | K | The Chang auxiliary potential and its supervision band. Referenced from \S3.1.2. |
| APA-C11 | sections/appendix_a_regularisation.tex | 357-363 | caption | K | The decode gallery behind the decode-floor claim, with the linear basis included. Self-contained. |
| APA-C12 | sections/appendix_a_regularisation.tex | 370-374 | caption | K | Decode fidelity against dimension, the exhibit S4-07 points at. Single-seed provenance disclosed. |
| APA-C13 | sections/appendix_a_regularisation.tex | 381-386 | caption | K | The phase-split forecast merit, the exhibit S4-15 narrates. Self-contained. |
| APA-10 | sections/appendix_a_regularisation.tex | 397-401 | para | T | A one-sentence list of the tables that follow, each of which has a caption. Pure connective tissue. |
| APA-C14 | sections/appendix_a_regularisation.tex | 405-412 | caption | K | The encoder and predictor configuration, counted from the released checkpoint. The reproducibility record. |
| APA-C15 | sections/appendix_a_regularisation.tex | 437-443 | caption | K | The per-estimator configuration table, the Gupta table-1 analogue. What makes the estimator suite reproducible. |
| APA-C16 | sections/appendix_a_regularisation.tex | 470-476 | caption | K | The single authoritative uncertainty accounting, six sources with their units and counts. Replaces three partial statements that disagreed with each other. |
| TBAS-C1 | sections/tables/table_baselines.tex | 9-25 | caption | K | Defines every compared family in one place. The matched-dimension claim is checkable only against this table. |
| TENK-C1 | sections/tables/table_enkf.tex | 15-25 | caption | K | The four estimator configurations side by side, which no other exhibit gives. Recently widened from a single-filter list. |
| TMEC-C1 | sections/tables/table_mechanism.tex | 5-11 | caption | K | The rollout-departure decomposition behind the protected-subspace mechanism. Referenced from \S4.3. |
| APB-01 | sections/appendix_b_sensing.tex | 8-15 | para | K | States the sensing conventions and how the recovery estimator is selected. Needed before any recovery number can be read. |
| APB-02 | sections/appendix_b_sensing.tex | 18-35 | para | K | Why the ordering is not an artefact of model-conditioned placement, with the target-blind rerun. Answers a real objection. |
| APB-03 | sections/appendix_b_sensing.tex | 38-47 | para | T | Restates the window-against-instant result already given in S4-20 and S4-21, with the same macros. The appendix adds only the enstrophy figure. |
| APB-04 | sections/appendix_b_sensing.tex | 50-57 | para | K | Reads the recovered-field panels and localises the loss to the wake structures. Connects the sensing result to physical space. |
| APB-C1 | sections/appendix_b_sensing.tex | 62-69 | caption | K | The recovered field against each family's own decode ceiling, which is the only fair comparison. The caption says so. |
| APB-05 | sections/appendix_b_sensing.tex | 74-81 | para | T | Restates the boundary failure of every static approach, already carried by S4-25 and table 12. The appendix repeats the numbers without adding evidence. |
| APB-06 | sections/appendix_b_sensing.tex | 84-95 | para | K | Justifies the delay stride and window as convention rather than tuning, and declines the false-nearest-neighbours criterion honestly. Methodologically careful. |
| APB-07 | sections/appendix_b_sensing.tex | 101-109 | para | K | The streaming realism protocol and the four-coefficient filter, both demoted from \S4.6. Real results, correctly placed. |
| APB-C2 | sections/appendix_b_sensing.tex | 114-117 | caption | K | The deployment-realism panels with the seed counts. Self-contained. |
| APB-C3 | sections/appendix_b_sensing.tex | 126-130 | caption | K | Relative peak error across the envelope, the scale-invariance exhibit \S4.6.3 points at. Self-contained. |
| APB-C4 | sections/appendix_b_sensing.tex | 138-141 | caption | K | The per-family end-to-end sweeps behind the own-stack claim. Single-seed provenance disclosed. |
| APB-08 | sections/appendix_b_sensing.tex | 150-161 | para | K | Explicit estimator-selection guidance in the Gupta idiom, ending in one sentence a practitioner can act on. The most useful paragraph in the appendices. |
| APC-01 | sections/appendix_c_calibration.tex | 5-23 | para | K | The full audit trail for the band scale, including the excluded test-selected value quoted only here. Integrity material that must stay in the paper. |
| APC-C1 | sections/appendix_c_calibration.tex | 30-38 | caption | K | The estimator ladder, the smoother comparison and the band calibration in one figure. Carries the static-inverse claim the abstract makes. |
