# Session 18 Plan: Baseline comparison, manuscript drafting, submission to JFM

Date: 2026-05-28
Lead: Carlos Sanmiguel Vila (INTA, UC3M)
Hardware: 2x RTX 6000 Blackwell (sm_120), bf16 mixed precision
Target venue: JFM, primary submission target. Realistic submission window: 6 weeks from session start.

-----

## Strategic frame

Session 18 closes the experimental phase of the paper and produces the manuscript. The session has three workstreams in series: a focused baseline comparison against Fukami AE and POD (one experiment, one week), the manuscript draft (three weeks across all sections), and figure polish (one to two weeks). Total: six weeks to submission.

No new encoder retraining of the JEPA. No new SHAP runs. No new sensor selection. The only new training is the Fukami AE and POD baseline models, both standard published architectures applied to your existing data.

The paper this session produces makes five claims, each with a section and a headline figure:

**Claim 1.** The latent trajectories of vortex-airfoil impact events trace organized arcs in a 3D projection, cluster by perturbation parameters, and pass through the impact frame as a smooth high-speed traversal (curvature minimum, speed maximum). The trajectory geometry is reproducible across independently trained encoders at the basis-invariant level.

**Claim 2.** Parameter recoverability from the latent concentrates at the impact frame, with asymmetric Gaussian decay (sigma_L = 10 frames, sigma_R = 54 frames). The impact frame is a privileged dynamical state where parameter information, surface pressure footprint, and near-body vorticity become co-observable.

**Claim 3 (the centerpiece).** Conditional Markov closure holds in physical observables. The Markov-only rollout from z_impact matches or beats the full-context rollout in C_L, wake vorticity impulse, and integrated wake enstrophy out to H = 16 frames in-distribution and OOD. Pre-impact temporal context is not just information-free; it is mildly harmful for short-horizon prediction.

**Claim 4.** The response-relevant coherent structures driving the encoded representation, identified by gradient-SHAP and validated by intervention, do not coincide with classical Q-criterion vortex cores (mean IoU < 0.2). They localize at shear layers, body-vortex interaction zones, and wake transition regions. This parallels the Cremades et al. finding for wall-bounded turbulence in a strongly transient parametric setting.

**Claim 5.** Sparse surface pressure (K = 8 to 16 sensors) with a nonlinear estimator (MLP-reg or TCN) recovers the impact-frame latent and the perturbation parameters sufficiently well to drive the Markov rollout within 30 percent of the oracle absolute physical-metric error. The deployment loop closes for forces, vorticity impulse, and wake enstrophy in-distribution.

The Session 16 and Session 17 findings supply the evidence. Session 18 adds the baseline comparison (Experiment B1) and produces the manuscript.

-----

## Experiment B1: Fukami AE and POD baseline comparison on physical Markov closure

### Why this experiment is necessary

The paper’s strongest claim is that the JEPA learns a predictive state, not just a low-dimensional reconstruction manifold. The natural reviewer challenge is “Fukami already showed that a 3-variable manifold suffices for vortex-gust interactions; what does your predictive framing add over their reconstruction framing?” Your D100 result (Session 14) shows JEPA absorbs the dataset 2.16x more efficiently than Fukami AE at matched d = 32 via epiplexity. This is a model-fitting argument. What is missing is the physical-metric comparison: does the Fukami AE latent, with a transformer predictor on top, also achieve conditional Markov closure on C_L, vorticity impulse, and enstrophy?

If the answer is no, the paper has a clean differentiation: JEPA is a predictive state, Fukami AE is a reconstruction manifold, and the two are distinguishable on physical-metric Markov closure. If the answer is yes, the paper’s differentiation shifts to coordinate-invariance (cross-seed properties) and structure discovery (non-Q SHAP localization), and the Markov closure becomes a shared property of both predictive architectures. Either outcome is publishable; the paper’s framing depends on which holds.

### Hypothesis

The Fukami AE at d = 64 with a transformer predictor on top will achieve weaker physical-metric Markov closure than the JEPA at the same horizon, with the gap most visible in C_L and wake enstrophy at H = 16. POD at d = 16, 32, 64 with the same transformer predictor will fall further behind.

### Inputs

Existing per-encounter vorticity fields, existing JEPA latents for reference, existing physical-metric pipeline from Session 17 Experiment 2.

### Procedure

**Part (a) Fukami AE training.**

Train the lift-augmented autoencoder from Fukami and Taira PRF 2025 on your data at d = 64. The architecture follows their published implementation: convolutional encoder, convolutional decoder, with the lift coefficient appended to the latent as an auxiliary supervised signal during training. Train for the same number of epochs and on the same train split as the JEPA.

Train identical d = 3 and d = 32 variants for the head-to-head ladder, since Fukami’s headline claim is the d = 3 result and your D81 (Session 11/12) used d = 32.

Verify the trained Fukami AE achieves reconstruction quality comparable to the published results on representative encounters before proceeding. If reconstruction is poor, debug the implementation before moving to the predictor training step. Do not move forward with a Fukami baseline that does not match the published reconstruction quality on this flow class.

**Part (b) POD baselines.**

Compute POD on the train-split vorticity fields using the snapshot method. Truncate at d = 16, 32, 64. Project Test B and Test C encounters onto the truncated POD basis. This is closed-form and fast.

**Part (c) Common transformer predictor on each baseline.**

To compare predictive states fairly, all three baselines need the same predictor architecture trained on the same task. Train a transformer predictor (the same architecture you used in JEPA: AdaLN-Zero conditioning on (G, D, Y), max_seq_len = 32, RoPE positional encoding) on top of:

- Fukami AE d = 3 latents
- Fukami AE d = 32 latents
- Fukami AE d = 64 latents
- POD d = 16 coefficients
- POD d = 32 coefficients
- POD d = 64 coefficients

Each predictor is trained with the same loss, the same H_roll = 8, the same optimizer, and the same number of training steps. Do not retune hyperparameters per baseline; use the JEPA training recipe verbatim. The fairness of the comparison depends on this. Hyperparameter optimization per baseline is a separate, longer experiment and is out of scope for this submission.

**Part (d) Markov closure in physical observables on each baseline.**

For each of the six baseline + predictor pairs, run the Markov-only and Full-context rollouts on Test B and Test C, decode every frame, compute the physical metrics from Session 17 Experiment 2 (C_L, wake vorticity impulse I_y^w, integrated wake enstrophy, spectral lambda ratio at H = 16 and H = 32).

The headline table for the paper is a 7-row by 4-column comparison: JEPA d = 64 plus six baselines, rows being baselines and columns being the four physical metrics at H = 16 on Test B. A second table reports the same on Test C.

**Part (e) Epiplexity decomposition on each baseline (optional, time-permitting).**

Reuse the Session 14 D100 epiplexity pipeline to report per-token loss-curve area for each baseline. This extends D100 from “JEPA absorbs the dataset 2.16x more efficiently than Fukami AE at d = 32” to “JEPA absorbs the dataset more efficiently than Fukami AE at every tested d.” This is a one-paragraph result and one supporting table.

### Outputs

```
outputs/session18/exp_b1/fukami_ae_d{3,32,64}/checkpoint.pt
outputs/session18/exp_b1/fukami_ae_reconstruction_check.json
outputs/session18/exp_b1/pod_basis_d{16,32,64}.npz
outputs/session18/exp_b1/predictor_d{3,16,32,64}/checkpoint.pt
outputs/session18/exp_b1/physical_closure_comparison.csv
outputs/session18/exp_b1/epiplexity_comparison.csv
outputs/session18/figures/exp_b1_markov_closure_baselines.png
outputs/session18/figures/exp_b1_K_curve_baselines.png
```

### Compute estimate

Fukami AE training: 3 models × 4 hours per model on RTX 6000 = 12 hours.
POD computation: 1 hour.
Transformer predictor training on top of each baseline: 6 models × 3 hours = 18 hours.
Rollouts and physical metric computation: 6 hours.
Epiplexity computation: 4 hours.
Total: about 40 hours wall time, easily within one week.

### Acceptance gate

The experiment does not have a pass/fail gate because both outcomes are publishable. Two cases:

**Case A:** JEPA wins on physical Markov closure at H = 16 by more than 20 percent absolute error reduction on C_L and enstrophy. The paper’s headline claim becomes “JEPA learns a predictive state that Fukami AE and POD do not match on physical observables.” Section 5 of the paper is built around the comparison table.

**Case B:** Fukami AE matches JEPA on physical Markov closure (within 20 percent). The paper’s headline claim shifts to “JEPA and Fukami AE both achieve conditional Markov closure on this flow; JEPA additionally provides cross-seed reproducibility at the trajectory level, response-relevant non-Q structure discovery, and superior pressure observability.” The differentiation is on the secondary findings, not on the Markov closure itself.

Document the outcome before proceeding to manuscript drafting. The manuscript’s framing depends on which case holds.

-----

## Manuscript drafting workstream

The drafting workstream runs in parallel with Experiment B1 from Day 3 onward. Sections 1, 2, 3, 4, and 7 of the manuscript can be drafted without B1 results; Section 5 (Markov closure) requires the B1 comparison table; Sections 6 and 8 (structures and discussion) can be drafted independently.

### Section structure (target JFM length: 8000 to 10000 words excluding methods)

**Section 1. Introduction. Target 1200 words.**

Open with the problem: parametric vortex-gust airfoil interaction, the practical importance for transient aerodynamics in extreme weather and bio-inspired flight, the challenge of forecasting and control with limited sensor access.

Frame the gap. Cite Fukami and Taira PRF 2025 and JFM 2025 for the existence of low-dimensional aerodynamic response manifolds. Cite Solera-Rico et al. Nat. Commun. 2024 for beta-VAE-plus-transformer ROMs. Cite Cremades et al. Nat. Commun. 2025 for the response-relevant coherent structures methodology. Cite Linot et al. for predictive-state framings in chaotic flows.

State the contribution. The paper shows that a JEPA learns a low-dimensional impact state that is dynamically sufficient for predicting the post-impact wake; that this state is observable from sparse surface pressure with a nonlinear estimator; and that the input regions driving the state are not the strongest vortices in the flow.

Preview the structure. Five claims as listed above, with Sections 3 through 7 supporting each in order.

**Section 2. Methodology. Target 1500 words.**

The flow case: NACA 0012 at alpha = 14 degrees, Re = 5000, 2D mid-plane of a 3D DNS, parametric perturbation by an impacting vortex with strength G, diameter D, and lateral offset Y.

The JEPA: encoder, predictor, decoder. Architecture summary, training objective (SSL loss with the SL recipe), conditioning (AdaLN-Zero on (G, D, Y)). Reference the appendix for full architecture details. Emphasize that the decoder is for visualization, not for the JEPA training objective.

The evaluation pipeline. Physical observables: C_L, C_D, wake vorticity impulse I_y^w, integrated wake enstrophy E_w, spectral lambda ratio. Bootstrap confidence intervals at 95 percent over 2000 resamples.

Sensor selection. Consensus method from D112 (TCSI plus MI plus LASSO plus qDEIM), K = 2 to K = 16.

Baseline methods. Fukami AE at d = 3, 32, 64. POD at d = 16, 32, 64. Common transformer predictor for all. Justification for the fairness protocol (no per-baseline hyperparameter tuning).

Limitations of the data: 2D mid-plane omega excludes bound circulation, so Wu’s impulse-lift theorem is not directly applicable; the wake vorticity impulse I_y^w is reported instead. Train, Test A, Test B, Test C splits.

**Section 3. Trajectory geometry. Target 1200 words.**

Open with the trajectory panel figure. Three projections (PCA-impact, PCA-pool, PLS-supervised). Report variance explained.

The arc structure. Pre-impact trajectory length, post-impact length, the impact frame as the transition point. Sign(G) clustering at the impact frame (silhouette 0.59 to 0.61).

The curvature signature. kappa(t) DIPS at impact (not peaks). Speed peaks at impact (1.33x baseline in Test C). The physical interpretation: the impact frame is a saddle-like region of the slow manifold, traversed quickly along a near-linear trajectory.

Cross-seed agreement. Median Spearman of normalised distance matrices 0.95, 10 of 10 representative encounters above 0.7. The trajectory geometry is canonical at the basis-invariant level even though the linear coordinates are not.

**Section 4. State-functional alignment at impact. Target 1500 words.**

The state-functional alignment hypothesis: at the impact instant, the perturbation parameters, the local vorticity field, and the surface pressure footprint align into a single observable state. Before impact, the parameters describe a far-field perturbation invisible to the airfoil. After impact, the parameters partially dissipate into the chaotic wake.

The Y recovery curve. Asymmetric Gaussian decay sigma_L = 10, sigma_R = 54. The asymmetry is the key result: Y becomes recoverable at vortex contact and remains recoverable for one impact window.

The cross-seed function transfer failure. Each seed self-fits Y at R^2 0.42 to 0.70, but the kernel ridge regressor does not transfer across seeds (R^2 -0.45 to -7.5). This is the strongest seed-arbitrariness statement: not only the linear basis, but the nonlinear functional form of the extraction, is seed-specific. The Y information is present in the data reproducibly; the encoder’s representation of that information is not.

The SHAP attribution decay. Per-encounter heterogeneity; the LE-disk metric is too coarse to capture the variability. Report representative encounters where the attribution sharply peaks at tau = 0, and note that the population statistic is weaker.

**Section 5. Physical Markov closure. The centerpiece. Target 1800 words.**

This is the headline section. It has two figures: the horizon-vs-error plot comparing Markov-only, AR-from-impact, and Full-context on Test B and Test C across the four physical metrics; and the baseline comparison table.

Open with the key finding: Markov-only matches or beats Full-context at H = 16 on all three primary metrics (C_L, I_y^w, enstrophy) on both Test B and Test C. The pre-impact temporal context, even when available, contributes no information for short-horizon prediction; for the Markov-beats-Full case, it actively introduces error.

The physical interpretation. The post-impact dynamics is well-described by the state at the impact instant. Pre-impact frames describe a far-field perturbation that has not yet interacted with the airfoil; including them in the predictor’s context introduces commitment errors that the Markov rollout avoids.

The Wu’s theorem caveat. The 2D mid-plane omega excludes bound circulation, so dI_y^w/dt vs C_L correlation is -0.03 on DNS itself. This bounds the impulse-lift dynamical-consistency check. The wake vorticity impulse closure result is still meaningful; it shows the wake dynamics is predictable from the impact state.

The baseline comparison (Experiment B1 results). Depending on the outcome:

- Case A: JEPA beats Fukami AE and POD at every d on all metrics. The Markov closure is a JEPA-specific property and the paper’s centerpiece result.
- Case B: Fukami AE at d = 64 matches JEPA on Markov closure. The paper’s centerpiece result becomes “JEPA achieves Markov closure with additional properties (cross-seed reproducibility, response-relevant structure discovery) that Fukami AE does not.” Section 5 ends with the shared finding; Sections 6 and 7 carry the differentiation.

**Section 6. Coherent structures from attribution. Target 1500 words.**

Open with the four-target SHAP panel figure (centroid_x, circulation_pos, peak_neg_omega, Y).

The non-Q-overlap finding. Mean IoU < 0.2 across all four targets. The response-relevant structures localize at shear layers, body-vortex interaction zones, and wake transition regions. They are not the vortex cores that Q-criterion identifies.

The physical interpretation. The lift on the airfoil is generated by the boundary-layer shear and by the pressure footprint of the impacting vortex, neither of which is strongest inside the vortex core. The encoder learned, through the predictive task, that the regions controlling the future state are where the airfoil’s boundary layer interacts with the impacting vortex’s shear layer.

The Cremades et al. parallel. Their finding for wall-bounded turbulence: classical coherent structures only partially overlap with regions important for future flow evolution. Your finding: the same holds for vortex-airfoil impact in a strongly transient parametric setting. Discuss the implications for active flow control: the SHAP structures identify the targets, not the Q-criterion vortices.

The Y structures. Asymmetric LE-region pattern with sign-flipping attribution. The Y sign-flip claim holds at the attribution-map sign level; not at the connected-component centroid level. Be precise about this distinction; reviewers will catch the difference.

Threshold sensitivity. 98th percentile is the sweet spot; structures are stable within +/- 1 percent. Report the sensitivity curve to preempt the reviewer question.

**Section 7. Closed-loop pressure observability. Target 1500 words.**

Open with the K-curve figure: physical-metric error as a function of K for Mode A (oracle), Mode B (pressure-z, oracle c), and Mode C (full pressure).

The headline result. At K = 8 with MLP-reg (or K = 16 with TCN), Mode C tracks Mode A within factor 0.7 to 1.3 in absolute physical-metric error. The deployment loop closes for forces, vorticity impulse, and wake enstrophy.

The nonlinear-estimator necessity. Linear ridge gives z R^2 = 0.034; MLP-reg gives 0.92. Report this clearly: the pressure-to-latent map is genuinely nonlinear, and the deployment story requires a nonlinear estimator. This is a methodological point worth emphasizing.

The non-monotonicity at K = 2. C_L factor 0.91 at K = 2 is better than at K = 4 or K = 8. Acknowledge this honestly and offer the most likely interpretation: K = 2 sensors at the leading edge cluster (sensors 11, 20 from D112) capture the dominant lift-relevant signal cleanly; intermediate K values add sensors that introduce noise to the estimator. Frame this as a finding worth investigating further, not a flaw to hide.

The Test C limitation. Pressure-to-z R^2 is negative at G = +4 OOD. The deployment fails out-of-distribution. Report this as a clean limitation; the paper does not claim OOD deployment.

**Section 8. Discussion. Target 800 words.**

Three subsections.

Subsection 8.1: The state-functional alignment as a fluid-mechanics statement. The impact instant is a privileged dynamical state, geometrically traversed as a saddle-like smooth pass-through, where the perturbation parameters and the post-impact dynamics meet in a single observable state. This is a statement about the dynamical system, not the JEPA architecture; the encoder discovered it.

Subsection 8.2: Connections to wall-bounded turbulence structure discovery. The Cremades et al. parallel. The non-Q-overlap finding suggests that response-relevant coherent structures (the regions important for predicting future flow evolution) are systematically different from classical vortex structures, in both wall-bounded turbulence and transient vortex-airfoil interaction. This may be a more general phenomenon.

Subsection 8.3: Limitations and future directions. The 2D mid-plane data limitation for Wu’s theorem. The single Re. The single airfoil. The seed-arbitrariness of the nonlinear extraction function, which bounds per-coordinate interpretability of JEPA latents. Extensions: other Re, other airfoils, 3D DNS data, active flow control using the SHAP structures as targets.

**Methods appendix. Target 2000 words.**

Full JEPA architecture: encoder backbone, latent dimension, predictor, decoder. Training: optimizer, learning rate schedule, batch size, total iterations, hardware. SSL loss formulation. SL decoder recipe with full hyperparameters.

Fukami AE implementation. POD computation. Common transformer predictor for baselines.

Sensor selection: TCSI, MI, LASSO, qDEIM, consensus method, sensor coordinates.

Physical observables: C_L from surface integration, C_D analogously, wake vorticity impulse I_y^w with the explicit integration domain, integrated wake enstrophy, spectral lambda ratio computation.

Bootstrap: 2000 resamples, 95 percent CI, paired or independent depending on the comparison.

Cross-seed protocol: Thrust 6 seed retrains, latent standardization, distance-matrix Spearman correlation.

Gradient-SHAP: 32 integration steps, phase-matched baseline, connected-component extraction at 98th percentile, intervention validation with Gaussian-blurred inpaint at sigma = 3 grid cells.

Closed-loop pipeline: pressure-to-z estimator (MLP-reg architecture), pressure-to-c estimator, Mode A/B/C rollout protocol.

-----

## Figure plan

The paper needs eight main-text figures, plus supplementary figures.

**Figure 1.** Problem and JEPA schematic. Vortex-gust impact geometry, parametric perturbation, JEPA encoder-predictor-decoder pipeline. One panel.

**Figure 2.** Trajectory geometry. Three-panel: 3D trajectory visualization with sign(G) clustering, curvature signature kappa(t) aligned by impact with the dip clearly visible, cross-seed distance Spearman per encounter (the 10 bars above 0.7).

**Figure 3.** State-functional alignment. Two-panel: parameter recovery R^2(tau) for G, D, Y with the asymmetric Gaussian fit for Y; cross-seed function transfer heatmap.

**Figure 4. The centerpiece.** Physical Markov closure. Four-panel grid: C_L absolute error vs horizon for Markov / AR / Full on Test B and Test C; same for wake enstrophy; same for I_y^w; spectral lambda ratio at H = 16 and H = 32. With shaded 95 percent bootstrap CIs.

**Figure 5.** Baseline comparison. Markov closure absolute error at H = 16 for JEPA d = 64 vs Fukami AE d = 3, 32, 64 vs POD d = 16, 32, 64. One panel per physical metric (C_L, I_y^w, enstrophy). Three panels total. Bars with error bars.

**Figure 6.** Coherent structures from SHAP. Four-target panel (centroid_x, circulation_pos, peak_neg_omega, Y). For each, three subpanels: the vorticity field, the SHAP map, the extracted connected components overlaid on the field. Q-criterion contours overlaid for direct visual comparison. This is the magazine-cover figure.

**Figure 7.** Q-overlap and Y sign-flip. Two-panel: IoU and overlap fraction distributions for each target; Y structure centroid as a function of Y with bootstrap CIs.

**Figure 8.** Closed-loop pressure observability. Two-panel: physical-metric error vs K for Mode A, B, C across the three primary metrics; sensor locations on the airfoil with the consensus selection highlighted.

Supplementary figures (S1 through S6): the conditioning paradox z-norm histograms, the full curvature profiles, the per-encounter trajectory panel for all 10 representative cases, the threshold sensitivity for SHAP structures, the K = 2 non-monotonicity diagnostic, the Test C closed-loop results.

-----

## Execution timeline

**Week 1: Experiment B1 Fukami AE and POD baseline comparison.**
Days 1 to 2: Fukami AE training at d = 3, 32, 64.
Day 3: POD basis computation.
Days 4 to 5: Transformer predictor training on top of each baseline.
Days 6 to 7: Rollouts, physical metric computation, comparison table.

**Week 2: Manuscript drafting first pass.**
Day 1: Section 2 (Methods, the most factual section, start here to build momentum).
Day 2: Section 3 (Trajectory geometry).
Day 3: Section 4 (State-functional alignment).
Day 4: Section 5 (Physical Markov closure), now with B1 results.
Day 5: Section 6 (Coherent structures).
Days 6 to 7: Section 7 (Closed-loop pressure) and Section 8 (Discussion).

**Week 3: Manuscript second pass plus Introduction.**
Day 1: Section 1 (Introduction, written last after all other sections are clear).
Days 2 to 5: Second pass through all sections for narrative flow, transitions, citations.
Days 6 to 7: Methods appendix.

**Week 4: Figure polish.**
Days 1 to 3: Main-text figures (1 through 8) at submission quality.
Days 4 to 5: Supplementary figures.
Days 6 to 7: Figure captions, integration with the manuscript.

**Week 5: Co-author review and revision.**
Send the manuscript to co-authors (Carolina if she contributes to the methodology section, any DNS collaborators for the data section, any consulting input). Allow one week for feedback. Revise.

**Week 6: Final polish and submission.**
Day 1: Final pass for typos, references, and figure quality.
Day 2: Submission preparation (cover letter, suggested reviewers, exclusion list, data availability statement, code repository link, supplementary materials package).
Day 3: Submit.
Days 4 to 7: Buffer for unforeseen issues.

-----

## What this session does not address

- No conditioning-dropout retrain of the predictor. The cond = 0 paradox is a one-paragraph discussion item, not a paper section.
- No cross-seed SHAP. Interesting but not necessary for the JFM submission; a follow-up paper or revision-stage addition.
- No persistent homology. Out of scope.
- No 3D DNS reanalysis to fix the Wu’s theorem issue. The 2D limitation is acknowledged and the wake vorticity impulse is reported as the alternative.
- No extension to other Re or other airfoils. These are the obvious next paper.

-----

## Decision points within the session

**End of Week 1 (after B1 results):** Decide which case the paper is in.

- Case A (JEPA beats Fukami AE on Markov closure): Section 5 leads with the comparison table, Sections 6 and 7 are supporting.
- Case B (Fukami AE matches JEPA on Markov closure): Section 5 reports the shared closure result, Sections 6 and 7 carry the differentiation.

Update the Section 5 outline and the Figure 5 design based on the outcome before drafting.

**End of Week 3 (after first complete draft):** Decide on venue.

- If the manuscript reads as a fluid-mechanics paper with broad cross-domain implications, submit to JFM.
- If the cross-domain implications dominate the narrative and the fluid-mechanics-specific findings feel like supporting evidence, consider Nat. Commun.
- Default: JFM, unless the manuscript itself argues for Nat. Commun.

**End of Week 5 (after co-author review):** Final venue lock and submission preparation.

-----

## D-entries to land in HANDOFF.md

- **D129:** Fukami AE and POD baseline comparison on physical Markov closure (Experiment B1).
- **D130:** Manuscript first draft completed (end of Week 2).
- **D131:** Figures at submission quality (end of Week 4).
- **D132:** Co-author review completed (end of Week 5).
- **D133:** Submission to JFM (or other venue per Week 3 decision).

-----

## Reproducibility

All scripts in scripts/session18/. The Fukami AE implementation should be a clean reimplementation of the published architecture, with the trained weights saved at outputs/session18/exp_b1/fukami_ae_d{3,32,64}/checkpoint.pt for reviewer verification.

The manuscript LaTeX source goes in manuscript/. Figures in figures/. Supplementary materials in supplementary/.

The code repository, when submitted, includes only the inference and evaluation scripts plus the trained-checkpoint download links, not the full training pipeline. The training pipeline is in the project repo, separately referenced.

-----

## Honest readiness assessment

You are ready to draft. The five sections of the paper have their evidence from Sessions 11 through 17. The one missing experiment (Fukami baseline) is well-scoped and one week of work. The manuscript drafting workflow is standard and the timeline is realistic.

Two risks worth flagging:

**Risk 1: The Fukami AE implementation may not match published reconstruction quality.** This is a common issue with reimplementing published architectures. Allocate buffer time in Week 1 for debugging. If the implementation does not match by end of Week 1, defer the baseline comparison and submit without it; the paper can claim that Fukami AE comparison is a follow-up, with the cross-seed and structure-discovery findings as the primary differentiation. This is a softer but still publishable paper.

**Risk 2: Co-author review may surface major revisions.** Build the buffer into Week 5. If major revisions are needed, the submission slips by 2 to 3 weeks; this is acceptable.

The session is realistic for a six-week submission window. The paper is strong enough for JFM. Begin Week 1 with Fukami AE implementation.

-----

## References

1. Fukami, K., Taira, K. “Grasping extreme aerodynamics on a low-dimensional manifold.” Phys. Rev. Fluids 10, 084703 (2025).
1. Fukami, K., Taira, K. “Compact reduced-order modeling of nonlinear vortical flows over time and parameter space.” J. Fluid Mech. 1021, A39 (2025).
1. Cremades, A., Hoyas, S., Vinuesa, R. “Classically studied coherent structures only paint a partial picture of wall-bounded turbulence.” Nat. Commun. 16, 10189 (2025).
1. Solera-Rico, A., et al. “β-Variational autoencoders and transformers for reduced-order modelling of fluid flows.” Nat. Commun. 15, 1361 (2024).
1. Finzi, M., Qiu, S., Jiang, Y., Izmailov, P., Kolter, J. Z., Wilson, A. G. “From Entropy to Epiplexity: Rethinking Information for Computationally Bounded Intelligence.” arXiv:2601.03220v2 (2026).