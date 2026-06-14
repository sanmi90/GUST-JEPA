# SESSION29 gate-review notes (my critical review of each track, reconciled to v2.1)

Not the subagents' self-reported verdicts verbatim: my scrutiny of each, with
reconciliation flags for the Track J/K integration. (Track D logic had to be
fixed, so no returned branch is taken at face value.)

## Track D (probe-class, F4): WEAK -- accepted. Claim stays LINEAR. (D197)

## Track B0 (clip leakage, F7): WEAK but immaterial at readout. (D198)

## Track G (stronger floors, F5/F6): STRONG -- accepted, with one flag.
- Predictive latent readout wake R^2 = +0.677 clears all four implemented floors:
  gdy +0.44, gdy_history +0.25, pressure_only +0.24, persistence +0.15. Latent
  also > fukami (+0.45) and pod (+0.32). Conclusion robust (latent clears floors
  under any reading).
- FLAG (reconcile at integration): G's bare-`gdy` floor (+0.44, nested
  GroupKFold alpha selection) differs from the PUBLISHED parametric wake floor
  `NumParamFloorWakeLinear` = -0.18 / `NumParamFloorWakeKrr` = -0.12 (same
  (G,D,Y)->wake, H16, test_b). NOT a true contradiction: the published floor's
  case-clustered CI is [-1.83, +0.77], which CONTAINS +0.44 -- the point-estimate
  gap is alpha-selection regime (CV-tuned vs fixed), and 10-case CIs are huge. DO
  NOT print both as "the (G,D,Y) floor": keep the published -0.18 as the bare
  parametric floor (tab:parametric_floor) and present Track G as the ADDITIONAL
  stronger floors (history/persistence/pressure), all cleared by the latent. The
  latent clears the parameter floor under both estimates.

## Track H (mechanism corroboration): WEAK -- accepted; CONFIRMS the paper, with a prose flag.
- Metric-independent diagnostics (kNN-distance, local-PCA residual) put the
  RECONSTRUCTIVE (fukami) rollout CLOSEST to the training manifold, JEPA farther,
  POD farthest -- the REVERSE of a "AE leaves the manifold more" magnitude story.
- This is NOT a contradiction: the paper ALREADY frames it as "small Euclidean
  drift of the reconstructive rollout with its large Mahalanobis departure ...
  along near-null directions" (intro:156, results:264-265). H's Euclidean-family
  result IS that "small Euclidean drift" half. The discriminating claim is the
  near-null departure SPECTRUM / fraction (sign p ~2e-13), which is direction-
  resolved and survives; the Mahalanobis MAGNITUDE is metric-dependent (fukami's
  tiny footprint lands in collapsed directions; dividing by near-null variance
  inflates Mahalanobis).
- FLAG (Track J prose): drop the "by an order of magnitude" / "leaves the manifold"
  MAGNITUDE language -- abstract.tex:29 ("its rollout leaves the manifold"),
  section_1_introduction.tex:153 ("leaves its training manifold by an order of
  magnitude"), section_4_results.tex:260 ("an order of magnitude"). Replace with
  the near-null-direction / departure-fraction framing the headline already uses.
  Add H's kNN + local-PCA as the conservative metric-independent check that
  motivates the direction-only claim. tab:latent_drift keeps rel-l2 + near-null
  frac; the Mahalanobis column stays only with the explicit near-null caveat
  already in the caption.

## Track A (baseline external validation, F1): accepted -- de-risks Table 1.
- Baseline forces in a defensible band of the lineage refs: mean C_L 0.761 vs
  Fukami 0.737 (+3.3%) / Rolandi 0.734 (+3.7%) / Gupta 0.763 (-0.3%) = IN BAND;
  mean C_D 0.253 within 1.7% of Fukami's production DNS. St 0.675 (internal).
- Honestly flagged: rms C_D +21.5% vs Fukami (beyond grid scatter; resolution/
  window sensitivity), rms C_L +9.6%. Rolandi/Gupta rms = NEEDS-LITERATURE (only
  their mean forces are published in the figure; NOT fabricated). Values sourced
  from repo-local FukamiGustRe5000.pdf.
- Use: a baseline-validation paragraph/row in S2.2 next to Table 1, reporting the
  mean-force agreement and flagging rms; submission still blocked on the partner
  solver-resolution rows + convergence panel, but the data itself is now
  externally validated.

## Track C-min (case-level slopegraph, F3): WEAK -- accepted, matches the paper.
- Headline representational wake result, per CASE (10 test_b cases), jepa vs
  fukami: JEPA better in 7/10. Encounter-level sign p=0.004; case-level Wilcoxon
  signed-rank p=0.024 (sig); conservative exact case-level sign test p=0.17 (NOT
  sig). Median per-case delta +38.2. Exactly the v2.1 GD-weak state.
- Slopegraph `outputs/session29/fig_case_slopegraph.pdf` (3/10 cases favour
  recon; green=jepa wins, red=recon) is the honest per-case figure that should
  replace the encounter-level bars (Track J figure plan).
- DATA-PATH NOTE (important for D/G): the HEADLINE closure probe is fit on
  POOLED-over-frames latents with DNS ground truth from `dns_physical_metrics.npz`
  (subagent reproduced the published dump to ~2e-11). The `per_frame_targets`
  wake_enstrophy used by `_s29_common.readout_xy` (Tracks D, G) is a DIFFERENT
  scale; R^2 is invariant under an affine target transform for a refit
  ridge-with-intercept probe, so D/G R^2 stand, but they are READOUT-FRAME numbers
  (jepa linear ~0.58), NOT the pooled-frame headline (0.79). Keep that distinction.

## Track I (causal pressure window, F6): STRONG -- accepted; re-base sensing on the causal window.
- State recovery (test_b, K=8 qDEIM placed on the strictly-pre-impact snapshot,
  KRR-RBF, nested GroupKFold-by-case, case-clustered CI):
  - `preimpact_m30_to_m1` (STRICTLY CAUSAL): jepa +0.822 > fukami +0.765 > pod
    +0.224 -- predictive most recoverable. CLEANER than the non-causal window.
  - `through_impact_m29_to_0`: jepa +0.830 > fukami +0.769 > pod +0.230.
  - `readout_m13_to_p16` (current NON-causal): jepa +0.625 < fukami +0.762 -- the
    ordering FLIPS (post-impact frames feed the reconstructive AE's
    reconstruction-aligned latent).
- The F6 critique is resolved: the predictive ordering not only survives a strictly
  causal pre-impact window, it STRENGTHENS. Params [G,D,Y] R^2 = 0.934, impact C_L
  R^2 = 0.778 (causal). Wake-via-latent stays the NON-RESULT (jepa +0.22, CIs span
  below 0; no claim).
- FLAG (Track J): RE-BASE the sensing headline on the causal `preimpact_m30_to_m1`
  window. The current SenseState* macros (jepa 0.78) are from the non-causal
  readout window with the paper's own taps; Track I uses causal-snapshot qDEIM +
  nested CV, so do NOT read the readout-window flip as "0.78 is wrong" -- read it
  as: a consistent causal placement makes the causal window both more defensible
  AND more favorable (jepa 0.82). Update the sensing numbers to the causal window
  (regenerate via pressure_causal.py -> numbers_parts), state the result strengthens
  under causality, keep the wide-10-case-CI caveat. This is a net WIN for the paper.

## Track B0.5 (frozen clip sensitivity, F7): STRONG -- closes F7, no B1 retrain.
- JEPA wake R^2: none 0.746 / per_encounter 0.796 / training_global 0.842; baselines
  fukami ~0.1, pod ~0 throughout. JEPA-minus-best-baseline advantage +0.63 / +0.70 /
  +0.77; sign preserved, max shift from the current pipeline 0.07 (< 0.10 tol).
  Enstrophy decreases monotonically as the clip tightens (sanity ok). The B0 leak
  does NOT move the wake headline -> B1 retrain unnecessary; report wake on
  physical units with this sensitivity as the robustness statement.
- KEY CATCH (target definition): B0.5 recomputed wake_enstrophy with the CANONICAL
  cache-box convention (`scripts/session16/exp2_build_targets.py`, the actual
  builder of the stored targets) and its per_encounter readout jepa R^2 = 0.796
  REPRODUCES the published 0.79 headline. So `_s29_common.readout_xy`'s
  `per_frame_targets` wake target (used by Tracks D and G, giving jepa ~0.58/0.68)
  is a DIFFERENT/degraded target. D and G internal comparisons hold (same target
  across families), but Track D's GBM-reversal verdict could be target-sensitive
  -> RE-RUN Track D on the canonical target to confirm before pinning title
  wording. G is STRONG by a wide margin (robust to target choice). The paper's
  "linear decodability" framing is conservative and safe under either D outcome.

## Track D CORRECTION (canonical target): STRONG, supersedes the D197 WEAK.
Re-ran probe_class_sweep with `--target-source canonical` (dns_physical_metrics,
the published-headline wake target). The predictive latent leads under EVERY probe
class by a wide margin (test_b wake R^2):
- linear: jepa +0.796 (= the published 0.79) > fukami +0.095 > pod -0.201
- kernel-ridge: jepa +0.846 > fukami +0.009 > pod -0.249
- mlp: jepa +0.402 > fukami -0.398, pod +0.172
- gbm: jepa +0.681 > fukami +0.069 > pod -0.138
VERDICT STRONG (no reversal under any probe). The GBM reversal reported in D197
was an ARTIFACT of the degraded per_frame_targets wake target (which gave jepa only
0.58 and let tree ensembles extract comparable signal from the baselines). On the
correct target the wake is readable from the predictive latent by a BROAD probe
class (linear, kernel, neural, tree), not merely linear -- so the manuscript's
"linear decodability" framing is CONSERVATIVE and could be upgraded to "broad probe
class" (still report the GBM/MLP point estimates and the wide 10-case CIs). This
supersedes D197. Outputs: probe_class_matrix_canonical.json /
probe_class_sweep_canonical.md. (Track G's STRONG holds a fortiori on the canonical
target, where the latent is ~0.80; G need not be re-run.)
