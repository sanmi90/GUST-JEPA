# Reviewer claim map (SESSION29.9)

Every claim the manuscript makes, the evidence it rests on, and the limitation
stated alongside it. Numbers are macro-bound (`paper/macros.tex`); see the cited
sections. This map reflects the honest (predictive-state design-study) branch
adopted after the C-full grouped-CV result.

| Claim | Evidence | Limitation stated |
|---|---|---|
| The wake, not only the scalar force, is the right endpoint | Figure 1 (staging), wake-enstrophy definition (sec 2), negative parametric floor on wake at H=16 (sec 4.1, `\NumFloorWake`) | The scalar forces do not exhaust the wake state; the wake observables are mid-plane and omit out-of-plane content |
| Wake-readable reduced states exist | Representational closure, predictive `\NumReprWakeJepaTf` vs no-wake baselines at/below zero (Table `tab:closure`(a)) | An encoded-field test, not a forecast; the separation is against the established no-wake baselines |
| Wake supervision supplies the instantaneous readability | Objective-free supervision-only encoder reads the wake as well (`\NumReprWakeSupOnly`); matched reconstructive control reads it comparably under case-disjoint retraining (`\NumCvFullReconMedian` vs `\NumCvFullJepaMedian`) (sec 4.2) | The predictive objective is NOT credited with making the wake readable |
| Anti-collapse supplies the in-distribution (on-manifold) geometry | supervision-only drift row is the most in-distribution (`\DriftMahaSupOnly`, near-null `\DriftSpecNearNullSupOnly`); shared with the predictive latent (sec 4.3, `tab:latent_drift`) | On-manifold geometry is not unique to the predictive objective |
| The predictive objective supplies dynamic usability | Lower relative-L2 rollout drift than supervision-only (`\DriftRelLtwoJepa` vs `\DriftRelLtwoSupOnly`); forecasts the wake where supervision-only and reconstruction do not (`\NumFcstWakeJepaReadout` vs `\NumFcstWakeSupOnly`); stable across retrains (`\NumCvFullJepaMin`-`\NumCvFullJepaMax`) (sec 4.2, 4.3) | The forecast is a rollout diagnostic, not a validated forecast |
| Reconstruction is unreliable for predictive use | Matched reconstructive control erratic across folds (`\NumCvFullReconMin`-`\NumCvFullReconMax`); near-null manifold departure under rollout (`\DriftSpecNearNullFukami`) (sec 4.2, 4.3) | Reconstruction can read the wake when explicitly supervised, and is useful for field recovery |
| Sparse wall pressure recovers the predictive state best | Cross-family state recovery at K=8, causal pre-impact window (`\SenseStateCausalJepaTf` vs others) (sec 4.7, App. B) | State recoverability, not wake recovery: pressure-to-wake is weak (`\SensePWakeJepaTf`); not a validated wall-pressure wake forecast |
| The no-gust shedding orbit is a single coherent loop in the predictive state | Whitened persistent homology, no-gust only (`\TopoBaseWhtMedJepa` vs `\TopoBaseWhtMedFukami`) (sec 4.3, App. A) | Gusted-encounter topology is metric-sensitive and NOT claimed |

## Scope boundaries (stated once, sec 5.2 / sec 6)
- Wall-normal offset Y is weakly resolved (training mass near Y=0).
- |G|=4 is a three-dimensional observability boundary, not a clean extrapolation.
- The rollout error exceeds the tolerance a closed-loop control demonstration would
  require; no validated forecast, controller, world model, or counterfactual model.
- DNS Table 1 resolution rows are partner-owned and gate submission until filled.
