# Physics Track P5: where the wake code lives (v2.1)

Three honest re-measurements on the v2.1 UNCONDITIONED latents (jepa_tf_noc / fukami / pod at d=64). D162/D163 are NOT assumed; every number below is AS MEASURED on v2.1. Where a result weakens vs the v2 (D162/D163) story it is flagged.

## Subspace-overlap null (CLAUDE.md)

The random-baseline mean cos^2 for two K-dim subspaces in d-dim ambient is K/d (~0.047 for K=3, d=64). Any pairwise cos^2 within ~0.01 of K/d is NOT overlap and must not be read as functional sharing. The grouping result below scores its leading-PC subspace overlap against this null.

## (i) Collective vs redundant code (D163 regenerated)

Per family, toward future wake enstrophy at H=16 on held-out test_b (targets joined by (case_id, encounter)): combination skill (ridge over all 64 coordinates, standardised on train), best single-coordinate skill, the GAP (combo - best single), forecast-beyond-forces partial correlation, and #coordinates with single |Spearman| > 0.5. A LARGE positive gap means the wake forecast is a COLLECTIVE (distributed) code, not carried by one coordinate.

- jepa_tf_noc: combo 0.801, best-single 0.442, GAP +0.359, n_strong 0, beyond-forces 0.791 (n_test_b=2688).
- fukami: combo 0.601, best-single 0.553, GAP +0.047, n_strong 1, beyond-forces 0.571 (n_test_b=2688).
- pod: combo 0.565, best-single 0.525, GAP +0.040, n_strong 1, beyond-forces 0.551 (n_test_b=2688).

JEPA seed robustness: gap +0.402 +- 0.091 over 4 seeds.


Functional grouping (jepa, D163 ~2 functions): 12 wake-active coordinates split into groups of sizes [3, 9]; leading-PC subspace cos^2 = 0.000 vs K/d null 0.016 -> overlap above null: False (the two groups' leading directions are at or below the K/d null, i.e. functionally distinct / not sharing a subspace -- consistent with D163's ~2 separate functions).


Paired jepa vs fukami wake forecast (sign test + case-clustered CI, NOT case-permutation): jepa errs less in 1323/2688 held-out rows (one-sided sign p = 0.797); mean |error| advantage -0.001 (case-clustered 95% CI [-0.061, 0.060] over 10 cases).


**D163 status (i):** the v2 manuscript reported the JEPA wake forecast as a collective code (large gap). On v2.1 the JEPA gap is +0.359: the collective-code result REPRODUCES.

## (ii) Energy-information curve (new)

Held-out wake-forecast R^2 (ridge on train, evaluated on test_b) vs the number of leading latent PCs retained (PCA fit on train post-impact latents). The KNEE is the smallest #PCs reaching 90% of the asymptotic (max-PC) skill. This quantifies how many latent PCs each family needs to CARRY the wake -- the participation-ratio-vs-distributed-code tension made numeric.

- jepa_tf_noc: knee at 32 PCs (asymptotic R^2 0.290).
- fukami: knee at 12 PCs (asymptotic R^2 0.254).
- pod: knee at 1 PCs (asymptotic R^2 0.087, retained energy 10.3%).

## (iii) Footprint of the wake-forecast direction (new; supersedes D163 decode)

The wake-forecast DIRECTION (the standardised ridge weight vector of the H=16 wake probe) is back-propagated through the FROZEN jepa encoder: saliency = |d (z . u) / d (input omega)|, averaged over 66 held-out impact-frame inputs (device: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition). The mean saliency map is energy-weighted overlap-scored against the thresholded |omega| mask and the Q-criterion vortex mask, vs a spatial-permutation chance baseline.

Q-criterion source: TRUE 2D Q from raw /u velocity (PREVENT_ROOT), reusing the _oneoff_pod_q_overlap / exp4_structures_shap form.

- |omega|: overlap 0.070 vs chance mean 0.099 (95% null [0.096, 0.102]), z = -18.90 -> **BELOW chance**: the footprint overlap sits below the lower null bound, so the wake-forecast saliency actively AVOIDS this structure (it lives off the high-magnitude cores / vortices, e.g. on shear-layer edges and lower-amplitude regions).
- Q: overlap 0.010 vs chance mean 0.011 (95% null [0.011, 0.012]), z = -5.69 -> **BELOW chance**: the footprint overlap sits below the lower null bound, so the wake-forecast saliency actively AVOIDS this structure (it lives off the high-magnitude cores / vortices, e.g. on shear-layer edges and lower-amplitude regions).

**D163 status (iii):** D163 claimed qualitatively that the code reads the LEV / shear layer. The number above is the honest test: NEITHER mask clears the chance null on v2.1 (both overlaps sit AT or BELOW chance), so the 'reads the LEV / shear layer' sentence is NOT supported by the saliency footprint and must be dropped. If anything the wake-forecast direction reads the input OFF the high-|omega| cores and Q vortices, consistent with the day's meta-pattern that latent-image / decode claims do not survive fair scrutiny on v2.1.

## Statistics + provenance

Held-out test_b (and test_b+test_c for the footprint); probes fitted on train post-impact frames only. Case-clustered conventions via stats_lib. The footprint overlap uses a spatial-permutation null (not a paired location test). Energy-information knee is a trend measure. Pipeline divisor 3*train_std = 10.9010 (split_v2p1). JEPA encoder checkpoint: outputs/runs/session28/jepa_tf_noc_d64_s42/encoder/checkpoint_iter020000.pt.
