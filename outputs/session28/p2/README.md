# Session 28 / Physics Track P2: the latent recovery clock and Gate GP2

The physical recovery clock tau_rec (v2: wake enstrophy re-enters the settled-
Baseline band and dwells one period) is read from
outputs/session28/physics/per_encounter_physics.npz. This script computes the LATENT
recovery clock from the predictive (jepa_tf_noc) latent trajectories and correlates
it with the physical one (Gate GP2). NOT git-committed; left for review.

## Latent tube

- Baseline case: Baseline; settled orbit points: 80 (effective dim 3.6 of 64).
- Mahalanobis tube radius (q95): 7.846.
- Euclidean cross-check radius (q95): 0.284.
- Recovery rule: occupancy window 56 frames (2.80 t/c), theta 0.8, min-window 28. Latent recovery uses physics_prep.occupancy_recovery on the tube-membership signal, MATCHED to the physical v2 clock (theta=0.8, window=56, min_window=28). The subharmonic shedding period is 1/(St_sub*dt_tc)~=59 frames (St_sub=0.3383); the physical clock rounds this to a 56-frame occupancy window and we keep that window for parity.
- The orbit is very low-rank, so the Mahalanobis whitening is ill-conditioned (see
  s46_regen). The gate statistic is a RANK correlation (insensitive to whitening
  magnitude) and a well-conditioned Euclidean tube is carried as a cross-check.

## Gate GP2 (Spearman of latent vs physical tau_rec, recovered-in-both, test_b)

- HEADLINE (Mahalanobis tube): Spearman rho = nan (n = 0; fewer than 3 recovered-in-both pairs; Spearman undefined).
- Euclidean cross-check: Spearman rho = nan (n = 0; fewer than 3 recovered-in-both pairs; Spearman undefined).

GP2 threshold = 0.7. **VERDICT: NOT EVALUABLE (too few recovered-in-both encounters)**.

latent-clock sentence DROPPED; the latent clock recovers too few encounters for a meaningful Spearman (recovered-in-both n=0). Physical DNS maps stand as the contribution.

Stated plainly: the latent recovery clock recovers too few test_b encounters
(0 recovered-in-both with the physical clock) for a
meaningful Spearman, so GP2 is not evaluable as a correlation. The
latent-clock sentence is DROPPED from S4.4. The physical tau_rec(G, D, Y) maps
stand as DNS physics and remain a contribution. See the diagnostic below for
the mechanism (the latent never re-enters the tight baseline tube).

## Latent closest-approach diagnostic (test_b)

Why the latent clock recovers as it does: the settled-Baseline orbit is a thin,
low-rank tube (Euclidean diameter 1.63 latent units; the
q95-self-spread tube radius is 0.17 orbit-diameters
Euclidean). The gusted test_b latents approach it only weakly:

- median minimum post-impact distance to the orbit: 5.32 orbit-diameters (closest any encounter gets: 0.70).
- median minimum post-impact Mahalanobis-to-distribution: 510.4 (tube radius 7.8); the
  whitening inflates off-orbit directions, so the Mahalanobis tube is unreachable.
- median relaxation ratio (last-frame / impact-frame distance): 0.86 (the latent barely drifts back).
- encounters reaching within 1 / 2 orbit-diameters at any post-impact frame: 4 / 5 of 42.

This is the honest mechanism, not a coding artefact: the latent does not return to
the baseline limit-cycle tube within the 120-frame window, consistent with the short
release cadence (D153) that censors most of the PHYSICAL clock too, and with the s46
Q2 finding that the predictive latent does not sit close to the baseline orbit.

## Recovered vs censored fractions (test_b)

| clock | n | recovered | censored | frac recovered |
|---|---|---|---|---|
| physical (v2) | 42 | 5 | 37 | 0.12 |
| latent (Mahalanobis) | 42 | 0 | 42 | 0.00 |
| latent (Euclidean) | 42 | 0 | 42 | 0.00 |

Status taxonomy: 'recovered_short_window' counts as recovered (a dwell was certified
on the available remainder); only 'censored' is censored. Censoring is never silently
excluded; encounters censored under EITHER clock are simply not in the paired
correlation, and the counts above make that explicit.

## Map data

Physical tau_rec(G, D, Y) over test_b/test_c and tau_rec vs |G| are in results.json
(physical_map_data) and drawn in fig_p2_recovery.{pdf,png} (left panel: the envelope
map; right panel: the latent-vs-physical clock scatter with the Spearman).
