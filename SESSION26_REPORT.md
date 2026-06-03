# Session 26 report: referee hardening of the JFM manuscript

Date: 2026-06-03. Scope: re-analysis of cached outputs, rewriting, and a
reproducibility package, in response to an external JFM-referee-grade review of
`paper/main.pdf`. No model training was performed. The paper runs on split v2.

## Headline outcome

The single most important result of this session is honest, not preservative: under
case-level clustering and a family-wide multiplicity correction, the **forecast**
wake-enstrophy advantage weakens to non-significant, while the **representational**
wake advantage and the entire mechanism chain (drift, topology, optimal transport,
scale decomposition) survive. The manuscript has been re-anchored accordingly: the
wake claim now rests on representation plus mechanism, with the forecast presented as
consistent confirmation rather than the load-bearing test, and the paper is reframed
around a fluid-mechanics principle (a transport-consistent latent metric) rather than
a representation-learning horse race.

## Gate outcomes

| Track | Gate | Outcome |
|-------|------|---------|
| 0 | Clean baseline build, artifact map, split + case map, error-array path | PASS |
| 1 | Case-clustered stats, Holm, floor; stats_summary | PASS (claims changed, see below) |
| 2 | Persistent-homology threshold + sampling robustness | PASS (softened "decisive") |
| 3 | Iy non-impulse, 2D-proxy chi_3D, omega_c sensitivity | PASS |
| 4 | Decoder confound: S4.6 scoped to the encode-decode ceiling at large scale | PASS |
| 5 | Baseline-tuning transparency (non-monotonicity, seed spread, AE config) | PASS |
| 6 | Headline reframe to the transport-consistency principle | PASS |
| 7 | World-model demoted to motivation; abstract ends on the positive result | PASS |
| 8 | Closed-loop pilot cut to an honest scope statement (no compute) | PASS |
| 9 | Economy + internal consistency (Table 4a R^2 column; consistency fix) | PASS |
| 10 | Reproducibility scaffolding + data-availability with DOI placeholder | PASS |
| 11 | Final verification and handoff | PASS |

Final build: latexmk exit 0, **41 pages**, 0 undefined references or citations, 0
em-dashes across the entire `paper/` tree, 0 overfull boxes in the final log. Every
new number traces to a committed file via `outputs/session26/new_numbers_manifest.tsv`
(all 13 source files verified git-tracked). Decision log: D164 to D175 in `HANDOFF.md`,
one entry per track.

## Claims that changed (honesty over preservation)

1. **Forecast wake advantage weakened (the load-bearing change).** The Markov-rollout
   wake-enstrophy paired improvement does not survive case-level clustering
   (case-clustered 95% CI [-4.5, +72.6] includes zero; case-level signed-rank p=0.10,
   7/10 cases; mixed-effects intercept p=0.10) and does not survive a Holm correction
   over the twelve paired tests (raw 0.044 -> Holm 0.44). It is now presented as
   consistent confirmation, not the decisive test. The **representational** wake
   advantage survives both (case-clustered CI [+12.5, +77.2]; signed-rank p=0.03; Holm
   0.017), and is the new anchor, together with the mechanism evidence.

2. **Forecast is not reliably above the conditioning floor at the case level.** The
   per-encounter forecast-minus-floor improvement has a case-clustered CI [-15.5, +31.4]
   (4/10 cases); the representational closure exceeds the floor in R^2 (0.75 vs 0.17)
   but its paired case-clustered interval grazes zero. The floor wording was narrowed
   accordingly; the variance-weighted R^2 gap remains the stronger statement.

3. **Topology "decisive" softened.** The generator-count separation is robust over a
   defensible noise-floor (2 to 15 percent) and sampling range and survives case-level
   clustering (10/10 cases, p<1e-3), but the exact Mann-Whitney p ranges over nine
   orders of magnitude across the threshold and sampling grid, so the order-of-magnitude
   p is reported as setting-dependent and "decisive" was removed.

Claims that held at the case level: the representational wake advantage, the latent
drift ratio, the topology generator-count separation, the optimal-transport Spearman
margin (9/10 cases), and the large-scale enstrophy tracking (JEPA 0.95 vs Fukami 0.77
at the case level).

## What was added or corrected

- S2.2: distinct-case counts (10 test_b, 4 test_c) and the non-independence note; the
  pre-registered primary endpoint (wake enstrophy); the Iy non-impulse caveat
  (r(dIy/dt, CL) ~ -0.05); the 2D-proxy statement with chi_3D ~ 0.20; a one-line
  omega_c sensitivity (collinearity > 0.90, advantage stable).
- Table 10: a Holm-p column and case-clustered wake CIs in the caption.
- Table 4(a): a representational wake-enstrophy R^2 column (the abstract's 0.75 now
  appears in the table it references).
- S4.3 + Appendix A: the persistent-homology threshold-and-sampling robustness grid
  citing Smith et al. (2024).
- S4.6: scoped every physical-space claim to the large-scale band and clarified the
  numbers are the encode-decode reconstruction ceiling, not a forecast rollout.
- S4.1 + S4.5 + methods: the reconstructive non-monotonicity as a drift consequence,
  the eightfold seed spread, and the AE best-config statement.
- Abstract, S1, S3.4, S4.3, S5.1, S6: the transport-consistency principle; world-model
  reduced to one motivation sentence (S1) and one caution clause (S5.4).
- Reproducibility scaffolding (README, LICENSE, CITATION.cff, .zenodo.json), the
  data-availability statement with a DOI placeholder, and tag v1.0.0-rc1.

## Residual risks

- The wake claim now leans on representation plus mechanism. A referee who insists the
  forward forecast is the only deployment-relevant metric will find it marginal; the
  paper is explicit that it is consistent but not decisive at the case level.
- n=10 test_b cases is a small number of clusters; the case-level signed-rank and
  mixed-effects tests have limited power, which is stated.
- The representational mixed-effects model did not converge (heavy-tailed deltas); the
  case-level evidence there rests on the signed-rank test and the cluster bootstrap.
- Test C is n=4 cases; all test_c statements remain reported-only, never used for
  selection.

## What remains for the collaborators (out of scope this session)

1. **Table 1 DNS resolution numbers** (the `[PENDING]` author-fill rows), untouched:
   free-stream Mach number or incompressible-limit confirmation; computational domain
   and spanwise extent Lz/c; element and solution-point counts; minimum wall-normal
   spacing and wall units; time step and maximum CFL; gust-release station x0/c.
2. **The grid and time-step sensitivity (convergence) study.**
3. **Reproducibility deposit decisions**: confirm the MIT license (and any INTA / UC3M
   / UPC / BSC institutional requirement), mint the Zenodo DOI from tag v1.0.0-rc1, and
   replace `10.xxxx/zenodo.PLACEHOLDER` in the data-availability statement,
   `.zenodo.json`, `CITATION.cff`, and `README.md`.
4. **Author-contributions block** (CRediT) is still a manuscript placeholder.

## Reproducing this session (CPU, no training)

```bash
source .venv/bin/activate && export PREVENT_ROOT=$HOME/PREVENT
python scripts/session26/track1_stats.py
python scripts/session26/track2_topology_robustness.py
python scripts/session26/track3_physics_caveats.py
```
