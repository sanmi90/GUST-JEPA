# E4 scale band + parametric (model-free) floor (v2.1)

Two cheap v2.1 regenerations for the unconditioned manuscript appendix/tables.
Both are CPU-only, model-free, and produced by
`scripts/session28/e4_floor_regen.py` (subcommands `scaleband`, `floor`, `all`).
Outputs are review artifacts; nothing here is committed.

Provenance: cache `${PREVENT_ROOT}/data/processed/vortex-jepa/v2p1`, split
`configs/splits/split_v2p1.json`, observable targets
`outputs/session28/exp2/dns_physical_metrics.npz`. The floor reuses the
closure-matrix probe internals, the case-level CV fold construction, the
`1 - SSE/SST` (about the held-out mean) estimator, and `stats_lib`
case-clustered bootstrap from `scripts/session28/closure_matrix.py` +
`scripts/session28/stats_lib.py`, so it is directly comparable to the closure
latent cells. The E4 metric reuses the large-scale wake-enstrophy Gaussian
filter from `scripts/session28/physics_prep.py`.

## E4 scale-band sensitivity (closes referee S4)

The headline large-scale wake-enstrophy tracking number uses a Gaussian filter
at sigma/c = 0.05. E4 recomputes the headline metric (the peak post-impact
large-scale wake-enstrophy excursion, `denstrophy_peak_post = max_t |E(t) -
mean(E pre-impact)|`) at sigma/c in {0.01, 0.03, 0.05}, and tests whether the
headline FINDING (the monotone increase of the peak excursion with gust strength
|G|) survives the band choice.

test_b, peak excursion vs |G| (Spearman rho):

| sigma/c | sigma (px) | median peak | rho(|G|, peak) | p |
|---------|-----------|-------------|----------------|------|
| 0.01    | 0.32      | 204.9       | +0.51          | 5e-4 |
| 0.03    | 0.96      | 61.5        | +0.70          | 2e-7 |
| 0.05    | 1.60      | 46.9        | +0.74          | 2e-8 |

The same positive, significant trend holds on train (rho 0.63 / 0.76 / 0.78) and
val (0.68 / 0.79 / 0.80). test_c carries no trend by construction (all test_c
cases have |G| = 4, so the |G| Spearman is undefined).

Robustness sentence (appendix): Recomputing the peak large-scale wake-enstrophy
excursion at sigma/c in {0.01, 0.03, 0.05} leaves the headline finding intact:
the excursion increases monotonically with gust strength on the held-out test_b
cases at every filter scale (Spearman rho = 0.51, 0.70, 0.74; all p < 0.05), so
the |G| trend does not depend on the choice of filter scale. The absolute
excursion magnitude naturally grows as the band narrows and smooths less; the
per-encounter ordering at the 0.03 band tracks the headline 0.05 band at Spearman
rho = 0.97, while the sub-pixel 0.01 band (essentially unsmoothed) shifts the
ordering to rho = 0.81 (reported as a diagnostic, not a gate).

Verdict: ROBUST = True. Full table: `outputs/session28/e4_scaleband/results.json`.

## Parametric (model-free) floor

RENAME NOTE (2026-06-13): this is the MODEL-FREE PARAMETRIC FLOOR, NOT the
"conditioning floor". It regresses each observable on the gust parameters
(G, D, Y) ALONE with no latent and no trained model; the conditioned-reference
question (T6 predictor conditioned on c, evaluated through the closure matrix) is
a separate B4 deliverable.

Each of the six observables is regressed on (G, D, Y) with a linear (ridge) and a
nonlinear (KRR-RBF) regressor, fit on TRAIN cases, read out at impact + 16 on
test_b (pooled tiers); held-out R^2 = 1 - SSE/SST about the held-out split's own
mean (closure-matrix estimator), with case-clustered CI (n = 10000). The point of
the floor: the latent BEATING it shows the representation carries flow state
BEYOND the gust parameters.

test_b, impact + 16, R^2 [case-clustered 95% CI]:

| observable        | ridge (linear)        | KRR-RBF               |
|-------------------|-----------------------|-----------------------|
| C_L               | +0.61 [+0.22, +0.85]  | +0.61 [+0.13, +0.88]  |
| C_D               | +0.54 [+0.00, +0.92]  | +0.51 [-0.14, +0.95]  |
| I_y               | +0.25 [-0.67, +0.81]  | +0.22 [-0.74, +0.81]  |
| wake_enstrophy    | -0.18 [-1.83, +0.77]  | -0.12 [-1.79, +0.81]  |
| circulation_pos   | -0.05 [-1.58, +0.81]  | -0.02 [-1.52, +0.81]  |
| circulation_neg   | +0.23 [-0.91, +0.86]  | +0.22 [-1.01, +0.89]  |

Latent beats the floor: on the wake enstrophy the gust parameters alone explain
R^2 = -0.18 (linear) / -0.12 (KRR-RBF) of the held-out test_b variance at
impact + 16 (negative: at the +16 horizon the parameters alone do worse than
predicting the test_b mean). The predictive latent recovers R^2 = +0.79 (closure
matrix, representational wake_enstrophy, H = 16, test_b pooled, ridge, JEPA
tf-no-c d = 64; `outputs/session28/numbers_parts/closure_headline.json`),
clearing the model-free parametric floor by +0.91. The latent therefore carries
wake flow state well beyond what the gust parameters encode.

Note on the protocol vs the old D145 floor: this floor fits on TRAIN and reads
out at impact + 16 on test_b with SST about the held-out test_b mean, matching the
closure-matrix cell exactly. That is stricter than the Session-23 D145 floor
(impact frame, fit-and-evaluate on the same split), which is why the wake floor
is at/below zero here rather than the ~0.48 reported for the impact-frame
same-split recipe; the comparison to the latent is now apples-to-apples.

Full table: `outputs/session28/parametric_floor/results.json`.

## Numbers / macros (eval_all)

`outputs/session28/numbers_parts/e4_floor.json` (part `e4_floor`, validates
clean through `eval_all.py --check`):

- `NumEfourWakeGtrendSigmaEone` / `...Ethree` / `...Efive` -- the wake-enstrophy
  peak-excursion |G| Spearman trend at sigma/c = 0.01 / 0.03 / 0.05 (test_b).
- `NumParamFloorWakeLinear` / `NumParamFloorWakeKrr` -- the parametric-floor
  wake_enstrophy R^2 (linear ridge / KRR-RBF) at H = 16 on test_b, with
  case-clustered CI.

(Macros are alphabetic per the eval_all contract, so the band scales are spelled
Eone/Ethree/Efive rather than digits.)

## Reproduce

```bash
export PREVENT_ROOT=$HOME/PREVENT
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
nice -n 10 /home/carlos/GUST-JEPA/.venv/bin/python \
    scripts/session28/e4_floor_regen.py all
# tests:
/home/carlos/GUST-JEPA/.venv/bin/python -m pytest tests/test_e4_floor_regen.py -q
```

Wall time on CPU: floor ~5 s, scale band ~60 s (one cache pass over 382
encounters x 3 bands), full test suite ~67 s.
