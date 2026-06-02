# vortex-causality

Information-theoretic / causal analysis toolkit for the **vortex-jepa** project,
plus the orchestration that turns the manuscript’s empirical findings into
estimator-independent information statements.

It packages two complementary lenses from the Lozano-Duran group, both
observational and both information-theoretic:

|tool                                                          |what it does                                                                                     |role in this paper                                                                                                        |
|--------------------------------------------------------------|-------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
|**aIND** (Arranz & Lozano-Duran, JFM 1000 A95, 2024)          |splits a field into the part informative about a target’s future and a residual that carries none|scalar specialisation here = **observability** `O = I(T;S)/H(T)`, the principled version of §4.7 and the forecast ordering|
|**SURD** (Martinez-Sanchez et al., Nat. Commun. 15:9296, 2024)|decomposes causality into **r**edundant / **u**nique / **s**ynergistic + information leak        |explains **why** the wake separates the encoders: the future wake is synergistic in (parameters, current wake state)      |
|**Causal-Forecast** (ALD-Lab, 2026)                           |links SURD components to the predictive limit (irreducible error)                                |external anchor: this manuscript’s claim, formalised                                                                      |

These are **observational**. They characterise the information structure of the
data and of the learned latents. They do **not** validate the predictor as a
counterfactual operator. The interventional test already failed (HANDOFF D154),
and that conclusion stands. Report results as “what causes the future observable”
and “which latent is observable of it”, never as an interventional/world-model
claim.

## Why this lens fits the paper

The manuscript’s headline is that the **wake** observables, not the scalar forces,
separate the predictive and reconstructive latents, and that a latent which keeps
only the lift signature fails on the wake. The information-theoretic reason:

- the **future lift** is nearly *uniquely* caused by the gust parameters (the
  forcing is imprinted at impact);
- the **future wake enstrophy** is *synergistic* in (parameters, current wake
  state): it needs the joint, not either alone.

So a reconstruction objective that retains amplitude/lift but not the
wake-state interaction cannot forecast the wake; a predictive objective that
retains exactly the future-relevant information can. `observability` shows the
predictive latent is the most informative of the future wake; `surd` shows that
information is irreducible to the parameters.

## Layout

```
infotheory/
  estimators.py      KSG mutual information, Kozachenko-Leonenko entropy,
                     histogram MI (Miller-Madow), surrogate nulls, robustness sweep
  observability.py   aIND-style O = I(T;S)/H(T); residual-information check
  surd.py            discrete SURD (n=2,3) + reference-impl adapter + staged stand-in
  io_vortex.py       bridge to the vortex-jepa cache/artifacts; synthetic design;
                     validate_against_known() gate
scripts/
  run_causal_analysis.py   orchestration: Block A (SURD on observables, the
                           de-risking core), Block B (latent observability),
                           Block C (staged SURD). --synthetic and --real modes.
tests/
  test_infotheory.py       8 gates: analytic Gaussian MI, independence null,
                           CMI chain rule, KL entropy, SURD XOR/COPY/UNIQUE,
                           3-source smoke + information conservation
CLAUDE.md            agentic task brief (Tracks M / D / C with acceptance gates)
```

## Install / run

No build step. numpy, scipy, scikit-learn only (h5py needed for `--real`).

```bash
# unit gates (no data needed)
python tests/test_infotheory.py            # -> "All 8 tests passed."

# end-to-end smoke on synthetic data with known ground truth (no data needed)
python scripts/run_causal_analysis.py --synthetic --surrogate 80 --bins 6

# the one-day de-risking experiment on REAL observables, no model in the loop
python scripts/run_causal_analysis.py --real \
    --split configs/splits/split_v1.json \
    --source-a G --source-b wake_enstrophy_impact \
    --targets wake_enstrophy_future CL_future \
    --bins 6 --surrogate 200 --out outputs_causal/derisk

# full real run once latents are exported and schema hooks confirmed
python scripts/run_causal_analysis.py --real \
    --split configs/splits/split_v1.json \
    --latent JEPA_d64=runs/.../latents.npz \
    --latent reconstructive_d64=runs/.../latents.npz \
    --latent POD_d64=runs/.../latents.npz \
    --pressure K8=runs/.../pressure_k8.npz \
    --bins 6 --surrogate 200 --out outputs_causal/full
```

## Reference implementations (cross-check headline numbers against these)

- SURD: <https://github.com/Computational-Turbulence-Group/SURD>
  (mirror <https://github.com/ALD-Lab/SURD>)
- state-aware SURD: <https://github.com/ALD-Lab/SURD-states>
- aIND: <https://github.com/Computational-Turbulence-Group/aIND>
- Causal-Forecast: <https://github.com/ALD-Lab/Causal-Forecast>

The in-repo `surd_discrete` is fast and unit-tested for the 2-3 source lattices
used here; it is for iteration, not to supersede the reference. Validate any
number that will be published against the reference SURD (`surd_reference`).

## Honest limitations (read before writing manuscript text)

- **Observational, not interventional.** See above and HANDOFF D154.
- **Sample size.** Synergy estimates are data-hungry and biased at small n. Run
  estimation on the pooled *training* trajectories (~226 encounters x 120 frames),
  not the 42 held-out encounters, and always report the surrogate null and the
  kNN-vs-histogram robustness spread.
- **Non-stationarity.** The gust is a transient off a limit cycle. A single
  stationary SURD mixes regimes; use the state-aware version applied per stage.
- **Out-of-distribution.** SURD will show a large information leak at |G| = 4,
  consistent with the 3-D observability boundary, so it adds little there.
- **Schema hooks.** `io_vortex.py` defaults to the names in HANDOFF.md but cannot
  know the exact dataset keys without the data. Confirm every `# SCHEMA HOOK` and
  let `validate_against_known()` (reproduces the 29.83 wake MAE) catch mistakes.

```

```