# CLAUDE.md  —  Session task brief: manuscript polish + information-theoretic causal analysis

This is an operational brief for an agentic Claude Code session working on the
**vortex-jepa** repository. It is written in the same spirit as the existing
HANDOFF decision log: every track has a rationale and an explicit, checkable
acceptance gate. Do not mark a track done until its gate passes.

The work splits into three independent tracks. **Track M** (manuscript polish)
and **Track D** (DNS author-fill scaffolding) are low-risk and can land
immediately. **Track C** (causal / information-theoretic analysis) is the
scientifically novel addition and is gated on a one-day de-risking experiment
before any manuscript length is committed.

-----

## Non-negotiables (apply to every track)

1. **No fabricated numbers, ever.** In particular, never invent DNS solver
   resolution values (minimum wall spacing, wall units, element/solution-point
   counts, time step, CFL, gust-release station, grid/time-step sensitivity).
   These are author-fill only. Track D scaffolds the *structure*; the authors
   supply the *values*.
1. **Honesty about scope.** The interventional test already failed (HANDOFF
   D154): the predictor is a conditional forward model, not a validated
   counterfactual / world model. SURD and IND are **observational**. Nothing in
   Track C may be written as evidence for an interventional or world-model claim.
   Frame causal results as “what causes the future observable” and “which latent
   is observable of it”, never as “the model intervenes correctly”.
1. **No em-dashes** anywhere in the manuscript or comments (existing house rule).
1. **Match repo conventions.** Reuse the canonical closure probe, the
   `(case_id, encounter)` alignment, normalise-in / denormalise-out, raw
   `(G, D, Y)` conditioning, and the fixed split. Do not introduce a parallel
   data path.
1. **Reproducibility.** Thread a fixed seed into every new script; record the
   exact command, the input checkpoints/metrics, and the output paths in a new
   HANDOFF decision entry (D158, D159, …). Keep the LaTeX build clean
   (`latexmk` exit 0, no undefined refs).
1. **Pre-register gates.** For any quantitative claim, write the acceptance
   threshold *before* running, exactly as the closed-loop pilot did.

-----

## Track M — manuscript polish (paste-ready, no new computation)

These address the referee-facing issues identified in review. They are text-only.

### M1. Retitle to match the abstract’s framing

The current title reads like a new dynamics model. The contribution is a
controlled comparison of encoder objectives judged by forward physical closure,
plus a drift mechanism. Propose 3 candidate titles in that register (e.g.
“What a reduced state must retain to forecast vortex-gust airfoil interactions:
a predictive-versus-reconstructive comparison at Re = 5000”), and update
`main.tex` once the researcher picks one.
**Gate:** title contains “predictive” and “closure” (or a synonym), and does not
promise an interventional/world model.

### M2. Reframe the world-model framing as motivation, not claim

In §1 and §3, introduce the action-conditioned world-model view explicitly as an
analogy that motivates the unconditional-encoder / conditioned-predictor split,
not as a property being claimed. Then §5.4’s negative interventional result reads
as “as anticipated, the interventional reading does not hold” rather than a
walk-back. Add a one-clause scope statement to the abstract.
**Gate:** §1 states the world-model framing is a motivation; §5.4 references that
framing; abstract carries the scope clause; diff touches only prose.

### M3. Fix the data-availability inconsistency

The Data Availability statement says code is “available on reasonable request”,
but Appendix B twice says “the released code”. Reconcile: if anything is
released, point to the repository and make both sentences consistent; otherwise
remove “released” from Appendix B.
**Gate:** `grep -n "released code" main.tex` and the data-availability statement
are mutually consistent.

### M4. Draw the boundary with the companion paper [31]

Add one sentence stating what is novel here versus the under-review companion
(there: controlled wakes, the compactness-vs-forecast tradeoff; here: the
parametric gust, the drift mechanism as the common cure, and its geometric and
topological characterisation).
**Gate:** a sentence citing [31] with an explicit “there … here …” contrast
exists in §1 or §5.1.

### M5. Consolidate the observability-boundary refrain and trim figures

The “|G| = 4 is an observability boundary, not a generalisation claim” statement
recurs in §2.1, §4.1, §4.6, §5.3, App. B. State it canonically once (§2.1) and
cross-reference elsewhere. Separately, move borderline figures (15, 16, 18) fully
into the appendix if not already, and tighten the densest multi-number sentences
in §4.1 into shorter declaratives.
**Gate:** the boundary statement appears once in full with `\S2.1`
cross-references elsewhere; figure count in the main body is reduced; build clean.

-----

## Track D — DNS author-fill scaffolding (blocked on collaborators for values)

The `[PENDING ...]` block in §2.2 is the single allowed author-fill. Do **not**
fill numbers. Instead:

### D-a. Convert the prose PENDING block into a labelled checklist table

Make a `\pending{}` table with one row per required quantity (free-stream Mach or
incompressible-limit confirmation; full domain and span Lz/c; element and
solution-point counts; minimum wall-normal spacing and wall units; time step and
max CFL; gust-release station x0/c; grid and time-step sensitivity result), each
with a blank value cell and a one-line “why a referee needs this” note. This
makes the gap auditable and gives the collaborators a fill-in form.
**Gate:** the table compiles, every value cell is a visible `\pending{}`, and the
DNS-vs-LES distinction over the source [10] is stated as a one-line methodological
note (DNS, no subgrid model, all scales resolved) without numeric claims.

-----

## Track C — information-theoretic causal analysis (the novel addition)

Uses the standalone `infotheory/` package shipped with this brief (drop it under
`src/` or keep it adjacent and add to PYTHONPATH). The package is method-agnostic,
unit-tested in isolation (`python tests/test_infotheory.py` → 8 passing gates),
and validated against analytic Gaussian MI and the canonical PID gates
(XOR=synergy, COPY=redundancy, UNIQUE=unique).

**Scientific framing.** Two complementary tools, both from the Lozano-Duran group
and both information-theoretic:

- **aIND** (Arranz & Lozano-Duran, JFM 2024): an informative/residual split of a
  field relative to a target. Its scalar specialisation here is the
  *information-theoretic observability* O = I(T;S)/H(T). This is the principled
  version of §4.7 (recoverability from the wall) and of the forecast ordering.
- **SURD** (Martinez-Sanchez, Arranz & Lozano-Duran, Nat. Commun. 2024): a
  redundant/unique/synergistic decomposition of causality. Its forecasting
  companion (ALD-Lab/Causal-Forecast, 2026) links these components to the
  theoretical limit of predictive performance (irreducible error), which is
  almost exactly this manuscript’s empirical claim.

**The thesis they jointly support.** The wake observables separate the encoders
because the *future wake* is synergistic in (gust parameters, current wake state),
whereas the *future lift* is more uniquely caused by the parameters. A
reconstruction objective that retains the lift signature but not the
wake-state interaction therefore cannot forecast the wake; a predictive objective
that retains exactly the information about the future can. IND/observability shows
the predictive latent is the most informative of the future wake; SURD shows why
that information is irreducible to the parameters alone.

### C0. (GATE-ZERO, do this first) One-day de-risking on observables only

Run Block A of `scripts/run_causal_analysis.py` in `--real` mode with NO model in
the loop: SURD of {future wake enstrophy, future lift} from {G, current wake
enstrophy}. This costs ~one day and decides whether the section is worth writing.

```
python scripts/run_causal_analysis.py --real \
    --split configs/splits/split_v1.json \
    --source-a G --source-b wake_enstrophy_impact \
    --targets wake_enstrophy_future CL_future \
    --bins 6 --surrogate 200 --out outputs_causal/derisk
```

**Gate (pre-registered):** the section proceeds **iff** the synergistic fraction
S[G + wake_now]/H(T) for the future wake is at least ~2x that for the future
lift, AND the future-lift unique fraction U[G]/H(T) exceeds its synergy, AND the
information leak is not so large (say < 0.7) that nothing is identifiable. If the
gate fails, write a two-line HANDOFF entry recording that SURD does not add value
on this dataset and stop Track C. (This is the same discipline as the failed
interventional gate: a negative result is a result, recorded, not buried.)

### C1. Wire the schema hooks and pass the validation gate

Before any latent-level number, make `infotheory/io_vortex.py` read the real
cache. Confirm every `# SCHEMA HOOK` against `src/data/episode_dataset.py` and the
`wake_observables/` writer (impact frame index, observable keys, params source,
partition map). Export per-family impact-frame latents to `.npz`
(`keys`, `z_impact`) from the existing eval pipeline (do not import the model into
the analysis package).
**Gate:** `io_vortex.validate_against_known()` reproduces the d64 representational
wake MAE 29.83 (Table 3a) within tolerance. If it does not, a hook is wrong; fix
before proceeding. `run_causal_analysis.py --real` runs this gate automatically.

### C2. Block B — latent observability table (principled §4.7)

Compute O = I(T;S)/H(T) for every family (JEPA d64/d32, Fukami d3/.., POD ..) and
the K8/K2 pressure sets, toward {future wake enstrophy, future lift, future neg.
circ.}, on held-out test_b, with surrogate nulls and the kNN/hist robustness
sweep.
**Gate (pre-registered):** for the future wake, O(JEPA) is the largest among the
three *latent* families and exceeds O(reconstructive) by a margin outside the
surrogate noise; for the future lift, the gap narrows (consistent with lift being
read directly from pressure, §4.7). Report all targets, not a selected subset
(matches the manuscript’s existing all-observables discipline). Add as one table +
one paragraph in §4.7, framed as the information-theoretic mirror of the
R^2 recoverability result.

### C3. Block C — staged SURD (state-aware) of the future wake

Quantify the causal-structure shift the manuscript asserts in §4.1: parameter-
dominated near impact, wake-state-dominated and more synergistic at the horizon.
Use the released **state-aware** SURD (ALD-Lab/SURD-states) for the headline
figure; the in-repo `surd_by_stage` stand-in is for fast iteration only.
**Gate:** the synergistic fraction of the future wake is non-decreasing from the
impact stage to the peak/recovery stage, OR, if it is not, the §4.1 sentence
asserting the shift is softened to match what the data show. Either way the text
and the SURD agree. Cross-check the headline numbers against the reference SURD
(`surd_reference`) before they enter the manuscript.

### C4. Decide placement: one figure in-text, or a companion paper

The manuscript is submission-ready and has carefully bounded its causal claims. Do
**not** expand the main text into an open-ended causal-discovery section. Two
acceptable outcomes:
(a) one tight table (Block B) + one figure (Block C) + one paragraph in §4, with
a sentence citing SURD, IND, and Causal-Forecast; or
(b) a short companion paper carrying the richest version (state-aware SURD across
the staged encounter on the training trajectories, with estimator robustness,
and the latent coordinates as SURD sources in a second pass).
**Gate:** the chosen option adds <= 1 table and <= 1 figure to the main body;
the abstract’s scope clause (M2) still bounds the claim; build clean.

-----

## Suggested landing order

1. Track M (M1–M5) and Track D-a in one pass: text only, immediate, low risk.
1. Track C0 (gate-zero de-risking) on observables alone. Stop if it fails.
1. If C0 passes: C1 (schema + validation gate) → C2 (observability) → C3 (staged
   SURD) → C4 (placement).
1. Write a HANDOFF entry per landed track (rationale, command, inputs, outputs,
   gate result), matching the existing D-numbered style.

## Definition of done

- `latexmk` exit 0, no undefined refs, no em-dashes, abstract within length.
- `python tests/test_infotheory.py` → all gates pass.
- Every causal number in the manuscript is reproduced within tolerance by
  `run_causal_analysis.py --real` and cross-checked against the reference SURD.
- HANDOFF updated. Scope honesty (observational, not interventional) preserved.