# SESSION 23 plan: take the draft from “analyses done” to “JFM-submittable”

This is a Claude Code session plan in your repo’s conventions (CLAUDE.md,
HANDOFF.md decision-log style, the SESSION20 track/gate/stub format). No time or
GPU limit. Most of this session is post-processing, data and code archaeology,
figure work, and manuscript surgery, so it runs almost entirely on CPU and the
L40S cards under `VORTEX_JEPA_ALLOW_NON_RTX6000=1`; the only encoder training is
the autoencoder seeds in Track C, and only if they do not already exist.

Organising principle. Session 20 (D132-D138) computed the analyses that now sit
in the draft: the 2x2 controls, held-out R^2, persistent homology, the OT field
metric and OT-geodesic alignment, the phase-amplitude limit cycle, the scale
decomposition, and the horizon sweep. The current manuscript (`main.tex`, the
build that produced `main_7.pdf`) has those results but is NOT submittable, for
seven concrete reasons a JFM reviewer in the Taira-adjacent community will catch.
This session closes all seven and adds the physics depth and the two reframings
that lift it from “strong ML-on-fluids” to “JFM fluid mechanics”. Every track
ends in an acceptance gate and a HANDOFF decision stub.

The seven gaps, mapped to tracks:

1. The methods are a placeholder (`\pending{}` solver/grid/Mach); Track A scaffolds
   the subsection and marks the DNS numbers as an author-fill block. -> Track A.
1. The six headline observables are never defined with equations. -> Track B.
1. The paper contradicts itself about the controls (abstract vs Outlook). -> Track K.
1. The conditioning-floor claim is contradicted by its own Table 4. -> Track D.
1. It is physics-light where JFM is physics-led (LEV/shear layer). -> Tracks E, F.
1. The case-vs-encounter accounting does not add up (84 cases, 320 encounters). -> Track B.
1. The pressure result is buried; the failed control pilot is foregrounded. -> Track H.

Plus: headline seed robustness (Track C), error maps over the gust axes (Track G),
the optional interventional test that earns the world-model framing (Track I),
the physics-led figure rebuild (Track J), and the full rewrite and compliance
pass (Track K).

Read first, in this order: CLAUDE.md, the latest HANDOFF.md entries, REWRITE_NOTES.md,
WORLD_MODEL_FRAMING.md, the current `main.tex`, and `outputs/session20/` (the
D132-D138 artifacts this session builds on). This plan assumes all of them.

A numbering note: the HANDOFF stubs below are written as D161+ as a contiguous
placeholder block. Renumber them to follow the last entry in your current
HANDOFF.md (Session 20 ended at D138; sessions 21 and 22 will have advanced it).

-----

## Pre-flight (20 min, blocks everything) and the questions to surface now

```bash
source .venv/bin/activate
export PREVENT_ROOT=$HOME/PREVENT WANDB_PROJECT=vortex-jepa
python -m src.training.sanity_checks --all                    # --require-gpu only if Track C trains
```

Confirm the Session 20 artifacts are present, because Tracks C-H read from them:

```bash
ls outputs/runs/session12/S12_E_d64/encoder/                  # production d=64
ls outputs/runs/session12/S12_E_d32/encoder/                  # matched d=32
ls outputs/runs/session14/thrust6/jepa_d64_seed{0,1,2}/encoder/  # JEPA seeds (Track C)
ls outputs/session18/exp_b1_test3/                            # B1 closure + drift + pressure CSVs
ls outputs/session20/persistent_homology/ outputs/session20/ot_field/ \
   outputs/session20/ot_alignment/ outputs/session20/phase_amplitude/ \
   outputs/session20/scale_decomp/ outputs/session20/horizon_sweep/   # D134-D138
ls outputs/session16/exp2/per_frame_targets/                  # per-frame DNS observables (Track B)
```

Confirmed facts (from the project lead) to document and verify against the data,
not to rediscover. Only the exact I_y convention is still to be read out of the
code (Track B); do not guess that one.

- **DNS, no subgrid model.** The runs are direct numerical simulations with no LES
  subgrid closure. Track A states this explicitly in Section 2.2 and keeps “DNS”
  throughout. The resolution numbers that justify DNS (`Delta n+` in the baseline
  wake and the rest of the solver-resolution block) are supplied by the project
  lead and left as an author-fill block; the agent does not invent them. The source
  at the same Re and alpha was an LES, so the authors should supply the resolution
  that makes DNS credible here.
- **The encoder consumes the mid-plane omega_z slice** (expected; Track A verifies
  it against the cache). The chi_3D diagnostic (Track F) then measures exactly what
  the mid-plane slice cannot see at |G|=4, which is the physical reason test_c
  degrades.
- **A case contains several encounters, already separated in preprocessing.** The
  84 cases yield the 320 encounters (226+28+42+24). Track B confirms the per-case
  count from the split manifest and states the relationship in Section 2.2; there
  is no arithmetic contradiction, only an unstated definition to add.
- **Autoencoder seed encoders should exist.** Track C verifies and reuses them and
  trains only if any are missing (the session’s only possible RTX work).
- **The full 3D vorticity fields exist.** Track F computes chi_3D directly from
  them; no need to derive omega from `/u`.
- **Still to read from the code, not guess: the exact I_y convention** (sign and
  region). Track B reads it from the probe-target code; the paper states what the
  code computes.

-----

## Track A (no compute, START FIRST, blocks the methods section): scaffold the numerical-method subsection

This writes the Section 2.2 “Numerical method” subsection from the confirmed
facts and definitions, and leaves a single clearly-marked author-fill block for
the DNS solver-resolution numbers, which the project lead supplies separately. The
agent does NOT mine the SOD2D configs for Mach, domain, mesh, `Delta n+`, time
step, CFL, or the release station; those are deferred to the authors. It is no
compute and leads because the methods are the first thing a reviewer checks and
because the rest of Section 2.2 (observables, splits) depends on this scaffold.

Write the subsection from the template in the revision plan, Part II.2: state that
the runs are direct numerical simulations with no subgrid model, that the encoder
consumes the mid-plane omega_z slice (name the span average once as the
alternative), the C_L and C_D definitions, and the Taylor-vortex profile with the
G, D, Y definitions and the t=0 convention. Insert the author-fill block verbatim:

> *[Authors to insert: free-stream Mach number; computational domain x/c, y/c,
> spanwise extent; element and solution-point counts; minimum wall-normal spacing
> and Delta n+ in the baseline wake; time step and maximum CFL; gust release
> station x0/c; and the grid and time-step sensitivity check. These DNS details
> are not yet filled in and will be added by the authors.]*

Two cheap verifications against the cache, to confirm the two facts the paper now
states (not to fill numbers):

```bash
python - <<'PY'
import h5py, glob, os
# Confirm the cache stores a mid-plane omega_z slice (expected shape ~ 192 x 96, no z axis),
# and record any solver attributes already present for the authors' later use.
f = sorted(glob.glob(os.path.join(os.environ['PREVENT_ROOT'],'**','*.h5'), recursive=True))[0]
with h5py.File(f,'r') as h:
    print('FILE', f, 'ROOT ATTRS', dict(h.attrs))
    for k,v in h.items():
        print(k, getattr(v,'shape',None), dict(getattr(v,'attrs',{})))
PY
```

The cache subdomain x/c in [-1.5, 4.5], y/c in [-1.5, 1.5] matches the data-driven
subdomain of the source paper exactly; cite that lineage in one sentence. Make
“DNS” and the mid-plane phrasing consistent across the abstract, Section 2.1,
Section 2.2, and Section 3.1 in one pass, so Track K does not have to.

ACCEPTANCE GATE: Section 2.2 contains the DNS and mid-plane statements, the
C_L/C_D and Taylor-vortex definitions, the cache-subdomain citation, and the
author-fill block, with no stray invented numbers; the mid-plane slice shape is
verified against the cache. The DNS resolution numbers remain a marked author-fill
block until the project lead supplies them; do not invent Mach, Delta n+, dt, CFL,
or x0/c.

HANDOFF stub: `### D161: Session 23 Track A -- numerical-method subsection scaffolded; DNS and mid-plane stated; resolution numbers left as an author-fill block per the project lead`.

-----

## Track B (no compute, START FIRST, blocks Section 2.2 and Table 2): observable definitions from the code, and the case/encounter reconciliation

Clears gaps 2 and 6. The six observables that carry the headline must be defined
with equations that match the code that produced the probe targets, not a guess.

Read the definitions out of the source:

```bash
# Find where the six per-frame observables are computed (the probe targets).
grep -rn 'wake_enstrophy\|enstrophy\|circulation\|impulse\|I_y\|Iy\|gamma_pos\|gamma_neg' \
  src/ scripts/ | grep -i 'def \|=.*sum\|trapz\|integrate'
sed -n '1,200p' $(grep -rl 'per_frame_targets\|build_targets\|compute_observables' src scripts | head -1)
```

For each observable record the exact formula, the wake-window bounds Omega_w, the
mask (solid plus one adjacent cell layer, per Section 2.2), whether values are
dimensional or chord-normalised, whether the same mask is used for every case, and
how H=16 is indexed relative to impact. Write the equation block (revision plan
Part II.3) to match the code. Then verify, do not assert:

```bash
python scripts/session23/verify_observable_defs.py \
  --targets outputs/session16/exp2/per_frame_targets/ \
  --fields ${VORTEX_JEPA_CACHE}/v2 \
  --tol 1e-6 \
  --out outputs/session23/observable_verify.json
# Re-derive each observable from the written equation on a few frames and compare
# to the stored target value; all six must match to float tolerance.
```

Confirm the counts. A case contains several encounters that preprocessing already
separated, and the 84 cases yield the 320 encounters (226+28+42+24), so there is
no contradiction, only a definition the paper omits. Read the split manifest to
confirm the per-case encounter count and the partition:

```bash
python - <<'PY'
import json, collections
m = json.load(open('outputs/data_pipeline/v1/manifest.json'))  # adjust path to the v2 split manifest
# confirm: distinct (G,D,Y) cases == 84; total encounters == 320; encounters per case;
# per-split encounter counts == 226/28/42/24 (test_a from training cases, test_b, test_c)
PY
```

Add one sentence to Section 2.2 stating the relationship, e.g. “The 84 cases
(distinct (G,D,Y) points) are separated in preprocessing into 320 encounter
windows of 120 frames, split 226 / 28 / 28-held-out (test_a) / 42 (test_b) /
24 (test_c).” Make the abstract and Section 2.2 numbers agree with the manifest.

ACCEPTANCE GATE: `verify_observable_defs.py` confirms all six written equations
reproduce the stored targets to 1e-6; the wake window, mask, units, and H=16
convention are stated; the 84-cases-to-320-encounters relationship is stated and
matches the manifest.

HANDOFF stub: `### D162: Session 23 Track B -- six observables defined to match the code and verified; case/encounter counts reconciled`.

-----

## Track C (light compute; trains only if AE seeds are missing): headline seed robustness

Session 20 Track A produced three seeds for the 2x2 control cells, but the
headline d=64 and d=32 JEPA / AE / POD comparison (Table 2) is single-seed. A
reviewer who sees the +-0.27 seed variance on the reconstructive CNN+ViT control
cell (current Table 5) will ask whether the headline is similarly fragile.

```bash
# JEPA seeds already exist (session14/thrust6). POD is deterministic. If AE seed
# encoders are missing, train two more (RTX-only, per CLAUDE.md):
python scripts/session9_train_fukami.py --gpu 0 --split v2 --latent-dim 64 --beta 0.01 \
    --tag-suffix AE_d64_seed1 --seed 1
python scripts/session9_train_fukami.py --gpu 1 --split v2 --latent-dim 64 --beta 0.01 \
    --tag-suffix AE_d64_seed2 --seed 2
# repeat at --latent-dim 32 if the d=32 row of Table 2 is also to be seed-reported

# Re-run the uniform B1 downstream eval over the seed encoders (reuse the chain):
python scripts/session23/exp_headline_seeds.py \
    --jepa-seeds outputs/runs/session14/thrust6/jepa_d64_seed{0,1,2} \
    --ae-seeds   outputs/runs/session*/AE_d64_seed{0,1,2} \
    --pod        outputs/session18/exp_b1_test3/pod_d64 \
    --closure-machinery outputs/session18/exp_b1_test3/ \
    --split v2 --horizon 16 \
    --out outputs/session23/headline_seeds/
```

Produce: Table 2 with mean +- standard deviation over three seeds, and the
per-seed paired JEPA-minus-AE wake-enstrophy improvement.

ACCEPTANCE GATE: report the paired wake-enstrophy improvement per seed. If it is
positive in all three seeds, add “The paired wake-enstrophy improvement is positive
in all three seeds” to Section 4.1. If a seed flips sign, report it plainly and
soften the headline to the median; do not hide it.

HANDOFF stub: `### D163: Session 23 Track C -- headline Table 2 seed-averaged; paired wake improvement [stable/variable] across seeds`.

-----

## Track D (no training): the stronger conditioning floor and the fairness fix

Clears gap 4. Session 20 Track B (D133) computed the held-out R^2 and fixed the
apples-to-oranges training-R^2 bug. This track strengthens the floor so it proves
what the paper claims, because the wake-enstrophy floor (R^2=0.482) currently sits
above the JEPA H=16 forecast (0.449), which makes “parameters alone cannot
generalise” false for that observable.

```bash
python scripts/session23/exp_conditioning_floor_plus.py \
    --split v2 \
    --inputs c phase c+phase nn krr_logocase \
    --observables CL CD Iy wake_enstrophy circ_pos circ_neg \
    --out outputs/session23/conditioning_floor_plus/
```

Add four floors beyond `c`-only: phase-only (the within-encounter phase/time index
tau), `c+phase`, nearest-neighbour-in-parameter-space, and kernel ridge with
leave-one-CASE-out cross-validation (group = case id, not encounter, to avoid
leakage across encounters of the same run). Then rewrite Table 4 and the Section
4.1 sentence around what the floor rules out (the gust parameters as the source of
the force and circulation gains; parameter interpolation failing on test_c), using
the replacement text in revision plan Part II.4.

ACCEPTANCE GATE: Table 4 has all five floor rows for all six observables on
train / test_b / test_c; the Section 4.1 over-claim is replaced; the wake-forecast
claim is explicitly rested on the paired test (Table 7) and the drift mechanism,
not on the floor.

HANDOFF stub: `### D164: Session 23 Track D -- conditioning floor strengthened (phase, c+phase, NN, leave-one-case-out KRR); 4.1 over-claim retired`.

-----

## Track E (light compute, the high-value physics addition): LEV and shear-layer tracking

Clears gap 5. This is the single change that turns the wake-enstrophy number into
a vortex and earns the JFM label. Reference: Odaka, Lopez-Doriga, Taira (JFM 1031
R3, 2026) and the PRF dataset paper. Builds on the Session 20 Track F large-scale
fields (sigma/c=0.05) you already have.

```bash
python scripts/session23/exp_lev_tracking.py \
    --dns-fields ${VORTEX_JEPA_CACHE}/v2 \
    --jepa-decoded outputs/runs/session12/S12_E_d64/encoder/decoder_specloss_recipe/recon \
    --fukami-decoded outputs/session18/exp_b1/fukami_recon \
    --pod-decoded outputs/session18/exp_b1/pod_recon \
    --large-scale outputs/session20/scale_decomp/ \
    --sigma-over-c 0.05 \
    --split v2 --frames impact H16 \
    --out outputs/session23/lev_tracking/
```

On the sigma/c=0.05 large-scale field, threshold the dominant suction-side
negative lobe, extract the LEV centroid (x_LEV, y_LEV), peak |omega_z|, and signed
LEV circulation Gamma_LEV, for DNS, JEPA decode, Fukami decode, and POD, at impact
and H=16 on test_b. (The extraction recipe is in revision plan Part V.1.)

Figure (feeds Track J Figure 5): x_LEV error vs horizon, Gamma_LEV error vs
horizon, and a scatter of per-encounter wake-enstrophy error against LEV-circulation
error. Earned sentence for Section 4.6 (template in Part V.1), filled with the
measured centroid distance, the retained-circulation fraction, and the Spearman
of wake-enstrophy error against LEV-circulation error.

ACCEPTANCE GATE: at H=16 on test_b the predictive decode has smaller LEV-centroid
distance and smaller LEV-circulation error than the reconstructive decode, and the
per-encounter wake-enstrophy error co-varies with LEV-circulation error
(Spearman > 0). If so, Section 4.6 becomes a statement about the leading-edge
vortex and shear layer rather than a scalar. If the predictive LEV error is not
smaller, that contradicts the wake-enstrophy probe gap and must be debugged before
claiming (cross-check against the D129/D131 probe-R^2 gap and the Track F result).

HANDOFF stub: `### D165: Session 23 Track E -- wake-enstrophy advantage localised to the LEV/shear-layer position and circulation`.

-----

## Track F (light compute, 3D fields confirmed present): the measured |G|=4 observability boundary

Clears the defensive test_c paragraph and turns the slice-vs-average point from
Track A into a number. The PRF source shows |G|>=4 becomes three-dimensional (the
bounding vortex pair destabilises before re-impingement); the full 3D vorticity
fields exist, so measure it directly.

```bash
python scripts/session23/exp_chi3d.py \
    --fields-3d ${VORTEX_JEPA_CACHE}/v2_3d   # 3D vorticity confirmed present; locate in the cache
    --split v2 \
    --out outputs/session23/chi3d/
```

Compute the spanwise-fluctuating enstrophy fraction
chi_3D(t) = int ||omega - <omega>_z||^2 dV / int ||omega||^2 dV (the content the
mid-plane encoder cannot see; formula and snippet in Part V.2), and plot
max_t chi_3D against |G| across the training range and at |G|=4. Because the
encoder is confirmed to read the mid-plane slice (Track A), chi_3D is the exact
measure of what it discards, so this also quantifies the slice-vs-span-average gap.

ACCEPTANCE GATE: a measurable jump in max_t chi_3D at |G|=4 relative to |G|<=3. If
present, it becomes an inset to Figure 1 (Track J) and the Section 2.1 / 5.3 test_c
sentences cite the number instead of citing the source paper alone. If the jump is
modest, report the curve as-is and keep the verbal argument; do not overstate it.

HANDOFF stub: `### D166: Session 23 Track F -- |G|=4 observability boundary measured by chi_3D; test_c degradation is a measured limit`.

-----

## Track G (no training): error maps over (G, D, Y, phi)

Replaces the deferred per-stratum table (current Section 4.4) with continuous trend
plots, which the small test set supports better than bins. Builds on the Session 20
Track E phase machinery.

```bash
python scripts/session23/exp_error_maps.py \
    --paired outputs/session18/exp_b1_test3/   # per-encounter JEPA and AE wake errors
    --phase  outputs/session20/phase_amplitude/  # shedding phase at impact
    --split v2 \
    --out outputs/session23/error_maps/
```

Define the per-encounter paired improvement Delta e = e_AE - e_JEPA on wake
enstrophy and plot it against G, D, Y, and the baseline shedding phase at impact
phi, four panels, each with a LOWESS curve and a bootstrapped band (statsmodels
`lowess`, bootstrap over encounters; snippet in Part VI.3). The phase axis is the
one the source papers care about (timing relative to the natural cycle); compute
phi at impact from the baseline limit cycle you already have.

ACCEPTANCE GATE: the four-panel figure exists with bootstrapped trend bands and a
one-paragraph reading of where the predictive advantage concentrates (expected: at
off-midplane Y, given the weak Y resolution). Feeds Track J and replaces the
“left to a study with more cases” sentence in Section 4.4.

HANDOFF stub: `### D167: Session 23 Track G -- error maps over the gust axes and shedding phase; Section 4.4 deferral removed`.

-----

## Track H (light re-plot + writing): promote pressure observability, demote the closed-loop pilot

Clears gap 7. This is where I diverge from the external critique, which wanted the
pressure material buried. Pressure observability is a main result and reinforces
the thesis; only the failed closed-loop control pilot is demoted. Split the two,
which are currently fused in Appendix B.

```bash
python scripts/session23/exp_pressure_observability_main.py \
    --pressure-csv outputs/session18/exp_b1_test3/pressure_recovery.csv \
    --split v2 \
    --out outputs/session23/pressure_main/
```

Promote one figure to the main text: cross-family latent recovery R^2 vs sensor
count K (current Fig. 13a) and the impact-C_L contrast (current Fig. 13b),
including the counter-intuitive result that the reconstructive d=3 latent is the
easiest to recover yet gives the worst lift estimate. Add the main-text subsection
“The predictive state is the most observable from the wall” (text in revision plan
Part IV). Keep the TCSI placement result and the closed loop in the appendix.
Reduce the closed-loop pilot to one labelled-limitation paragraph (oracle equals
estimator, 18% within band against an 80% target, so the bottleneck is the rollout
not the estimator). Remove “model-based control” from any headline claim.

ACCEPTANCE GATE: the pressure-observability subsection exists in the main text with
its promoted figure; the closed-loop pilot is one clearly-labelled limitation
paragraph; “model-based control” no longer appears as a headline claim in the
abstract or conclusion; the abstract and conclusion end on observability plus
forecastability (Track K applies the wording).

HANDOFF stub: `### D168: Session 23 Track H -- pressure observability promoted to a main result; closed-loop pilot demoted to a one-paragraph limitation`.

-----

## Track I (no training, STRETCH, optional, does not gate): the measured interventional test

Earns the world-model framing in the Introduction by measuring it rather than
asserting it. Reference for the method: Wang, Kou, Noack, Zhang (JFM 1035 A18,
2026), Galerkin-based Granger causality, as the causal-inference template.

```bash
python scripts/session23/exp_intervention.py \
    --predictor outputs/session18/exp_b1_test3/unified_predictor \
    --latents outputs/session14/latents/S12_E_d64 \
    --dns-fields ${VORTEX_JEPA_CACHE}/v2 \
    --perturb G --delta 0.5 \
    --split v2 \
    --out outputs/session23/intervention/
```

Perturb the conditioning c -> c + delta_c along one axis (delta G), roll the
predictor forward, and compare the predicted change in each observable to the
measured change between matched simulation encounters that differ only in G by
delta G. Report the correlation between predicted and measured response across the
parameter grid.

ACCEPTANCE GATE (optional): if the predictor’s response to a parameter intervention
matches the simulation’s across the grid, the Introduction keeps the interventional
world-model language as a result and cites the GGC paper. If it does not match,
soften Section 1 to “conditional forward model” and report the test as a limitation.
Either outcome is reportable; this track does not block the others.

HANDOFF stub: `### D169: Session 23 Track I (optional) -- interventional response [matches/does not match] simulation; world-model framing [earned/softened]`.

-----

## Track J (depends on E, F, G, H outputs): the physics-led figure rebuild and JFM conventions

The reference papers are figure-led; your first flow-physics panels are very late
and the pressure figures dilute the story. Rebuild to five physics-led main figures
plus appendix figures (the plan is in revision plan Part VII).

```bash
python scripts/session23/build_figures.py \
    --inputs outputs/session20 outputs/session23 \
    --style jfm \
    --out paper/figures/
```

Main figures: Fig 1 flow and dataset (configuration, Taylor vortex, G/D/Y, impact
definition, four stages, the split, plus the chi_3D inset from Track F); Fig 2 the
matched protocol (JEPA/AE/POD into one predictor and probe family, with the
predictive-vs-reconstructive training contrast folded in as a sub-panel); Fig 3
headline closure (a forest plot or heatmap of held-out R^2 plus the paired
per-encounter wake panel, paired result made primary); Fig 4 drift mechanism
(horizon dependence + Mahalanobis ratio + persistent-H1 count in one figure); Fig 5
transport and physical wake (vorticity stages + OT field distance + large-scale
wake enstrophy + the new LEV centroid/circulation error from Track E).

Repair the specific defects (a reviewer will list these): the clipped Fig. 9
x-axis label; the tiny, disconnected baseline cycle in Fig. 8 (normalise axes or
inset so the return to orbit is legible; label the phase wrap); the missing
early-horizon curves in Fig. 6 (label unavailable horizons rather than blank
space); one consistent family label everywhere (“predictive (JEPA)”,
“reconstructive AE”, “POD”; retire “recon.” and “Fukami”); identical spatial
extent, colourbar, ticks, and airfoil scaling across all vorticity panels with
x/c, y/c labelled at least once per figure. Export vector PDF, minimum line width
0.5 pt, RGB, captions beneath, figures cited in order.

Add the two venue-convention items the reference papers have and the draft lacks:
encoder and predictor architecture tables (convert the Appendix A prose to layer
tables in the style of Tran et al. Table 1), and the scale-decomposition equations
written out in Section 4.6 in the style of Odaka et al. (3.1)-(3.4). Produce the
JFM graphical abstract: airfoil + incoming vortex + three small latent rollouts
labelled by colour only, no dense text.

ACCEPTANCE GATE: the five main figures are regenerated to spec, all listed defects
are fixed (verify the Fig. 9 label is not clipped and the Fig. 8 baseline cycle is
legible by rendering and reading the PDFs), the architecture tables and scale-decomp
equations are in the source, and the graphical abstract exists as a standalone
vector file.

HANDOFF stub: `### D170: Session 23 Track J -- figures rebuilt physics-led to JFM conventions; defects repaired; graphical abstract and architecture tables added`.

-----

## Track K (the integration track, depends on A, B, C, D, E, F, G, H): the rewrite and compliance pass

Applies the paste-ready rewrites (revision plan Part III) and clears gap 3, the
most dangerous one, the self-contradiction about the controls.

Fix the controls contradiction first. Grep the source for the three contradictory
phrasings and unify them on the single claim (revision plan Part II.1):

```bash
grep -n 'immediate next step\|set out the matched\|set out the controls\|next step is to run those controls' main.tex
# Outlook must not say the controls are still to be run; they are in Table 5.
```

Then apply, in order: the abstract (revision plan III.1, single paragraph, about
250 words, ending on observability plus forecastability); the contributions
paragraph (III.2, including the “without being trained to” OT-alignment contrast
with Tran et al.); the mixed-ordering results sentence (III.3); the paired-test
headline ordering (III.4, lead with the paired statistic, demote the wide marginal
interval to a parenthetical); the discussion scope paragraph (III.5); the
conclusion (III.6).

Compliance pass (revision plan Part VIII): single-paragraph abstract at or under
250 words; remove the manual Key words line unless the JFM class requires it; add
competing-interests, data-availability, funding, and author-contributions sections;
audit arXiv / under-review citations and, in particular, make the SIGReg
(characteristic-function regulariser) description in Appendix A fully self-contained
so reproducibility does not depend on the unreviewed LeWM preprint; prepare the
AI-use declaration with tool, version, dates, and description.

ACCEPTANCE GATE:

- No internal contradiction about the controls remains: the grep above returns
  nothing in the Outlook, and the abstract, Section 4.5, Section 5.2, and the
  conclusion all use the single claim.
- The abstract is one paragraph at or under 250 words.
- Every `\pending{}` in the manuscript is cleared, with one allowed exception:
  the DNS solver-resolution author-fill block in Section 2.2, which the project
  lead supplies separately. Keep that block clearly marked (an author-fill note,
  not invented numbers) and flag it in the HANDOFF note as the one remaining
  author input. Any other value still unavailable after Tracks A and B is listed
  in the questions HANDOFF note, not invented.
- A clean build with no undefined references or citations (the Section 2.2
  author-fill note is expected and is not a build error):
  
  ```bash
  latexmk -pdf -interaction=nonstopmode main.tex && grep -i 'undefined\|multiply defined' main.log
  ```

HANDOFF stub: `### D171: Session 23 Track K -- manuscript rewritten and compliant; controls contradiction removed; clean build, no pending`.

-----

## Dependency graph and scheduling

```
Pre-flight --+--> Track A (methods archaeology) .... no compute, START FIRST, documents the confirmed methods facts [CPU + data]
             +--> Track B (observable defs + counts) no compute, START FIRST [CPU + data/code]
             +--> Track C (headline seeds) .......... light; trains AE seeds only if missing [RTX0+RTX1 iff training]
             +--> Track D (conditioning floor+) ..... no training [CPU/L40S]
             +--> Track E (LEV tracking) ............ light, needs Track-F-style large-scale fields [light GPU]
             +--> Track F (chi_3D boundary) ......... light, needs 3D fields [light GPU/CPU]
             +--> Track G (error maps) ............. no training [CPU]
             +--> Track H (pressure promote/demote)  light re-plot + writing [CPU]
             +--> Track I (intervention, OPTIONAL) .. no training [CPU/L40S]
                         |
   Tracks E,F,G,H ---> Track J (figure rebuild) ..... depends on their outputs [CPU]
   Tracks A,B,C,D,E,F,G,H ---> Track K (rewrite + compliance) .... integrates everything [CPU, latexmk]
```

Launch A, B, D, E, F, G, H, and (optional) I in parallel the moment pre-flight
passes; they read existing artifacts or the data and need no encoder training.
Start Track C first only if the AE seed encoders are missing, since that is the
session’s only RTX work (encoder training stays RTX-only per CLAUDE.md; the rest
runs on L40S and CPU under the bypass). Track J waits on E, F, G, H. Track K
integrates everything and is last.

## What this session produces for the manuscript

- Section 2.2 numerical-method subsection scaffolded; DNS (no subgrid model) and
  mid-plane caching stated; the DNS resolution numbers left as an author-fill block
  for the project lead (Track A).
- The six observables defined with equations that match and are verified against
  the code; case/encounter counts reconciled (Track B).
- Table 2 seed-averaged with the paired wake improvement reported across seeds
  (Track C).
- Table 4 conditioning floor strengthened and the Section 4.1 over-claim retired
  (Track D).
- Section 4.6 upgraded from latent-space argument to a measured statement about the
  LEV and shear-layer position and circulation (Track E), with the |G|=4 test_c
  degradation converted to a measured chi_3D observability boundary (Track F).
- Section 4.4 deferral replaced by error maps over the gust axes and shedding phase
  (Track G).
- Pressure observability promoted to a main result; the closed-loop pilot demoted
  to one labelled limitation (Track H).
- Optionally, the world-model framing earned by a measured interventional test, or
  softened if it fails (Track I).
- Five physics-led figures rebuilt to JFM conventions with the defects repaired,
  plus architecture tables, the scale-decomposition equations, and a graphical
  abstract (Track J).
- Abstract, contributions, results claims, discussion scope, and conclusion
  rewritten; the controls self-contradiction removed; a clean, compliant build
  (Track K).

Every track has a HANDOFF decision stub (D161-D171, renumbered to your log) so the
session drops straight into the decision-log format.

## A note on honesty and the no-slop bar, in your project’s style

JFM at this level does not forgive invented numbers or claims the data does not
support, and three things in this session are exactly where slop would creep in.
Guard them:

- The SOD2D methods values (Track A) and the observable equations (Track B) come
  from the actual configs and the actual code. If a value is not in any artifact,
  it stays `\pending{}` and goes to the human as a blocking question. Do not fill a
  plausible Mach number, a plausible Delta n+, or a guessed I_y sign to make the
  section look finished. A reviewer will recompute these.
- Four tracks can return a weak or negative result: Track C (a seed may flip the
  wake improvement), Track E (the LEV error may not separate, which would
  contradict the probe gap and must be debugged, not written around), Track F (the
  chi_3D jump may be modest), and Track I (the intervention may not match). Each
  gate above says what to claim if strong and what to claim if weak. Follow them.
  The manuscript already hedges to “predictive family” where the evidence is
  partial; keep it honest. Rewrite the draft to match the result, never the result
  to match the draft, exactly as the decision log did when earlier plan gates
  failed.
- The pressure promotion (Track H) is a genuine result; the closed-loop pilot is a
  genuine negative. Report both as what they are. The strength of the paper is that
  the predictive state is both forecastable and observable, and that the one thing
  that does not yet work, closed-loop control, is bounded by the rollout and said
  so plainly.