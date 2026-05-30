# Session 21B: figures (Path A, target ~8 main-text figures)

Paste into Claude Code with the repo open. FIRST read `SESSION21_STYLE_GUIDE.md`
in full; it is the visual contract and every spec below assumes it. This is the
largest-effort, highest-payoff session on the “looks like JFM” axis.

All figures are produced from EXISTING saved arrays (rollouts, decodes, latents,
closure tables, per-encounter errors). The Session 20 analyses
(`scripts/session20/exp_*.py`) already computed everything; locate their outputs
rather than recomputing. Do not retrain. Match or exceed the reference figures.

Family colour key, fixed across the WHOLE paper (define once, import everywhere):
predictive/JEPA, reconstructive/Fukami, POD, and a neutral DNS/DNS-oracle colour.
Use the same four colours in every figure and the legend.

Net figure plan for the main text (about 8): one architecture schematic (Fig 2
kept, Figs 1 and 3 simplified/merged); Fig 4 closure (rerendered); Fig 5 horizon
(de-cluttered); compact persistent-homology figure; NEW FIG A observable traces;
NEW FIG B parameter space; NEW FIG C encounter-as-cycle (replaces Figs 7,10,11,12);
Fig 8 OT (de-cluttered); Fig 9 scale decomposition (de-cluttered). Deployment Fig
14 stays (see Session 21C); Figs 13 and 15 move out.

========================================================================

## SPEC 1 – Fig 4 rerender (the broken centrepiece closure figure)

========================================================================
Problem: rotated/overlapping axis text, bar-chart style. This is the main result
figure and currently looks broken.

Purpose (one idea): held-out forward closure separates the families, most clearly
on the wake observable.
Inputs: held-out closure values + bootstrap 95% CIs at H=16 for every family
(d-variants), on test_b and test_c, for the six observables (the arrays behind
Table 2/3 and current Fig 4).
Layout: 2 rows (test_b top, test_c bottom) x 3 columns (force {C_L,C_D}, impulse
{I_y}, wake {enstrophy, circulations}). Dots with vertical 95% CI whiskers, NOT
bars. Families by the fixed colour key, consistent marker per family. Draw the
DNS-oracle lower bound as a horizontal reference line in each panel. Log y only in
the wake panel if range demands. All axis labels horizontal and legible. No
in-figure title; no “Track”/“D-” labels.
Caption: state that error bars are bootstrap 95% CIs over n=42 (test_b) and n=24
(test_c), and that the wake panel shows the clearest separation.

========================================================================

## SPEC 2 – NEW FIG A: predicted observables through an encounter

========================================================================
Model: Fukami, Nakao & Taira Fig 3 (lift histories with glyphs + snapshot strip).
This is the most compelling figure the paper lacks: it shows the closure the
tables only quantify.

Purpose (one idea): under rollout, the predictive trace tracks the simulation
through the LEV peak and into recovery while the reconstructive trace flattens or
diverges.
Inputs: for THREE representative test_b encounters (a weak gust, a strong-positive
gust, a strong-negative gust; pick by |G| and sign from the (G,D,Y) labels), the
simulation traces C_L(t), wake-enstrophy(t), and signed circulation(t) over t from
about -10 to +40 frames around impact, plus the Markov-rollout predictions of
JEPA, Fukami, and POD for the same window (the rollouts already saved for the
closure tables).
Layout: small multiples, 3 rows (the three encounters) x 3 columns (C_L,
wake-enstrophy, circulation). In each cell: simulation as a bold solid reference
line, the three family predictions as the fixed colour key. Mark encounter stages
with the numbered glyphs 1..4 (the SAME glyphs as NEW FIG C) at the stage frames.
Optionally show the all-test_b envelope faintly behind the highlighted encounter
(as FNT does). Shared legend once.
Caption: name the three encounters by (G,D,Y); state that the predictive rollout
tracks the wake observables through impact and recovery while the reconstructive
rollout flattens, and that stage glyphs match NEW FIG C.

OPTIONAL complement (turns the not-done Section 4.4 analysis into a real result;
the authors invited additions): add a 4th column or a small inset, “closure vs
gust regime”: held-out wake-enstrophy error (or R2) versus signed G, points
coloured by family, using the per-encounter errors and (G,D,Y) labels you already
have. This is cheap and pre-empts the referee question about where closure holds.

========================================================================

## SPEC 3 – NEW FIG B: the parameter-space sampling

========================================================================
Model: Fukami, Nakao & Taira Fig 3 lower-right 3D (G,D,Y) scatters.

Purpose (one idea): the split is stratified, the |G|=4 set is the OOD boundary,
and training mass concentrates near Y=0 (which is why Y is weakly resolved).
Inputs: the (G,D,Y) coordinates and split label for all 84 cases / the encounter
partition (train, test_b interior, test_b boundary, test_c).
Layout: either a single 3D (G,D,Y) scatter or a 2x2 of 2D projections (G-D, G-Y,
D-Y). Points coloured by split (four colours). Make three things visually obvious:
the test_b stratification, the |G|=4 OOD boundary, and the Y=0 concentration.
Caption: state the partition counts (226/28/42/24) and that the Y=0 concentration
explains the weak-Y recoverability reported later.

========================================================================

## SPEC 4 – NEW FIG C: “the encounter as a cycle”  (CENTREPIECE)

========================================================================
Model: Smith et al. Fig 5 exactly (latent loop + phase + snapshot columns with
numbered stage glyphs). Replaces current Figs 7, 10, 11, 12.

Purpose (one idea): the gust encounter is a single closed cycle in the predictive
latent; it departs the baseline limit cycle, executes LEV growth and shedding, and
returns, and the predictive decode tracks the staged flow while the reconstructive
decode collapses.
Inputs: predictive latent trajectory for one representative test_b encounter
projected to its leading two PCs, with the baseline (no-gust) limit cycle overlaid
(from exp_phase_amplitude.py); the phase theta(t) along the orbit (Hilbert analytic
signal of the two leading PCs); simulation, JEPA-decode, and Fukami-decode
vorticity snapshots at the four stages; the per-snapshot OT field distance (from
exp_ot_field_and_alignment.py).
Layout (Smith Fig 5 grammar):
(a) latent loop in (PC1, PC2): baseline limit cycle as a light closed ring, the
gust trajectory overlaid, four numbered stage glyphs 1..4 around it
(1 baseline/far-field, 2 LEV growth at impact, 3 LEV+TEV shedding at peak
load, 4 recovery), a direction arrow.
(b) phase theta(t) vs t/c (frames relative to impact), 0 to 2pi, with the four
stages marked by dashed verticals carrying the same glyphs.
(c,d,e) three columns of mid-plane vorticity snapshots at stages 1..4:
(c) simulation, (d) JEPA decode, (e) Fukami decode. Vorticity convention per
the style guide (red-blue, black airfoil, fixed range, shared colourbar).
Annotate each snapshot with its OT field distance in a corner (NOT SSIM).
Same numbered glyphs label the rows.
Caption: name the case by (G,D,Y); one sentence that this single figure carries
the limit cycle, the phase reduction, the staged encounter, and the
reconstruction comparison; state that the reconstructive decode collapses while
the predictive decode localises the impingement.

========================================================================

## SPEC 5 – Compact persistent-homology figure (de-clutter current Fig 6)

========================================================================
Purpose (one idea): the predictive encoding is a single clean loop; the
reconstructive encoding fragments (median 1 vs ~4 generators, p=4.4e-8).
Inputs: two representative Vietoris-Rips H1 persistence diagrams (predictive and
reconstructive, simulation-encoded) and the loop-count histogram (from
exp_persistent_homology.py).
Layout: two persistence diagrams + the generator-count histogram with the p-value
annotated minimally. DELETE the paragraph baked into the current figure; move it
entirely to the caption. Move the H1-lifetime-versus-horizon panel (the
confounded/negative result) to the appendix or reduce to one body sentence.
Caption: carries the method (Vietoris-Rips, what an H1 generator is), the counts,
and the p-value.

========================================================================

## SPEC 6 – De-clutter Figs 5, 8, 9 (text to captions, strip internal labels)

========================================================================

- Fig 5 (horizon sweep): remove the in-image title “Track G: held-out closure…”
  Keep the R2=0.5 dotted reference line. Move all explanation to the caption.
- Fig 8 (OT): remove panel sub-titles “D-i:” and “D-ii:”; describe both panels in
  the caption instead. Keep the OT-field-distance-vs-SSIM comparison and the
  Shepard panel; render in the latent-map style of the guide.
- Fig 9 (scale decomposition): move the baked-in “|G|=4: flow is 3-D (mid-plane
  decomp. incomplete)” annotation into the caption. Keep the large-scale field
  triptych and the staged-enstrophy curves.

========================================================================

## SPEC 7 – Schematics (Figs 1-3 -> simplify toward OT Fig 4 style)

========================================================================
Model: OT Fig 4 (horizontal, encoder/decoder as single lettered glyphs, one core
idea visualised, no loss equations on the diagram).

- Fig 1: strip the five loss-term annotations (L_pred, L_roll, L_SIGReg, L_wake,
  L_obs) into the caption/methods. Keep only the data path
  (omega -> encoder E -> z -> predictor P -> z_hat) with the frozen visualisation
  decoder shown greyed and the conditioning c = (G,D,Y) entering ONLY the predictor
  as a short labelled arrow.
- Fig 2 (predictive vs reconstructive): keep as the conceptual centrepiece; this
  is the clearest schematic. Consider making it the single architecture figure.
- Fig 3 (fairness protocol): merge the “shared predictor, shared conditioning”
  idea into Fig 2 (show the shared predictor once) and CUT Fig 3, or demote it to
  the appendix. Three schematics for one architecture is more than the references
  spend.
- Optional elegant unifier: one schematic that doubles as the world-model diagram
  (encoder -> predictor(state | action) -> decoder, with the gust as the
  intervention a), carrying both the architecture and the world-model framing.

========================================================================

## Output and checks

========================================================================

- Vector output (PDF/EPS) for all line/scatter figures; 300+ dpi raster only for
  the vorticity-snapshot panels.
- DESIGN AT PRINT SIZE for the JFM A4 layout: follow the “Figure sizing for the
  JFM A4 LaTeX layout” section of the style guide. Measure \textwidth from the
  compiled document, build each figure at that physical width (or a fraction),
  include at scale 1, and set fonts to 7 to 8 pt and markers/line weights for
  legibility at column width. Do not design large and let LaTeX shrink it.
- Identical family colours and identical vorticity colourbar range/label across
  every figure.
- The numbered stage glyphs 1..4 are byte-identical between NEW FIG A and NEW FIG C.
- After building, confirm the main-text figure count is about 8 and every figure
  obeys the seven rules in the style guide (no prose in figures, no internal
  labels, shared keys, dots-not-bars, captions carry interpretation).