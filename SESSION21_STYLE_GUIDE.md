# Taira-group figure and narrative style guide

Distilled from four exemplar figures read directly from the reference papers:
Tran, Yeh & Taira (JFM 1027 A24) Fig 4 (schematic) and Fig 7 (latent-space
centrepiece); Smith et al. (JFM 980 A18) Fig 5 (encounter-as-cycle); Fukami,
Nakao & Taira (JFM 992 A17) Fig 3 (lift histories over parameter space). This
file is the visual contract for Session 21B. Every new or rerendered figure must
obey it. The goal is to match or exceed these papers, not approximate them.

## The seven rules (non-negotiable)

1. No prose inside a figure. No sentence-length titles, no explanatory boxes that
   are really paragraphs. Axis labels, short legends, and at most one short
   annotation per panel. Everything else goes in the caption. (Current draft
   violates this in Figs 5, 6, 8, 9.)
1. No internal vocabulary anywhere in a figure or axis: not “Track G”, not “D-i”
   or “D-ii”, not “A1”..“A5”, not “test_a/b/c” raw if a worded label reads better,
   not session numbers. Reader-facing names only.
1. One shared visual key ties the abstract space to the physical flow. The
   exemplars use lettered or numbered glyphs (circles with A..G in OT Fig 7;
   circles with 1..4 in Smith Fig 5) that appear identically on the latent
   scatter, the phase curve, and the flow snapshots. Adopt numbered stage glyphs
   1..4 for the encounter and reuse the same glyphs across every panel and figure
   that shows the encounter.
1. Encoders/predictors are simple lettered glyphs, never layer stacks with loss
   equations. OT Fig 4 draws the encoder as a single green trapezoid labelled E
   and the decoder as a peach trapezoid labelled D, with one red line in the
   latent box for the core idea. Flow path left to right. Loss terms live in the
   caption or methods, not on the diagram.
1. Show the encounter as a time sequence of snapshots, not a single frame. Smith
   Fig 5 and FNT Fig 3 both lay out a column/row of vorticity snapshots at the
   marked stages. A single snapshot is never sufficient for a transient.
1. Dots with error-bar whiskers, not bar charts, for quantitative comparison
   across families. Taira figures essentially never use bar charts. Consistent
   family colour across the entire paper. A reference line (DNS-oracle lower
   bound) drawn horizontally. Log axis only where the dynamic range demands it.
1. Captions carry all interpretation, in one to three sentences, declarative.
   Smith Fig 5 caption is a single sentence naming the case. OT Fig 7 caption is
   two sentences. Write captions that explain the panels and state the takeaway,
   never a paragraph of analysis (that is body text).

## Vorticity-snapshot convention (keep, make identical everywhere)

Red-blue diverging colormap, airfoil drawn as a solid black body, fixed symmetric
range across every snapshot figure, single shared colourbar labelled with the
normalised vorticity. This is already the draft’s convention and matches the
references; the only fix is to make the range and colourbar label byte-identical
across Figs and to annotate each snapshot with its OT field distance (not SSIM)
in a corner, per the OT-reframing in the main text.

## Latent-space-as-map convention (OT Fig 7, the centrepiece grammar)

When plotting a latent scatter or loop: clean axes (e.g. PC1, PC2 or xi_1, xi_2),
colour encodes a continuous physical scalar with a small in-corner colourbar,
marker shape can encode a discrete parameter, numbered/lettered glyphs with short
leader lines tag representative cases, and at most one single-line arrow annotation
states the trend (OT Fig 7b: “Decreasing separation bubble size in response to
actuation”). Place representative-case flow structure adjacent (OT Fig 7c stacks
isocontours; Smith Fig 5c/d stack snapshots).

## Figure sizing for the JFM A4 LaTeX layout (critical)

These figures go into a single-column JFM Standard Article on A4 compiled with
jfm.cls. Design every figure at the physical size it will print, so nothing is
rescaled by \includegraphics and fonts never shrink below legibility. Rescaling a
large canvas down is the failure mode behind the current “diagnostic-dump” look.

- Measure the real text width once. Put \the\textwidth in the compiled document
  and read it (jfm.cls on A4 is roughly 5 inches / about 33 pica; use the MEASURED
  value, do not guess). Build full-width figures at that width and partial-width
  figures at the chosen fraction. Include with width=\textwidth (or the fraction)
  at scale 1, so design size equals print size.
- Match the document fonts and sizes. In matplotlib: font.size 8, axes.labelsize 8
  to 9, tick labelsize 7, legend.fontsize 7. Use a serif/TeX font to match the body
  (usetex, or mathtext with a Computer-Modern-like font) so figure text and paper
  text look consistent.
- Line and marker weights for print at 1:1: lines.linewidth about 1.0, markersize
  about 4 to 5, thin error-bar caps, axis spines about 0.8. The numbered stage
  glyphs and family markers must stay distinguishable at column width; prefer a few
  clear glyphs to many faint points.
- Multi-panel figures (NEW FIG A small multiples, NEW FIG C) must fit within
  \textwidth by \textheight at final size; lay panels out with constrained_layout
  or tight_layout and explicit spacing, never by scaling a big canvas down.
- Vector PDF for all line/scatter figures; 300+ dpi raster only for vorticity
  snapshot panels, generated at final panel size (not a large image downscaled).
- Acceptance test per figure: view it at the intended column width and confirm the
  smallest text is legible and the markers separate. If it is not legible at print
  size, the figure is wrong no matter how it looks zoomed in.

## Narrative and abstract style

- Abstract: one paragraph, about 230 to 270 words, confident, ends on
  significance. Lead with the question and the result; do not spend a third of it
  on caveats. (Replacement abstract already drafted in the review; use it with the
  DNS/SOD2D wording.)
- “Key words:” line after the abstract. OT paper uses “machine learning, big data,
  separated flows”; Smith uses “vortex shedding, low-dimensional models, machine
  learning”. Use the latter three.
- Section rhythm matches these papers: a tight intro that states the gap and the
  three contributions; a methods section that names the flow solver and
  configuration up front (provenance before cache layout); results that lead each
  subsection with the question it answers; a discussion that states scope honestly
  but crisply.
- Sentences: precise but readable. Split any sentence running more than about
  three lines. The reference papers rarely stack more than two subordinate
  clauses.
- Declarative captions and topic sentences. Each results subsection opens by
  telling the reader what question the subsection settles.