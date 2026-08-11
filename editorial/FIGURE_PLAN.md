# FIGURE_PLAN.md

## Session 39 (comparison-led reframe): the redesign figure plan (target 13 main)

Per paper_redesign.md section 5, the main-text figure target is 13, mapped to
the six-beat structure. This pass is figure-engineering (matplotlib regen; the
GPU compute tracks T1/T3/T4 feed panels here) and carries the D-A and D-B
demotions, which are figure-coupled and MUST move prose + figure together:

| new # | content | source (existing) | needs |
|---|---|---|---|
| 1 | gust sweep: lift + vorticity | fig:staging | keep |
| 2 | parameter space + splits | fig:paramspace | relabel |
| 3 | pipeline schematic: 3 families -> probe + forecaster | rework fig 4 + fig 22 panel | TikZ redraw |
| 4 | conditioning: PR + peak-load readability, **L and W axes only** | fig:cube trimmed (drop N cols) | regen (D-A) |
| 5 | task-dependent readability heatmap, trimmed rows | fig:readability | regen |
| 6 | dimension axis: peak lift, probe dilution, decoded SSIM | fig:dimrace | keep |
| 7 | decoded instants, 3 phases, 3 families + **POD row (T2)** | fig:decode + T2 | T2 (POD decode) |
| 8 | physics: atlas, DMD, **per-coord spectra (T1)**, **portraits (T3)** | fig:atlas + T1 + T3 | T1,T3 (GPU) |
| 9 | forecastability: shared-op bars, horizon curves w/ **phase split (T5)**, direct-vs-AR | fig:forecast + T5 | T5 |
| 10 | rollout geometry: near-null + drift (or fold into 8) | fig:mechanism_hroll | regen |
| 11 | sensors traded for delays + recovery grid | fig:trade + tab:recovery | keep |
| 12 | filter tracking across envelope + sign asymmetry | fig:hero/cl_envelope/envelope condensed | regen |
| 13 | assimilation vs dimension grid | fig:dimsgrid | keep |

TO APPENDICES (D-A/D-B, figure-coupled, do WITH the prose move):
- D-A: fig:phi (Chang construction) + the N columns of fig:cube; s4_a N-cube
  prose trims to one CLN sentence, N detail -> appendix.
- D-B: fig:centerpiece(c ladder, d smoother, e NIS), fig:relerr, fig:deploy
  (streaming, d=4 panel) -> appendix; s4_d ladder/NIS/smoother/streaming prose
  -> appendix with one pointer in sec:res_estimation. (The §5.4 "Choosing the
  estimator" discussion synthesis already moved to appendix_b_sensing.tex.)
- Also to appendix: fig 18 peak-error scaling, figs 19/20 per-family phase and
  sensor sweeps, fig 23 recovered-field, fig 22 training-contrast (unless used in 3).

Compute tracks feeding these (paper_redesign.md section 6): T1 per-coordinate
Welch PSD spectral flatness (GPU; gates the D-E "broadband" wording); T2 POD
decode row (light); T3 latent PC1-PC2 portraits (GPU); T4 decoded-forecast SSIM
(GPU); T5 phase-split forecast re-aggregation (light, existing records).

---

# FIGURE_PLAN.md (Session 38, Stage 5)

Dispositions per the editorial memo section 7 (target 11-12 main figures, 5
main tables). Numbering = the Stage 0 concordance (main.aux). Status DONE =
landed this session; TODO(M3x) = needs the figure-engineering pass with
Carlos available for the layout calls (D302/D305 land there too).

| # | Label | Disposition | Status |
|---|---|---|---|
| 1 | fig:staging | KEEP | done (no change; v2p1 asset, see provenance note below the table) |
| 2 | fig:paramspace | KEEP; split aliases in labels | TODO (M3, relabel pass) |
| 3 | fig:phi | MOVE to appendix (Chang construction) | TODO (M3) |
| 4+5 | fig:method(+arch/eval), fig:estimation_loop | REDRAW as one pipeline schematic | TODO (M3b) |
| 6 | fig:cube | REDRAW: flatten to two panels, single (a)/(b) labels | TODO (M3a) |
| 7 | fig:readability | KEEP | done (no change) |
| 8 | fig:decode | MOVE to appendix | TODO (M3) |
| 9 | fig:dimrace | KEEP | done (no change) |
| 10 | fig:readability_matrix | DROP (duplicates tab:closure; panel b was D310-superseded) | DONE |
| 11+12(b,c) | fig:forecast, fig:mechanism_hroll | REDRAW as one forecasting figure; 12(a) joins the schematic or appendix | TODO (M3c) |
| 13 | fig:trade | KEEP | done (no change) |
| 14 | fig:hero | KEEP; archive-signed case_id headers REPLACED by |G|, D, Y (G sign withheld pending the s3.5 sign audit) | DONE (regenerated) |
| 15+16(a,b)+17(a) | fig:envelope, fig:centerpiece(a,b), fig:relerr(a) | REDRAW as one envelope/phase figure | TODO (M3d) |
| 16(c)+20 | fig:centerpiece(c), fig:dimsgrid | REDRAW as ladder + dimension grid, no in-panel prose | TODO (M3e); interim: 16(e) "test A" -> "validation" and ladder annotations at the prose precision DONE (regenerated) |
| 16(d,e), 17(b) | | MOVE to appendix/supplementary | TODO (M3) |
| 18 | fig:ownstack | MOVE to appendix/supplementary | TODO (M3) |
| 19 | fig:deploy | MOVE to appendix/supplementary | TODO (M3) |
| 21 | fig:atlas | REDRAW slim (DMD spectrum + one atlas panel), stays in 4.1 per the memo recommendation; D302 confirm | TODO (M3f) |
| 22-25 | appendix figs (incl. fig:recon, fig:pooling_cost already moved to supplementary S3) | remainder to supplementary | partially DONE |

Provenance note (2026-07-18 audit): fig:staging is built from `fig_staging_v2p1.pdf`
(scripts/session29, v2p1 cache) while every other data figure is v2p2. This is
data-identical, not stale: the v2p2 cache symlinks the 87 v2p1 encounters and the
staging sweep draws only pre-existing cases, so a v2p2 regen would reproduce the
same fields. Deliberate; do not regenerate.

Caption contract for every kept/redrawn figure (checked at the M3 pass):
content; split and n; uncertainty convention; the single primary inference.
Labels legible at JFM column width at 100 per cent.

Standing FIGURE-TODOs carried in-source: none remaining in tex (the
readability-matrix TODO died with the figure; the centerpiece precision
TODO is resolved by the regeneration; the TikZ architecture asset still
says "direct REX (quantile LSTM, h=512)", rename to the \DirectFC wording
at the M3b schematic redraw).

## Session 38 figure model-provenance audit (Carlos directive: best models everywhere)

Sweep of all 28 built assets (24 data figures + 4 TikZ schematics): per-figure
generating script, data inputs, and model generation verified.

CONFIRMED CURRENT (no action): fig_cl_envelope_traces (two-stage b177 on vec),
fig_da_centerpiece (b177/vec), fig_cube_* (Track C cube, its own experiment),
fig_da_dims_grid (D261 vec lineage), fig_decode_panels (vec + CLN),
fig_deployment (b177 d4), fig_dimension_race (S34/35 ladders), fig_forecast40
(vec REX), fig_forecastability (rex2_cov on vec), fig_ownstack + fig_relative
_errors (da_phase_eval on vec), fig_phi_panel / fig_staging / fig_paramspace
(data-only), TikZ schematics. IMPORTANT subtlety: the session-33 physics
re-runs keep LEGACY KEYS pointing at vec latents (spectrum_dmd + manifold
atlas "jepa_pool" -> latents_jepa_pool_vec; mechanism "jepa_wake_pool" ->
latents_jepa_pool_vec), so fig:atlas and tab:mechanism/fig:mechanism_hroll
ARE the flagship generation despite their labels.

DELIBERATE PREVIOUS-GENERATION (kept, spatial-era by necessity): Gate O
recovery (tab:recovery) and fig_pooling_cost compare pooled vs SPATIAL
latents; the vec flagship has no spatial latent, so the spatial-era model is
the only possible substrate (pooling-losslessness record).

STALE, FIXED THIS SESSION:
- fig:envelope (15): read the session32 jepa_pool envelope while tab:envelope
  quotes the vec envelope. Re-pointed to envelope_vec.json / jepa_pool_vec;
  the off-scale open-loop annotation made data-driven; regenerated.
- fig:hero (14): traces were the jepa_pool generation (D220-era dump).
  Re-dumped on the vec flagship (same frozen filter recipe, median-rule
  picks per |G| in {1, 1.5, 2}) -> outputs/session38/hero_traces_vec.json;
  regenerated with de-archived |G|, D, Y headers.
- fig:trade (13) annotation: divergence-boundary note came from the session32
  jepa_pool envelope; re-pointed to the vec envelope and regenerated.
- Supplementary decode gallery (fig:recon + fig:field_recovery): decoded the
  jepa_pool latents; re-pointed to the vec latents + vec OSP taps
  (regeneration running at write time).

RESOLVED (no numbers change needed): the Track T story. The paper's macros
were ALREADY on vec re-runs made at the session-33 freeze (parts read
track_t_grid_vec.json / track_t3_vec.json; t2b_reduced_filter.json is
itself the vec run, model field checked) but fig:trade still plotted the
ORIGINAL jepa_pool grid: figure/table generation mismatch, same class as
the envelope. Fixed: fig_track_t re-pointed to the same vec artifacts the
numbers pipeline uses; regenerated. BONUS VERIFICATION: an independent
protocol-identical re-run of the full K x W grid on the vec flagship
(outputs/session38/track_t_recovery_grid_vec.json, GPU 1) REPRODUCES the
frozen session-33 vec grid to numerical noise (max cell delta < 0.001) and
replicates GATE T2 STRONG with the same strong cells and the same
(K=1, W=30) pick. Against the old jepa_pool grid, all 24 vec cells sit
uniformly lower by 0.02-0.07 with identical ordering: the trade structure
is generation-robust.

## Session 38 correction (d = 4 lineage reading)

An earlier session-38 note framed the T6 deployment number
(\PoneDFourFilterImpact = 0.782, lift-focused CLN-rexpred d4 with its own
tuned stack, five seeds) as "winning" at d = 4. WRONG, and Carlos caught
it: the paper's own dims grid posts the WAKE lineage at d4 at R2 0.782 /
RMSE 0.298 (best recipe eobs), equal R2 and better RMSE, and the shared
band-1.77 rex_enkf arm agrees on the ordering (wake 0.75 vs lift-focused
0.43 through the wall). The consistent reading across every table: at
d = 4 the WAKE-supervised state is the better wall-estimated one; the
lift-focused lineage's 0.782 is the self-contained-deployable-package
viability result (own co-trained forecaster, dimension-insensitive peak
readability), not a superiority claim. No paper text asserts the wrong
version; the correction applies to the session-38 exploratory commentary
(commit message of the band-1.77 set included the confounded framing).

## 2026-08-04: framework float split, near-body schematic added (S3-C1, S31-01)

Body figures 13 -> 15. Both changes are Carlos's, from PDF comments.

**Split (S3-C1).** The combined framework float, two full-width TikZ panels
under one caption, became two figures. It could only ever be placed at a page
top, and one caption had to carry both a training-time architecture and a
deployment loop. Now:

| figure | source | placed |
|---|---|---|
| `fig:method_arch` (alias `fig:method`) | `tikz/fig1_jepa_architecture.pdf` | \S3.1, unchanged position |
| `fig:estimation_loop` | `tikz/fig_estimation_loop.pdf` | \S3.3, next to its first use |

No artwork was regenerated; the two PDFs are the ones the subfigures already
carried. The alias `fig:method` is kept on the architecture float so the
appendix A reference still resolves. Call sites updated: the panel letters in
`section_3_methods.tex` (two), `s3_4_estimators.tex:80` (`\ref{...}b` on what is
now a whole figure) and `appendix_a_regularisation.tex`. This also closes
REVIEW_LOG finding 7 for this float, the multi-label case where three `\ref`s
resolved to one number.

**New (S31-01).** `tikz/fig_nearbody_head.tex` -> `fig:nearbody_head`, in \S3.1.
The head had a field panel (`fig:phi`, appendix A) but no schematic of the
pipeline from raw fields to the $80$-dimensional target. Standalone-compilable
with `pdflatex`, same colour convention as `fig_estimation_loop.tex`.

Sizing note for anything drawn next: the placer scales artwork to `\linewidth`
= 384 pt, so a wide standalone loses label size in proportion. The first draft
was 593 pt natural, a $0.65$ scale that put `\scriptsize` under 5 pt. It is now
484 pt ($0.79$), reached by tightening the node geometry and by deleting two
long annotation strings that the caption already carried. `fig_estimation_loop`
sits at $0.70$ and `fig1_jepa_architecture` at $0.74$ for comparison.

### 2026-08-11: figure 5 type size (S31-01, comment landed on S3-C1)

Carlos: increase the smallest type, move text out of the artwork into the
caption if it does not fit, make the caption self-contained. The comment was
anchored to S3-C1 by the reader's fallback rule, since it sat above figure 4's
tag on the page, but it was about figure 5.

Nothing in the artwork is now set below `\footnotesize`. Three lines were cut,
all of which the caption already carried or now carries: the Laplace problem
under `auxiliary potential`, the mask formula under `feathered band`, and the
`8x4 signed patch energies (+) 16-bin radial spectrum` line under the target
bar. `at frame t` went too, since the group label above the row already says
per frame, and `small MLP` moved into the caption.

Effective type size at `\linewidth`: was `\scriptsize` at scale $0.793$, so
$5.5$ pt. Now `\footnotesize` at $0.802$, so $6.4$ pt. The scale barely moved,
because after the cuts the width is bound by the top row and the target box
rather than by the deleted text; the gain is the size step, not the geometry.
Tightening the node widths and gaps bought only 6 pt of natural width, which is
worth knowing before anyone tries the same lever again.

The caption now carries the definitions the artwork gave up ($\phi_L$ as the
exterior Laplace problem, $w$ as the feathered mask with its formula) and is
split into three labelled parts, per frame / geometry only / trained, matching
the three regions of the drawing.
