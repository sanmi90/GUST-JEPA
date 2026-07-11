# FIGURE_PLAN.md (Session 38, Stage 5)

Dispositions per the editorial memo section 7 (target 11-12 main figures, 5
main tables). Numbering = the Stage 0 concordance (main.aux). Status DONE =
landed this session; TODO(M3x) = needs the figure-engineering pass with
Carlos available for the layout calls (D302/D305 land there too).

| # | Label | Disposition | Status |
|---|---|---|---|
| 1 | fig:staging | KEEP | done (no change) |
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
