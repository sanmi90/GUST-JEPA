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
