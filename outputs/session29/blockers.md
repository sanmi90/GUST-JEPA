# SESSION29 blockers (reconciled to v2.1; updated 2026-06-14 post-reframe)

Track K target: empty of hard items except explicitly accepted non-submission ones.

## HARD (submission-blocking) -- author/collaborator-owned, NOT runnable here
- [ ] **DNS Table 1 incomplete**: 7 `\pending{}` rows in section_2_flow_and_data.tex
  (Mach/incompressible, domain + Lz/c, element/solution-point counts, near-wall
  resolution, timestep/CFL, gust-release station x0/c, grid+time-step sensitivity).
  Collaborator-owned (Miro, Lehmkuhl); package drafted
  scripts/session28/DNS_COLLABORATOR_PACKAGE.md. Track A baseline validation
  de-risks the DATA (mean forces in band of Fukami/Rolandi/Gupta) but the solver
  rows + convergence panel are still required. (F1)
- [ ] **Zenodo DOI** minted from tag v1.0.0-rc2 (then dropped into README,
  .zenodo.json, CITATION.cff, data-availability); final license/CRediT/funding.

## RESOLVED this session (SESSION29, all committed; see GATE_REVIEW_NOTES.md)
- [x] **F4 probe-class** (Track D, canonical): broad-probe readability, predictive
  latent leads under linear/kernel/MLP/GBM. STRONG.
- [x] **F4 objective vs supervision** (Track E): readability is supervision-driven
  (control 0.92 >= jepa 0.80); predictive objective owns the forecast (0.50 vs
  0.27) + is compatible where reconstruction suppresses it. Title reframed to
  "wake-supervised predictive latent"; sentinel PASS; forecast dissociation.
- [x] **F7 preprocessing** (B0 + B0.5): leak immaterial at readout; wake advantage
  preprocessing-robust; no B1 retrain needed.
- [x] **F6 pressure causality** (Track I): ordering holds and sharpens under a
  strictly pre-impact window (jepa 0.82); sensing reported causal.
- [x] **F5/F6 floors** (Track G): latent clears all physical floors.
- [x] **F3 case-level** (C-min): per-case slopegraph in S4.1; GD-weak stated honestly.
- [x] **mechanism magnitude** (Track H): "order of magnitude" dropped; near-null
  direction claim + kNN/local-PCA corroboration.
- [x] **F1 baseline validation** (Track A): mean forces in band of external refs.
- [x] **F5 horizon** (Track F): predictive latent leads at every H in {4,8,16,32}.

## ACCEPTED non-submission items (author decision)
- Main-text result figure count is 11 (plan target ~7-8). Reflects the Phase D v2.1
  curation plus the new case-level slopegraph; further culling (move e.g. the
  decode strip or a mechanism panel to the appendix) is an author layout decision,
  not a correctness blocker.
- Track C-full (grouped-CV retrain) running as the optional case-level robustness
  upgrade; the case-level result is already reported (slopegraph, GD-weak).
