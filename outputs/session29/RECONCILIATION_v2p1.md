# SESSION29 reconciliation to v2.1 (decided 2026-06-14)

SESSION29 was authored against the v2 manuscript (`main_13.pdf`, 38 pp., split
v2, 84 cases / 226 train enc) and its HANDOFF stubs D179-D195. That premise is
SUPERSEDED: the repo is on the completed v2.1 unconditioned rebuild (`paper/main.tex`,
34 pp., split **v2p1**, 85 cases / 229 train enc, HANDOFF through **D196**, tag
`v1.0.0-rc2`). Author decision (2026-06-14): **reconcile to v2.1**, **cheap gates
first** (no GPU retrain until the analytical gates show one is needed).

## Mapping applied to every track
- Manuscript under repair: `paper/main.tex` (v2.1), NOT `main_13.pdf`.
- Split / cache: `configs/splits/split_v2p1.json`, cache `v2p1`. NEVER `--split v2`.
- Reproduction-first: the v2.1 numbers already reproduce deterministically
  (`eval_all.py` -> `emit_macros.py`, GI gate passed). "Reproduce main_13" is
  void; instead we VERIFY the v2.1 `outputs/session28/numbers.json` and extend it.
- Numbers authority: keep `outputs/session28/numbers.json` + `paper/macros.tex`.
  New SESSION29 numbers land as new `numbers_parts/<analysis>.json` records
  (alphabetic macros) so the existing pipeline absorbs them.
- HANDOFF: SESSION29 stubs D179-D195 COLLIDE with the live D178-D196. SESSION29
  entries continue at **D197+** (renumbered from the plan's stubs).
- Output dir: `outputs/session29/` (figures, JSON, reports), `outputs/runs/session29/`.
- Tooling reuse: prefer existing session28 scripts (closure_matrix, sensing_cf,
  stats_lib, drift_ce1, topology_ce2, spectrum_dmd, undisturbed_validation,
  latents under `outputs/session28/latents/`). Create session29 scripts only for
  genuinely-new analyses.

## Canonical dataset accounting (v2.1; SESSION29.5 P0.2)
The canonical counts are **85 cases / 382 encounter windows** (229 train, 87
validation, 42 test_b, 24 test_c), frozen in `configs/splits/split_v2p1.json`
under the `summary` block and emitted to the manuscript ONLY through manifest
macros (`scripts/session28/emit_dataset_manifest_part.py` ->
`numbers.json` -> `paper/macros.tex`: `\NumCasesTotal`, `\NumEncTotal`,
`\NumEncTrain`, `\NumEncVal`, `\NumEncTestB`, `\NumEncTestC`). No literal count
appears in the section sources (narrative_qc hard gate).

The 84 -> 85 change versus v2 is a **training case-set change, not a no-gust
accounting change**: v2.1 ADDS two train cases (`G+1.00_D1.50_Y-0.10`,
`G-1.50_D1.50_Y+0.00`) and DROPS one (`G-2.00_D1.50_Y+0.10`), all in the train
split, from the finer-dt run3 regeneration (HANDOFF D177). test_b and test_c are
frozen identical to v2; the undisturbed Baseline (no-gust, calibration reference)
case is unchanged and remains in train. Net +1 case, +3 train encounters
(226 -> 229) and the validation count rises 86 -> 87.

## Already resolved by the v2.1 rebuild (SESSION29 VERIFIES, does not redo)
- **F2** dataset accounting: v2p1 uses 85 cases / 229 train consistently; GI gate
  confirmed zero count contradictions (now fully macro-bound, see above).
- **F3** statistical unit: the paper already reports BOTH encounter-level Holm
  (survives) and case-level (does not); GD declared WEAK; case-clustered CIs.
- **Track 0** numbers authority: numbers.json + macros.tex already in force.
- Most of **Track H** mechanism: headline is the departure-spectrum (near-null
  fraction), not the fragile Mahalanobis 9x; topology limited to the no-gust
  cycle; spectrum/DMD done. SESSION29 adds kNN + local-PCA corroboration only.

## Genuinely open (this session's work, cheap/frozen-encoder first)
- **Track D** probe-class robustness (linear vs kernel/MLP/GBM): can a nonlinear
  probe close the JEPA-vs-reconstructive wake gap? Pins "linear" vs "broad". HIGH.
- **Track B0/B0.5** per-encounter clip leakage diagnosis + frozen sensitivity.
- **Track G** stronger floors (phase, history, persistence) beyond G,D,Y.
- **Track I** causal pressure window (current sensing W=30 ends at impact+16).
- **Track A** baseline external validation vs Rolandi2025 / Gupta2023 / Fukami2025.
- **Track C-min** case-level slopegraph figure (stats largely done in B6).
- **Track H** kNN + local-PCA corroboration of the manifold departure.
- DEFERRED pending gate outcomes + explicit go-ahead (GPU): **Track E** control
  grid, **Track F** horizon matrix, **Track B1/C-full** retrains.

## Standing blockers carried from v2.1 (GO gate, author/collaborator-owned)
- DNS Table 1: 7 `\pending{}` rows + grid/time-step sensitivity (collaborators).
- Zenodo DOI from tag v1.0.0-rc2; license/CRediT/funding.
