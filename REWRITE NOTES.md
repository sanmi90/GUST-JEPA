# Rewrite notes and claim trace

This documents the structural and stylistic rewrite of the GUST-JEPA manuscript
toward the idiom of the Taira-group extreme-aerodynamics literature, what was
changed and why, and the honesty flags a reader or reviewer would otherwise
raise. Read this before editing the `.tex`.

The rewrite is a scaffold, not a finished paper. Prose for everything the
current evidence supports is written to submission quality. Everything that
requires a not-yet-run experiment or a not-yet-computed number is marked, in the
source, with `\pending{...}` (visible red text in the compiled PDF) or a
`% RESULT PENDING` comment. Search the tree for `pending` to get the full to-do
list.

## 1. Structure: eight sections to six

The reference papers (Smith et al. 2024 JFM; Fukami, Nakao & Taira 2024 JFM;
Fukami & Taira 2025 JFM; Tran, Yeh & Taira 2026 JFM; Odaka, Lopez-Doriga &
Taira 2026 JFM) share one skeleton: Introduction with the literature woven in,
a physics-first setup section, one compact methods section, Results organised by
physical question, and short Concluding remarks. The mapping applied here:

- Introduction (was 1) absorbs Related work (was 2). No standalone related-work
  section; the lineage is three paragraphs of the intro.
- Flow configuration and data (new, was the physics buried in 3.1 plus a new
  physics-first treatment). Leads with the staged encounter and the limit-cycle
  baseline, and reframes the |G|=4 case as a 2D-to-3D observability boundary.
- Predictive latent representation and evaluation protocol (was Methods, 3).
  Compressed; the SIGReg/PLDM material is demoted to one paragraph plus an
  appendix pointer.
- Results (was 4 Failure modes, 5 Full-scale results, 6 Decoder, 7 Ablations),
  reorganised into closure, horizon, mechanism, parameter/phase, controls,
  physical-space. Subsections are titled by physical question, never by
  experiment ID.
- Discussion (was 8) and Conclusions (new short section).
- The pressure-estimation, sensor-placement, and closed-loop pilot material
  moves to Discussion (control pathway, as a limitation) and Appendix C.

## 2. Honesty corrections (the important ones)

These are changes a careful reviewer would otherwise force, made proactively.

1. **The headline R^2 = 0.835 was a training-set fit, not held-out.** The source
   table comment confirms it (“Reporting train R^2 because the rollout closure
   metric is defined on the training distribution”). That rationale does not
   hold: the same rollout predictions used for the held-out test_b MAE give a
   held-out R^2 directly. The rewrite demotes the training R^2 to a clearly
   labelled “training-set fit, for reference only” table, promotes the held-out
   test_b MAE to the headline, and adds a placeholder table for the held-out
   R^2 that must be computed. The abstract now reports the held-out finding (a
   2.4 to 3.0 times reduction in wake-enstrophy error) rather than the train R^2.
   ACTION: compute test_b/test_c R^2 from the existing rollout predictions and
   fill `tab:b1_r2_heldout`.
1. **The central claim was confounded; it is now hedged to match the evidence.**
   The JEPA encoder differs from the reconstructive baseline in three ways at
   once: the predictive objective, the CNN+ViT architecture, and an 80-dim wake
   head (the reconstructive baseline has only a lift head). The largest reported
   advantage is on the wake observables the JEPA encoder alone was supervised on.
   The rewrite therefore claims a property of the “predictive latent family”,
   not of “the objective in isolation”, and adds Section 4.5 specifying the
   decisive control: a 2x2 of {predictive, reconstructive} x {CNN, CNN+ViT} at
   d=64 with auxiliary heads matched across all cells, plus a no-wake-head JEPA.
   ACTION: run that 2x2 (>=3 seeds) and the no-wake variant; fill
   `tab:controls_2x2`; then set the claim strength accordingly.
1. **Train-vs-held-out reversal on the integrated impulse.** On the training fit,
   POD scores best on I_y (0.799 vs JEPA 0.562). On held-out MAE, JEPA leads on
   I_y too (1.901 vs 2.245). The rewrite states this honestly and uses it to
   motivate reporting held-out R^2. Do not repeat the original’s “POD wins on
   I_y” as if it were a held-out result; it is a training-fit result.
1. **DNS vs LES.** The manuscript called the data DNS throughout. The dataset
   source (Fukami, Smith & Taira 2025 Phys. Rev. Fluids) describes
   spanwise-periodic large-eddy simulations of this exact configuration. The
   rewrite uses the neutral “numerical simulation” and flags the discrepancy in
   a comment in `section_2_flow_and_data.tex`. ACTION: confirm whether the data
   are DNS or LES and set the term once.
1. **Conditioning-floor comparison was apples-to-oranges.** The original
   compared the floor’s held-out R^2 against JEPA’s training R^2. The rewrite
   leaves the last column of `tab:conditioning_floor` for JEPA’s held-out R^2
   (pending item 1).
1. **Split version standardised.** The original mixed split_v1.2 (41 cases, 138
   encounters) in Section 1.2 with split_v2 (84 cases, 226/28/42/24) in the
   Methods and abstract. The rewrite uses split_v2 throughout, consistent with
   the n=42 test_b in the results tables. All sha256 strings and lock-path
   references are removed.

## 3. What was cut

- The implementation diary: Session 7/8/9, the D-/E-/F-/R-numbers, CLAUDE.md and
  HANDOFF.md, W&B run metadata, output-path references, the RTX 6000 Blackwell /
  sm_120 / L40S / sm_89 hardware lines, the `VORTEX_JEPA_ALLOW_NON_RTX6000`
  bypass, “production cell”, “smoke scale”, the split sha256. None of this
  belongs in a JFM paper.
- The second paper. Section 1.2’s “Contributions 1/2/3” were a different study
  (SIGReg vs PLDM controlled-collapse, with a different headline metric: Test B
  delta over a (c,t) baseline, and a smoke-to-full-scale participation-ratio
  inversion). That entire narrative is removed from the main text. The
  anti-collapse regulariser is now one methods sentence (“we use SIGReg; the
  choice is second order vs VICReg at this scale”), with the comparison deferred
  to Appendix A. If you want to publish the regulariser study, it is its own
  paper; do not staple it back on.
- The pre-results “Failure modes” framing (eight subsections) is dissolved; the
  two load-bearing diagnostics (latent drift, conditioning floor) move into
  Methods/Results, the rest to the appendix.

## 4. New analyses scaffolded (methods written, results pending)

Each is drawn from a specific reference and repairs a specific weakness. The
first three are post-processing of latents and decoded fields you already have
(no retraining); they can run in parallel with the 2x2. Method text is in the
Results section; results are marked pending.

1. **Persistent homology of latent trajectories** (Smith et al. 2024), in
   Section 4.3. Turns the scalar drift ratio into a topological invariant: the
   predictive rollout should preserve the encounter’s persistent H1 cycle while
   the reconstructive rollout’s fragments. Tool: ripser or giotto-tda on the
   four latent point clouds (DNS-encoded and rolled-out, JEPA and AE).
1. **Optimal-transport field dissimilarity** (Tran et al. 2026), in Section 4.6.
   Replaces the misleading structural-similarity score in the reconstruction
   comparison (the AE’s high SSIM is bulk-zero agreement with a collapsed field).
   Unbalanced OT, signed vorticity split into +/- parts and summed, Sinkhorn via
   the POT library.
1. **OT-geodesic vs latent-distance alignment** (Tran et al. 2026), in Section
   4.3. The geometric mechanism for why the predictive rollout stays
   on-manifold: its latent metric should track the physical transport geometry
   better than the reconstructive one.
1. **Limit-cycle and phase-amplitude reading** (Fukami, Nakao & Taira 2024), in
   Section 4.4. Defines “recovery” precisely (return to the baseline limit cycle)
   and connects the predictor to the sensitivity-function control for these
   flows. Unifies with the persistent cycle of item 1.
1. **Scale decomposition and force-element interpretation** (Odaka et al. 2026;
   Motoori & Goto 2019), in Section 4.6. Ties the wake-observable advantage to
   the leading-edge vortex and shear layer rather than to a scalar. Scale
   decomposition is cheap post-processing; the force-element computation needs
   the velocity fields and is optional.

An optional sixth (causal / information-flow among latent directions and
observables, after Wang et al. 2026) is mentioned only in passing; it overlaps
with the closure result and is not needed.

## 5. Full pending list (search the source for `pending`)

- Compute held-out test_b/test_c R^2 -> `tab:b1_r2_heldout`, abstract, 4.1.
- Fill JEPA held-out R^2 column in `tab:conditioning_floor`.
- Run the 2x2 (objective x architecture, auxiliary-matched, >=3 seeds) and the
  no-wake-head JEPA -> `tab:controls_2x2`; set the central claim strength.
- Horizon sweep H in {1,4,8,16,32,64} -> closure-vs-H figure (4.2).
- Persistent homology -> persistence diagrams + H1-lifetime-vs-horizon (4.3).
- OT-geodesic alignment -> correlation plot (4.3).
- Limit-cycle / phase-amplitude -> latent orbit + return-to-orbit (4.4).
- Parameter- and phase-stratified closure (4.4).
- OT field dissimilarity for reconstruction (4.6).
- Scale decomposition of the wake-observable advantage (4.6).
- Confirm DNS vs LES and set the term (2.1).

## 6. Build

Layout matches the repo convention: `main.tex` at the root, everything under
`sections/` (`sections/*.tex`, `sections/tables/`, `sections/figures/`).
Build: `pdflatex main && bibtex main && pdflatex main && pdflatex main`.
The section files are `.tex` (ready to `\input` directly); if your pipeline
generates `.tex` from `.md`, point it at these or rename.

`PREVIEW_main.pdf` in this folder is a rendered proof (16 pages) built in a
sandbox that lacked `lmodern`, so it was compiled with Times (`mathptmx`)
instead; your `main.tex` keeps `lmodern`. The preview is only to show the
structure renders and the tables and figures place correctly.

## 7. Citation keys to verify

`refs.bib` is minimal and just lets the document compile. The fluid-mechanics
entries (fukami2025prf, fukami2023, fukami2025, fukami2024control, smith2024,
tran2026, odaka2026, solera2024, motoori2019) are filled from the actual papers;
verify volume and page numbers. The ML entries reuse the original draft’s keys
(lecun2022, ijepa, vjepa2, lewm, lejepa, pldm, vicreg) and several arXiv IDs in
the original were placeholders, so check them. jones2022 and mohamed2023 are
intro-only and should be confirmed or replaced.