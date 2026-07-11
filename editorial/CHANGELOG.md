# CHANGELOG.md (JFM rewrite program; structural moves, wording changes, open author decisions)

## Session 36 (2026-07-10/11), numbers-frozen gate

- M5 strong-effect bar dropped (cite-or-drop; % REVIEW-CLAIM at the Gate O
  paragraph); tracer to zero hits.
- Nomenclature migrated to paper/nomenclature.tex; archive split names only
  at the s2.2 definitional site; JEPA confined to s1; leakage-free reduced
  to one definitional sentence.
- tab:closure merit column: suited-operator (Xmerit*, h8) replaced by
  shared-operator (XmeritSh*, pre-registered h16; D310 null branch); caption
  rebuilt horizon-truthful; s4 merit paragraph and s5.5 seed-variance
  passage reworded (% REVIEW-CLAIM).
- Parameter-only floor re-run on v2p2 (M2b), Methods sentences confirmed.

## Session 37 Stage 3 (2026-07-11), structural moves

- Results reassembled into four subsections:
  4.1 Constructing a physically useful state (s4_a construction + s4_b
      decode floor + "What the coefficient state carries" + the physics
      subsubsection: DMD/shedding-clock and atlas paragraphs + fig:atlas);
  4.2 Compression and forecastability (new v4/s4_b2_dimension.tex, split
      out of s4_b: dimension tiers + probe-dilution + fig:dimrace; the
      distributed-code and spatial-trade paragraphs; s4_c forecasting;
      "Why the state stays usable under rollout");
  4.3 What wall pressure observes (retitled "What the wall can see",
      labels preserved, sec:res_wallobs added);
  4.4 Sequential estimation and operating limits (tracking + envelope +
      s4_d estimator ladder).
  All absorbed \subsection headers demoted to \subsubsection; every label
  preserved; sec:res_code alias moved with the distributed-code content
  and two references re-pointed (s4_a, s5).
- Appendices: A (architecture/regularisation/UQ) and B (sensing) unchanged;
  NEW in-paper appendix C = calibration disclosure (was D.3, kept in-paper
  per the memo/D305 recommendation); appendix D dissolved.
- NEW paper/supplementary.tex (JFM class, xr cross-refs to main, S-numbered):
  S1 forecaster ledger (was D.1), S2 failure modes (was D.2), S3
  supplementary figures (was appendix C content: decode gallery fig:recon,
  pooling-cost fig:pooling_cost), S4 suited-operator merit table (NEW,
  documents the superseded protocol; tab:suited_merit from the retained
  Xmerit* macros). In-paper references to moved content replaced by plain
  "supplementary material" mentions (5 sites).
- main.tex section retitles: 2 "Flow configuration, data and endpoints";
  3 "Reduced states, forecasting and wall-pressure estimation".
- Both targets compile: main 49pp rc=0, supplementary 3pp rc=0, zero
  undefined references in either log.

## Open author decisions

- D302 fig:atlas slim placement (Stage 5; memo recommends keep slim in 4.1).
- D305 calibration-audit placement: implemented as in-paper appendix C per
  the recommendation; confirm or move to supplementary at the Stage 5 STOP.
- Carlos-owned: session35 branch merge, DNS Table 1 (tab:dns_pending, 7
  \pending{} rows), Zenodo DOI, license/CRediT/funding.
