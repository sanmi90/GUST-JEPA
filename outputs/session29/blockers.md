# SESSION29 blockers (reconciled to v2.1, 2026-06-14)

Hard blockers prevent JFM submission. Soft/accepted items are non-blocking.

## HARD (submission-blocking)
- [ ] **DNS Table 1 incomplete**: 7 `\pending{}` rows in `paper/sections/section_2_flow_and_data.tex`
  (Mach/incompressible confirmation, domain + Lz/c, element/solution-point counts,
  near-wall resolution, timestep/CFL, gust-release station x0/c, grid+time-step
  sensitivity). Collaborator-owned (Miro, Lehmkuhl). Package drafted
  `scripts/session28/DNS_COLLABORATOR_PACKAGE.md`. (F1, Track A)

## OPEN (analytical, this session decides; not yet blocking)
- [ ] **Track D** probe-class: is the wake advantage broad-probe or linear-only? (F4)
- [ ] **Track B0** per-encounter clip leakage: causal or future-frame-dependent? (F7)
- [ ] **Track I** pressure window non-causal (W=30 ends at impact+16); causal
  ordering must survive or pressure stays appendix state-recoverability. (F6)
- [ ] **Track G** stronger floors: does the latent clear phase/history/persistence? (F5/F6)
- [ ] **Track A** baseline external validation vs Rolandi/Gupta/Fukami not yet run. (F1)

## RESOLVED by the v2.1 rebuild (verified, see RECONCILIATION_v2p1.md)
- [x] F2 dataset accounting (85 cases / 229 train, consistent; GI gate clean).
- [x] F3 case vs encounter unit (both reported; GD weak; case-clustered CIs).
- [x] Numbers authority (numbers.json + macros.tex; GI gate passed).
- [x] Mechanism headline de-risked (departure-spectrum, not Mahalanobis 9x;
      topology no-gust-cycle only; spectrum/DMD). kNN/local-PCA corroboration pending.

## ACCEPTED non-submission items (author-owned)
- Zenodo DOI from tag v1.0.0-rc2; final license/CRediT/funding.
