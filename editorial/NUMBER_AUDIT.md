# NUMBER_AUDIT.md (Session 38, Stage 6)

Every quoted result number in the build is macro-bound; the audit is
enforced by tooling rather than by a hand list:

1. scripts/session35/trace_numbers.py (build gate): PASS at ZERO hand-typed
   numerals. Every decimal in the content tex flows from macros.tex /
   macros_v3.tex or is a whitelisted protocol/configuration constant
   (trace_whitelist.json, each entry justified; the whitelist forbids
   result values by policy).
2. scripts/session36/audit_numbers.py: PASS. 835 macros in macros_v3.tex
   against 735 numbers.json entries (value + ci_lo/ci_hi convention), zero
   value mismatches, zero missing, zero orphans. numbers.json regeneration
   is byte-identical modulo the provenance-commit line (checked at the
   Session 36 close).
3. Per-table part mapping and split generation: editorial/PROVENANCE.md
   section 2 (all v2.2 except the disclosed tab:prepsens). Seed counts and
   n are carried per entry in the part JSONs (seed_mean/seed_sd/n).
4. Encounter counts and gust boundaries quoted in prose are the split-v2p2
   manifest constants (integers, whitelisted as a class; manifest
   configs/splits/split_v2p2.json).

Residual conventions a reviewer should know:
- Precision differs across parts by design (p1 filter headlines %.3f, dap
  ladder %.2f, envelope %.1f/%.2f); within any one comparison the precision
  is uniform. The figure ladder annotations were aligned to the prose
  precision at the Session 38 regeneration (memo catch 4).
- The two >20pt overfull boxes in the log are inside the pre-existing TikZ
  schematic assets, queued for the M3b redraw (FIGURE_PLAN.md); the four
  wide tables (envelope, mechanism, filter_error, recovery) were wrapped in
  \fittab this session and no longer overflow.
- Word counts at close: abstract 249/250, s1 1235/1300, s6 376/450 inside
  budget; s3 4641 under the D268-relaxed 5000; s2 1957 and s5 1801 above
  their original targets and s4 6566 above 5300, accepted under D268
  (clarity over budget; only genuine redundancy was pruned).
