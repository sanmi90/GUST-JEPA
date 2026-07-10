# SESSION 36 REPORT (2026-07-10/11): JFM rewrite phase 1, closed at the numbers-frozen gate

Branch `jfm-rewrite-v2` (off session35-manuscript-v4). Governing spec
SESSION_36.md; upstream program archived at editorial/upstream/. HANDOFF
D266 + D310-D312. All session-close gates GREEN.

## What landed

Phase 0 (475b852): program adopted; D301-D319 reserved as the rewrite
block; Carlos resolved D301 (recompute merit under the shared operator),
D303 + D306 (adopt front_matter_rewrite.tex; new title "Predictive
reduced-order states for wall-pressure estimation of extreme vortex
gust--airfoil interactions", binds at Session 37 Stage 4), L1 (combined
session packaging), L2 (main.pdf IS main_25; concordance 1:1).

Stage 0 (45b2e80): four audit scripts at scripts/session36/ (audit_numbers
wraps the trace_numbers gate; word-boundary language linter; refs checker;
budget wordcount). editorial/MANUSCRIPT_AUDIT.md: concordance verified
against main.aux (25 figures / 13 tables); the memo's 15 catches located at
file:line; main text ~17.1k words vs the 12.5k target. M5 CLOSED by
evidence: no before-results commit declares the 0.2 strong-effect bar, so
the bar clause was dropped (cite-or-drop; % REVIEW-CLAIM; conclusion
unchanged); tracer now passes at zero hits.

Stage 1 (5acae84): editorial/CLAIM_MAP.md (four primary claims anchored;
all supported; one disclosed test-selection instance, appendix-confined)
and editorial/PROVENANCE.md (everything v2.2 except the deliberate
tab:prepsens disclosure; five of six M2 targets confirm-and-retag).
D304 RETIRED by evidence: the duplicated "near 3.0" is discrete-ladder
quantization (both D = 1.0 and 1.5 first exceed a 0.5 divergence rate at
grid |G| = 3; D = 1.5 reaches exactly 0.5 already at |G| = 2), not a macro
bug; Stage 4 rewrites the false contrast.

Stage 2 (954635c): nomenclature migrated to hand-maintained
paper/nomenclature.tex (\PredState, \LiftState, \DirectFC, \FnoiseKF,
\TwoStageKF, \LinLatKF, \StaticInv, \ValSplit/\TestSplit/\BoundarySplit).
Archive split names survive only at the s2.2 definitional site; JEPA
confined to the s1 lineage (+ TikZ assets); "leakage-free" reduced to the
definitional sentence + "pressure-only estimator"; cube codes reduced to
legend pairings. flagship/specialist: zero survivors.

Track M1 (ac79505, D310): the s3.2.1 operator (LSTM h512, 9 quantiles,
6000 iters) fit identically on all ten tab:closure families x 3 seeds on
frozen v2p2 caches (six caches newly encoded from frozen checkpoints).
GATE = NULL branch: the wake-headed top block holds but is a statistical
tie (H=16: AeWake 0.574, JepaWake 0.561, SupOnly 0.443; overlapping
case-clustered CIs); mid-block ordering genuinely changes (JepaNowake
collapses to -0.426, FukamiWake rises to 0.162, Pod falls to 0.036).
Horizon finding: the co-trained vector predictor leads at h8 (0.755) and
is OVERTAKEN by the shared direct forecaster at h16 (0.418 vs 0.561),
converting the caption claim into a horizon-dependent statement that
reinforces the error-accumulation thesis. Disposition applied: tab:closure
merit column now quotes XmeritSh* at the pre-registered horizon sixteen
(the old caption said sixteen over h8 values; both horizons now truthful);
suited-operator Xmerit* stays macro-bound for the Stage 3 supplementary
table; s4 + s5.5 passages reworded with % REVIEW-CLAIM.

Track M2 (D311): five targets confirmed v2.2 by artifact args (mechanism,
DMD/atlas, distributed-code gap, topology, K x W recovery grid); the
parameter-only floor was the single pre-v2.2 item and was RE-RUN
(scripts/session36/param_floor_v2p2.py): both Methods sentences CONFIRM
(wake-enstrophy floor 0.151 at the H=16 window, 0.687 at impact). Stage 4
caution: circulation_neg stays parameter-explainable at H=16 (0.730), so
the floor claim remains wake-enstrophy-specific.

## Gates at close

latexmk rc=0, 50pp; zero undefined refs/citations; tracer PASS (0 hits);
macros/json cross-check 835 macros / 735 entries, zero mismatches/orphans;
regeneration byte-identical modulo provenance-commit lines; language linter
66 hits, all belonging to the declared Stage 4 language table (was 85);
zero em-dashes; wordcount unchanged (compression is Stage 4).

## Deviations from the upstream specs (disclosed)

- The M1 fits carry JSON provenance (git commit, config, gpu_name) instead
  of per-fit W&B runs, following the session-34 latent_rex/rex_tune
  precedent for downstream operator fits.
- STOP 2 (ledger approval) was folded into the session-close review at
  Carlos's "keep going"; both ledgers are unreviewed and flagged for his
  read-through.

## Open for Session 37 (Stage 3 + 4)

Restructure Results to the four target subsections (the v4 s4_a-d subfiles
already mirror them; promotion-and-prune); prose rewrite to budgets with
the language table and the remaining memo catches (1, 2, 5, 6, 7, 9-14);
front matter binds (D303/D306 adopted; numbers to macros; the abstract
divergence-count clause disappears with the draft); Discussion
three-mechanism reorganisation; supplementary.tex created (suited-operator
table moves there). Session 38: Stage 5 figures (M3a-M3f redraws;
fig:readability_matrix drops or regenerates from the m1 part; fig 14/16e
FIGURE-TODOs), Stage 6 consistency, freeze. Carlos-owned unchanged: merge
decision, DNS Table 1, Zenodo DOI, license/CRediT/funding; D302/D305
decided at the Stage 5 STOP.
