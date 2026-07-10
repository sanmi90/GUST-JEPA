# Claude Code session v2: JFM rewrite of main_25

Supersedes `JFM_rewrite_session.md`. Changes from v1: a provenance stage (v2.1 vs v2.2 numbers), a macro-based nomenclature migration stage, Python audit tooling (`scripts/`), per-section word budgets as gates, per-figure surgery specs, the corrected language table, and explicit author decision stubs. Paste the master prompt below at the repository root. Requires the full source tree (sections/, macros.tex, macros_v3.tex, numbers.json, bibliography, figure scripts, figstyle.py).

Compute-dependent items (shared-operator merit, v2.2 re-runs, figure regeneration) are NOT done by this session; they live in `SESSION_MS_manuscript_compute.md` and this session only flags them.

---

## Master prompt

```text
You are editing a fluid-mechanics manuscript for submission to Journal of
Fluid Mechanics. Work as a conservative scientific editor, not as a
co-author who may invent results.

GOAL
Restructure and rewrite the manuscript around one question: can a reduced
state of an extreme vortex-gust encounter retain the wake dynamics, be
advanced in time, and be estimated from sparse wall pressure? Maintain the
distinction between three operations in every claim: representation (what
the encoded state contains), prediction (what survives advancement), and
estimation (what wall pressure recovers). Never use one as evidence for
another. Preserve the central attribution: observable supervision supplies
readability and the protected latent geometry; the multi-step predictive
objective supplies rollout stability, not instantaneous readability.

NON-NEGOTIABLE RULES
1.  Never invent, recompute, round or silently alter a numerical result.
2.  numbers.json -> macros is the single source of truth. If prose, a
    caption or a table disagrees with a macro, insert `% REVIEW-NUMBER:`
    with both values and report it. Never repair silently.
3.  Never add a citation absent from the BibTeX files.
4.  Never change a scientific conclusion without `% REVIEW-CLAIM:`.
5.  Comment protocol: `% REVIEW-NUMBER:`, `% REVIEW-CLAIM:`,
    `% FIGURE-TODO:`, `% PROVENANCE-TODO:`. Every flag also goes into the
    corresponding editorial/*.md ledger.
6.  Preserve labels, refs and figure filenames; controlled renames only.
7.  British spelling, JFM notation, and NO em-dashes anywhere (use commas,
    colons or parentheses). This applies to new prose and to any sentence
    you touch.
8.  No statement may justify a model or configuration choice by test-set
    results. Selection statements cite validation only.
9.  Split names: use the aliases validation (test_a), in-distribution test
    (test_b), boundary test (test_c), defined once in section 2.2.
10. Report physical gust sign G everywhere; archive case identifiers
    (s = -G) appear only in the data appendix.
11. Compile after every structural stage with `latexmk -pdf main` (and
    `latexmk -pdf supplementary` once it exists).
12. git: branch `jfm-rewrite-v2`; one commit per stage with a message
    listing files moved/edited and open flags.

STAGE 0: BASELINE AND AUDIT
- Branch, build, save build/baseline.pdf. Record latexmk warnings.
- Create scripts/audit_numbers.py, scripts/lint_language.py,
  scripts/check_refs.py, scripts/wordcount.py from the specifications at
  the end of this prompt. Run all four.
- Write editorial/MANUSCRIPT_AUDIT.md: per-section word counts against the
  budgets below; figure/table inventory; acronym inventory with counts;
  PENDING/TODO/placeholder items; repeated numerical claims; broken refs;
  every hit from the banned-language list; every em-dash.
- STOP. Show the audit and wait for approval.

STAGE 1: CLAIM MAP AND PROVENANCE
- editorial/CLAIM_MAP.md: the four primary claims (state construction;
  compression; common-operator forecasting; wall-pressure estimation),
  each with its exact figure/table/equation and split. Mark unsupported or
  multiply-stated claims; do not repair.
- editorial/PROVENANCE.md: for every main-text number and figure, tag the
  data split and model generation (v2.2: 102 cases / 450 encounters /
  symmetric boundary test; or v2.1: 85-case split) from numbers.json
  provenance fields, run manifests or W&B references. Items to check
  explicitly: near-null/Mahalanobis mechanism (table 7, fig 12a);
  parameter-only floor; latent atlas and DMD spectrum (fig 21);
  distributed-code gap; topology; pressure-recovery pillar (table 8,
  fig 13); preprocessing pillar (table 13, already disclosed as v2.1 and
  the model for disclosure wording). Anything untraceable or v2.1 gets
  `% PROVENANCE-TODO:` and an entry for the compute session.
- STOP. Show both ledgers and wait for approval.

STAGE 2: NOMENCLATURE MIGRATION (macro-level, before any prose rewrite)
- Add text macros to macros.tex and replace throughout:
    \PredState   -> wake-supervised predictive state   (was CLW/flagship)
    \LiftState   -> lift-focused predictive state       (was CLN/specialist)
    \DirectFC    -> direct multi-horizon forecaster     (was REX/TiRex)
    \FnoiseKF    -> forecast-noise filter               (was REX-EnKF)
    \TwoStageKF  -> two-stage filter
    \LinLatKF    -> linear latent filter                (was LAE-KF)
    \StaticInv   -> static delay-embedded inverse       (was static E_obs)
    \ValSplit / \TestSplit / \BoundarySplit for the split aliases.
- Cube codes (CL, CLN, ...) survive only on figure 6/7 axes plus a legend
  row in the corresponding table.
- JEPA appears only in the section 1 and section 3 lineage discussion.
- Replace the "leakage-free" framing with one definitional sentence in
  section 3.3, thereafter "the pressure-only estimator".
- Compile; grep-verify zero stray occurrences of the old names outside
  figure assets and the appendix legend.

STAGE 3: STRUCTURE
- Reorganise into: 1 Introduction; 2 Flow configuration, data and
  endpoints; 3 Reduced states, forecasting and wall-pressure estimation;
  4 Results (4.1 Constructing a physically useful state; 4.2 Compression
  and forecastability; 4.3 What wall pressure observes; 4.4 Sequential
  estimation and operating limits); 5 Discussion; 6 Concluding remarks.
- Mapping: 4.1 absorbs current 4.1, the decode part of 4.2, 4.3, and the
  DMD/shedding-clock part of 4.10; 4.2 absorbs the dimension part of 4.2,
  4.4, 4.5 and the distributed-code gap; 4.3 absorbs 4.6 and 4.6.1; 4.4
  absorbs 4.7, 4.8, 4.9.
- Appendices in-paper: A architecture/regularisation/UQ; B estimator
  configurations and sensing; C calibration audit (current D.3, kept in
  the paper). Create supplementary.tex for: forecaster ledger (D.1),
  failure modes (D.2), topology, preprocessing robustness, streaming and
  noise grids, spatial-latent trade, suited-operator comparison, atlas
  panels.
- Move, never delete. Compile both targets.

STAGE 4: PROSE REWRITE
- Paragraph pattern: physical question, evidence, mechanism, implication.
- At most two headline numbers per paragraph; the rest to tables/captions.
- Word budgets (gates, checked with scripts/wordcount.py): abstract 250;
  s1 1300; s2 1600; s3 2600; s4.1 1300; s4.2 1200; s4.3 1000; s4.4 1800;
  s5 1200; s6 450.
- Use the pre-approved drafts in front_matter_rewrite.tex for the title,
  abstract, introduction ending and concluding remarks (author-approved
  wording; bind the literals to macros).
- Language table (replace everywhere, including captions):
    flagship -> wake-supervised predictive state
    specialist -> lift-focused predictive state
    load-bearing -> necessary under the present objective
    knob-free -> with a single validation-calibrated noise scale and no
                 test-set tuning        [NOT "without additional tuning"]
    protocol-clean -> selected without reference to the test set
    refuted -> not supported by the validation results
    catastrophic -> define the numerical criterion once, then "divergent"
    honest / honesty requires / honest surprise / we say so plainly ->
                 delete; state the result
    settle/settled -> establish / examine
    buys -> provides / confers
    earns its keep -> state the numbers
    erratic -> non-monotonic in dimension and irreproducible across seeds
    own-stack -> per-family end-to-end
    as-built model -> the co-trained predictor
    kit strength -> default configuration
    pre-registered -> fixed in advance of evaluation (say once, in s3.5)
    dimension-invariant -> approximately unchanged over the tested
                 dimensions (nonlinear probe), linearly accessible at d=4
    beyond any wall-limited filter -> not recovered by the tap counts,
                 delay windows and estimators considered here
    three-dimensional observability boundary -> an indication of a
                 three-dimensional observability limitation (n = 4 cases)
- Mandatory claim repairs (insert % REVIEW-CLAIM at each):
    a) s4.7 "highest boundary closure (0.84 against 0.66 to 0.85)": false
       superlative (0.85 exceeds 0.84). Rewrite as the combination claim:
       closure among the highest, lowest divergence rate (0.72 vs
       0.93-0.97), consistency across gust strengths, least-degraded wake
       readout.
    b) s4.8 "near unity (0.6) ... nearly twenty (15.2)": state numbers
       plainly.
    c) s4.8 "near a ratio of 3.0 ... and near 3.0": flag % REVIEW-NUMBER,
       likely macro bug; needs author values (stub D304).
    d) s4.9 static inverse 0.83 vs fig 16c 0.825: bind to one macro.
    e) s4.8 RMSE 0.68 at |G|=3 vs table 11 columns {1,2,4}: reconcile.
    f) Abstract/anywhere: no "no divergences" claim for the
       in-distribution test set (fig 16c shows 2/42); scope to the
       boundary set or omit.
    g) Recoverability interval 0.120 [0.096,0.145]: keep once in Results.
    h) Reconcile the envelope narrative in one sentence: median tracking
       holds through |G| = 4 while uncertainty consistency degrades from
       |G| of about 3.
    i) The static-inverse honesty clause must survive into s6.
- Reduce repetition between Results, Discussion, Conclusions: Discussion
  re-quotes no Results number; it references sections.

STAGE 5: FIGURES AND TABLES
- editorial/FIGURE_PLAN.md mapping every figure to KEEP / REDRAW /
  COMBINE / MOVE / DROP, using this plan:
    keep: 1, 2, 7, 9, 13, 14
    redraw: 6 (flatten to two panels, single (a)/(b) labelling);
            4+5 -> one pipeline schematic;
            11+12(b,c) -> one forecasting figure;
            15+16(a,b)+17(a) -> one envelope/phase-error figure;
            16(c)+20 -> ladder + dimension-grid figure, no in-panel prose,
            test-peek annotation moves to appendix C text;
            21 -> slim to DMD spectrum + one atlas panel, placed in s4.1.
    drop: 10 (duplicates table 6).
    move: 3, 8, 12(a), 16(d,e), 17(b), 18, 19, 22-25 to appendix or
          supplementary per the memo.
- Do not regenerate scientific plots unless the script and data are
  present (figstyle.py conventions); otherwise emit % FIGURE-TODO with the
  exact spec for the compute session (Track M3).
- Every kept caption states: content; split and n; uncertainty convention;
  the single primary inference. Enlarge labels; remove redundant in-panel
  annotations.
- Tables: fill nothing in table 1; it remains a blocker until the authors
  supply dns_metadata.yaml values. Table 6 merit column: pending decision
  D301 (recompute under the shared operator, Track M1) either relabel the
  column explicitly as suited-operator with the confound stated, or leave
  a % REVIEW-NUMBER placeholder for the recomputed values.
- Target: 11-12 main figures, 5 main tables.

STAGE 6: CONSISTENCY
- Rerun all four scripts. Write editorial/NUMBER_AUDIT.md: every quoted
  R2, RMSE, seed count, encounter count, gust boundary, with its source
  macro or table.
- Nomenclature grep: zero survivors of the old names; zero em-dashes;
  G-sign convention: no archive-signed identifier outside the data
  appendix.
- Broken-reference scan; chktex if available.
- Visual diff of final PDF against build/baseline.pdf: missing figures,
  clipped tables, illegible panels.

DELIVERABLES
Compiling main + supplementary; editorial/MANUSCRIPT_AUDIT.md, CLAIM_MAP.md,
PROVENANCE.md, FIGURE_PLAN.md, NUMBER_AUDIT.md, CHANGELOG.md (structural
moves, wording changes, unresolved author decisions).

ACCEPTANCE CHECKLIST
- The paper's one question is statable after page 1.
- Abstract: one motivation, one method, three findings, one limit,
  physical units alongside R2, <= 250 words, no divergence-count claims
  for the in-distribution set.
- Every primary claim has one main figure or table; all cross-family
  forecast claims use the shared operator.
- The static-inverse result appears in the conclusions.
- No placeholder, no PENDING DNS value, no test-selected claim, no
  % PROVENANCE-TODO older than the compute session, no em-dash.
- PDF compiles cleanly; main figures legible at page size.

SCRIPT SPECIFICATIONS (create verbatim, then adapt paths)

scripts/audit_numbers.py
------------------------
import json, re, csv, pathlib, sys
SEC = pathlib.Path("sections")
NUM = re.compile(r"(?<![\w.])\d+\.\d+(?![\w.])")
MACRO = re.compile(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}")
macros = {}
for mf in ["macros.tex", "macros_v3.tex"]:
    p = pathlib.Path(mf)
    if p.exists():
        macros.update(dict(MACRO.findall(p.read_text())))
try:
    numbers = json.loads(pathlib.Path("numbers.json").read_text())
except FileNotFoundError:
    numbers = {}
rows = []
for tex in sorted(SEC.glob("*.tex")):
    for i, line in enumerate(tex.read_text().splitlines(), 1):
        if line.lstrip().startswith("%"):
            continue
        for m in NUM.finditer(line):
            rows.append([tex.name, i, m.group(0), line.strip()[:120]])
with open("editorial/number_literals.csv", "w", newline="") as f:
    csv.writer(f).writerows([["file", "line", "literal", "context"], *rows])
# cross-check macros against numbers.json values (string containment)
mismatch = [(k, v) for k, v in macros.items()
            if k in numbers and str(numbers[k]) not in v]
print(f"{len(rows)} numeric literals; {len(mismatch)} macro/json mismatches")
for k, v in mismatch:
    print("MISMATCH", k, v, "json:", numbers[k])

scripts/lint_language.py
------------------------
import pathlib, re, sys
BANNED = ["flagship", "specialist", "load-bearing", "knob-free",
          "protocol-clean", "refuted", "catastrophic", "honest",
          "settle", "buys", "earns its keep", "erratic", "own-stack",
          "as-built", "kit strength", "pre-registered",
          "dimension-invariant", "wall-limited filter", "prove",
          "boundary" ]  # 'boundary' reported, not banned: review hits
EMDASH = ["\u2014", "---"]
hits = 0
for tex in sorted(pathlib.Path(".").rglob("*.tex")):
    if "build" in tex.parts: continue
    for i, line in enumerate(tex.read_text(errors="ignore").splitlines(), 1):
        low = line.lower()
        for b in BANNED:
            if b in low:
                print(f"{tex}:{i}: [{b}] {line.strip()[:100]}"); hits += 1
        for e in EMDASH:
            if e in line and not line.lstrip().startswith("%"):
                print(f"{tex}:{i}: [EM-DASH] {line.strip()[:100]}"); hits += 1
print(f"{hits} language hits")

scripts/check_refs.py
---------------------
import re, pathlib
log = pathlib.Path("main.log")
if log.exists():
    t = log.read_text(errors="ignore")
    for pat, name in [(r"Reference `([^']+)' undefined", "UNDEF-REF"),
                      (r"Citation `([^']+)' undefined", "UNDEF-CITE"),
                      (r"Label `([^']+)' multiply defined", "MULTI-LABEL")]:
        for m in re.finditer(pat, t):
            print(name, m.group(1))
else:
    print("run latexmk first")

scripts/wordcount.py
--------------------
import subprocess, pathlib, shutil, re, sys
BUDGET = {"intro": 1300, "config": 1600, "methods": 2600, "results1": 1300,
          "results2": 1200, "results3": 1000, "results4": 1800,
          "discussion": 1200, "conclusions": 450}
for tex in sorted(pathlib.Path("sections").glob("*.tex")):
    if shutil.which("texcount"):
        out = subprocess.run(["texcount", "-1", "-sum", str(tex)],
                             capture_output=True, text=True).stdout.strip()
        n = int(re.split(r"[+\s]", out)[0] or 0)
    else:
        body = re.sub(r"%.*|\\\w+(\[[^\]]*\])?(\{[^{}]*\})?", " ",
                      tex.read_text())
        n = len(body.split())
    key = next((k for k in BUDGET if k in tex.stem), None)
    flag = ""
    if key and n > BUDGET[key]:
        flag = f"  OVER budget {BUDGET[key]}"
    print(f"{tex.name}: {n} words{flag}")

Begin with Stage 0 only. Show the audit and wait for approval before
touching scientific prose.
```

---

## Author decision stubs (append to HANDOFF.md at the next free D numbers)

- **D301** Table 6 merit column: recompute under the shared direct forecaster (Track M1) vs restructure table 6 to representation-tier only with merit living in the s4.2 figure. Recommended: recompute.
- **D302** Figure 21: slim (DMD spectrum + one atlas panel) kept in s4.1 vs moved to appendix. Recommended: keep slim in s4.1.
- **D303** Approve the abstract, introduction ending and concluding remarks drafts in `front_matter_rewrite.tex` (or edit before Stage 4 binds them).
- **D304** Supply the intended per-core-diameter half-divergence gust ratios for s4.8 ("near 3.0 ... near 3.0" is presumed a macro bug).
- **D305** Calibration audit (current appendix D.3): keep as in-paper appendix C (recommended, integrity) vs supplementary.
- **D306** Title choice among the three options in `front_matter_rewrite.tex`.
