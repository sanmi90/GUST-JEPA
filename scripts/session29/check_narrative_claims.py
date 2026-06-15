"""SESSION29.9 narrative-claim audit (predictive-state design-study frame).

Three checks over the main sections + the sensing appendix:

1. HARD-FAIL (positive context): overclaim phrases that must not appear asserting
   the claim. The forecast/controller/world-model/causal phrases are negation-aware
   (allowed inside a negated / limitation clause, e.g. "not a validated forecast");
   the readability/topology/pressure overclaims fail wherever they appear.
2. WARN: phrases acceptable only in bounded contexts; reported for manual triage
   into OK / EDIT / LIMITATION in outputs/session29_9/narrative_claim_audit.md.
3. REQUIRED concepts: the manuscript must express the division-of-labour and the
   scope boundaries; absence fails the build.

Exit non-zero on any hard-fail or missing required concept.

Usage:
    python scripts/session29/check_narrative_claims.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SECT = REPO / "paper" / "sections"
OUT = REPO / "outputs" / "session29_9" / "narrative_claim_audit.md"

SECTIONS = [
    "abstract", "section_1_introduction", "section_2_flow_and_data",
    "section_3_methods", "section_4_results", "section_5_discussion",
    "section_6_conclusions", "appendix_b_sensing",
]

# Always-fail overclaims (no acceptable positive context).
HARD_FAIL_ALWAYS = [
    "predictive objective supplies readability",
    "predictive objective makes the wake readable",
    "renders wake structure",
    "single recurrent cycle",
    "all gusted encounters are single loops",
    "pressure recovers the wake",
    "closed-loop controller demonstrated",
    # SESSION29.10 overclaim guards
    "alone forecasts",
    "exact deployment mirror",
    "wake recovery from wall pressure",
]
# Fail only in a POSITIVE (non-negated) context; OK inside a negated/limitation clause.
HARD_FAIL_IF_POSITIVE = [
    "validated forecast",
    "closed-loop controller",
    "causal model",
    "world model",
]
NEG = re.compile(
    r"\b(no|not|without|never|do not|does not|cannot|rather than|nor|neither|"
    r"motivat|analogy|future work|remains?)\b", re.I)

WARN = ["baseline", "reconstruction suppresses", "controller", "forecastable",
        "on-manifold", "wake-readable"]

# Required concepts (regex; case-insensitive, whitespace-flexible).
REQUIRED = {
    "supervision supplies readability":
        r"supervision[^.]{0,80}(supplies|supplied)[^.]{0,40}readab",
    "anti-collapse supplies in-distribution geometry":
        r"anti-collapse[^.]{0,80}(geometry|in distribution|in-distribution|manifold)",
    "predictive objective supplies dynamic usability":
        r"predictive objective[^.]{0,120}(dynamic|forecast|rollout|trajectory)",
    "state recoverability, not wake recovery":
        r"(state[^.]{0,60}not[^.]{0,30}wake|not[^.]{0,30}wake recover|"
        r"does not recover the wake)",
    "not a validated forecast":
        r"(not[^.]{0,40}validated forecast|rather than[^.]{0,40}validated forecast)",
    "not a closed-loop controller":
        r"(not[^.]{0,40}closed-loop|closed-loop[^.]{0,60}(future work|beyond|"
        r"limitation|would require))",
}


def main() -> None:
    blob = {}
    full = []
    for s in SECTIONS:
        p = SECT / f"{s}.tex"
        if not p.exists():
            continue
        lines = [ln for ln in p.read_text().splitlines()
                 if not ln.strip().startswith("%")]
        blob[s] = lines
        full.append("\n".join(lines))
    corpus = "\n".join(full).lower()

    hard = []
    for s, lines in blob.items():
        for i, ln in enumerate(lines, 1):
            low = ln.lower()
            window = (lines[i - 2].lower() if i >= 2 else "") + " " + low
            for ph in HARD_FAIL_ALWAYS:
                if ph in low:
                    hard.append((s, i, ph, ln.strip()[:90]))
            for ph in HARD_FAIL_IF_POSITIVE:
                if ph in low and not NEG.search(window):
                    hard.append((s, i, ph + " (positive)", ln.strip()[:90]))

    warns = []
    for s, lines in blob.items():
        for i, ln in enumerate(lines, 1):
            low = ln.lower()
            for ph in WARN:
                if ph in low:
                    warns.append((s, i, ph, ln.strip()[:90]))

    missing = [name for name, rx in REQUIRED.items()
               if not re.search(rx, corpus, re.I | re.S)]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    md = ["# SESSION29.9 narrative-claim audit\n",
          f"- hard-fail overclaims: {len(hard)}",
          f"- required concepts present: {len(REQUIRED) - len(missing)}/{len(REQUIRED)}",
          f"- warnings (manual triage): {len(warns)}", ""]
    if hard:
        md += ["## HARD FAILS"] + [f"- {s}:{i} `{ph}` -> {t}" for s, i, ph, t in hard]
    if missing:
        md += ["", "## MISSING REQUIRED CONCEPTS"] + [f"- {m}" for m in missing]
    md += ["", "## WARNINGS (mark OK / EDIT / LIMITATION)"]
    md += [f"- [ ] {s}:{i} `{ph}` -> {t}" for s, i, ph, t in warns]
    OUT.write_text("\n".join(md) + "\n")

    print(f"[claims] hard-fails={len(hard)} missing-required={len(missing)} "
          f"warnings={len(warns)} -> {OUT}")
    if hard or missing:
        for s, i, ph, t in hard:
            print(f"  HARD-FAIL {s}:{i} {ph}")
        for m in missing:
            print(f"  MISSING   {m}")
        sys.exit(1)


if __name__ == "__main__":
    main()
