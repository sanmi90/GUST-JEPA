"""SESSION29.2 narrative QC: the JFM-narrative acceptance gates in one pass.

Consolidates the plan's check_abstract_length / audit_narrative_overclaims /
check_jfm figure-count / style checks. Reports, does not edit.

Hard gates (drive the PASS/FAIL verdict, all are correctness/style locks):
- Abstract: prose word count in [150, 240] (target ~180-230).
- Overclaim: forbidden phrases must not appear in a POSITIVE (non-negated) context.
- Style: 0 em-dashes in the main body (no free-typed result decimals).

Advisory (reported, does NOT fail the verdict):
- Figures: the JFM-discipline aspiration is <= 8 main-text figure floats (appendix
  excluded). The author has consciously exceeded it: the SESSION29.2 figure cull
  chose option (a) (move only the named diagnostics to the supplementary appendix,
  preserve the rest) and SESSION29.3 added the physical-staging Figure 1, for an
  accepted budget of 12 main floats. The advisory stays visible so a future edit
  that adds an unbudgeted figure is flagged, but a soft style preference must not
  flip the hard verdict and mask a real correctness regression.
- Pending: \\pending{} count in the main body (the DNS Table 1 rows are expected
  until the simulation collaborators supply the resolution numbers).

Output: outputs/session29/narrative/narrative_qc.md (+ JSON).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SECT = REPO / "paper" / "sections"
OUT_MD = REPO / "outputs" / "session29" / "narrative" / "narrative_qc.md"
OUT_JSON = REPO / "outputs" / "session29" / "narrative" / "narrative_qc.json"

# forbidden POSITIVE claims (allowed only when negated / in a limitation clause)
FORBIDDEN = [
    "each encounter as a single persistent cycle",
    "renders wake structure",
    "counterfactual model",
    "faithful carrier of the wake",
]
# a line is "allowed" for forbidden-phrase purposes if it negates the claim
NEG = re.compile(r"\b(no|not|without|never|do not|does not|cannot|rather than)\b", re.I)
MAIN_SECTIONS = [
    "section_1_introduction",
    "section_2_flow_and_data",
    "section_3_methods",
    "section_4_results",
    "section_5_discussion",
    "section_6_conclusions",
]


def abstract_words() -> int:
    t = (SECT / "abstract.tex").read_text()
    lines = [ln for ln in t.splitlines() if not ln.strip().startswith("%")]
    body = " ".join(lines)
    if "Extreme" in body:
        body = body[body.index("Extreme") :]
    body = re.sub(r"\\[A-Za-z]+", " ", body)  # drop latex macros
    body = re.sub(r"[${}^\\]", " ", body)
    return len(body.split())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-abstract-words", type=int, default=240)
    ap.add_argument(
        "--jfm-figure-aspiration",
        type=int,
        default=8,
        help="JFM-discipline target for main figure floats (advisory only)",
    )
    ap.add_argument(
        "--accepted-figure-budget",
        type=int,
        default=12,
        help="author-accepted main figure floats (advisory ceiling, see module docstring)",
    )
    args = ap.parse_args()

    report = {}
    # 1. abstract length
    aw = abstract_words()
    report["abstract_words"] = aw
    report["abstract_ok"] = 150 <= aw <= args.max_abstract_words

    # 2. overclaim audit (positive, non-negated occurrences in main body)
    overclaims = []
    for s in MAIN_SECTIONS:
        p = SECT / f"{s}.tex"
        if not p.exists():
            continue
        plines = p.read_text().splitlines()
        for i, ln in enumerate(plines, 1):
            if ln.strip().startswith("%"):
                continue
            # negation may wrap from the previous line; check a 2-line window
            window = (plines[i - 2] if i >= 2 else "") + " " + ln
            for phrase in FORBIDDEN:
                if phrase in ln.lower() and not NEG.search(window):
                    overclaims.append(
                        {
                            "file": s,
                            "line": i,
                            "phrase": phrase,
                            "text": ln.strip()[:100],
                        }
                    )
    report["overclaims"] = overclaims
    report["overclaim_ok"] = len(overclaims) == 0

    # 3. main-text figure count: count figure FLOATS (\begin{figure}), not
    # \includegraphics, so a multi-panel figure (e.g. the method arch+protocol
    # figure) counts as one. Appendix sections are excluded. Advisory only.
    n_floats = 0
    graphics = []
    for s in MAIN_SECTIONS:
        p = SECT / f"{s}.tex"
        if p.exists():
            txt = p.read_text()
            n_floats += len(re.findall(r"\\begin\{figure\*?\}", txt))
            graphics += re.findall(r"includegraphics[^{]*\{([^}]*)\}", txt)
    report["main_figures"] = n_floats
    report["main_figure_paths"] = [Path(f).name for f in graphics]
    report["figures_within_aspiration"] = n_floats <= args.jfm_figure_aspiration
    report["figures_within_budget"] = n_floats <= args.accepted_figure_budget

    # 4. style: em-dashes, pending in main body
    emdash = pending = 0
    for s in MAIN_SECTIONS:
        p = SECT / f"{s}.tex"
        if not p.exists():
            continue
        txt = p.read_text()
        emdash += len(re.findall(r"[—–]", txt))
        pending += txt.count("\\pending{")
    report["emdash"] = emdash
    report["pending_main"] = pending
    report["style_ok"] = emdash == 0

    # hard verdict: correctness/style locks only (figure budget is advisory)
    gates = {k: report[k] for k in ("abstract_ok", "overclaim_ok", "style_ok")}
    report["all_gates_pass"] = all(gates.values())

    n = report["main_figures"]
    if report["figures_within_aspiration"]:
        fig_note = f"within JFM aspiration (<= {args.jfm_figure_aspiration})"
    elif report["figures_within_budget"]:
        fig_note = (
            f"over JFM aspiration (<= {args.jfm_figure_aspiration}), within the "
            f"author-accepted budget (<= {args.accepted_figure_budget}); see docstring"
        )
    else:
        fig_note = (
            f"OVER the accepted budget (<= {args.accepted_figure_budget}): an "
            f"unbudgeted figure was added, cull or re-justify"
        )

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2))
    lines = [
        "# SESSION29.2 narrative QC\n",
        "## Hard gates (drive the verdict)",
        f"- abstract words: {aw} (gate <= {args.max_abstract_words}): "
        f"{'PASS' if report['abstract_ok'] else 'FAIL'}",
        f"- overclaim positive occurrences: {len(overclaims)}: "
        f"{'PASS' if report['overclaim_ok'] else 'FAIL'}",
        f"- em-dashes (main): {emdash}: "
        f"{'PASS' if report['style_ok'] else 'FAIL'}",
        "",
        f"**HARD GATES: {'PASS' if report['all_gates_pass'] else 'FAIL'}**",
        "",
        "## Advisory (reported, does not fail the verdict)",
        f"- main figure floats: {n} ({fig_note})",
        f"- pending (main): {pending} (DNS Table 1 rows, expected)",
    ]
    if overclaims:
        lines += ["", "## overclaim hits"] + [
            f"- {o['file']}:{o['line']} `{o['phrase']}` -> {o['text']}"
            for o in overclaims
        ]
    if not report["figures_within_budget"]:
        lines += ["", "## main figures (over accepted budget)"] + [
            f"- {n}" for n in report["main_figure_paths"]
        ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
