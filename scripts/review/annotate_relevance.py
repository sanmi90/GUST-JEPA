#!/usr/bin/env python3
"""Relevance colour-coding overlay for the manuscript (branch review-colour-map).

Wraps every prose paragraph and every figure/table caption of the body and the
appendices in

    \\rb{<class>}{<ID>}{<rationale>}...text...\\re

so that reviewmarks.tex can colour it and hang a margin note beside it. The
manuscript text itself is never rewritten: --strip removes every marker and
restores the files exactly.

Three modes:

    --inventory   parse the sources, assign stable IDs, and write the ledger
                  skeleton (editorial/RELEVANCE_MAP.md) plus a preview file
    --apply       read the ledger and insert the markers
    --strip       remove every marker

IDs are <unit>-<nn> for paragraphs and <unit>-C<n> for captions, numbered in
source order within each unit, so they stay stable as long as no block is
added or removed. Re-run --inventory after any structural edit; it preserves
the class and rationale of every ID that still exists.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / "paper"
LEDGER = REPO / "editorial" / "RELEVANCE_MAP.md"
PREVIEW = REPO / "editorial" / "RELEVANCE_PREVIEW.md"

# Source files in reading order, with their ID prefix. The tables live in their
# own files but are part of the section that inputs them; they keep their own
# prefix so an ID always names exactly one file.
UNITS: list[tuple[str, str]] = [
    ("AB", "sections/abstract.tex"),
    ("S1", "sections/section_1_introduction.tex"),
    ("S2", "sections/section_2_flow_and_data.tex"),
    ("S3", "sections/section_3_methods.tex"),
    ("S31", "sections/v4/s3_1_chang_head.tex"),
    ("S33", "sections/v4/s3_3_rex.tex"),
    ("S34", "sections/v4/s3_4_estimators.tex"),
    ("S35", "sections/v4/s3_5_protocol.tex"),
    ("S4", "sections/section_4_results.tex"),
    ("S4A", "sections/v4/s4_a_construction.tex"),
    ("S4B", "sections/v4/s4_b_reconstruction.tex"),
    ("S4B2", "sections/v4/s4_b2_dimension.tex"),
    ("S4C", "sections/v4/s4_c_prediction.tex"),
    ("S4D", "sections/v4/s4_d_assimilation.tex"),
    ("S5", "sections/section_5_discussion.tex"),
    ("S6", "sections/section_6_conclusions.tex"),
    ("APA", "sections/appendix_a_regularisation.tex"),
    ("APB", "sections/appendix_b_sensing.tex"),
    ("APC", "sections/appendix_c_calibration.tex"),
    # NOT annotated: sections/appendix_c_supplementary_figures.tex is input by
    # supplementary.tex, not by main.tex, so it is already outside the paper;
    # sections/protocol_box.tex is input by nothing at all.
    ("TBAS", "sections/tables/table_baselines.tex"),
    ("TCLO", "sections/tables/table_closure.tex"),
    ("TSSI", "sections/tables/table_critical_ssim.tex"),
    ("TDNS", "sections/tables/table_dns_metadata.tex"),
    ("TENK", "sections/tables/table_enkf.tex"),
    ("TENV", "sections/tables/table_envelope.tex"),
    ("TFAM", "sections/tables/table_family_filter.tex"),
    ("TFIL", "sections/tables/table_filter_error.tex"),
    ("TMEC", "sections/tables/table_mechanism.tex"),
    ("TOBS", "sections/tables/table_obs_critical.tex"),
    ("TREC", "sections/tables/table_recovery.tex"),
]

CLASSES = {"K", "T", "A", "S", "D"}

# Environments whose body is display mathematics: a prose paragraph broken by
# one of these is still one paragraph, so the marker spans it.
MATH_ENVS = {
    "equation", "equation*", "align", "align*", "gather", "gather*",
    "multline", "multline*", "eqnarray", "eqnarray*", "displaymath", "split",
}

# A depth-0 line starting with one of these is structural, not prose.
STRUCTURAL = re.compile(
    r"\\(section|subsection|subsubsection|paragraph|input|include|label|"
    r"bigskip|medskip|smallskip|vspace|hspace|clearpage|newpage|par\b|"
    r"FloatBarrier|nolinenumbers|runninglinenumbers|centering|noindent\s*$|"
    r"begin|end|item|toprule|midrule|bottomrule|hline|multicolumn|caption)"
)

BEGIN_RE = re.compile(r"\\begin\{([^}]*)\}")
END_RE = re.compile(r"\\end\{([^}]*)\}")


def strip_comment(line: str) -> str:
    """Drop a trailing LaTeX comment, honouring \\%."""
    out = []
    i = 0
    while i < len(line):
        c = line[i]
        if c == "\\" and i + 1 < len(line):
            out.append(line[i : i + 2])
            i += 2
            continue
        if c == "%":
            break
        out.append(c)
        i += 1
    return "".join(out)


def match_brace(text: str, open_idx: int) -> int:
    """Index of the brace matching the one at open_idx, or -1."""
    depth = 0
    i = open_idx
    while i < len(text):
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


class Block:
    def __init__(self, bid, kind, start, end, preview):
        self.bid = bid
        self.kind = kind          # "para" or "caption"
        self.start = start        # 0-based line index of first content line
        self.end = end            # 0-based line index of last content line
        self.preview = preview


def parse_file(path: Path, prefix: str) -> list[Block]:
    """Return the annotatable blocks of one source file, in source order.

    Markers are stripped first, so a file that is already annotated parses to
    exactly the same blocks as a clean one; marker insertion never adds a line.
    """
    lines = strip_text(path.read_text()).splitlines()
    depth = 0
    env_stack: list[str] = []
    blocks: list[Block] = []
    para_n = 0
    cap_n = 0

    cur: list[int] = []          # line indices of the current prose run
    pending_math_gap = False     # a display-math env just closed at depth 0

    def flush(force_new: bool = True):
        nonlocal cur, para_n
        if not cur:
            return
        para_n += 1
        text = " ".join(strip_comment(lines[i]).strip() for i in cur)
        text = re.sub(r"\s+", " ", text).strip()
        blocks.append(
            Block(f"{prefix}-{para_n:02d}", "para", cur[0], cur[-1], text[:160])
        )
        cur = []

    i = 0
    while i < len(lines):
        raw = lines[i]
        body = strip_comment(raw).strip()

        if depth == 0:
            # A caption can appear only inside a float, so at depth 0 the only
            # things that matter are prose runs and environment openings.
            if not raw.strip():
                if cur and not pending_math_gap:
                    flush()
                i += 1
                continue
            if not body:
                # comment-only line: transparent. Several paragraphs carry a
                # % REVIEW-CLAIM block mid-paragraph, and splitting there would
                # invent a paragraph break that the PDF does not have.
                i += 1
                continue

            begins = BEGIN_RE.findall(body)
            if begins:
                env = begins[0]
                if env in MATH_ENVS and cur:
                    # keep the run open across the display; the continuation
                    # paragraph decides whether it really continues
                    pending_math_gap = True
                else:
                    flush()
                    pending_math_gap = False
            elif STRUCTURAL.match(body):
                flush()
                pending_math_gap = False
            else:
                if pending_math_gap:
                    # continuation only when the text resumes mid-sentence
                    if re.match(r"^[a-z]", body):
                        cur.append(i)
                    else:
                        flush()
                        cur.append(i)
                    pending_math_gap = False
                else:
                    cur.append(i)

        else:
            # inside an environment: look for captions
            if "\\caption" in body:
                joined_start = i
                text = "\n".join(lines[i:])
                m = re.search(r"\\caption\s*(\[[^\]]*\])?\s*\{", text)
                if m:
                    open_idx = m.end() - 1
                    close_idx = match_brace(text, open_idx)
                    # \caption{} panel labels on subfigures carry no prose and
                    # are skipped; only captions with text are annotatable
                    if close_idx > 0 and text[open_idx + 1 : close_idx].strip():
                        cap_n += 1
                        n_lines = text[:close_idx].count("\n")
                        preview = re.sub(
                            r"\s+", " ", strip_comment(text[open_idx + 1 : open_idx + 200])
                        )
                        blocks.append(
                            Block(
                                f"{prefix}-C{cap_n}",
                                "caption",
                                joined_start,
                                joined_start + n_lines,
                                preview.strip()[:160],
                            )
                        )

        # update depth after processing the line
        for env in BEGIN_RE.findall(body):
            env_stack.append(env)
            depth += 1
        for env in END_RE.findall(body):
            if env_stack:
                env_stack.pop()
            depth = max(0, depth - 1)

        i += 1

    flush()
    return blocks


# --------------------------------------------------------------------------
# ledger


CLASS_TABLE = """<!-- GENERATED SKELETON, then hand-classified. Regenerate the ID, file, line
and type columns with `python scripts/review/annotate_relevance.py --inventory`,
which preserves the class and rationale of every ID that still exists. Insert the
markers with --apply, remove every marker with --strip. -->

# Relevance map

Branch `review-colour-map`. One row per annotatable block of the manuscript body
and appendices: every prose paragraph and every figure or table caption that
`main.tex` typesets. The overlay that renders it is `paper/reviewmarks.tex`.

| | class | colour | meaning |
|---|---|---|---|
| **K** | keep | green | stays in the body as it is |
| **T** | trim | orange | could be removed; the paper survives without it |
| **A** | annex | purple | belongs in an appendix, not the body |
| **S** | supp | brown | belongs in the supplementary material |
| **D** | delete | red | can be deleted outright, nothing is lost |

`note` is the rationale printed in the margin beside the block. It is LaTeX, so
escape `_`, `%`, `&` and `#`, and it must not contain the `|` character.

## Caveats

Two paragraphs separated only by a `%` comment parse as one block, because a
comment swallows its own line ending and the two really are one paragraph in the
PDF. `sections/tables/table_dns_metadata.tex` is generated by
`scripts/session29/render_table1_from_yaml.py`, so re-rendering it drops that
marker; re-run `--apply` afterwards.
`sections/appendix_c_supplementary_figures.tex` is input by `supplementary.tex`
rather than `main.tex`, and `sections/protocol_box.tex` is input by nothing at
all, so neither is annotated.

"""


def ledger_header(rows: list[str]) -> str:
    """Header plus a tally recomputed from the rows themselves."""
    from collections import Counter

    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    total = Counter(c[4] for c in cells)
    per_file: dict[str, Counter] = {}
    for c in cells:
        per_file.setdefault(c[1], Counter())[c[4]] += 1

    out = [CLASS_TABLE, "## Tally\n", "| class | blocks |", "|---|---|"]
    for k in "KTASD":
        out.append(f"| {k} | {total.get(k, 0)} |")
    out.append(f"| **total** | **{len(rows)}** |")
    out.append("\nBlocks not marked keep, by file:\n")
    out.append("| file | T | A | S | D |")
    out.append("|---|---|---|---|---|")
    for f in sorted(per_file):
        c = per_file[f]
        if not any(c[k] for k in "TASD"):
            continue
        out.append(f"| `{f}` | {c['T'] or ''} | {c['A'] or ''} | {c['S'] or ''} | {c['D'] or ''} |")
    out.append("\n| ID | file | lines | type | class | note |")
    out.append("|---|---|---|---|---|---|")
    return "\n".join(out) + "\n"


def read_ledger() -> dict[str, tuple[str, str]]:
    """ID -> (class, note) from the existing ledger, if any."""
    if not LEDGER.exists():
        return {}
    out = {}
    for line in LEDGER.read_text().splitlines():
        if not line.startswith("|") or line.startswith("|---") or "| ID |" in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 6:
            continue
        bid, _f, _l, _t, cls, note = cells[:6]
        out[bid] = (cls, note)
    return out


def cmd_inventory() -> int:
    old = read_ledger()
    rows = []
    previews = []
    for prefix, rel in UNITS:
        path = PAPER / rel
        if not path.exists():
            print(f"missing: {rel}", file=sys.stderr)
            continue
        blocks = parse_file(path, prefix)
        previews.append(f"\n## {rel}  ({prefix}, {len(blocks)} blocks)\n")
        for b in blocks:
            cls, note = old.get(b.bid, ("", ""))
            rows.append(
                f"| {b.bid} | {rel} | {b.start + 1}-{b.end + 1} | {b.kind} | "
                f"{cls} | {note} |"
            )
            previews.append(f"- **{b.bid}** ({b.kind}, L{b.start+1}-{b.end+1}) {b.preview}")

    LEDGER.write_text(ledger_header(rows) + "\n".join(rows) + "\n")
    PREVIEW.write_text(
        "<!-- generated by scripts/review/annotate_relevance.py --inventory -->\n"
        "# Block previews\n" + "\n".join(previews) + "\n"
    )
    n_new = sum(1 for r in rows if r.split("|")[5].strip() == "")
    print(f"{len(rows)} blocks -> {LEDGER.relative_to(REPO)} ({n_new} unclassified)")
    return 0


# --------------------------------------------------------------------------
# apply / strip

RB_RE = re.compile(r"\\rb\{")
# \re must not match the \re of \ref, \relax, \renewcommand ... An earlier
# version used a plain string replace and silently turned every \ref{} in the
# manuscript into f{}. The negative lookahead is the whole guard.
RE_RE = re.compile(r"\\re(?![a-zA-Z@])")


def strip_text(text: str) -> str:
    """Remove every marker. Exactly inverts apply: markers are inserted inline,
    never on lines of their own, so stripping restores the file byte for byte."""
    out = text
    while True:
        m = RB_RE.search(out)
        if not m:
            break
        idx = m.end() - 1
        for _ in range(3):
            close = match_brace(out, idx)
            if close < 0:
                raise SystemExit("unbalanced \\rb argument")
            nxt = out.find("{", close + 1)
            if nxt < 0 or out[close + 1 : nxt].strip():
                idx = close
                break
            idx = nxt
        out = out[: m.start()] + out[idx + 1 :]
    return RE_RE.sub("", out)


def cmd_strip() -> int:
    n = 0
    for _prefix, rel in UNITS:
        path = PAPER / rel
        if not path.exists():
            continue
        before = path.read_text()
        after = strip_text(before)
        if after != before:
            path.write_text(after)
            n += 1
    print(f"stripped markers from {n} files")
    return 0


def cmd_apply(strict: bool) -> int:
    ledger = read_ledger()
    if not ledger:
        raise SystemExit("no ledger; run --inventory first")

    total = 0
    skipped = []
    for prefix, rel in UNITS:
        path = PAPER / rel
        if not path.exists():
            continue
        original = path.read_text()
        text = strip_text(original)
        path.write_text(text)          # parse_file reads from disk
        blocks = parse_file(path, prefix)
        lines = text.split("\n")

        # apply from the end so earlier line indices stay valid
        edits = []
        for b in blocks:
            cls, note = ledger.get(b.bid, ("", ""))
            if cls not in CLASSES:
                skipped.append(b.bid)
                continue
            note = note.strip()
            if not note:
                skipped.append(b.bid)
                continue
            edits.append((b, cls, note))

        for b, cls, note in sorted(edits, key=lambda e: e[0].start, reverse=True):
            marker = f"\\rb{{{cls}}}{{{b.bid}}}{{{note}}}"
            # Inline insertion only: no line is added, so the line numbers in
            # the ledger stay valid and --strip restores the file exactly.
            if b.kind == "para":
                lines[b.start] = marker + lines[b.start]
                lines[b.end] = lines[b.end] + "\\re"
            else:
                # caption: wrap the argument, not the \caption command
                chunk = "\n".join(lines[b.start : b.end + 1])
                m = re.search(r"\\caption\s*(\[[^\]]*\])?\s*\{", chunk)
                open_idx = m.end() - 1
                close_idx = match_brace(chunk, open_idx)
                chunk = (
                    chunk[: open_idx + 1]
                    + marker
                    + chunk[open_idx + 1 : close_idx]
                    + "\\re"
                    + chunk[close_idx:]
                )
                lines[b.start : b.end + 1] = chunk.split("\n")
            total += 1

        applied = "\n".join(lines)
        path.write_text(applied)

        # Round-trip guard: the overlay must be removable without a trace.
        if strip_text(applied) != text:
            path.write_text(original)
            raise SystemExit(f"round-trip check failed on {rel}; file restored")

    print(f"applied {total} markers")
    if skipped:
        print(f"skipped {len(skipped)} unclassified blocks: {', '.join(skipped[:12])}"
              + (" ..." if len(skipped) > 12 else ""))
        if strict:
            return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--inventory", action="store_true")
    g.add_argument("--apply", action="store_true")
    g.add_argument("--strip", action="store_true")
    ap.add_argument("--strict", action="store_true",
                    help="--apply exits non-zero if any block is unclassified")
    args = ap.parse_args()
    if args.inventory:
        return cmd_inventory()
    if args.strip:
        return cmd_strip()
    return cmd_apply(args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
