"""P4 number tracer: fail the build on hand-typed numerals in the manuscript.

Scans the content tex files of the v4 build (sections, tables, captions,
abstract; NOT the auto-generated macros files) for numerals that do not come
from the macros pipeline. Whitelisted contexts (SESSION_35.md section 8):

- display equation environments (equation, align, gather, eqnarray, \\[ \\]):
  true equation constants live there;
- super/subscript digit tokens in inline math (R^2, C_L, t^2, x_1);
- dates (4-digit years 1900-2099) and explicit whitelist regexes in
  scripts/session35/trace_whitelist.json (physical constants like NACA 0012,
  Re = 5000, alpha = 14 deg; each entry carries a justification comment).

Inline math IS scanned: hand-typed result values hide there. Structural
command arguments (\\cite, \\ref, \\label, \\includegraphics, tabular specs,
\\multicolumn, \\cmidrule, lengths) are stripped before scanning.

Exit code 1 on any hit; each hit prints file:line with context. Run:
    python -m scripts.session35.trace_numbers [--tex-root paper]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WHITELIST_JSON = REPO_ROOT / "scripts/session35/trace_whitelist.json"

CONTENT_GLOBS = [
    "sections/*.tex",
    "sections/tables/*.tex",
    "sections/captions/*.tex",
    "protocol_box.tex",
]
EXCLUDE_NAMES = re.compile(r"^macros")

DISPLAY_ENVS = ("equation", "align", "gather", "eqnarray", "multline", "array")

STRIP_COMMANDS = [
    # command with one brace argument whose content is structural, not prose
    r"\\(?:cite|citep|citet|citealp|ref|eqref|autoref|cref|Cref|label|input|include|"
    r"bibliography|bibliographystyle|graphicspath|pgfplotsset|definecolor)\*?"
    r"(?:\[[^\]]*\])?\{[^{}]*\}",
    r"\\includegraphics(?:\[[^\]]*\])?\{[^{}]*\}",
    r"\\(?:vspace|hspace|setlength|addtolength|rule|resizebox)\*?\{[^{}]*\}",
    r"\\begin\{tabular\}\{[^{}]*\}",
    r"\\begin\{[a-zA-Z*]+\}(?:\[[^\]]*\])?",
    r"\\end\{[a-zA-Z*]+\}",
    r"\\multicolumn\{\d+\}\{[^{}]*\}",
    r"\\multirow\{\d+\}\{[^{}]*\}",
    r"\\cmidrule(?:\([^)]*\))?\{[^{}]*\}",
    r"\\(?:toprule|midrule|bottomrule)(?:\[[^\]]*\])?",
    r"\\pending\{[^{}]*\}",  # author-owned placeholders are exempt by definition
]

SUPERSUB = re.compile(r"[\^_]\{?\d+\}?")
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
NUMERAL = re.compile(r"\d+(?:\.\d+)?")


def strip_display_math(text: str) -> str:
    for env in DISPLAY_ENVS:
        text = re.sub(
            rf"\\begin\{{{env}\*?\}}.*?\\end\{{{env}\*?\}}", " ", text, flags=re.S
        )
    text = re.sub(r"\\\[.*?\\\]", " ", text, flags=re.S)
    return text


def strip_comments(text: str) -> str:
    out = []
    for line in text.split("\n"):
        # keep \% but cut from unescaped %
        m = re.search(r"(?<!\\)%", line)
        out.append(line[: m.start()] if m else line)
    return "\n".join(out)


def scan_file(path: Path, whitelist: list[re.Pattern]) -> list[tuple[int, str, str]]:
    raw = path.read_text()
    text = strip_comments(raw)
    text = strip_display_math(text)
    for pat in STRIP_COMMANDS:
        text = re.sub(pat, " ", text)
    text = SUPERSUB.sub(" ", text)
    text = YEAR.sub(" ", text)
    for pat in whitelist:
        text = pat.sub(" ", text)
    hits = []
    for i, line in enumerate(text.split("\n"), start=1):
        for m in NUMERAL.finditer(line):
            lo = max(0, m.start() - 30)
            hi = min(len(line), m.end() + 30)
            hits.append((i, m.group(), line[lo:hi].strip()))
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex-root", default="paper")
    ap.add_argument("--max-print", type=int, default=60)
    args = ap.parse_args()

    root = REPO_ROOT / args.tex_root
    wl_blob = json.loads(WHITELIST_JSON.read_text()) if WHITELIST_JSON.exists() else {}
    whitelist = [re.compile(e["pattern"]) for e in wl_blob.get("entries", [])]

    skip = {e["file"] for e in wl_blob.get("generated_files", [])}
    files: list[Path] = []
    for g in CONTENT_GLOBS:
        files += [p for p in sorted(root.glob(g))
                  if not EXCLUDE_NAMES.match(p.name)
                  and str(p.relative_to(root)) not in skip]

    total = 0
    printed = 0
    for f in files:
        hits = scan_file(f, whitelist)
        total += len(hits)
        for line_no, num, ctx in hits:
            if printed < args.max_print:
                print(f"{f.relative_to(REPO_ROOT)}:{line_no}: [{num}] ...{ctx}...")
                printed += 1
    if total:
        print(f"\n[trace_numbers] FAIL: {total} numeral(s) outside the macros "
              f"pipeline ({printed} shown). Macro-ize or whitelist with "
              f"justification in {WHITELIST_JSON.name}.")
        return 1
    print("[trace_numbers] PASS: no hand-typed numerals found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
