"""Convert paper/sections/*.md to .tex for the manuscript build.

Handles the conventions used in this project:
- `# Section ...` (title line)        -> stripped (main.tex provides \\section)
- `## N.M Title`                       -> \\subsection{Title}
- `### N.M.K Title`                    -> \\subsubsection{Title}
- `**bold**`                           -> \\textbf{bold}
- `*italics*` (not inside math)        -> \\emph{italics}
- inline `code`                        -> \\texttt{code}
- pipe tables                          -> longtable (best-effort, only for
                                          sections 1 and 2 that may still have them)
- horizontal rule `---` (markdown)     -> stripped
- "LaTeX-friendly markdown ..." preamble lines -> stripped

Leaves alone:
- $math$, $$math$$
- \\cite{key}, \\input{...}, \\begin{...}, \\ref{}
- existing LaTeX commands written in the source
"""
import re
from pathlib import Path

REPO = Path("/home/carlos/GUST-JEPA")
SECTIONS_DIR = REPO / "paper/sections"


def convert_md(text: str) -> str:
    lines = text.splitlines()
    out_lines = []
    in_preamble_paragraph = False
    skipped_title = False

    i = 0
    while i < len(lines):
        line = lines[i]

        # Strip the first H1 title line
        if not skipped_title and line.startswith("# "):
            skipped_title = True
            i += 1
            continue

        # Strip the "LaTeX-friendly markdown..." preamble paragraph
        if line.strip().startswith("LaTeX-friendly markdown"):
            # skip until blank line
            while i < len(lines) and lines[i].strip():
                i += 1
            i += 1  # skip the blank
            continue

        # Skip horizontal rules
        if line.strip() == "---":
            i += 1
            continue

        # Section headers
        m = re.match(r"^### +(?:\d+(?:\.\d+)*\s+)?(.+)$", line)
        if m:
            out_lines.append(rf"\subsubsection{{{m.group(1).strip()}}}")
            i += 1
            continue
        m = re.match(r"^## +(?:\d+(?:\.\d+)*\s+)?(.+)$", line)
        if m:
            out_lines.append(rf"\subsection{{{m.group(1).strip()}}}")
            i += 1
            continue

        # Pipe table block: collect contiguous pipe-delimited lines
        if line.strip().startswith("|"):
            tbl = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                tbl.append(lines[i])
                i += 1
            out_lines.append(convert_pipe_table(tbl))
            continue

        # Inline markdown -> LaTeX
        s = line
        # `code` -> \texttt{code}; do this before bold/italic since backticks won't
        # contain underscores that get rewritten. Also escape _ inside texttt content.
        def _texttt(m):
            inner = m.group(1).replace("_", r"\_").replace("%", r"\%").replace("#", r"\#")
            return r"\texttt{" + inner + "}"
        s = re.sub(r"`([^`]+)`", _texttt, s)
        # **bold**
        s = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", s)
        # *italics* (but not ** which we already consumed)
        s = re.sub(r"(?<![*\\])\*([^*\n]+?)\*(?!\*)", r"\\emph{\1}", s)
        # Outside-of-texttt underscores in technical content like r2(z -> CL_future).
        # Wrap obvious identifiers token-by-token: any run [A-Za-z]+_[A-Za-z0-9_]+
        # that is NOT already inside a TeX command argument should be in math.
        # Simplest heuristic: if a line has both a free `_` AND a free `>` (the
        # `r2(z -> CL_future)` pattern from sections 1-2), inline-escape underscores
        # outside TeX command arguments.
        # We do a targeted token replacement: A_B style identifiers (no math)
        s = _escape_freestanding_underscores(s)

        out_lines.append(s)
        i += 1

    text = "\n".join(out_lines).rstrip() + "\n"
    return text


def _escape_freestanding_underscores(s: str) -> str:
    """Escape underscores that are NOT inside math delimiters $...$ or inside a
    LaTeX command argument like \\texttt{...}, \\cite{...}, \\input{...}.

    Strategy: walk the string in chunks. Inside $...$ we leave _ alone (math).
    Inside the argument of \\texttt or \\cite or \\input or \\ref or \\label
    we leave _ alone (handled by texttt-escape above for texttt; for cite/ref
    the bib key uses _ verbatim and is fine because pdflatex defers reading the
    arg until aux time). Elsewhere, _ in text mode needs \\_.
    """
    out = []
    i = 0
    in_math = False
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == "$":
            in_math = not in_math
            out.append(ch)
            i += 1
            continue
        if not in_math and ch == "\\":
            # Already-escaped `\_`: pass through as-is, do NOT re-escape.
            if i + 1 < n and s[i + 1] == "_":
                out.append("\\_")
                i += 2
                continue
            # detect \cmd{...} where we want to skip the braced argument
            m = re.match(r"\\([A-Za-z]+)\s*\{", s[i:])
            if m:
                cmd = m.group(1)
                # find matching close brace, accounting for nesting
                start = i + m.end()
                depth = 1
                j = start
                while j < n and depth > 0:
                    if s[j] == "{":
                        depth += 1
                    elif s[j] == "}":
                        depth -= 1
                    j += 1
                # copy through verbatim (including any underscores)
                out.append(s[i:j])
                i = j
                continue
            # other \cmd without brace, e.g. \\
            out.append(ch)
            i += 1
            continue
        if not in_math and ch == "_":
            out.append(r"\_")
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def convert_pipe_table(lines: list[str]) -> str:
    # First row = header, second row = separator (|---|---|), rest = data
    # Best effort tabular; uses booktabs.
    rows = []
    for L in lines:
        # split on |, drop the leading/trailing empties
        parts = [p.strip() for p in L.strip().strip("|").split("|")]
        rows.append(parts)
    if len(rows) < 2:
        return "\n".join(lines)  # malformed, pass through
    header = rows[0]
    body = rows[2:]  # skip the separator row

    # determine col alignment from the separator row
    seps = rows[1]
    aligns = []
    for sep in seps:
        sep = sep.strip()
        if sep.startswith(":") and sep.endswith(":"):
            aligns.append("c")
        elif sep.endswith(":"):
            aligns.append("r")
        elif sep.startswith(":"):
            aligns.append("l")
        else:
            aligns.append("l")
    align_str = " ".join(aligns)

    def cells(row):
        # escape & in cells, leave LaTeX commands alone
        cleaned = []
        for c in row:
            c2 = re.sub(r"`([^`]+)`", r"\\texttt{\1}", c)
            c2 = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", c2)
            c2 = c2.replace("%", r"\%")
            cleaned.append(c2)
        return " & ".join(cleaned)

    out = []
    out.append(r"\begin{table}[h]")
    out.append(r"  \centering")
    out.append(r"  \small")
    out.append(rf"  \begin{{tabular}}{{{align_str}}}")
    out.append(r"    \toprule")
    out.append("    " + cells(header) + r" \\")
    out.append(r"    \midrule")
    for row in body:
        out.append("    " + cells(row) + r" \\")
    out.append(r"    \bottomrule")
    out.append(r"  \end{tabular}")
    out.append(r"\end{table}")
    return "\n".join(out)


def main():
    files = sorted(SECTIONS_DIR.glob("section_*.md")) + [SECTIONS_DIR / "abstract.md"]
    for src in files:
        text = src.read_text()
        tex = convert_md(text)
        dst = src.with_suffix(".tex")
        dst.write_text(tex)
        print(f"  {src.name:<40} -> {dst.name} ({len(tex)} chars)")


if __name__ == "__main__":
    main()
