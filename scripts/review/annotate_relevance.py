#!/usr/bin/env python3
"""Relevance colour-coding overlay for a LaTeX manuscript.

Wraps every prose paragraph and every figure/table caption of a document in

    \\rb{<class>}{<ID>}{<rationale>}...text...\\re

so that the companion style file (paper/reviewmarks.tex) can colour the block
and hang a margin note beside it. Five classes: K keep, T could be removed,
A move to an appendix, S move to the supplementary material, D delete.

The manuscript text is never rewritten. Markers are inserted inline, so no line
is added and the ledger's line numbers stay valid, and --strip restores the
sources byte for byte. --apply round-trips its own output through --strip and
refuses to write if anything differs.

    --inventory   walk the document, assign stable IDs, write the ledger
                  skeleton and a preview file. Preserves the class and
                  rationale of every ID that still exists.
    --apply       read the ledger and insert the markers.
    --strip       remove every marker.

The set of files comes from the \\input/\\include graph of the configured root
documents, in source order, so nothing has to be listed by hand and a file that
no document reaches is never annotated. Configuration lives in
scripts/review/relevance.json; see that file, and README.md beside it, for how
to point this at a different project.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parent / "relevance.json"

CLASSES = {"K", "T", "A", "S", "D"}
CLASS_NAMES = {
    "K": ("keep", "green", "stays in the body as it is"),
    "T": ("trim", "orange", "could be removed; the paper survives without it"),
    "A": ("annex", "purple", "belongs in an appendix, not the body"),
    "S": ("supp", "brown", "belongs in the supplementary material"),
    "D": ("delete", "red", "can be deleted outright, nothing is lost"),
}

# Environments whose body is display mathematics: a prose paragraph broken by
# one of these is still one paragraph, so the marker spans it.
MATH_ENVS = {
    "equation", "equation*", "align", "align*", "gather", "gather*",
    "multline", "multline*", "eqnarray", "eqnarray*", "displaymath", "split",
}

# List environments. TeX ends a paragraph at a blank line, not at \end{itemize},
# so prose that resumes straight after a displayed list is still that paragraph
# and must stay one block: splitting it would renumber every later block in the
# file and silently misalign the ledger. Unlike MATH_ENVS the continuation is
# not required to start lowercase, and a blank line after the list still ends
# the block, which is exactly TeX's own rule.
LIST_ENVS = {"enumerate", "itemize", "description"}

# A depth-0 line starting with one of these is structural, not prose.
STRUCTURAL = re.compile(
    r"\\(section|subsection|subsubsection|paragraph|input|include|label|"
    r"bigskip|medskip|smallskip|vspace|hspace|clearpage|newpage|par\b|"
    r"FloatBarrier|nolinenumbers|runninglinenumbers|centering|noindent\s*$|"
    r"begin|end|item|toprule|midrule|bottomrule|hline|multicolumn|caption)"
)

BEGIN_RE = re.compile(r"\\begin\{([^}]*)\}")
END_RE = re.compile(r"\\end\{([^}]*)\}")
INPUT_RE = re.compile(r"\\(?:input|include)\s*\{([^}]*)\}")
RB_RE = re.compile(r"\\rb\{")
# \re must not match the \re of \ref, \relax, \renewcommand ... An early version
# used a plain string replace and silently turned every \ref{} in the manuscript
# into f{}. The negative lookahead is the whole guard.
RE_RE = re.compile(r"\\re(?![a-zA-Z@])")


# ---------------------------------------------------------------- config


class Config:
    def __init__(self, path: Path):
        raw = json.loads(path.read_text())
        self.path = path
        self.repo = _find_repo_root(path)
        self.roots = [self.repo / r for r in raw["roots"]]
        self.ledger = self.repo / raw.get("ledger", "RELEVANCE_MAP.md")
        self.preview = self.repo / raw.get("preview", "RELEVANCE_PREVIEW.md")
        self.exclude = [re.compile(p) for p in raw.get("exclude", [])]
        self.prefixes: dict[str, str] = raw.get("prefixes", {})
        self.generated = set(raw.get("generated", []))


def _find_repo_root(config_path: Path) -> Path:
    for parent in config_path.parents:
        if (parent / ".git").exists():
            return parent
    return config_path.parents[1]


def auto_prefix(stem: str, taken: set[str]) -> str:
    """Initials of the stem, deduplicated. Only used for unlisted files."""
    parts = [p for p in re.split(r"[_\-.]", stem) if p]
    cand = "".join(p[0] for p in parts).upper()[:6] or stem[:3].upper()
    base, n = cand, 2
    while cand in taken:
        cand, n = f"{base}{n}", n + 1
    return cand


def discover(cfg: Config) -> list[tuple[str, Path, str]]:
    """(prefix, absolute path, path relative to the root's directory), in source
    order, over the \\input graph of every configured root."""
    out: list[tuple[str, Path, str]] = []
    seen: set[Path] = set()
    taken: set[str] = set()

    def walk(tex: Path, base: Path) -> None:
        if tex in seen or not tex.exists():
            if not tex.exists():
                print(f"warning: {tex} does not exist", file=sys.stderr)
            return
        seen.add(tex)
        text = strip_comments(tex.read_text())
        for target in INPUT_RE.findall(text):
            child = base / target
            if child.suffix != ".tex":
                child = child.with_suffix(".tex")
            rel = child.relative_to(base).as_posix()
            if any(p.search(rel) for p in cfg.exclude):
                continue
            if child.exists() and child not in seen:
                stem = child.stem
                prefix = cfg.prefixes.get(stem) or auto_prefix(stem, taken)
                taken.add(prefix)
                out.append((prefix, child, rel))
            walk(child, base)

    for root in cfg.roots:
        walk(root, root.parent)
    return out


# ------------------------------------------------------------- LaTeX bits


def strip_comment(line: str) -> str:
    """Drop a trailing LaTeX comment, honouring \\%."""
    out, i = [], 0
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


def strip_comments(text: str) -> str:
    return "\n".join(strip_comment(l) for l in text.splitlines())


def match_brace(text: str, open_idx: int) -> int:
    """Index of the brace matching the one at open_idx, or -1."""
    depth, i = 0, open_idx
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


def strip_text(text: str) -> str:
    """Remove every marker. Exactly inverts apply."""
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


class Block:
    def __init__(self, bid: str, kind: str, start: int, end: int, preview: str):
        self.bid = bid
        self.kind = kind          # "para" or "caption"
        self.start = start        # 0-based index of the first content line
        self.end = end            # 0-based index of the last content line
        self.preview = preview


def parse_text(text: str, prefix: str) -> list[Block]:
    """The annotatable blocks of one source file, in source order.

    Markers are stripped first, so an already-annotated file parses to exactly
    the same blocks as a clean one; marker insertion never adds a line.
    """
    lines = strip_text(text).splitlines()
    depth, env_stack = 0, []
    blocks: list[Block] = []
    para_n = cap_n = 0
    cur: list[int] = []
    pending_math_gap = False

    def flush() -> None:
        nonlocal cur, para_n
        if not cur:
            return
        para_n += 1
        body = " ".join(strip_comment(lines[i]).strip() for i in cur)
        blocks.append(
            Block(f"{prefix}-{para_n:02d}", "para", cur[0], cur[-1],
                  re.sub(r"\s+", " ", body).strip()[:160])
        )
        cur = []

    for i, raw in enumerate(lines):
        body = strip_comment(raw).strip()

        if depth == 0:
            if not raw.strip():
                if cur and not pending_math_gap:
                    flush()
                continue
            if not body:
                # Comment-only line: transparent. A comment swallows its own
                # line ending, so two "paragraphs" split by one are a single
                # paragraph in the PDF and must stay a single block.
                continue

            begins = BEGIN_RE.findall(body)
            if begins:
                if begins[0] in MATH_ENVS and cur:
                    pending_math_gap = True
                elif begins[0] in LIST_ENVS and cur:
                    pass  # span the list; only a blank line ends the block
                else:
                    flush()
                    pending_math_gap = False
            elif STRUCTURAL.match(body):
                flush()
                pending_math_gap = False
            else:
                if pending_math_gap and not re.match(r"^[a-z]", body):
                    flush()
                cur.append(i)
                pending_math_gap = False

        elif "\\caption" in body:
            chunk = "\n".join(lines[i:])
            m = re.search(r"\\caption\s*(\[[^\]]*\])?\s*\{", chunk)
            if m:
                open_idx = m.end() - 1
                close_idx = match_brace(chunk, open_idx)
                # \caption{} panel labels on subfigures carry no prose
                if close_idx > 0 and chunk[open_idx + 1 : close_idx].strip():
                    cap_n += 1
                    preview = re.sub(
                        r"\s+", " ", strip_comment(chunk[open_idx + 1 : open_idx + 200])
                    )
                    blocks.append(
                        Block(f"{prefix}-C{cap_n}", "caption", i,
                              i + chunk[:close_idx].count("\n"), preview.strip()[:160])
                    )

        for env in BEGIN_RE.findall(body):
            env_stack.append(env)
            depth += 1
        for _env in END_RE.findall(body):
            if env_stack:
                env_stack.pop()
            depth = max(0, depth - 1)

    flush()
    return blocks


# ------------------------------------------------------------------ ledger

ROW_RE = re.compile(r"^\| (\S+) \| ([^|]*)\| ([^|]*)\| ([^|]*)\|([^|]*)\|([^|]*)\|$")


def read_ledger(cfg: Config) -> dict[str, tuple[str, str]]:
    """ID -> (class, note) from the existing ledger, if any."""
    if not cfg.ledger.exists():
        return {}
    out = {}
    for line in cfg.ledger.read_text().splitlines():
        m = ROW_RE.match(line)
        if m and m.group(1) != "ID":
            out[m.group(1)] = (m.group(5).strip(), m.group(6).strip())
    return out


def ledger_header(cfg: Config, rows: list[str]) -> str:
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    total = Counter(c[4] for c in cells)
    per_file: dict[str, Counter] = {}
    for c in cells:
        per_file.setdefault(c[1], Counter())[c[4]] += 1

    roots = ", ".join(f"`{r.relative_to(cfg.repo)}`" for r in cfg.roots)
    out = [
        "<!-- GENERATED SKELETON, then hand-classified. Regenerate the ID, file,",
        "line and type columns with",
        "`python scripts/review/annotate_relevance.py --inventory`, which preserves",
        "the class and rationale of every ID that still exists. Insert the markers",
        "with --apply, remove every marker with --strip. -->",
        "",
        "# Relevance map",
        "",
        f"One row per annotatable block reachable from {roots}: every prose",
        "paragraph and every figure or table caption the document typesets. The",
        "overlay that renders it is `paper/reviewmarks.tex`.",
        "",
        "| | class | colour | meaning |",
        "|---|---|---|---|",
    ]
    for k in "KTASD":
        name, colour, meaning = CLASS_NAMES[k]
        out.append(f"| **{k}** | {name} | {colour} | {meaning} |")
    out += [
        "",
        "`note` is the rationale printed in the margin beside the block. It is",
        "LaTeX, so escape `_`, `%`, `&` and `#`, and it must not contain `|`.",
        "",
        "## Caveats",
        "",
        "Two paragraphs separated only by a `%` comment parse as one block,",
        "because a comment swallows its own line ending and the two really are one",
        "paragraph in the PDF.",
    ]
    if cfg.generated:
        gen = ", ".join(f"`{g}`" for g in sorted(cfg.generated))
        out += [
            "",
            f"Generated, so re-running its generator drops the marker and --apply",
            f"has to be re-run: {gen}.",
        ]
    out += ["", "## Tally", "", "| class | blocks |", "|---|---|"]
    for k in "KTASD":
        out.append(f"| {k} | {total.get(k, 0)} |")
    out.append(f"| **total** | **{len(rows)}** |")
    out += ["", "Blocks not marked keep, by file:", "",
            "| file | T | A | S | D |", "|---|---|---|---|---|"]
    for f in sorted(per_file):
        c = per_file[f]
        if not any(c[k] for k in "TASD"):
            continue
        out.append(f"| `{f}` | {c['T'] or ''} | {c['A'] or ''} | "
                   f"{c['S'] or ''} | {c['D'] or ''} |")
    out += ["", "| ID | file | lines | type | class | note |",
            "|---|---|---|---|---|---|"]
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ modes


def cmd_inventory(cfg: Config) -> int:
    old = read_ledger(cfg)
    rows, previews = [], []
    for prefix, path, rel in discover(cfg):
        blocks = parse_text(path.read_text(), prefix)
        if not blocks:
            continue
        previews.append(f"\n## {rel}  ({prefix}, {len(blocks)} blocks)\n")
        for b in blocks:
            cls, note = old.get(b.bid, ("", ""))
            rows.append(f"| {b.bid} | {rel} | {b.start + 1}-{b.end + 1} | "
                        f"{b.kind} | {cls} | {note} |")
            previews.append(f"- **{b.bid}** ({b.kind}, L{b.start+1}-{b.end+1}) {b.preview}")

    cfg.ledger.parent.mkdir(parents=True, exist_ok=True)
    cfg.ledger.write_text(ledger_header(cfg, rows) + "\n".join(rows) + "\n")
    cfg.preview.write_text(
        "<!-- generated by scripts/review/annotate_relevance.py --inventory -->\n"
        "# Block previews\n" + "\n".join(previews) + "\n"
    )
    unclassified = sum(1 for r in rows if r.split("|")[5].strip() not in CLASSES)
    print(f"{len(rows)} blocks -> {cfg.ledger.relative_to(cfg.repo)} "
          f"({unclassified} unclassified)")
    stale = set(old) - {r.split("|")[1].strip() for r in rows}
    if stale:
        print(f"{len(stale)} ledger rows no longer exist and were dropped: "
              f"{', '.join(sorted(stale)[:10])}")
    return 0


def cmd_strip(cfg: Config) -> int:
    n = 0
    for _prefix, path, _rel in discover(cfg):
        before = path.read_text()
        after = strip_text(before)
        if after != before:
            path.write_text(after)
            n += 1
    print(f"stripped markers from {n} files")
    return 0


def cmd_apply(cfg: Config, strict: bool) -> int:
    ledger = read_ledger(cfg)
    if not ledger:
        raise SystemExit("no ledger; run --inventory first")

    total, skipped, touched_generated = 0, [], []
    for prefix, path, rel in discover(cfg):
        original = path.read_text()
        text = strip_text(original)
        lines = text.split("\n")

        edits = []
        for b in parse_text(text, prefix):
            cls, note = ledger.get(b.bid, ("", ""))
            if cls not in CLASSES or not note.strip():
                skipped.append(b.bid)
                continue
            edits.append((b, cls, note.strip()))
        if not edits:
            if text != original:
                path.write_text(text)
            continue

        # from the end, so earlier line indices stay valid
        for b, cls, note in sorted(edits, key=lambda e: e[0].start, reverse=True):
            marker = f"\\rb{{{cls}}}{{{b.bid}}}{{{note}}}"
            if b.kind == "para":
                lines[b.start] = marker + lines[b.start]
                lines[b.end] = lines[b.end] + "\\re"
            else:
                chunk = "\n".join(lines[b.start : b.end + 1])
                m = re.search(r"\\caption\s*(\[[^\]]*\])?\s*\{", chunk)
                open_idx = m.end() - 1
                close_idx = match_brace(chunk, open_idx)
                chunk = (chunk[: open_idx + 1] + marker
                         + chunk[open_idx + 1 : close_idx] + "\\re"
                         + chunk[close_idx:])
                lines[b.start : b.end + 1] = chunk.split("\n")
            total += 1

        applied = "\n".join(lines)
        if strip_text(applied) != text:
            raise SystemExit(f"round-trip check failed on {rel}; nothing written")
        path.write_text(applied)
        if rel in cfg.generated:
            touched_generated.append(rel)

    print(f"applied {total} markers")
    for rel in touched_generated:
        print(f"note: {rel} is generated; re-running its generator drops the marker")
    if skipped:
        print(f"skipped {len(skipped)} unclassified blocks: {', '.join(skipped[:12])}"
              + (" ..." if len(skipped) > 12 else ""))
        if strict:
            return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--inventory", action="store_true")
    g.add_argument("--apply", action="store_true")
    g.add_argument("--strip", action="store_true")
    g.add_argument("--files", action="store_true",
                   help="list the discovered files and their ID prefixes")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--strict", action="store_true",
                    help="--apply exits non-zero if any block is unclassified")
    args = ap.parse_args()

    cfg = Config(args.config)
    if args.files:
        for prefix, _path, rel in discover(cfg):
            print(f"{prefix:6s} {rel}")
        return 0
    if args.inventory:
        return cmd_inventory(cfg)
    if args.strip:
        return cmd_strip(cfg)
    return cmd_apply(cfg, args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
