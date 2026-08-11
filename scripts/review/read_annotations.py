#!/usr/bin/env python3
"""Read PDF comments back out of an annotated build and attach them to block IDs.

The overlay prints a unique identifier at the head of every paragraph and
caption, so a comment anywhere on the page can be traced to the block it
belongs to. Annotate the overlay PDF in any reader that writes standard PDF
annotations, hand the file back, and this turns it into a list keyed by ID:

    python scripts/review/read_annotations.py reviewed.pdf

Sticky notes, highlights, underlines, strikeouts, squiggles, free text, ink and
boxes are all read. Link annotations, which hyperref puts on every cross
reference, are ignored. For a highlight or a strikeout the underlying text is
quoted too, so the comment arrives with what it was pointing at.

    --format md|json   report format (default md)
    --out FILE         write the report here instead of stdout
    --apply-classes    where a comment starts with a bare class letter
                       (K, T, A, S or D, optionally followed by ':'), set that
                       class on the block's ledger row. Nothing else in the
                       ledger is touched, and every change is printed.

Needs only qpdf and pdftotext, both of which the build already relies on. No
Python dependencies beyond the standard library.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parent / "relevance.json"

# hyperref puts a /Link on every cross reference; /Popup is the window of
# another annotation, never a comment in its own right.
IGNORED = {"/Link", "/Popup"}
# Annotations that carry text under them, via /QuadPoints.
TEXT_MARKUP = {"/Highlight", "/Underline", "/StrikeOut", "/Squiggly"}
ID_TOKEN = re.compile(r"\[([A-Z][A-Z0-9]*-C?\d+)\]")
BARE_ID = re.compile(r"^([A-Z][A-Z0-9]*-C?\d+)$")
# Caption identifiers carry a C. A caption is printed BELOW its float, which is
# what makes a comment written on the artwork ambiguous; see resolve().
CAPTION_ID = re.compile(r"^[A-Z][A-Z0-9]*-C\d+$")
CLASSES = {"K", "T", "A", "S", "D"}
LEAD_CLASS = re.compile(r"^\s*([KTASD])\b\s*[:.\-]?\s*(.*)$", re.S)


# ----------------------------------------------------------------- geometry


class Word:
    __slots__ = ("text", "x0", "y0", "x1", "y1")

    def __init__(self, text, x0, y0, x1, y1):
        self.text, self.x0, self.y0, self.x1, self.y1 = text, x0, y0, x1, y1

    @property
    def key(self):
        """Reading order within a page: line first, then left to right."""
        return (round(self.y0 / 4), self.x0)


# Glyphs with no Unicode mapping come out of pdftotext as raw control bytes,
# which XML 1.0 forbids; a maths-heavy page is enough to break the parse.
BAD_XML_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def page_words(pdf: Path) -> list[tuple[float, float, list[Word]]]:
    """(width, height, words) per page, in pdftotext's top-left coordinates."""
    xml = subprocess.run(
        ["pdftotext", "-bbox-layout", str(pdf), "-"],
        capture_output=True, text=True, check=True).stdout
    root = ET.fromstring(BAD_XML_CHARS.sub("", xml))
    ns = {"x": "http://www.w3.org/1999/xhtml"}
    pages = []
    for page in root.iter("{http://www.w3.org/1999/xhtml}page"):
        words = [
            Word(w.text or "", float(w.get("xMin")), float(w.get("yMin")),
                 float(w.get("xMax")), float(w.get("yMax")))
            for w in page.iter("{http://www.w3.org/1999/xhtml}word")
        ]
        pages.append((float(page.get("width")), float(page.get("height")), words))
    if not pages:  # older poppler without the xhtml namespace
        for page in root.iter("page"):
            words = [Word(w.text or "", float(w.get("xMin")), float(w.get("yMin")),
                          float(w.get("xMax")), float(w.get("yMax")))
                     for w in page.iter("word")]
            pages.append((float(page.get("width")), float(page.get("height")), words))
    _ = ns
    return pages


def overlaps(a: tuple, b: tuple, slack: float = 1.0) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return (ax0 < bx1 + slack and bx0 < ax1 + slack
            and ay0 < by1 + slack and by0 < ay1 + slack)


# --------------------------------------------------------------- annotations


def pdf_annotations(pdf: Path) -> list[dict]:
    """Every non-ignored annotation, with its page index and rectangle."""
    raw = subprocess.run(["qpdf", "--json", str(pdf)],
                         capture_output=True, text=True, check=True).stdout
    doc = json.loads(raw)
    objs = doc["objects"]

    def deref(ref):
        return objs.get(ref) if isinstance(ref, str) else ref

    out = []
    for page_index, page in enumerate(doc["pages"]):
        pobj = deref(page["object"]) or {}
        for ref in pobj.get("/Annots", []) or []:
            a = deref(ref)
            if not isinstance(a, dict):
                continue
            subtype = a.get("/Subtype")
            if subtype in IGNORED:
                continue
            contents = _pdf_string(a.get("/Contents"))
            quads = [float(q) for q in a.get("/QuadPoints", [])] or None
            rect = [float(v) for v in a.get("/Rect", [0, 0, 0, 0])]
            if not contents and subtype not in TEXT_MARKUP:
                continue  # an empty drawing carries no message
            out.append({
                "page": page_index,
                "subtype": (subtype or "?").lstrip("/"),
                "contents": contents,
                "author": _pdf_string(a.get("/T")),
                "subj": _pdf_string(a.get("/Subj")),
                "date": _pdf_date(_pdf_string(a.get("/M"))),
                "rect": rect,
                "quads": quads,
            })
    return out


def _pdf_string(v) -> str:
    """qpdf's JSON v1 hands back a decoded string; v2 prefixes u: (unicode) or
    b: (hex bytes). Nothing is stripped beyond that: a comment may legitimately
    begin and end with a bracket."""
    if not isinstance(v, str):
        return ""
    if v.startswith("u:"):
        return v[2:]
    if v.startswith("b:"):
        try:
            return bytes.fromhex(v[2:]).decode("utf-8", "replace")
        except ValueError:
            return v[2:]
    return v


def _pdf_date(v: str) -> str:
    m = re.match(r"D:(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})", v or "")
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)} {m.group(4)}:{m.group(5)}" if m else ""


# --------------------------------------------------------------- resolution


def resolve(pdf: Path) -> list[dict]:
    pages = page_words(pdf)
    annots = pdf_annotations(pdf)

    # Every bracketed ID on every page, in reading order, so an annotation can
    # be traced to the block whose tag most recently precedes it.
    anchors: list[list[tuple[tuple, str]]] = []
    for _w, _h, words in pages:
        found = []
        for word in words:
            m = ID_TOKEN.search(word.text)
            if m:
                found.append((word.key, m.group(1)))
        anchors.append(sorted(found))

    last_of_page = {}
    running = None
    for i, found in enumerate(anchors):
        last_of_page[i] = running
        if found:
            running = found[-1][1]

    for a in annots:
        pw, ph, words = pages[a["page"]] if a["page"] < len(pages) else (0, 0, [])

        # PDF space is bottom-left origin; pdftotext is top-left.
        def flip(x0, y0, x1, y1):
            return (min(x0, x1), ph - max(y0, y1), max(x0, x1), ph - min(y0, y1))

        boxes = []
        if a["quads"]:
            q = a["quads"]
            for i in range(0, len(q) - 7, 8):
                xs, ys = q[i:i + 8:2], q[i + 1:i + 8:2]
                boxes.append(flip(min(xs), min(ys), max(xs), max(ys)))
        else:
            boxes.append(flip(*a["rect"]))

        covered = [w for w in words if any(overlaps((w.x0, w.y0, w.x1, w.y1), b)
                                           for b in boxes)]
        covered.sort(key=lambda w: w.key)
        a["quoted"] = " ".join(w.text for w in covered).strip()

        anchor_key = min((b[1], b[0]) for b in boxes)          # (top, left)
        anchor_key = (round(anchor_key[0] / 4), anchor_key[1])
        # An ID inside the covered text wins: the comment is on the tag itself.
        # A comment written by carry_comments stamps its block into /Subj, so
        # its geometry does not have to be interpreted at all.
        stamped = BARE_ID.match((a.get("subj") or "").strip())
        inline = [ID_TOKEN.search(w.text).group(1) for w in covered
                  if ID_TOKEN.search(w.text)]
        if stamped:
            a["block"] = stamped.group(1)
        elif inline:
            a["block"] = inline[0]
        else:
            page_anchors = anchors[a["page"]]
            prior = [bid for key, bid in page_anchors if key <= anchor_key]
            later = [bid for key, bid in page_anchors if key > anchor_key]
            a["block"] = prior[-1] if prior else last_of_page.get(a["page"])
            # A float prints its artwork ABOVE its caption, so a comment written
            # on a figure sits after the preceding block's tag and before the
            # caption's own, and reading order hands it to whatever text
            # precedes the figure. Geometry cannot settle it: a comment on the
            # last line of that preceding paragraph lands in the same gap, and
            # the artwork of a vector figure carries real text of its own, so
            # "is there text in between" does not separate the two cases. Say so
            # instead of guessing, and let --move fix the ones that guess wrong.
            if later and CAPTION_ID.match(later[0]) and a["block"] != later[0]:
                a["ambiguous"] = later[0]
        a["page_label"] = a["page"] + 1
        _ = pw
    return annots


# ------------------------------------------------------------------ output


def report_md(annots: list[dict], pdf: Path) -> str:
    by_block: dict[str, list[dict]] = {}
    for a in annots:
        by_block.setdefault(a["block"] or "(unattached)", []).append(a)

    out = [f"# Comments read from `{pdf.name}`", "",
           f"{len(annots)} annotation(s) over {len(by_block)} block(s). "
           "Attached by the nearest preceding block identifier printed in the PDF. "
           "A comment drawn on a figure precedes that figure's caption tag, so it "
           "attaches to the block above; those are flagged.",
           ""]
    for block in sorted(by_block, key=lambda b: (b == "(unattached)", b)):
        out.append(f"## {block}")
        for a in by_block[block]:
            who = f" ({a['author']})" if a["author"] else ""
            when = f", {a['date']}" if a["date"] else ""
            out.append(f"- **{a['subtype']}**, p.{a['page_label']}{who}{when}")
            if a.get("ambiguous"):
                out.append(f"  - ⚠ sits above the caption of **{a['ambiguous']}**; "
                           f"if it is about that float, "
                           f"`carry_comments.py --move {block} {a['ambiguous']}`")
            if a["quoted"]:
                q = a["quoted"]
                out.append(f"  - on: “{q[:300]}{'...' if len(q) > 300 else ''}”")
            if a["contents"]:
                for line in a["contents"].splitlines():
                    if line.strip():
                        out.append(f"  - > {line.strip()}")
        out.append("")
    return "\n".join(out)


def apply_classes(annots: list[dict], ledger: Path) -> int:
    rows = ledger.read_text().splitlines()
    row_re = re.compile(r"^\| (\S+) \| ([^|]*)\| ([^|]*)\| ([^|]*)\|([^|]*)\|([^|]*)\|$")
    wanted: dict[str, str] = {}
    for a in annots:
        if not a["block"]:
            continue
        m = LEAD_CLASS.match(a["contents"] or "")
        if m and m.group(1) in CLASSES:
            wanted[a["block"]] = m.group(1)

    changed = 0
    for i, line in enumerate(rows):
        m = row_re.match(line)
        if not m or m.group(1) not in wanted:
            continue
        new, old = wanted[m.group(1)], m.group(5).strip()
        if new == old:
            continue
        rows[i] = (f"| {m.group(1)} | {m.group(2)}| {m.group(3)}| {m.group(4)}| "
                   f"{new} | {m.group(6).strip()} |")
        print(f"  {m.group(1)}: {old} -> {new}")
        changed += 1
    if changed:
        ledger.write_text("\n".join(rows) + "\n")
    unknown = set(wanted) - {row_re.match(r).group(1) for r in rows if row_re.match(r)}
    for bid in sorted(unknown):
        print(f"  warning: {bid} is not a ledger row", file=sys.stderr)
    print(f"{changed} class(es) changed in {ledger}")
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--format", choices=["md", "json"], default="md")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--apply-classes", action="store_true")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = ap.parse_args()

    if not args.pdf.exists():
        raise SystemExit(f"{args.pdf} does not exist")

    annots = resolve(args.pdf)
    if not annots:
        print("no comments found in this PDF", file=sys.stderr)
        return 0

    text = (json.dumps(annots, indent=2, ensure_ascii=False)
            if args.format == "json" else report_md(annots, args.pdf))
    if args.out:
        args.out.write_text(text + "\n")
        print(f"{len(annots)} comment(s) -> {args.out}")
    else:
        print(text)

    if args.apply_classes:
        cfg = json.loads(args.config.read_text())
        repo = next((p for p in args.config.resolve().parents if (p / ".git").exists()),
                    args.config.resolve().parents[1])
        apply_classes(annots, repo / cfg.get("ledger", "RELEVANCE_MAP.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
