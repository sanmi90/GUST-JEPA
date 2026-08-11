#!/usr/bin/env python3
"""Carry PDF comments across a rebuild.

latexmk regenerates the PDF from scratch, so any comment written into it by a
reader is destroyed. This keeps the comments in a sidecar keyed by block
identifier and puts them back after the build, so the PDF stays disposable and
the comments do not:

    python scripts/review/carry_comments.py --save    paper/main.pdf
    cd paper && latexmk -pdf main.tex
    python scripts/review/carry_comments.py --restore paper/main.pdf

--save merges: re-saving the same PDF twice adds nothing, and comments already
in the sidecar survive a save from a PDF that no longer carries them. A comment
whose block has disappeared from the manuscript is kept but reported, and is
not restored until its block comes back.

Restored comments are re-anchored on the block's printed identifier, not on
their original coordinates, so they land in the right place after the text
reflows. A comment that has been acted on should be closed with --resolve so it
stops coming back.

    --save PDF          read comments out of PDF into the sidecar
    --restore PDF       write unresolved comments into PDF, in place
    --resolve ID [...]  close every open comment on these blocks
    --reopen ID [...]   the inverse
    --move FROM TO      re-attach comments on one block to another, all of
                        them or, with --match TEXT, only those whose text
                        contains TEXT. A
                        comment drawn on a figure sits above that figure's
                        caption tag, so reading order attaches it to the block
                        before the float; read_annotations flags that case
                        rather than guessing, and this is the correction.
    --list              show the sidecar

Needs only qpdf and pdftotext. No Python dependencies.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_annotations import BARE_ID, ID_TOKEN, page_words, resolve  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parent / "relevance.json"
SIDECAR_DEFAULT = "editorial/pdf_comments.json"

ICON = 20.0          # sticky-note icon size, points

# How an addressed comment is drawn: a different icon, a muted colour and a
# visible prefix, so it reads as closed in any viewer without disappearing.
OPEN_STYLE = ("/Comment", "1 0.82 0.25")
DONE_STYLE = ("/Note", "0.60 0.72 0.62")
DONE_PREFIX = "[done] "
ICON_RISE = 3.0      # lift it into the interline space, so the icon
                     # sits on the tag's own line instead of spilling
                     # over the line below
# The icon sits clear of the text, just left of the block's printed identifier.
# It does not need to touch the tag: every restored comment carries its block in
# /Subj, which read_annotations trusts ahead of any geometry. Geometry is only
# how a HUMAN-placed comment gets resolved, and a human puts it where they mean.
ICON_GAP = 4.0
# Two comments on one block used to get byte-identical rectangles, and poppler
# draws a fixed-size icon at the rect origin, so the second landed exactly on
# the first and was invisible. Stack them down the empty margin instead.
ICON_STACK = ICON + 3.0


def repo_root(start: Path) -> Path:
    for p in start.resolve().parents:
        if (p / ".git").exists():
            return p
    return start.resolve().parents[1]


def load(sidecar: Path) -> dict:
    if sidecar.exists():
        return json.loads(sidecar.read_text())
    return {"comments": []}


def save(sidecar: Path, data: dict) -> None:
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def key(c: dict) -> tuple:
    """Identity of a comment, for merging on save and for skipping on restore.

    The DONE_PREFIX has to come off first. A resolved comment is written into
    the PDF with the prefix but stored in the sidecar without it, so comparing
    the raw contents makes every resolved comment look new to --restore, which
    then appends another copy on each rebuild.
    """
    contents = (c.get("contents") or "").strip()
    if contents.startswith(DONE_PREFIX.strip()):
        contents = contents[len(DONE_PREFIX.strip()):].strip()
    return (c.get("block"), c.get("author", ""), contents)


# ----------------------------------------------------------------- save


def cmd_save(pdf: Path, sidecar: Path) -> int:
    data = load(sidecar)
    have = {key(c) for c in data["comments"]}
    added = 0
    for a in resolve(pdf):
        contents = a["contents"]
        was_done = contents.startswith(DONE_PREFIX)
        if was_done:
            contents = contents[len(DONE_PREFIX):]
        c = {
            "block": a["block"],
            "author": a["author"],
            "date": a["date"],
            "subtype": a["subtype"],
            "contents": contents,
            "quoted": a["quoted"][:200],
            "resolved": was_done,
        }
        if key(c) in have:
            continue
        data["comments"].append(c)
        have.add(key(c))
        added += 1
        print(f"  + {c['block']}: {c['contents'][:70]}")
    save(sidecar, data)
    total = len(data["comments"])
    open_n = sum(1 for c in data["comments"] if not c["resolved"])
    print(f"{added} new; sidecar holds {total} comment(s), {open_n} open "
          f"-> {sidecar}")
    return 0


# --------------------------------------------------------------- restore


def _pdf_string(s: str) -> str:
    """A literal string when ASCII, a UTF-16BE hex string otherwise."""
    if all(ord(ch) < 128 for ch in s):
        out = s.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        return f"({out})"
    return "<FEFF" + s.encode("utf-16-be").hex().upper() + ">"


def _anchor_boxes(pdf: Path) -> dict[str, tuple[int, float, float, float, float]]:
    """block ID -> (page index, anchor box in PDF bottom-left space).

    The margin note prints the identifier bare ("S1-08") while the paragraph
    prints it bracketed ("[S1-08]"). The margin is preferred, so the icon lands
    beside the note in the margin instead of on top of the prose. With
    \reviewnotesfalse there is no margin note and the bracketed tag is used.
    """
    margin: dict[str, tuple] = {}
    inline: dict[str, tuple] = {}
    for page_index, (w, h, words) in enumerate(page_words(pdf)):
        by_line: dict[int, float] = {}
        for word in words:
            by_line[word.key[0]] = max(by_line.get(word.key[0], 0.0), word.x1)
        for word in words:
            # The margin note's identifier line ends with the right-aligned
            # class word, so the widest x on that line is the note column's
            # right edge. Deriving it this way keeps the icon placement
            # independent of \marginparwidth.
            line_right = by_line[word.key[0]]
            box = (page_index, word.x0, h - word.y1, word.x1, h - word.y0,
                   line_right)
            m = ID_TOKEN.search(word.text)
            if m:
                inline.setdefault(m.group(1), box)
                continue
            m = BARE_ID.match(word.text.strip())
            if m:
                margin.setdefault(m.group(1), box)
    return {**inline, **margin}


def cmd_restore(pdf: Path, sidecar: Path) -> int:
    data = load(sidecar)
    todo = list(data["comments"])
    if not todo:
        print("no comments to restore")
        return 0

    # Idempotence. --restore is meant to run straight after a build, but a
    # build that latexmk decides to skip leaves the previous restore in place,
    # and restoring again would duplicate every comment. Anything already in
    # the file is left alone.
    present = {key(a) for a in resolve(pdf)}
    todo = [c for c in todo if key(c) not in present]
    if not todo:
        print(f"{len(present)} comment(s) already in {pdf}; nothing to do")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "work.pdf"
        subprocess.run(["qpdf", "--qdf", "--object-streams=disable",
                        str(pdf), str(work)], check=True)

        anchors = _anchor_boxes(work)
        pages = json.loads(subprocess.run(["qpdf", "--json", str(work)],
                                          capture_output=True, text=True,
                                          check=True).stdout)["pages"]
        page_obj = [int(p["object"].split()[0]) for p in pages]

        text = work.read_bytes().decode("latin-1")
        next_id = max(int(m.group(1))
                      for m in re.finditer(r"^(\d+) 0 obj\n", text, re.M)) + 1

        # Seed the per-block count from what is already in the file, so a
        # restore that adds only the second comment of a block still puts it in
        # the second slot rather than on top of the first.
        placed: dict[str, int] = Counter(a["block"] for a in resolve(pdf))

        new_objects, per_page, restored, orphan = [], {}, 0, []
        for c in todo:
            if c["block"] not in anchors:
                orphan.append(c)
                continue
            pi, x0, y0, _x1, y1, line_right = anchors[c["block"]]
            # Just outside the margin note, on the row of its identifier. The
            # overlay leaves empty page out there, so the icon never covers
            # prose or the note itself, whatever size a viewer decides to draw
            # it at (poppler ignores /Rect and draws a fixed 20pt).
            rx0 = line_right + ICON_GAP
            rx1 = rx0 + ICON
            _ = x0
            n = placed[c["block"]]
            placed[c["block"]] = n + 1
            ry1 = y1 + ICON_RISE - n * ICON_STACK
            ry0 = ry1 - ICON
            icon, colour = DONE_STYLE if c["resolved"] else OPEN_STYLE
            shown = (DONE_PREFIX if c["resolved"] else "") + c["contents"]
            body = (f"<< /Type /Annot /Subtype /Text /Name {icon} "
                    f"/Rect [{rx0:.2f} {ry0:.2f} {rx1:.2f} {ry1:.2f}] "
                    f"/Contents {_pdf_string(shown)} "
                    f"/T {_pdf_string(c['author'] or 'reviewer')} "
                    f"/Subj {_pdf_string(c['block'])} "
                    f"/C [{colour}] /F 4 "
                    f"/P {page_obj[pi]} 0 R >>")
            if c.get("date"):
                body = body[:-3] + f" /M {_pdf_string('D:' + re.sub(chr(58) + '|-| ', '', c['date']) + '00')} >>"
            new_objects.append((next_id, body))
            per_page.setdefault(pi, []).append(next_id)
            next_id += 1
            restored += 1
            _ = y0

        # add the new references to each page's /Annots array
        for pi, ids in per_page.items():
            obj_no = page_obj[pi]
            m = re.search(rf"^{obj_no} 0 obj\n(.*?)\nendobj", text, re.S | re.M)
            if not m:
                raise SystemExit(f"page object {obj_no} not found in the QDF file")
            body = m.group(1)
            refs = "".join(f"\n    {i} 0 R" for i in ids)
            if "/Annots [" in body:
                new_body = body.replace("/Annots [", "/Annots [" + refs, 1)
            else:
                new_body = body.replace("<<", "<<\n  /Annots [" + refs + "\n  ]", 1)
            text = text[:m.start(1)] + new_body + text[m.end(1):]

        appended = "".join(f"{n} 0 obj\n{b}\nendobj\n" for n, b in new_objects)
        # Cut at the cross-reference TABLE, not at the "xref" inside the final
        # "startxref": cutting at the latter leaves the stale table in place, so
        # qpdf reads it, never sees the appended objects, and silently drops the
        # references to them. The table is wrong for everything after the first
        # edit anyway, so it goes and qpdf is asked to reconstruct it.
        cut = text.rfind("\nxref\n")
        if cut < 0:
            raise SystemExit("no cross-reference table in the QDF file")
        m = re.search(r"trailer\s*(<<.*?>>)\s*startxref", text[cut:], re.S)
        trailer = m.group(1) if m else "<< /Root 1 0 R >>"
        text = (text[:cut] + "\n" + appended
                + "trailer " + trailer + "\nstartxref\n0\n%%EOF\n")

        broken = Path(tmp) / "broken.pdf"
        broken.write_bytes(text.encode("latin-1"))
        fixed = Path(tmp) / "fixed.pdf"
        r = subprocess.run(["qpdf", str(broken), str(fixed)],
                           capture_output=True, text=True)
        if not fixed.exists():
            sys.stderr.write(r.stderr)
            raise SystemExit("qpdf could not rebuild the file; PDF left untouched")
        shutil.copyfile(fixed, pdf)

    print(f"restored {restored} comment(s) into {pdf}")
    for c in orphan:
        print(f"  not restored, block {c['block']} is no longer in the document: "
              f"{c['contents'][:60]}")
    return 0


# ------------------------------------------------------------------ misc


def cmd_mark(sidecar: Path, blocks: list[str], resolved: bool) -> int:
    data = load(sidecar)
    n = 0
    for c in data["comments"]:
        if c["block"] in blocks and c["resolved"] != resolved:
            c["resolved"] = resolved
            n += 1
            print(f"  {c['block']}: {'resolved' if resolved else 'reopened'}")
    save(sidecar, data)
    print(f"{n} comment(s) updated")
    return 0


def cmd_move(sidecar: Path, src: str, dst: str, match: str | None = None) -> int:
    """Re-attach comments on one block to another.

    Needed because a comment drawn on a figure sits above that figure's caption
    tag, so reading order attaches it to the block before the float. The reader
    flags the case rather than guessing; this is the correction.

    Without --match every comment on src moves. That is wrong whenever a block
    collected several comments and only some were misattributed, so --match
    takes a case-insensitive substring of the comment text and moves only those.
    """
    data = load(sidecar)
    moved = [c for c in data["comments"] if c["block"] == src
             and (match is None or match.lower() in (c["contents"] or "").lower())]
    if not moved:
        extra = f" matching {match!r}" if match else ""
        print(f"no comment attached to {src}{extra}")
        return 1
    for c in moved:
        c["block"] = dst
        print(f"  {src} -> {dst}: {c['contents'][:60]}")
    save(sidecar, data)
    print(f"{len(moved)} comment(s) moved; re-run --restore to redraw them "
          f"on {dst}")
    return 0


def cmd_list(sidecar: Path) -> int:
    data = load(sidecar)
    if not data["comments"]:
        print("sidecar is empty")
        return 0
    for c in data["comments"]:
        flag = "done" if c["resolved"] else "open"
        who = f" ({c['author']})" if c["author"] else ""
        print(f"[{flag}] {c['block']}{who}: {c['contents']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--save", type=Path, metavar="PDF")
    g.add_argument("--restore", type=Path, metavar="PDF")
    g.add_argument("--resolve", nargs="+", metavar="ID")
    g.add_argument("--reopen", nargs="+", metavar="ID")
    g.add_argument("--move", nargs=2, metavar=("FROM", "TO"))
    ap.add_argument("--match", metavar="TEXT",
                    help="with --move, only comments whose text contains TEXT")
    g.add_argument("--list", action="store_true")
    ap.add_argument("--sidecar", type=Path)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = ap.parse_args()

    root = repo_root(args.config)
    if args.sidecar:
        sidecar = args.sidecar
    else:
        cfg = json.loads(args.config.read_text()) if args.config.exists() else {}
        sidecar = root / cfg.get("comments", SIDECAR_DEFAULT)

    if args.save:
        return cmd_save(args.save, sidecar)
    if args.restore:
        return cmd_restore(args.restore, sidecar)
    if args.resolve:
        return cmd_mark(sidecar, args.resolve, True)
    if args.reopen:
        return cmd_mark(sidecar, args.reopen, False)
    if args.move:
        return cmd_move(sidecar, *args.move, match=args.match)
    return cmd_list(sidecar)


if __name__ == "__main__":
    raise SystemExit(main())
