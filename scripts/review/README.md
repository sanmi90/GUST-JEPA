# Relevance overlay

A reusable way to answer, on the page itself, "what has to stay in this paper
and what can go". Every prose paragraph and every figure or table caption is
coloured by its value to the manuscript, carries a unique identifier, and gets
a margin note saying why.

| | class | colour | meaning |
|---|---|---|---|
| **K** | keep | green | stays in the body as it is |
| **T** | trim | orange | could be removed; the paper survives without it |
| **A** | annex | purple | belongs in an appendix, not the body |
| **S** | supp | brown | belongs in the supplementary material |
| **D** | delete | red | can be deleted outright, nothing is lost |

It is an **overlay, not an edit**. Markers go in inline, so no line is added and
the ledger's line numbers stay valid; `--strip` restores the sources byte for
byte; `--apply` round-trips its own output through `--strip` and refuses to
write if anything differs. With the master switch off, the PDF is the clean
manuscript at its original page size.

## Two pieces

- `paper/reviewmarks.tex` renders the overlay. `\input` it from the preamble.
- `scripts/review/annotate_relevance.py` puts the markers in and takes them out,
  driven by `scripts/review/relevance.json` and the ledger.

## Everyday use

```bash
python scripts/review/annotate_relevance.py --files      # what will be annotated
python scripts/review/annotate_relevance.py --inventory  # (re)build the ledger skeleton
#   ... fill in the class and note columns of editorial/RELEVANCE_MAP.md ...
python scripts/review/annotate_relevance.py --apply      # insert the markers
cd paper && latexmk -pdf main.tex
```

`--apply --strict` exits non-zero if any block is still unclassified, which is
what to use in a check script. `--strip` removes everything.

To change one block's verdict, edit its row in the ledger and re-run `--apply`;
the step strips before it inserts, so it is idempotent and safe to repeat.

After editing the manuscript, re-run `--inventory`. It preserves the class and
rationale of every ID that still exists, refreshes the line numbers, and reports
any ledger row whose block has disappeared.

## Switches

At the top of `paper/reviewmarks.tex`:

| switch | effect |
|---|---|
| `\reviewmarksfalse` | master off: clean manuscript, original page size |
| `\reviewcolourfalse` | keep the notes and tags, drop the text colouring |
| `\reviewnotesfalse` | keep the colouring, drop the margin notes |
| `\reviewidsfalse` | keep everything, drop the inline `[ID]` tags |

## Commenting on the PDF and getting the comments back

Annotate the overlay build in any reader that writes **standard PDF
annotations**, then:

```bash
python scripts/review/read_annotations.py reviewed.pdf
python scripts/review/read_annotations.py reviewed.pdf --out editorial/COMMENTS.md
python scripts/review/read_annotations.py reviewed.pdf --apply-classes
```

The report is keyed by block identifier, not by page or coordinate. Every
paragraph and caption prints its `[ID]` in the PDF, so each comment is attached
to the block whose tag most recently precedes it, and a highlight or strikeout
arrives with the text it was drawn over. Comments therefore survive a rebuild:
the coordinates move, the identifiers do not.

### Comments drawn on a figure

A float prints its artwork **above** its caption, so a comment written on a
figure sits after the preceding block's tag and before the caption's own, and
reading order hands it to whatever text precedes the figure. Geometry cannot
settle this: a comment on the last line of that preceding paragraph lands in the
same gap, and a vector figure carries real text of its own, so "is there text in
between" separates nothing.

The reader does not guess. It attaches by reading order as usual and flags the
case:

```
## S3-06
- **Text**, p.13 (asolera)
  - ⚠ sits above the caption of **S31-C1**; if it is about that float,
    `carry_comments.py --move S3-06 S31-C1`
```

`--move FROM TO` re-attaches every comment on one block to another; run
`--restore` afterwards to redraw them beside the new block's tag. Two ways to
avoid needing it: write the comment just **below** the caption tag of the figure
you mean, or on the caption text itself, which wins outright because the ID is
then inside the annotated region.

This applies only to comments you write. Restored ones record their block in
`/Subj` and never depend on geometry.

`--apply-classes` reads a comment that *begins* with a bare class letter and
sets that class on the block's ledger row. `D`, `T: too long`, `A - move this`
all work; `Keep this` does not, deliberately, since it does not start with a
standalone letter. Nothing else in the ledger is touched and every change is
printed, so `git diff` is the review.

Sticky notes, highlights, underlines, strikeouts, squiggles, free text, ink and
boxes are all read. `/Link` annotations are ignored, which matters because
`hyperref` puts one on every cross-reference (454 of them in this document).

Needs only `qpdf` and `pdftotext`, both already required by the build. No
Python dependencies.

### Keeping comments across a rebuild

`latexmk` regenerates the PDF from scratch, so anything a reader wrote into it
is destroyed. Rebuild through the wrapper instead and the comments survive:

```bash
scripts/review/rebuild.sh                        # paper/main.tex
scripts/review/rebuild.sh paper/supplementary.tex
scripts/review/rebuild.sh paper/main.tex -g      # extra latexmk arguments
```

It saves the comments, builds, and puts them back. The build runs before the
restore and the script stops on failure, so a broken build never produces a PDF
with comments stamped into it. The same three steps by hand:

```bash
python scripts/review/carry_comments.py --save    paper/main.pdf
cd paper && latexmk -pdf main.tex
python scripts/review/carry_comments.py --restore paper/main.pdf
```

Comments live in the sidecar named by `relevance.json` (`comments`), keyed by
block identifier, so a restored comment is re-anchored on its block after the
text reflows rather than pinned to a coordinate that has moved. `--save` merges,
so nothing is duplicated and nothing already recorded is lost. `--restore` is
idempotent, which matters because latexmk sometimes decides a rebuild is
unnecessary and leaves the previous restore in place.

Closing a comment you have acted on:

```bash
python scripts/review/carry_comments.py --resolve S1-08
python scripts/review/carry_comments.py --reopen  S1-08
python scripts/review/carry_comments.py --list
```

A closed comment is not deleted. It comes back in a muted green with a
different icon and its text prefixed `[done]`, so the record of what was raised
and dealt with stays in the document. Open comments stay yellow.

Restored comments are placed just outside the margin note, on the row of its
identifier, where the overlay leaves the page empty. That is deliberate:
poppler ignores the annotation rectangle and draws a fixed-size icon, so
anything placed against the prose covers a few characters. Placement carries no
meaning anyway, since each restored comment records its block in `/Subj`, which
the reader trusts ahead of any geometry.

### Handing the mechanics to a subagent

None of the commands above need the model that is rewriting the manuscript. They
need a shell and the ability to copy text out of a PDF, and their output is
bulky: a full read is every comment ever written, most of them already closed.
Running them in the main context spends it on plumbing.

`.claude/agents/pdf-notes.md` defines a Sonnet subagent that owns exactly this
plumbing. Delegate to it with `subagent_type: "pdf-notes"`:

```
read the annotations on paper/main.pdf and report the open ones verbatim
resolve S34-01, S34-02 and S34-C1, then restore into paper/main.pdf
move the comment on S3-06 that mentions "separate (a) and (b)" to S3-C1
```

It has `Bash` and `Read` and nothing else, so it cannot edit a `.tex` file even
if asked. Three rules keep the division honest:

- it reproduces `/Contents` verbatim, never paraphrased, because the caller acts
  on the exact words and a summarised comment is a lost comment;
- it never decides what a comment means or whether it was addressed, and closes
  only the identifiers it is explicitly given;
- it passes on the reader's `⚠` ambiguity flags rather than resolving them,
  since choosing between a paragraph and the figure below it is a judgement
  about content.

The split is worth keeping even when the mechanics look trivial. The failure
mode this workflow has actually hit is a comment silently lost or attached to
the wrong block, and both come from the plumbing, not from the writing.

### Readers that work

| platform | reader | notes |
|---|---|---|
| Ubuntu | **Evince / Papers** | What this workflow is used with, and verified end to end. Right-click, "Add text annotation", type, then Ctrl+S. Sticky notes and highlights only, which is all that is needed. |
| Ubuntu | **Okular** | A wider annotation set (strikeout, squiggle, free text). Press Ctrl+S after annotating; older versions keep notes in `~/.local/share/okular/docdata/` rather than in the file. |
| Ubuntu | **Firefox** | Zero install: open the PDF, use the highlight, text and draw tools, then Save. |
| Android | **MuPDF viewer** (F-Droid, AGPL) | The open-source pick. Built on the MuPDF engine, which writes real PDF annotation objects into the file rather than a sidecar database. Highlight, underline, strikeout and ink; check whether your build also offers a sticky note. Verify with the round-trip test below before a full pass. |
| Android | **KOReader** (F-Droid, AGPL) | Also MuPDF-backed and more capable, but it keeps annotations in a `.sdr` sidecar folder by default, which this reader cannot see. You must turn on writing them back into the PDF, or nothing comes through. |
| Android | **Xodo** | Proprietary, but the most complete free annotation set if the open-source options fall short. |
| Android | **Adobe Acrobat Reader** | Proprietary. Notes, highlight, strikeout. Save, do not "share a flattened copy". |

A highlight carrying a note is enough for this workflow: the reader takes the
comment from the annotation's `/Contents` and the quoted text from what the
highlight covers, so an app with no sticky-note tool is still usable.

Annotating on a phone does not mean editing the working PDF. `--save` takes any
path, and comments merge into the sidecar by block identifier, so the usual
route is to copy the built PDF to the device, annotate it there, copy it back
anywhere at all, and run

```bash
python scripts/review/carry_comments.py --save ~/Downloads/main.pdf
scripts/review/rebuild.sh
```

The comments land on their blocks in the freshly built PDF. The annotated copy
is then disposable.

Evince and Okular both render through poppler, so the open and closed styles
show as intended there: a yellow speech bubble against a muted green note. A
reader that draws its own icon regardless still shows the `[done]` prefix.

Evince keeps the file open while you work on it. After `rebuild.sh` replaces the
PDF underneath, press Ctrl+R (or reopen) to see the new one; if Evince asks to
save on exit after that, say no, or it will write the stale in-memory copy back
over the rebuilt file.

**Avoid Xournal++**: its PDF export flattens annotations into the page content,
so they cannot be read back. The same is true of anything offering to "print to
PDF" or "flatten" on the way out.

If a reader turns out to keep its comments in a sidecar database rather than in
the file, `read_annotations.py` will simply report none, which is the test.
Round-trip one comment before doing a full pass.

Copy `paper/reviewmarks.tex` and the `scripts/review/` directory, then:

1. Point `roots` in `relevance.json` at your top-level `.tex` file (or files;
   a list is allowed, so a paper and its supplement can share one ledger).
2. Set `ledger` and `preview` to wherever the ledger should live.
3. Add preamble-only includes to `exclude` (macro files, notation files).
4. `\input` `reviewmarks.tex` from the preamble and put `\reviewlegend` at the
   head of the body.

Everything else follows: the file set comes from the `\input`/`\include` graph
of the roots, in source order, so nothing is listed by hand and a file no
document reaches is never annotated. ID prefixes are the initials of each file
stem, deduplicated; pin a prettier or historically stable one by adding it to
`prefixes`.

The style file is class-agnostic. It widens the page relative to whatever the
class set, so the text block keeps its position and measure. Retune by
`\providecommand`-ing any of `\reviewextrawidth`, `\reviewnotewidth`,
`\reviewnotegutter` before the `\input`.

## Things worth knowing

- **`\re` is not `\ref`.** The stripper uses a negative lookahead. An early
  version used a plain string replace and turned every `\ref{}` in the
  manuscript into `f{}` across thirty files. If you touch `strip_text`, keep
  the round-trip assertion in `--apply`.
- **Colour needs two nudges.** `\rb` runs in vertical mode at a paragraph head,
  where `\color` does not reach the paragraph that follows, and `\marginnote`
  leaves the surrounding text back at black. `\leavevmode` and a second
  `\color` after the note fix both. Captions never had the problem because they
  are already in horizontal mode.
- **Markers inside `\caption{}` must be robust.** `\rb` and `\re` are declared
  with `\DeclareRobustCommand` because the `caption` package double-scans the
  argument and writes it to the `.lof`; a fragile conditional there breaks brace
  balance.
- **A `%` comment does not end a paragraph.** It swallows its own line ending,
  so two source "paragraphs" separated only by a comment are one paragraph in
  the PDF, and the parser keeps them as one block.
- **Generated files.** List them under `generated` in the config. `--apply`
  still marks them and then reminds you that re-running their generator drops
  the marker.
- **Number gates.** If the project lints for hand-typed numerals, teach the
  linter to strip `\rb{}{}{}` first; section references inside a rationale are
  commentary, not manuscript numbers. Here that is one rule in
  `scripts/session35/trace_numbers.py`.
