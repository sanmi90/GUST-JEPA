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

### Readers that work

| platform | reader | notes |
|---|---|---|
| Ubuntu | **Okular** | Best annotation set. After annotating press Ctrl+S; older versions keep notes in `~/.local/share/okular/docdata/` unless you save into the file. |
| Ubuntu | **Evince / Papers** | Already installed on GNOME. Sticky notes and highlights only. Save with "Save a Copy". |
| Ubuntu | **Firefox** | Zero install: open the PDF, use the highlight, text and draw tools, then Save. |
| Android | **Xodo** | The reliable free option. Notes, highlight, strikeout, free text. |
| Android | **Adobe Acrobat Reader** | Notes, highlight, strikeout. Save, do not "share a flattened copy". |

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
