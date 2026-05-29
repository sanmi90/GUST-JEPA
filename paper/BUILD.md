# Manuscript build instructions

The manuscript is authored as Markdown in `sections/*.md` and assembled into
LaTeX via a small converter, then compiled with the standard `pdflatex` +
`bibtex` cycle.

## One-shot build

```bash
cd $REPO  # /home/carlos/GUST-JEPA
source .venv/bin/activate

# 1. Markdown -> LaTeX
python scripts/_oneoff_md_to_tex.py

# 2. LaTeX -> PDF (four passes: pdflatex, bibtex, pdflatex, pdflatex)
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex   main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

The output is `paper/main.pdf`.

## Source files

- `paper/main.tex`               LaTeX document skeleton, preamble, section
                                 includes, bibliography call.
- `paper/refs.bib`               Bibliography (stub entries; replace with full
                                 records before camera-ready).
- `paper/sections/abstract.md`   Section sources in Markdown.
- `paper/sections/section_N_*.md`
- `paper/sections/tables/*.tex`  Already-LaTeX tables included via `\input`.
- `paper/sections/figures/`      TikZ figures (`tikz/`) and matplotlib
                                 outputs (`results/`).
- `paper/HEADLINE_NUMBERS.md`    Canonical reference for every numerical
                                 claim in the manuscript. Not included in the
                                 PDF; used to check consistency across
                                 sections.

## Conventions

- Source Markdown is LaTeX-friendly: math in `$...$`, `\cite{key}`,
  `\input{...}`, `\begin{...}` are passed through verbatim.
- Backtick `code` becomes `\texttt{code}`.
- `**bold**` and `*italics*` are converted.
- Pipe tables in Markdown become `booktabs` tabular blocks.
- The converter escapes underscores in plain text but leaves them alone in
  math mode and inside the arguments of `\texttt`, `\cite`, `\ref`, `\label`,
  `\input`.
- No em-dashes anywhere; per the project style.

## What still needs human attention

- The 4 overfull horizontal boxes are typesetting nuisances, not content
  errors. They surface as warnings; the PDF still builds. Tighten the
  offending paragraphs by hand if you want a clean log.
- The two hyperref Warning lines come from math in the `\title{}`, which
  hyperref refuses to encode into the PDF bookmark title. Either strip the
  math from the title or wrap it in `\texorpdfstring{...}{...}` if you want
  the warnings gone.
- `refs.bib` entries are stub bibliographic records; replace each entry with
  the canonical citation before submission.
- Sections 1 (introduction) and 2 (related work) carry over from the v1
  partition draft and reference some v1 numbers. A numbers-pass against
  `HEADLINE_NUMBERS.md` is wanted but not yet done.
