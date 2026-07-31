#!/usr/bin/env bash
#
# Rebuild a document without losing the comments written into its PDF.
#
#   scripts/review/rebuild.sh                       # paper/main.tex
#   scripts/review/rebuild.sh paper/supplementary.tex
#   scripts/review/rebuild.sh paper/main.tex -g     # extra latexmk arguments
#
# Comments live in the sidecar named by scripts/review/relevance.json, keyed by
# block identifier, so they are re-anchored after the text reflows rather than
# pinned to coordinates. Saving before the build means a comment added since the
# last cycle is picked up; restoring after it means the fresh PDF carries every
# open comment plus the closed ones, drawn in the muted "done" style.
#
# The build runs before the restore and the script stops on failure, so a broken
# build never produces a PDF with comments stamped into it.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
carry="$here/carry_comments.py"

tex="${1:-paper/main.tex}"
[ $# -gt 0 ] && shift
[ "${tex##*.}" = "tex" ] || tex="${tex%.pdf}.tex"

if [ ! -f "$tex" ]; then
  echo "no such document: $tex" >&2
  exit 1
fi
pdf="${tex%.tex}.pdf"
dir="$(dirname "$tex")"
base="$(basename "$tex")"

python="${PYTHON:-python}"

if [ -f "$pdf" ]; then
  echo "==> saving comments from $pdf"
  "$python" "$carry" --save "$pdf"
else
  echo "==> no existing $pdf, nothing to save"
fi

echo "==> building $tex"
( cd "$dir" && latexmk -pdf -interaction=nonstopmode "$@" "$base" )

echo "==> restoring comments into $pdf"
"$python" "$carry" --restore "$pdf"
