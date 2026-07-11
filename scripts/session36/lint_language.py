"""Session 36 language linter (upstream spec + memo v2 extensions, paths adapted).

Scans the manuscript tex tree for (a) the banned-language list of the JFM
rewrite program (editorial/upstream/editorial_memo_v2.md section 8) and
(b) em-dashes in non-comment lines. Word-boundary regexes are used instead
of the upstream substring match so that e.g. 'prove' does not hit 'improve'
(precision adaptation, noted in MANUSCRIPT_AUDIT.md).

'boundary' is REPORTED but not banned (legitimate uses: boundary test split,
boundary layer); its hits are labelled REVIEW so the editorial pass can vet
each one.

Run from the repo root: .venv/bin/python scripts/session36/lint_language.py
Exit code 1 on any BANNED or EM-DASH hit (REVIEW hits do not fail).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / "paper"

BANNED = [
    r"flagship",
    r"specialist",
    r"load[- ]bearing",
    r"knob[- ]free",
    r"protocol[- ]clean",
    r"refut\w*",
    r"catastrophic\w*",
    r"honest\w*",
    r"\bsettl\w+",
    r"\bbuys\b",
    r"earns its keep",
    r"erratic\w*",
    r"own[- ]stack",
    r"\bas-built\b",
    r"kit strength",
    r"pre[- ]registered",  # allowed exactly once in s3.5 after Stage 4; every hit reported
    r"dimension[- ]invariant",
    r"wall[- ]limited filter",
    r"\bprove[sdn]?\b",
    r"carries particular force",
    r"celebrat\w*",
]
REVIEW = [r"\bboundar\w+"]
EMDASH = ["—", "---"]

SKIP_PARTS = {"build", "upstream"}
SKIP_NAMES = re.compile(r"^macros")


def main() -> int:
    banned_re = [(b, re.compile(b, re.IGNORECASE)) for b in BANNED]
    review_re = [(b, re.compile(b, re.IGNORECASE)) for b in REVIEW]
    hits = 0
    review_hits = 0
    for tex in sorted(PAPER.rglob("*.tex")):
        if set(tex.parts) & SKIP_PARTS or SKIP_NAMES.match(tex.name):
            continue
        for i, line in enumerate(tex.read_text(errors="ignore").splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("%"):
                continue
            rel = tex.relative_to(REPO)
            for label, rx in banned_re:
                if rx.search(line):
                    print(f"{rel}:{i}: BANNED [{label}] {stripped[:100]}")
                    hits += 1
            for label, rx in review_re:
                if rx.search(line):
                    print(f"{rel}:{i}: REVIEW [{label}] {stripped[:100]}")
                    review_hits += 1
            for e in EMDASH:
                if e in line:
                    print(f"{rel}:{i}: EM-DASH {stripped[:100]}")
                    hits += 1
    print(f"[lint_language] {hits} banned/em-dash hits; {review_hits} review-only hits")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
