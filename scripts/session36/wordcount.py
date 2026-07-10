"""Session 36 word-count gate (upstream spec, budgets keyed to actual files).

Counts prose words per manuscript section with texcount (fallback: crude
regex strip) and compares against the JFM rewrite budgets
(editorial/upstream/claude_code_jfm_rewrite_v2.md Stage 4).

The Results budget is checked as one aggregate (5300 = 1300 + 1200 + 1000 +
1800) until the Stage 3 restructure splits section 4 into the four target
subsections; after the restructure, add the per-subsection files to GROUPS.

Run from the repo root: .venv/bin/python scripts/session36/wordcount.py
Exit code 1 if any group exceeds its budget (informational at Stage 0).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SEC = REPO / "paper/sections"
V4 = SEC / "v4"

# group -> (budget, [files])
GROUPS: dict[str, tuple[int, list[Path]]] = {
    "abstract": (250, [SEC / "abstract.tex"]),
    "s1_introduction": (1300, [SEC / "section_1_introduction.tex"]),
    "s2_flow_and_data": (1600, [SEC / "section_2_flow_and_data.tex"]),
    "s3_methods": (
        2600,
        [SEC / "section_3_methods.tex"] + sorted(V4.glob("s3_*.tex")),
    ),
    "s4_results (aggregate 4.1-4.4)": (
        5300,
        [SEC / "section_4_results.tex"] + sorted(V4.glob("s4_*.tex")),
    ),
    "s5_discussion": (1200, [SEC / "section_5_discussion.tex"]),
    "s6_conclusions": (450, [SEC / "section_6_conclusions.tex"]),
}


def count_words(tex: Path) -> int:
    if shutil.which("texcount"):
        out = subprocess.run(
            ["texcount", "-1", "-sum", str(tex)], capture_output=True, text=True
        ).stdout.strip()
        m = re.match(r"(\d+)", out)
        if m:
            return int(m.group(1))
    body = re.sub(r"%.*|\\\w+(\[[^\]]*\])?(\{[^{}]*\})?", " ", tex.read_text(errors="ignore"))
    return len(body.split())


def main() -> int:
    over = 0
    for name, (budget, files) in GROUPS.items():
        total = 0
        detail = []
        for tex in files:
            if not tex.exists():
                continue
            n = count_words(tex)
            total += n
            detail.append(f"{tex.name}={n}")
        flag = ""
        if total > budget:
            flag = f"  OVER budget {budget} by {total - budget}"
            over += 1
        print(f"{name}: {total} words (budget {budget}){flag}")
        print(f"    {'; '.join(detail)}")
    print(f"[wordcount] {over} groups over budget")
    return 1 if over else 0


if __name__ == "__main__":
    sys.exit(main())
