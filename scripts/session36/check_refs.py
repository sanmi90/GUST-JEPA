"""Session 36 reference checker (upstream spec, paths adapted).

Parses paper/main.log (and paper/supplementary.log once it exists) for
undefined references, undefined citations and multiply defined labels.

Run AFTER latexmk: .venv/bin/python scripts/session36/check_refs.py
Exit code 1 on any hit or if the log is missing.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOGS = [REPO / "paper/main.log", REPO / "paper/supplementary.log"]

PATTERNS = [
    (r"Reference `([^']+)' on page .* undefined", "UNDEF-REF"),
    (r"Citation `([^']+)' on page .* undefined", "UNDEF-CITE"),
    (r"Reference `([^']+)' undefined", "UNDEF-REF"),
    (r"Citation `([^']+)' undefined", "UNDEF-CITE"),
    (r"Label `([^']+)' multiply defined", "MULTI-LABEL"),
]


def main() -> int:
    hits = 0
    seen_log = False
    for log in LOGS:
        if not log.exists():
            continue
        seen_log = True
        text = log.read_text(errors="ignore")
        found = set()
        for pat, name in PATTERNS:
            for m in re.finditer(pat, text):
                key = (name, m.group(1))
                if key in found:
                    continue
                found.add(key)
                print(f"{log.name}: {name} {m.group(1)}")
                hits += 1
    if not seen_log:
        print("[check_refs] no log found; run latexmk -pdf main in paper/ first")
        return 1
    print(f"[check_refs] {hits} problems")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
