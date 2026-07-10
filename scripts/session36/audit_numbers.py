"""Session 36 Stage 0 number audit (upstream spec, paths adapted).

Three jobs:
1. Wrap the existing build gate scripts/session35/trace_numbers.py (the
   authoritative hand-typed-numeral detector; never forked here) and report
   its verdict.
2. Dump every numeric decimal literal in the manuscript content files to
   editorial/number_literals.csv with file:line context (raw inventory,
   whitelisted or not, for the editorial ledgers).
3. Cross-check paper/macros_v3.tex against outputs/session33/numbers.json:
   every JSON entry's formatted value must equal the macro body, and every
   \\newcommand in macros_v3.tex must trace back to a JSON entry.

Run from the repo root: .venv/bin/python scripts/session36/audit_numbers.py
Exit code 1 if the tracer fails or any macro/json mismatch is found.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / "paper"
NUMBERS_JSON = REPO / "outputs/session33/numbers.json"
OUT_CSV = REPO / "editorial/number_literals.csv"

CONTENT_GLOBS = [
    "sections/*.tex",
    "sections/tables/*.tex",
    "sections/captions/*.tex",
    "sections/v4/*.tex",
    "protocol_box.tex",
]
NUM = re.compile(r"(?<![\w.])\d+\.\d+(?![\w.])")
NEWCMD = re.compile(r"\\(?:newcommand|providecommand)\{\\(\w+)\}\{([^{}]*)\}")


def fmt_value(value, fmt: str) -> str:
    try:
        return fmt % value
    except TypeError:
        return str(value)


def literal_dump() -> int:
    rows = []
    for glob in CONTENT_GLOBS:
        for tex in sorted(PAPER.glob(glob)):
            if tex.name.startswith("macros"):
                continue
            for i, line in enumerate(tex.read_text(errors="ignore").splitlines(), 1):
                if line.lstrip().startswith("%"):
                    continue
                for m in NUM.finditer(line):
                    rows.append(
                        [str(tex.relative_to(REPO)), i, m.group(0), line.strip()[:120]]
                    )
    OUT_CSV.parent.mkdir(exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "line", "literal", "context"])
        w.writerows(rows)
    print(f"[literals] {len(rows)} decimal literals dumped to {OUT_CSV.relative_to(REPO)}")
    return len(rows)


def macro_crosscheck() -> int:
    numbers = json.loads(NUMBERS_JSON.read_text())["numbers"]
    macros_text = (PAPER / "macros_v3.tex").read_text()
    defined = dict(NEWCMD.findall(macros_text))
    mismatches = []
    missing = []
    json_macros = set()
    for key, entry in numbers.items():
        macro = entry.get("macro")
        if not macro:
            continue
        # emit_macros_v3 convention: <Macro> from value, <Macro>lo/<Macro>hi
        # from ci_lo/ci_hi, all sharing the entry's fmt
        expected = [(macro, entry["value"])]
        for suffix, ci_key in (("lo", "ci_lo"), ("hi", "ci_hi")):
            if ci_key in entry:
                expected.append((macro + suffix, entry[ci_key]))
        for name, value in expected:
            json_macros.add(name)
            if name not in defined:
                missing.append((key, name))
                continue
            expect = fmt_value(value, entry.get("fmt", "%.2f"))
            if defined[name].strip() != expect.strip():
                mismatches.append((key, name, defined[name], expect))
    orphans = [m for m in defined if m not in json_macros]
    print(
        f"[macros] {len(defined)} defined in macros_v3.tex; {len(numbers)} json entries; "
        f"{len(mismatches)} value mismatches; {len(missing)} json macros undefined; "
        f"{len(orphans)} tex macros without a json entry"
    )
    for key, macro, got, expect in mismatches:
        print(f"  MISMATCH {macro} (json {key}): tex='{got}' json='{expect}'")
    for key, macro in missing:
        print(f"  MISSING  {macro} (json {key}) not defined in macros_v3.tex")
    for m in orphans:
        print(f"  ORPHAN   {m} defined in macros_v3.tex, no numbers.json entry")
    return len(mismatches) + len(missing)


def main() -> int:
    tracer = subprocess.run(
        [sys.executable, "-m", "scripts.session35.trace_numbers"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    print(tracer.stdout.strip().splitlines()[-1] if tracer.stdout.strip() else "[tracer] no output")
    literal_dump()
    bad = macro_crosscheck()
    if tracer.returncode != 0:
        print("[audit_numbers] FAIL: tracer found hand-typed numerals")
        return 1
    if bad:
        print(f"[audit_numbers] FAIL: {bad} macro/json problems")
        return 1
    print("[audit_numbers] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
