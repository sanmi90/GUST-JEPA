"""Session 33 provenance harness: merge numbers_parts -> numbers.json (v3 paper).

Same accretion model and validation as scripts/session28/eval_all.py (parts of
the form {"part": name, "numbers": {NAME: RECORD}}), with two changes:
  1. paths: outputs/session33/numbers_parts -> outputs/session33/numbers.json;
  2. a HARD macro-collision check against the v2.1 paper/macros.tex: main.tex
     inputs both macro files during the v3 transition and \\providecommand is
     silently first-wins, so any name collision must fail here, loudly.

Usage:
    python -m scripts.session33.eval_all_v3            # merge + validate + write
    python -m scripts.session33.eval_all_v3 --check    # validate only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PARTS_DIR = REPO / "outputs" / "session33" / "numbers_parts"
OUT_PATH = REPO / "outputs" / "session33" / "numbers.json"
V21_MACROS = REPO / "paper" / "macros.tex"

ALLOWED_KEYS = {
    "macro", "value", "fmt", "ci_lo", "ci_hi", "seed_mean", "seed_sd", "n",
    "split", "endpoint", "probe", "observable", "horizon", "run_tags",
    "source", "note", "unit",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def git_commit() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
    )
    return out.stdout.strip() if out.returncode == 0 else "UNKNOWN"


def v21_macro_names() -> set[str]:
    if not V21_MACROS.exists():
        return set()
    text = V21_MACROS.read_text()
    return set(re.findall(r"\\providecommand\{\\([A-Za-z]+)\}", text))


def validate_record(part: str, name: str, rec: dict) -> list[str]:
    errs = []
    if "value" not in rec:
        errs.append(f"{part}:{name}: missing 'value'")
    unknown = set(rec) - ALLOWED_KEYS
    if unknown:
        errs.append(f"{part}:{name}: unknown keys {sorted(unknown)}")
    macro = rec.get("macro")
    if macro is not None and not macro.isalpha():
        errs.append(f"{part}:{name}: macro '{macro}' must be alphabetic")
    return errs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true", help="Validate only; do not write.")
    args = p.parse_args()

    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    parts = sorted(PARTS_DIR.glob("*.json"))
    numbers: dict[str, dict] = {}
    macros_seen: dict[str, str] = {}
    errors: list[str] = []
    v21 = v21_macro_names()

    for pp in parts:
        try:
            blob = json.loads(pp.read_text())
        except json.JSONDecodeError as e:
            errors.append(f"{pp.name}: invalid JSON ({e})")
            continue
        part_name = blob.get("part", pp.stem)
        for name, rec in blob.get("numbers", {}).items():
            errors.extend(validate_record(part_name, name, rec))
            if name in numbers:
                errors.append(f"duplicate number name '{name}' (part {part_name})")
                continue
            macro = rec.get("macro")
            if macro:
                if macro in macros_seen:
                    errors.append(f"duplicate macro '{macro}' ({macros_seen[macro]} vs {name})")
                elif macro in v21:
                    errors.append(
                        f"macro '{macro}' ({name}) COLLIDES with v2.1 paper/macros.tex; "
                        f"providecommand is silent-first-wins -- rename it"
                    )
                else:
                    macros_seen[macro] = name
            rec.setdefault("source", part_name)
            numbers[name] = rec

    if errors:
        print(f"[eval_all_v3] {len(errors)} validation error(s):")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    provenance = {
        "generated_iso": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "v21_macros_file": str(V21_MACROS.relative_to(REPO)),
        "v21_macros_sha256": sha256_file(V21_MACROS) if V21_MACROS.exists() else None,
        "n_parts": len(parts),
        "parts": [pp.name for pp in parts],
    }
    payload = {"_provenance": provenance, "numbers": numbers}
    print(
        f"[eval_all_v3] {len(numbers)} numbers from {len(parts)} part(s); "
        f"{len(macros_seen)} macro-bound; {len(v21)} v2.1 macros collision-checked"
    )
    if args.check:
        print("[eval_all_v3] check mode: not writing")
        return
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[eval_all_v3] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
