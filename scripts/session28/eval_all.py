"""Session 28 provenance harness: the ONE evaluation entry point (master plan A5).

Every paper-bound quantity lives in outputs/session28/numbers.json, keyed by a
stable name, with full provenance: value, intervals, seed statistics, split,
endpoint, probe, run tags, the eval-protocol hash, and the git commit. The
manuscript only ever sees these numbers through paper/macros.tex (rendered by
scripts/session28/emit_macros.py), so a number that is not in numbers.json
cannot appear in the paper (referee Track 0; closes B3/B4 structurally).

Accretion model: each track script (closure_matrix.py, the Phase-C tracks,
undisturbed_validation.py, ...) writes a PART file into
outputs/session28/numbers_parts/<part>.json of the form

    {"part": "<name>", "numbers": {NAME: RECORD, ...}}

where RECORD carries at least {"value": float} and optionally
{"macro": str, "fmt": str, "ci_lo": float, "ci_hi": float, "seed_mean": float,
 "seed_sd": float, "n": int, "split": str, "endpoint": str, "probe": str,
 "observable": str, "horizon": int, "run_tags": [str], "source": str}.

This script validates the parts, refuses duplicate names and duplicate macros,
stamps provenance, and writes the merged numbers.json.

Usage:
    python scripts/session28/eval_all.py            # merge + validate + write
    python scripts/session28/eval_all.py --check    # validate only, no write
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PARTS_DIR = REPO / "outputs" / "session28" / "numbers_parts"
OUT_PATH = REPO / "outputs" / "session28" / "numbers.json"
PROTOCOL = REPO / "configs" / "eval_protocol_v2p1.yaml"
MANIFEST_RUNS = REPO / "scripts" / "session28" / "manifest_runs.yaml"

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


def validate_record(part: str, name: str, rec: dict) -> list[str]:
    errs = []
    if "value" not in rec:
        errs.append(f"{part}:{name}: missing 'value'")
    unknown = set(rec) - ALLOWED_KEYS
    if unknown:
        errs.append(f"{part}:{name}: unknown keys {sorted(unknown)}")
    macro = rec.get("macro")
    if macro is not None and not macro.isalpha():
        errs.append(f"{part}:{name}: macro '{macro}' must be alphabetic (LaTeX command name)")
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
                    errors.append(
                        f"duplicate macro '{macro}' ({macros_seen[macro]} vs {name})"
                    )
                else:
                    macros_seen[macro] = name
            rec.setdefault("source", part_name)
            numbers[name] = rec

    if errors:
        print(f"[eval_all] {len(errors)} validation error(s):")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    provenance = {
        "generated_iso": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "protocol_file": str(PROTOCOL.relative_to(REPO)),
        "protocol_sha256": sha256_file(PROTOCOL) if PROTOCOL.exists() else None,
        "manifest_runs_sha256": (
            sha256_file(MANIFEST_RUNS) if MANIFEST_RUNS.exists() else None
        ),
        "n_parts": len(parts),
        "parts": [pp.name for pp in parts],
    }
    # The protocol-freeze commit, rendered as \CommitHash in the manuscript.
    numbers.setdefault(
        "protocol_commit",
        {"macro": "CommitHash", "value": provenance["git_commit"][:12], "fmt": "%s",
         "source": "eval_all.py"},
    )

    payload = {"_provenance": provenance, "numbers": numbers}
    print(f"[eval_all] {len(numbers)} numbers from {len(parts)} part(s); "
          f"{len(macros_seen) + 1} macro-bound")
    if args.check:
        print("[eval_all] check mode: not writing")
        return
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[eval_all] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
