"""DNS metadata submission gate (SESSION29.8 S0).

A JFM submission cannot contain pending DNS-resolution rows. This checks
paper/dns_metadata.yaml:

  --mode draft       reports the pending fields and exits 0 (session may continue).
  --mode submission  exits non-zero if ANY required field is null / empty /
                     "[PENDING]" / "TBD", so the submission build fails until the
                     simulation collaborators supply every value.

The script does not fabricate values; it only verifies the YAML is complete.

Usage:
    python scripts/session29/check_dns_metadata.py --mode draft
    python scripts/session29/check_dns_metadata.py --mode submission
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
YAML = REPO / "paper" / "dns_metadata.yaml"
OUT = REPO / "outputs" / "session29" / "audits" / "dns_metadata_status.md"

# Actual pending fields are YAML null (-> Python None). The string tokens defensively
# catch typed placeholders. "none" is NOT pending: it is the real value for
# subgrid_model (no subgrid-scale model in a DNS).
PENDING_TOKENS = {None, "", "[pending]", "pending", "tbd", "tba", "n/a"}

# Required solver fields (filled facts) and the row keys (partner-owned).
REQUIRED_SOLVER = ["name", "discretisation", "subgrid_model", "airfoil_wall_condition",
                   "polynomial_order"]


def is_pending(v) -> bool:
    return (v is None) or (str(v).strip().lower() in PENDING_TOKENS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["draft", "submission"], default="draft")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    meta = yaml.safe_load(YAML.read_text())
    solver = meta.get("solver", {})
    rows = meta.get("rows", [])

    pending = []
    for f in REQUIRED_SOLVER:
        if is_pending(solver.get(f)):
            pending.append(f"solver.{f}")
    for r in rows:
        if is_pending(r.get("value")):
            pending.append(f"rows.{r.get('key', r.get('quantity'))}")

    n_total = len(REQUIRED_SOLVER) + len(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DNS metadata status (SESSION29.8 S0)",
        f"- mode: {args.mode}",
        f"- required fields: {n_total}; pending: {len(pending)}",
        "",
        "## Pending fields" if pending else "## All fields filled",
    ]
    lines += [f"- {p}" for p in pending]
    out_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    if args.mode == "submission" and pending:
        print(f"\n[check_dns_metadata] SUBMISSION BLOCKED: {len(pending)} DNS field(s) "
              f"still pending. Fill paper/dns_metadata.yaml and re-render Table 1.",
              file=sys.stderr)
        sys.exit(1)
    if args.mode == "draft" and pending:
        print(f"\n[check_dns_metadata] DRAFT: {len(pending)} DNS field(s) pending "
              f"(partner-owned). Submission build will fail until filled.")


if __name__ == "__main__":
    main()
