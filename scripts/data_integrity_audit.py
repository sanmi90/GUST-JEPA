"""Data integrity audit for vortex-jepa cached encounters.

Scans every (case, encounter) pair listed in a split manifest and flags:
- NaN/Inf in cached C_L, C_D, p_wall, omega_z
- Missing cache files
- Missing raw files
- Anomalous omega_z magnitudes (above hard cap or near zero)

Emits a JSON integrity manifest plus, optionally, a *_clean.json companion split
that excludes flagged encounters from train_encounter_indices,
test_a_encounter_indices, and test_b/test_c valid_encounter_indices.

Typical workflows
-----------------

After integrating new run3 cases into a v1.x split (e.g. via
``build_split_manifest_v1p5.py``) and preprocessing them, run:

    python scripts/data_integrity_audit.py \
        --split configs/splits/split_v1p5.json \
        --out-manifest outputs/session14/data_integrity/integrity_manifest.json \
        --write-clean-split

This produces both the manifest and ``split_v1p5_clean.json`` next to the
input split. The manifest is what you share with whoever owns the DNS
simulations so they can re-run the broken encounters.

For a one-off check of the current production split:

    python scripts/data_integrity_audit.py --split configs/splits/split_v1.json

Flags
-----

``--omega-hard-cap``: maximum permitted ``|omega_z|`` value in the cache.
Default 10000 (per CLAUDE.md "omega_z magnitude scale at Re=5000").
``--omega-min-max``: any encounter whose max ``|omega_z|`` falls below this
threshold is flagged as "near zero" (suspicious for an active gust case).
Default 1.0.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


REPO = Path(__file__).resolve().parents[1]


def scan_encounter(
    case_id: str,
    k: int,
    case: dict,
    cache_root: Path,
    prevent_root: Path,
    omega_hard_cap: float,
    omega_min_max: float,
) -> dict:
    """Audit one (case_id, encounter_index) pair against its cache file.

    Returns a dict with keys:
      encounter_index, issues (list[str]), n_nan_CL, n_nan_CD, n_nan_p_wall,
      n_nan_omega_z, max_abs_omega, cache_present, raw_present.
    The ``issues`` list is empty if the encounter is clean.
    """
    info = {
        "encounter_index": int(k),
        "issues": [],
        "n_nan_CL": 0,
        "n_nan_CD": 0,
        "n_nan_p_wall": 0,
        "n_nan_omega_z": 0,
        "max_abs_omega": None,
        "cache_present": False,
        "raw_present": False,
    }
    cache_p = cache_root / case_id / f"encounter_{k:02d}.h5"
    info["cache_present"] = cache_p.exists()
    if cache_p.exists():
        try:
            with h5py.File(cache_p, "r") as f:
                cl = f["C_L"][:].astype(np.float32)
                cd = f["C_D"][:].astype(np.float32)
                pw = f["p_wall"][:].astype(np.float32)
                om = f["omega_z"][:].astype(np.float32)
            info["n_nan_CL"] = int(np.isnan(cl).sum() + np.isinf(cl).sum())
            info["n_nan_CD"] = int(np.isnan(cd).sum() + np.isinf(cd).sum())
            info["n_nan_p_wall"] = int(np.isnan(pw).sum() + np.isinf(pw).sum())
            info["n_nan_omega_z"] = int(np.isnan(om).sum() + np.isinf(om).sum())
            finite_om = om[np.isfinite(om)]
            info["max_abs_omega"] = float(np.abs(finite_om).max()) if finite_om.size else None
            if info["n_nan_CL"] > 0:
                info["issues"].append("nan_C_L")
            if info["n_nan_CD"] > 0:
                info["issues"].append("nan_C_D")
            if info["n_nan_p_wall"] > 0:
                info["issues"].append("nan_p_wall")
            if info["n_nan_omega_z"] > 0:
                info["issues"].append("nan_omega_z")
            if info["max_abs_omega"] is not None:
                if info["max_abs_omega"] > omega_hard_cap:
                    info["issues"].append(f"omega_above_cap_{info['max_abs_omega']:.0f}")
                if info["max_abs_omega"] < omega_min_max:
                    info["issues"].append(f"omega_near_zero_{info['max_abs_omega']:.4f}")
        except Exception as e:
            info["issues"].append(f"cache_open_error_{type(e).__name__}")
    else:
        info["issues"].append("cache_missing")

    raw_p = prevent_root / case.get("relative_path", "")
    info["raw_present"] = raw_p.exists()
    if not info["raw_present"]:
        info["issues"].append("raw_missing")
    return info


def audit_split(
    split_path: Path,
    cache_root: Path,
    prevent_root: Path,
    omega_hard_cap: float = 10000.0,
    omega_min_max: float = 1.0,
) -> dict:
    """Run the integrity audit over every case in ``split_path``.

    Returns a manifest dict ready to JSON-dump. Each case_id maps to a
    list of per-encounter dicts; the top-level ``flagged_encounters`` list
    contains only the flagged rows for quick consumption.
    """
    with split_path.open() as f:
        split = json.load(f)
    flagged: list[dict] = []
    case_records: dict[str, dict] = {}
    issue_tag_counts: Counter = Counter()
    for case_id, c in sorted(split["cases"].items()):
        n_enc = int(c["n_encounters_full"])
        rows = []
        for k in range(n_enc):
            info = scan_encounter(
                case_id, k, c, cache_root, prevent_root, omega_hard_cap, omega_min_max
            )
            rows.append(info)
            if info["issues"]:
                flagged.append({
                    "case_id": case_id,
                    "split": c["split"],
                    **info,
                })
                for tag in info["issues"]:
                    # Strip numeric suffix to count "omega_above_cap_3777" as one bucket
                    short = tag.rsplit("_", 1)[0] if tag.rsplit("_", 1)[-1].replace(".", "").replace("-", "").isdigit() else tag
                    issue_tag_counts[short] += 1
        case_records[case_id] = {
            "split": c["split"],
            "n_encounters_full": n_enc,
            "source_group": c["source_group"],
            "G": c["G"],
            "D": c["D"],
            "Y": c["Y"],
            "relative_path": c["relative_path"],
            "encounters": rows,
        }
    n_total = sum(c["n_encounters_full"] for c in split["cases"].values())
    return {
        "created_iso": datetime.now(timezone.utc).isoformat(),
        "audit_tool": "scripts/data_integrity_audit.py",
        "split_audited": str(split_path.relative_to(REPO)) if split_path.is_absolute() else str(split_path),
        "split_manifest_version": split.get("manifest_version"),
        "thresholds": {"omega_hard_cap": omega_hard_cap, "omega_min_max": omega_min_max},
        "n_encounters_total": n_total,
        "n_encounters_flagged": len(flagged),
        "n_encounters_clean": n_total - len(flagged),
        "flag_tag_counts": dict(issue_tag_counts),
        "flagged_encounters": flagged,
        "case_records": case_records,
    }


def write_clean_split(split_path: Path, manifest: dict, out_path: Path) -> dict:
    """Write a copy of ``split_path`` with flagged encounters excluded.

    For train cases, drops the flagged indices from both
    ``train_encounter_indices`` and ``test_a_encounter_indices``.
    For test_b/test_c cases, adds a ``valid_encounter_indices`` field listing
    only the non-flagged encounters.
    Recomputes the ``summary`` block accordingly.
    """
    with split_path.open() as f:
        split = json.load(f)
    flagged_per_case: dict[str, set[int]] = {}
    for r in manifest["flagged_encounters"]:
        flagged_per_case.setdefault(r["case_id"], set()).add(int(r["encounter_index"]))

    clean = dict(split)
    clean["manifest_version"] = (split.get("manifest_version") or "split") + "_clean"
    clean["created_iso"] = datetime.now(timezone.utc).isoformat()
    clean.setdefault("source_inventory", {})["integrity_audit"] = {
        "audit_manifest": str(out_path.with_name(out_path.name.replace(".json", "")) ).replace(str(REPO) + "/", ""),
        "n_excluded_encounters": manifest["n_encounters_flagged"],
        "flag_criteria": (
            "NaN/Inf in C_L, C_D, p_wall, omega_z; cache missing; raw missing; "
            f"|omega_z| > {manifest['thresholds']['omega_hard_cap']} or < {manifest['thresholds']['omega_min_max']}"
        ),
    }

    clean_cases: dict[str, dict] = {}
    n_drop = {"train": 0, "test_a": 0, "test_b": 0, "test_c": 0}
    for case_id, c_orig in split["cases"].items():
        c = dict(c_orig)
        bad = flagged_per_case.get(case_id, set())
        if c["split"] == "train":
            n_drop["train"] += sum(1 for i in c_orig["train_encounter_indices"] if i in bad)
            n_drop["test_a"] += sum(1 for i in c_orig["test_a_encounter_indices"] if i in bad)
            c["train_encounter_indices"] = [i for i in c["train_encounter_indices"] if i not in bad]
            c["test_a_encounter_indices"] = [i for i in c["test_a_encounter_indices"] if i not in bad]
        elif c["split"] in ("test_b", "test_c"):
            valid = [i for i in range(c["n_encounters_full"]) if i not in bad]
            c["valid_encounter_indices"] = valid
            n_drop[c["split"]] += len(bad)
        clean_cases[case_id] = c
    clean["cases"] = clean_cases

    counts = {"train": 0, "test_b": 0, "test_c": 0}
    enc_counts = {"train": 0, "test_a": 0, "test_b": 0, "test_c": 0}
    for c in clean_cases.values():
        s = c["split"]
        counts[s] = counts.get(s, 0) + 1
        if s == "train":
            enc_counts["train"] += len(c["train_encounter_indices"])
            enc_counts["test_a"] += len(c["test_a_encounter_indices"])
        elif s == "test_b":
            enc_counts["test_b"] += len(c.get("valid_encounter_indices", list(range(c["n_encounters_full"]))))
        elif s == "test_c":
            enc_counts["test_c"] += len(c.get("valid_encounter_indices", list(range(c["n_encounters_full"]))))
    clean["summary"] = {
        "n_cases_total": len(clean_cases),
        "n_cases_train": counts["train"],
        "n_cases_test_b": counts["test_b"],
        "n_cases_test_c": counts["test_c"],
        "n_encounters_train": enc_counts["train"],
        "n_encounters_test_a": enc_counts["test_a"],
        "n_encounters_test_b": enc_counts["test_b"],
        "n_encounters_test_c": enc_counts["test_c"],
        "n_encounters_total_in_splits": sum(enc_counts.values()),
        "n_excluded_train": n_drop["train"],
        "n_excluded_test_a": n_drop["test_a"],
        "n_excluded_test_b": n_drop["test_b"],
        "n_excluded_test_c": n_drop["test_c"],
    }
    return clean


def main(argv: Iterable[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--split", type=Path, required=True,
                   help="Path to the split JSON to audit.")
    p.add_argument("--out-manifest", type=Path, default=None,
                   help="Where to write the integrity manifest JSON. "
                        "Default: outputs/session14/data_integrity/<split_stem>_integrity.json.")
    p.add_argument("--write-clean-split", action="store_true",
                   help="Also write a *_clean.json sibling that drops flagged encounters.")
    p.add_argument("--prevent-root", type=Path,
                   default=Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT"))))
    p.add_argument("--cache-root", type=Path, default=None,
                   help="Override the cache root. Default: PREVENT_ROOT/data/processed/vortex-jepa/v1")
    p.add_argument("--omega-hard-cap", type=float, default=10000.0)
    p.add_argument("--omega-min-max", type=float, default=1.0)
    args = p.parse_args(argv)

    cache_root = args.cache_root or (args.prevent_root / "data" / "processed" / "vortex-jepa" / "v1")
    out_manifest = args.out_manifest or (
        REPO / "outputs" / "session14" / "data_integrity" / f"{args.split.stem}_integrity.json"
    )
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    print(f"[audit] scanning {args.split} ...")
    manifest = audit_split(
        args.split, cache_root, args.prevent_root,
        omega_hard_cap=args.omega_hard_cap, omega_min_max=args.omega_min_max,
    )
    with out_manifest.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[audit] manifest -> {out_manifest}")
    print(f"[audit] {manifest['n_encounters_flagged']} flagged of "
          f"{manifest['n_encounters_total']} encounters")

    if manifest["flagged_encounters"]:
        print("\n[audit] flagged encounters:")
        for r in manifest["flagged_encounters"]:
            print(f"  [{r['split']:<7}] {r['case_id']:<22} enc_{r['encounter_index']:02d} "
                  f"issues={r['issues']}")

    if args.write_clean_split:
        out_clean = args.split.with_name(args.split.stem + "_clean.json")
        clean = write_clean_split(args.split, manifest, out_manifest)
        with out_clean.open("w") as f:
            json.dump(clean, f, indent=2)
        print(f"\n[audit] clean split -> {out_clean}")
        print(f"[audit] excluded counts: {clean['summary']['n_excluded_train']} train, "
              f"{clean['summary']['n_excluded_test_a']} test_a, "
              f"{clean['summary']['n_excluded_test_b']} test_b, "
              f"{clean['summary']['n_excluded_test_c']} test_c")


if __name__ == "__main__":
    main()
