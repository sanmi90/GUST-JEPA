"""SESSION 31 Track 0.A data certification.

Certifies that partition ``v2p2`` is fit for the canonical retrain before any
training starts: normalisation constants, per-encounter ``(case_id, encounter)``
alignment, case-level split disjointness, and wall-pressure alignment.

The pure check helpers below take already-extracted values so they are unit
testable; the ``main`` CLI walks the cache, gathers those values, and writes
``data_cert.json``.

Note on the val split (reconciliation with the plan's pre-flight wording): the
locked v2/v2.1/v2.2 design holds the validation set as a contiguous
*encounter-level* holdout *within* the train cases (train encounters then a
held-out tail), not as a separate set of cases. Train-cases and val-cases are
therefore the same cases by construction (CLAUDE.md: "Contiguous holdout only").
The leakage-critical property is that the EVAL sets (test_b, test_c) share no
case with train, which they do not. ``check_split_disjoint`` certifies the
case-level disjointness of {train, test_b, test_c} and surfaces val as the
intended within-train holdout rather than failing it.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

_SPLITS = ("train", "test_b", "test_c")
_DATASETS = ("omega_z", "p_wall", "C_L", "C_D")
_PARAM_TOL = 1e-6


def check_split_disjoint(split_manifest: dict) -> dict:
    """Certify case-level disjointness of {train, test_b, test_c}.

    Also cross-checks the top-level ``test_b_cases`` / ``test_c_cases`` lists
    against the per-case ``split`` field and reports the val-within-train cases.
    """
    cases = split_manifest["cases"]
    sets = {s: {cid for cid, c in cases.items() if c["split"] == s} for s in _SPLITS}

    overlaps = {}
    ok = True
    reasons: list[str] = []
    for i, a in enumerate(_SPLITS):
        start = i + 1
        for b in _SPLITS[start:]:
            ov = sorted(sets[a] & sets[b])
            overlaps[f"{a}|{b}"] = ov
            if ov:
                ok = False
                reasons.append(f"case overlap {a}|{b}: {ov}")

    # consistency of the top-level lists with the per-case split field
    for s, key in (("test_b", "test_b_cases"), ("test_c", "test_c_cases")):
        if key in split_manifest:
            listed = set(split_manifest[key])
            if listed != sets[s]:
                ok = False
                missing = sorted(sets[s] - listed)
                extra = sorted(listed - sets[s])
                reasons.append(f"{key} disagrees with split=={s}: missing {missing}, extra {extra}")

    val_within_train = sorted(
        cid for cid, c in cases.items() if c["split"] == "train" and c.get("val_encounter_indices")
    )

    return {
        "ok": ok,
        "reasons": reasons,
        "overlaps": overlaps,
        "n_per_split": {s: len(sets[s]) for s in _SPLITS},
        "val_within_train_cases": val_within_train,
        "val_note": (
            "val is a contiguous encounter-level holdout within the train cases "
            "(by design); it is not a separate case set."
        ),
    }


def check_normalisation(
    train_std: float,
    ssim_L: float,
    expected_std: float,
    expected_L: float,
    tol: float = 1e-3,
) -> dict:
    """Assert the pipeline normalisation constants match the plan's pins."""
    reasons: list[str] = []
    if abs(train_std - expected_std) > tol:
        reasons.append(f"train_std {train_std} != expected {expected_std} (tol {tol})")
    if abs(ssim_L - expected_L) > tol:
        reasons.append(f"ssim_L {ssim_L} != expected {expected_L} (tol {tol})")
    return {"ok": not reasons, "reasons": reasons, "train_std": train_std, "ssim_L": ssim_L}


def check_encounter_alignment(
    shapes: dict,
    attrs: dict,
    expected: dict,
    n_frames_expected: int,
) -> dict:
    """Per-encounter alignment: all datasets present, frame counts equal, params match."""
    reasons: list[str] = []
    for name in _DATASETS:
        if name not in shapes:
            reasons.append(f"missing dataset {name}")
            continue
        n = shapes[name][0]
        if n != n_frames_expected:
            reasons.append(f"{name} frame count {n} != {n_frames_expected}")

    if str(attrs.get("case_id")) != str(expected["case_id"]):
        reasons.append(f"case_id {attrs.get('case_id')} != {expected['case_id']}")
    for p in ("G", "D", "Y"):
        if abs(float(attrs.get(p, float("nan"))) - float(expected[p])) > _PARAM_TOL:
            reasons.append(f"{p} {attrs.get(p)} != {expected[p]}")

    return {"ok": not reasons, "reasons": reasons}


def check_pressure_alignment(
    p_wall_shape: tuple,
    n_frames: int,
    n_surface_expected: int = 192,
) -> dict:
    """Wall pressure is (n_frames, n_surface) and frame-aligned to the field."""
    reasons: list[str] = []
    if len(p_wall_shape) != 2:
        reasons.append(f"p_wall ndim {len(p_wall_shape)} != 2")
        return {"ok": False, "reasons": reasons}
    if p_wall_shape[0] != n_frames:
        reasons.append(f"p_wall frames {p_wall_shape[0]} != {n_frames}")
    if p_wall_shape[1] != n_surface_expected:
        reasons.append(f"p_wall surface pts {p_wall_shape[1]} != {n_surface_expected}")
    return {"ok": not reasons, "reasons": reasons}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _resolve_cache_dir(partition: str) -> Path:
    cache_root = os.environ.get("VORTEX_JEPA_CACHE")
    if cache_root is None:
        prevent = Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
        cache_root = prevent / "data" / "processed" / "vortex-jepa"
    return Path(cache_root) / partition


def _manifest_norm_constants(pipeline_manifest: dict) -> tuple[float, float]:
    """Pull (train_std, ssim_L) from the omega-pipeline manifest, tolerant of keys."""
    std = pipeline_manifest.get("train_stats", {}).get("std")
    if std is None:
        std = pipeline_manifest.get("train_std")
    L = (
        pipeline_manifest.get("ssim_data_range_L")
        or pipeline_manifest.get("ssim_data_range")
        or pipeline_manifest.get("ssim_L")
    )
    return float(std), float(L)


def main() -> None:
    ap = argparse.ArgumentParser(description="SESSION 31 Track 0.A data certification")
    ap.add_argument("--partition", default="v2p2")
    ap.add_argument("--split-manifest", default="configs/splits/split_v2p2.json")
    ap.add_argument("--pipeline-manifest", default="outputs/data_pipeline/v2p2/manifest.json")
    ap.add_argument("--expected-std", type=float, default=3.5396)
    ap.add_argument("--expected-L", type=float, default=8.487)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument(
        "--check",
        nargs="+",
        default=["normalisation", "alignment", "split_disjoint", "pressure_alignment"],
    )
    ap.add_argument("--out", default="outputs/session31/data_cert.json")
    args = ap.parse_args()

    import h5py

    repo = Path(__file__).resolve().parents[2]

    def _abs(p):
        p = Path(p)
        return p if p.is_absolute() else repo / p

    split_manifest = json.load(open(_abs(args.split_manifest)))
    pipeline_manifest = json.load(open(_abs(args.pipeline_manifest)))
    cache_dir = _resolve_cache_dir(args.partition)
    cases = split_manifest["cases"]

    report: dict = {"partition": args.partition, "checks": {}}

    if "normalisation" in args.check:
        std, L = _manifest_norm_constants(pipeline_manifest)
        report["checks"]["normalisation"] = check_normalisation(
            std, L, args.expected_std, args.expected_L, args.tol
        )

    if "split_disjoint" in args.check:
        report["checks"]["split_disjoint"] = check_split_disjoint(split_manifest)

    if "alignment" in args.check or "pressure_alignment" in args.check:
        align_fail: list[str] = []
        press_fail: list[str] = []
        n_enc = 0
        for cid, c in cases.items():
            for k in range(int(c["n_encounters_full"])):
                n_enc += 1
                enc_path = cache_dir / cid / f"encounter_{k:02d}.h5"
                key = f"{cid}/{k:02d}"
                if not enc_path.exists():
                    align_fail.append(f"{key}: file missing")
                    continue
                with h5py.File(enc_path, "r") as g:
                    shapes = {n: g[n].shape for n in _DATASETS if n in g}
                    attrs = {a: g.attrs[a] for a in ("case_id", "G", "D", "Y") if a in g.attrs}
                    p_shape = g["p_wall"].shape if "p_wall" in g else ()
                    n_frames = int(g["C_L"].shape[0]) if "C_L" in g else 0
                if "alignment" in args.check:
                    r = check_encounter_alignment(
                        shapes,
                        attrs,
                        expected={"case_id": cid, "G": c["G"], "D": c["D"], "Y": c["Y"]},
                        n_frames_expected=n_frames or 120,
                    )
                    if not r["ok"]:
                        align_fail.append(f"{key}: {'; '.join(r['reasons'])}")
                if "pressure_alignment" in args.check:
                    r = check_pressure_alignment(p_shape, n_frames or 120)
                    if not r["ok"]:
                        press_fail.append(f"{key}: {'; '.join(r['reasons'])}")
        if "alignment" in args.check:
            report["checks"]["alignment"] = {
                "ok": not align_fail,
                "n_encounters": n_enc,
                "n_fail": len(align_fail),
                "failures": align_fail[:25],
            }
        if "pressure_alignment" in args.check:
            report["checks"]["pressure_alignment"] = {
                "ok": not press_fail,
                "n_encounters": n_enc,
                "n_fail": len(press_fail),
                "failures": press_fail[:25],
            }

    report["all_ok"] = all(v.get("ok") for v in report["checks"].values())

    out_path = _abs(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out_path, "w"), indent=2, default=str)

    for name, res in report["checks"].items():
        status = "PASS" if res.get("ok") else "FAIL"
        extra = ""
        if "n_fail" in res:
            extra = f" ({res['n_fail']}/{res['n_encounters']} fail)"
        print(f"  [{status}] {name}{extra}")
    print(f"all_ok={report['all_ok']}; wrote {out_path}")


if __name__ == "__main__":
    main()
