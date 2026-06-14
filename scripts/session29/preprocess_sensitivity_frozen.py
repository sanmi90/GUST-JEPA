"""SESSION29 Track B0.5: frozen-encoder preprocessing sensitivity of the wake result (v2.1).

Referee F7 / follow-on to Track B0. Track B0 found the per-encounter p99.99 clip
of the omega pipeline is future-dependent (full-window p99.99 is ~11% above the
causal threshold) but its effect at the wake readout is tiny (~0.005% of impact-
window cells re-clipped). Wake enstrophy is SQUARED vorticity, so it is the most
clip-sensitive observable in the project. This track asks, with FROZEN encoders
and NO retrain: is the JEPA-vs-baseline held-out wake advantage INVARIANT to
reasonable TRAINING-ONLY preprocessing choices of the TARGET observable?

Method (frozen latents, recompute only the TARGET):
  Recompute the wake_enstrophy target at the readout frame impact+H (H=16) under
  three clip treatments of the PHYSICAL (raw-omega) cache field, then re-fit the
  frozen-latent -> wake Ridge probe per family and compare held-out test_b R^2.

Clip treatments (the pipeline spatial DROP mask is applied in ALL three, then):
  - none            : no amplitude clip (closest to the project's stored target,
                      which itself used no amplitude clip; differs only by the
                      drop mask, 54 of whose cells fall in the wake box).
  - per_encounter   : clip |omega| to the per-encounter p99.99 (the CURRENT
                      pipeline; future-dependent, the Track B0 leak).
  - training_global : clip |omega| to ONE threshold = p99.99 of masked |omega|
                      over all TRAIN encounters only (a leak-free training-only
                      clip; the same threshold is then applied to test_b).

Wake enstrophy convention is the project canonical one (scripts/session16/
exp2_build_targets.py == scripts/session16/exp1b_decode_axes.py): on the raw-scale
(192, 96) field, abs_o = |omega|, integrate (abs_o * wake_box_mask)^2 * dx * dy
over the wake box x in [0.5, 4.0], y in [-1.0, 1.0]. The amplitude clip is applied
to the raw field BEFORE squaring (clipping caps the |omega| of the biggest cores,
which lowers enstrophy monotonically).

Probe (matched to Track D / Track G readout regime): readout frame = impact+H,
one row per encounter, X = frozen latent at impact+H (cm.readout_xy), y = the
recomputed wake_enstrophy target at impact+H. Fit Ridge on TRAIN with
GroupKFold(5)-by-case alpha selection, evaluate once on test_b, group by case_id,
score = case-clustered wake-enstrophy R^2 (cm.case_clustered_r2_ci). Families:
jepa_tf_noc (the predictive latent), fukami, pod (baselines/context).

Verdict: STRONG iff the JEPA-minus-baseline wake advantage keeps its SIGN and
approximate magnitude across all three treatments (preprocessing-robust; B1
retrain NOT required). WEAK iff some training-only clip choice materially moves
the advantage (report preprocessing sensitivity as a result; downgrade the wake
headline; B1 retrain warranted).

Outputs: outputs/session29/preprocess_sensitivity_frozen.json (+ provenance) and
.md. CPU-only (the GPU is busy training); the encoders are frozen so no model is
loaded, only the per-family frozen-latent npz + the raw cache omega field.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))  # _s29_common
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root for `src`
import _s29_common as cm  # noqa: E402

REPO = cm.REPO
OUT_JSON = REPO / "outputs" / "session29" / "preprocess_sensitivity_frozen.json"
OUT_MD = REPO / "outputs" / "session29" / "preprocess_sensitivity_frozen.md"
MANIFEST = REPO / "outputs" / "data_pipeline" / "v2p1" / "manifest.json"

# Canonical wake-enstrophy geometry (scripts/session16/exp2_build_targets.py).
X_EXTENT = (-1.5, 4.5)
Y_EXTENT = (-1.5, 1.5)
NX, NY = 192, 96
DX = (X_EXTENT[1] - X_EXTENT[0]) / NX
DY = (Y_EXTENT[1] - Y_EXTENT[0]) / NY
WAKE_X = (0.5, 4.0)
WAKE_Y = (-1.0, 1.0)

TREATMENTS = ["none", "per_encounter", "training_global"]
FAMILIES = ["jepa_tf_noc", "fukami", "pod"]
BASELINES = ["fukami", "pod"]  # the JEPA advantage is JEPA minus the best of these
CLIP_PERCENTILE = 99.99
HORIZON = 16
# "Approximate magnitude" tolerance: the JEPA-minus-best-baseline advantage is
# called preserved if it keeps its sign AND stays within this absolute R^2 band of
# the per_encounter (current-pipeline) advantage across both other treatments.
ADVANTAGE_TOL = 0.10


def wake_box_mask() -> np.ndarray:
    """Boolean (192, 96) mask of the canonical wake box (True = inside box)."""
    x = np.linspace(X_EXTENT[0] + DX / 2, X_EXTENT[1] - DX / 2, NX)
    y = np.linspace(Y_EXTENT[0] + DY / 2, Y_EXTENT[1] - DY / 2, NY)
    in_x = (x >= WAKE_X[0]) & (x <= WAKE_X[1])
    in_y = (y >= WAKE_Y[0]) & (y <= WAKE_Y[1])
    return (in_x[:, None] & in_y[None, :]).astype(bool)


def apply_drop_mask(field: np.ndarray, drop_mask: np.ndarray) -> np.ndarray:
    """Zero the pipeline spatial-drop cells. field (..., 192, 96), drop_mask (192,96)."""
    out = field.copy()
    out[..., drop_mask] = 0.0
    return out


def clip_amplitude(field: np.ndarray, threshold: float) -> np.ndarray:
    """Symmetric amplitude clip to [-threshold, +threshold]. Inf = no clip."""
    if not np.isfinite(threshold):
        return field
    return np.clip(field, -threshold, threshold)


def wake_enstrophy_frame(omega_frame_raw: np.ndarray, box_mask: np.ndarray) -> float:
    """Canonical wake enstrophy on one (192, 96) raw-scale (already-treated) field.

    abs, restrict to the wake box, square, integrate by dx*dy. Matches
    scripts/session16/exp2_build_targets.py line 88 (abs then square then sum).
    """
    abs_o = np.abs(omega_frame_raw)
    wake_field = abs_o * box_mask
    return float((wake_field**2).sum() * DX * DY)


def cache_root() -> Path:
    root = os.environ.get("VORTEX_JEPA_CACHE")
    if not root:
        prevent = os.environ.get("PREVENT_ROOT")
        if not prevent:
            raise EnvironmentError(
                "neither VORTEX_JEPA_CACHE nor PREVENT_ROOT is set; cannot locate "
                "the v2.1 cache for the raw omega field"
            )
        root = str(Path(prevent) / "data" / "processed" / "vortex-jepa")
    return Path(root) / "v2p1"


def _encounter_file(cache: Path, case_id: str, enc: int) -> Path:
    f = cache / case_id / f"encounter_{enc:02d}.h5"
    if not f.exists():
        raise FileNotFoundError(f"missing cache encounter: {f}")
    return f


def ordered_rows(split: str, horizon: int):
    """Per-encounter rows in cm.readout_xy order/skip-logic (single alignment source).

    Uses the jepa family ONLY for its (case_id, encounter, impact_frame) alignment
    and the readout-frame validity test fr < n_frames; the target y is the raw-cache
    wake enstrophy, recomputed per treatment, NOT the latent.
    """
    fam = cm.load_family(cm.FAMILY_TAGS["jepa_tf_noc"], split)
    cid, enc, imp = fam["case_ids"], fam["encounter_indices"], fam["impact_frame"]
    n_frames = fam["z_full"].shape[1]
    rows = []
    for i in range(len(cid)):
        impact = int(imp[i])
        fr = impact + horizon
        if fr >= n_frames:
            continue
        rows.append(
            {
                "case_id": str(cid[i]),
                "encounter": int(enc[i]),
                "impact": impact,
                "readout_frame": fr,
            }
        )
    if not rows:
        raise ValueError(f"{split}: no valid readout rows at horizon {horizon}")
    return rows


def compute_training_global_threshold(pipe, drop_mask: np.ndarray, cache: Path) -> float:
    """ONE leak-free p99.99 over masked |omega| across all TRAIN encounters.

    Streamed per encounter (no giant concat): an exact global p99.99 over kept
    cells of all train frames via a single pooled percentile on the concatenated
    masked-|omega| values. To bound memory we subsample at most ``cap`` kept values
    per encounter uniformly; with ~229 train encounters x 120 frames x ~18k kept
    cells this would be ~5e8 values, so we pool per-encounter |omega| over kept
    cells and take the global percentile on the pooled stack.
    """
    keep = ~drop_mask
    rows = ordered_rows("train", HORIZON)
    # Pool ALL kept |omega| values across train encounters (full window, matching
    # the per-encounter manifest convention which uses the full [0:120] window).
    pooled = []
    for r in rows:
        f = _encounter_file(cache, r["case_id"], r["encounter"])
        with h5py.File(f, "r") as h:
            omega = np.asarray(h["omega_z"], dtype=np.float32)  # (120,192,96)
        vals = np.abs(omega)[:, keep].ravel()  # kept cells, all frames
        pooled.append(vals)
    allvals = np.concatenate(pooled)
    return float(np.percentile(allvals, CLIP_PERCENTILE))


def build_targets(split, rows, treatment, pipe, drop_mask, box_mask, cache, global_thr):
    """Recompute the wake_enstrophy target at the readout frame, one per row.

    Returns y (n,) aligned to ``rows``. The raw-cache omega field at the readout
    frame is (1) drop-masked, (2) amplitude-clipped per the treatment, then fed to
    the canonical wake-enstrophy reduction. Per-encounter clip uses the manifest's
    per-encounter p99.99; training_global uses the single train-pooled threshold;
    none uses no clip. Fails loudly on a non-finite target.
    """
    y = np.empty(len(rows), dtype=float)
    for i, r in enumerate(rows):
        f = _encounter_file(cache, r["case_id"], r["encounter"])
        with h5py.File(f, "r") as h:
            frame = np.asarray(h["omega_z"][r["readout_frame"]], dtype=np.float32)  # (192,96)
        frame = apply_drop_mask(frame, drop_mask)
        if treatment == "none":
            thr = float("inf")
        elif treatment == "per_encounter":
            thr = pipe.get_threshold(r["case_id"], r["encounter"])
        elif treatment == "training_global":
            thr = global_thr
        else:
            raise ValueError(f"unknown treatment {treatment!r}")
        frame = clip_amplitude(frame, thr)
        y[i] = wake_enstrophy_frame(frame, box_mask)
    if not np.isfinite(y).all():
        raise ValueError(f"{split}/{treatment}: non-finite wake-enstrophy target")
    return y


def fit_eval_ridge(Xtr, ytr, gtr, Xte, yte, gte, seed):
    """Ridge latent->y, GroupKFold(5)-by-case alpha selection, case-clustered R^2 CI."""
    pipe = Pipeline([("scale", StandardScaler()), ("model", Ridge())])
    grid = {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
    n_groups = len(np.unique(gtr))
    cv = GroupKFold(n_splits=min(5, n_groups))
    gs = GridSearchCV(pipe, grid, scoring="r2", cv=cv, n_jobs=-1)
    gs.fit(Xtr, ytr, groups=gtr)
    yp = gs.predict(Xte)
    r2, lo, hi = cm.case_clustered_r2_ci(yte, yp, gte, seed=seed)
    if not np.isfinite(r2):
        raise ValueError("ridge: non-finite R^2 on test_b")
    cases = np.unique(gte)
    per_case_mae = float(np.mean([np.mean(np.abs(yte[gte == c] - yp[gte == c])) for c in cases]))
    return dict(
        r2=float(r2),
        ci_lo=float(lo),
        ci_hi=float(hi),
        per_case_mae=per_case_mae,
        best_alpha=float(gs.best_params_["model__alpha"]),
        n_features=int(Xtr.shape[1]),
        n_test_b=int(len(yte)),
        n_cases_test_b=int(len(np.unique(gte))),
    )


def aligned_latents(tag, split, rows, horizon):
    """Frozen-latent X at impact+H aligned to ``rows`` (cm.readout_xy order).

    cm.readout_xy already enforces the same skip-logic, so re-using it and trusting
    the identical row order would work; instead we index the family explicitly by
    (case_id, encounter) to GUARANTEE X and the recomputed y index the same
    encounter even if a family's npz ever drifts in row order.
    """
    fam = cm.load_family(tag, split)
    z, cid, enc, imp = (
        fam["z_full"],
        fam["case_ids"],
        fam["encounter_indices"],
        fam["impact_frame"],
    )
    lut = {(str(cid[i]), int(enc[i])): i for i in range(len(cid))}
    X = np.empty((len(rows), z.shape[2]), dtype=float)
    for j, r in enumerate(rows):
        key = (r["case_id"], r["encounter"])
        if key not in lut:
            raise KeyError(f"latent {tag}/{split}: missing {key}")
        idx = lut[key]
        fr = int(imp[idx]) + horizon
        X[j] = z[idx, fr, :]
    return X


def advantage_record(per_treatment_family_r2: dict) -> dict:
    """JEPA-minus-best-baseline wake advantage per treatment + STRONG/WEAK verdict.

    per_treatment_family_r2[treatment][family] = test_b R^2. The advantage is
    jepa_tf_noc minus max(fukami, pod). STRONG iff across ALL treatments the
    advantage keeps the SIGN of the per_encounter (current-pipeline) advantage AND
    stays within ADVANTAGE_TOL of it. Pure function (unit-testable).
    """
    adv = {}
    for t, fam_r2 in per_treatment_family_r2.items():
        base = max(fam_r2[b] for b in BASELINES if b in fam_r2)
        best_base = max((b for b in BASELINES if b in fam_r2), key=lambda b: fam_r2[b])
        adv[t] = {
            "jepa_r2": float(fam_r2["jepa_tf_noc"]),
            "best_baseline": best_base,
            "best_baseline_r2": float(base),
            "advantage": float(fam_r2["jepa_tf_noc"] - base),
        }
    ref = adv["per_encounter"]["advantage"]
    ref_sign = np.sign(ref)
    sign_ok = all(np.sign(adv[t]["advantage"]) == ref_sign for t in adv)
    mag_ok = all(abs(adv[t]["advantage"] - ref) <= ADVANTAGE_TOL for t in adv)
    return {
        "per_treatment": adv,
        "reference_treatment": "per_encounter",
        "reference_advantage": float(ref),
        "advantage_tol": ADVANTAGE_TOL,
        "max_abs_advantage_shift": float(max(abs(adv[t]["advantage"] - ref) for t in adv)),
        "sign_preserved": bool(sign_ok),
        "magnitude_preserved": bool(mag_ok),
        "branch": "STRONG" if (sign_ok and mag_ok) else "WEAK",
        "b1_retrain_needed": not (sign_ok and mag_ok),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--treatments", nargs="+", default=TREATMENTS)
    ap.add_argument("--families", nargs="+", default=FAMILIES)
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from src.data.omega_pipeline import OmegaPipeline

    if not MANIFEST.exists():
        raise FileNotFoundError(f"missing omega pipeline manifest: {MANIFEST}")
    pipe = OmegaPipeline.from_manifest(MANIFEST)
    drop_mask = pipe.mask.cpu().numpy().astype(bool)  # True = zeroed cell
    box_mask = wake_box_mask()
    cache = cache_root()

    if drop_mask.shape != (NX, NY):
        raise ValueError(f"drop mask shape {drop_mask.shape} != ({NX},{NY})")

    inputs = [MANIFEST]
    inputs += [cm.LATENTS / cm.FAMILY_TAGS[f] / "train.npz" for f in args.families]

    rows = {sp: ordered_rows(sp, args.horizon) for sp in ("train", "test_b")}

    if args.dry_run:
        print(f"[dry-run] manifest {MANIFEST}")
        print(
            f"[dry-run] drop-mask cells {int(drop_mask.sum())}, "
            f"wake-box cells {int(box_mask.sum())}"
        )
        print(f"[dry-run] drop cells inside wake box: {int((drop_mask & box_mask).sum())}")
        for sp in ("train", "test_b"):
            n = len(rows[sp])
            ncases = len(set(r["case_id"] for r in rows[sp]))
            print(
                f"[dry-run] {sp}: {n} rows, {ncases} cases "
                f"(readout frame impact+{args.horizon})"
            )
        print(f"[dry-run] treatments {args.treatments}, families {args.families}")
        print(
            "[dry-run] training_global threshold NOT computed in dry-run "
            "(would stream the train cache)"
        )
        return

    # training_global threshold (single, leak-free; computed once over TRAIN only).
    global_thr = float("inf")
    if "training_global" in args.treatments:
        global_thr = compute_training_global_threshold(pipe, drop_mask, cache)
        print(f"training_global p{CLIP_PERCENTILE} threshold (train-pooled): {global_thr:.4f}")

    # Recompute targets per treatment, per split.
    targets = {sp: {} for sp in ("train", "test_b")}
    for sp in ("train", "test_b"):
        for t in args.treatments:
            targets[sp][t] = build_targets(
                sp, rows[sp], t, pipe, drop_mask, box_mask, cache, global_thr
            )

    # Frozen latents per family, aligned to the row order, plus case groups.
    groups = {sp: np.asarray([r["case_id"] for r in rows[sp]]) for sp in ("train", "test_b")}
    latents = {sp: {} for sp in ("train", "test_b")}
    for sp in ("train", "test_b"):
        for f in args.families:
            latents[sp][f] = aligned_latents(cm.FAMILY_TAGS[f], sp, rows[sp], args.horizon)

    # Fit/eval Ridge per (treatment, family).
    results = {t: {} for t in args.treatments}
    per_treatment_family_r2 = {t: {} for t in args.treatments}
    for t in args.treatments:
        for f in args.families:
            rec = fit_eval_ridge(
                latents["train"][f],
                targets["train"][t],
                groups["train"],
                latents["test_b"][f],
                targets["test_b"][t],
                groups["test_b"],
                args.seed,
            )
            results[t][f] = rec
            per_treatment_family_r2[t][f] = rec["r2"]
            print(
                f"{t:16s} {f:14s} R2={rec['r2']:+.3f} "
                f"[{rec['ci_lo']:+.3f},{rec['ci_hi']:+.3f}] "
                f"alpha={rec['best_alpha']:g} MAE/case={rec['per_case_mae']:.4g}"
            )

    # Target sanity: enstrophy must be monotone non-increasing as the clip tightens
    # (none >= training_global/per_encounter), since clipping only LOWERS |omega|.
    target_summary = {}
    for sp in ("train", "test_b"):
        target_summary[sp] = {
            t: {
                "mean": float(np.mean(targets[sp][t])),
                "median": float(np.median(targets[sp][t])),
                "max": float(np.max(targets[sp][t])),
            }
            for t in args.treatments
        }

    verdict = None
    if all(t in per_treatment_family_r2 for t in TREATMENTS) and all(
        f in per_treatment_family_r2[TREATMENTS[1]] for f in (["jepa_tf_noc"] + BASELINES)
    ):
        verdict = advantage_record({t: per_treatment_family_r2[t] for t in TREATMENTS})

    payload = {
        "_provenance": cm.provenance(inputs, seed=args.seed),
        "config": {
            "observable": "wake_enstrophy",
            "horizon": args.horizon,
            "treatments": args.treatments,
            "families": args.families,
            "baselines": BASELINES,
            "clip_percentile": CLIP_PERCENTILE,
            "advantage_tol": ADVANTAGE_TOL,
            "wake_box": {"x": list(WAKE_X), "y": list(WAKE_Y)},
            "regime": (
                "readout-frame impact+H, frozen latent X = z[impact+H], "
                "y = raw-cache wake enstrophy recomputed per clip treatment "
                "(pipeline drop mask applied in all 3), Ridge with "
                "GroupKFold(5)-by-case alpha selection, fit train, eval test_b, "
                "case-clustered bootstrap CI"
            ),
            "training_global_threshold": global_thr,
            "drop_mask_cells": int(drop_mask.sum()),
            "drop_cells_in_wake_box": int((drop_mask & box_mask).sum()),
            "note": (
                "The project's stored wake_enstrophy target (exp2_build_targets) "
                "used NO amplitude clip AND NO pipeline drop mask; the 'none' "
                "treatment here adds the drop mask (54 of whose cells fall in the "
                "wake box) so all three treatments share the same spatial support."
            ),
        },
        "target_summary": target_summary,
        "results": results,
        "per_treatment_family_r2": per_treatment_family_r2,
        "verdict": verdict,
    }
    cm.write_artifact(OUT_JSON, payload)
    _write_md(args, results, target_summary, verdict, global_thr)
    print(f"\nwrote {OUT_JSON}\nwrote {OUT_MD}")
    if verdict:
        print(
            f"VERDICT: {verdict['branch']} | ref advantage (per_encounter) "
            f"{verdict['reference_advantage']:+.3f} | max |shift| "
            f"{verdict['max_abs_advantage_shift']:.3f} | sign_preserved "
            f"{verdict['sign_preserved']} | mag_preserved {verdict['magnitude_preserved']} "
            f"| B1 retrain needed: {verdict['b1_retrain_needed']}"
        )


def _write_md(args, results, target_summary, verdict, global_thr) -> None:
    lines = [
        "# Track B0.5: frozen-encoder preprocessing sensitivity of the wake result "
        "(v2.1, wake enstrophy, H={}, test_b)\n".format(args.horizon),
        "Frozen latents, NO retrain. The wake-enstrophy TARGET at readout frame "
        "impact+{} is recomputed under three TRAINING-ONLY clip treatments of the "
        "raw-cache omega field (pipeline drop mask applied in all three), then the "
        "frozen-latent -> wake Ridge probe (GroupKFold(5)-by-case alpha selection, "
        "fit train, eval test_b, case-clustered CI) is re-fit per family.\n".format(args.horizon),
        "Clip treatments: `none` (no amplitude clip), `per_encounter` (current "
        "pipeline p99.99, future-dependent), `training_global` (one leak-free "
        "p99.99 over all TRAIN encounters; train-pooled threshold "
        "= {:.4f}).\n".format(global_thr),
        "## test_b wake-enstrophy R^2 (family x clip treatment)\n",
        "| family | " + " | ".join(args.treatments) + " |",
        "|---|" + "---|" * len(args.treatments),
    ]
    for f in args.families:
        cells = []
        for t in args.treatments:
            r = results[t][f]
            cells.append(f"{r['r2']:+.3f} [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]")
        tag = "**jepa_tf_noc** (predictive)" if f == "jepa_tf_noc" else f
        lines.append(f"| {tag} | " + " | ".join(cells) + " |")

    lines.append("\n## JEPA-minus-best-baseline wake advantage per treatment\n")
    if verdict:
        lines.append("| treatment | JEPA R^2 | best baseline | baseline R^2 | advantage |")
        lines.append("|---|---|---|---|---|")
        for t, a in verdict["per_treatment"].items():
            lines.append(
                f"| {t} | {a['jepa_r2']:+.3f} | {a['best_baseline']} | "
                f"{a['best_baseline_r2']:+.3f} | {a['advantage']:+.3f} |"
            )
        lines.append(
            "\nReference advantage (current pipeline, per_encounter): "
            f"{verdict['reference_advantage']:+.3f}. "
            f"Max |advantage shift| across treatments: "
            f"{verdict['max_abs_advantage_shift']:.3f} "
            f"(tolerance {verdict['advantage_tol']}). "
            f"Sign preserved: {verdict['sign_preserved']}. "
            f"Magnitude preserved: {verdict['magnitude_preserved']}.\n"
        )

    lines.append("## Target sanity (enstrophy lowers monotonically as clip tightens)\n")
    lines.append("| split | stat | " + " | ".join(args.treatments) + " |")
    lines.append("|---|---|" + "---|" * len(args.treatments))
    for sp in ("train", "test_b"):
        for stat in ("mean", "median", "max"):
            cells = [f"{target_summary[sp][t][stat]:.4g}" for t in args.treatments]
            lines.append(f"| {sp} | {stat} | " + " | ".join(cells) + " |")

    if verdict:
        lines.append("\n## Verdict\n")
        if verdict["branch"] == "STRONG":
            lines.append(
                "**STRONG.** The JEPA-vs-baseline held-out wake advantage keeps its "
                "sign and approximate magnitude across all three training-only clip "
                "treatments (`none`, `per_encounter`, `training_global`). The wake "
                "headline is preprocessing-robust; the Track B0 clip leak does not "
                "move the result. A B1 retrain under a leak-free clip is NOT "
                "required.\n"
            )
        else:
            lines.append(
                "**WEAK.** A training-only clip choice materially moves the "
                "JEPA-minus-baseline wake advantage (sign flip or shift beyond the "
                f"tolerance {verdict['advantage_tol']}). Report preprocessing "
                "sensitivity as a result and downgrade the wake headline; a B1 "
                "retrain under a leak-free (training_global) clip is warranted.\n"
            )
    OUT_MD.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
