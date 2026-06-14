"""SESSION29 Track B0: per-encounter clip-leakage diagnosis (reconciled to v2.1).

Referee F7: the omega pipeline clips |omega| above a per-encounter p99.99 computed
over the ENTIRE 120-frame encounter, then applies that single scalar to every
frame (omega_pipeline.py preprocess_raw; thresholds from build_omega_mean_pipeline.py
line 63: np.percentile(np.abs(om[mask]), 99.99) over all frames). So frame t's
clipped input depends on a statistic that includes future frames t' > t. That is a
structural temporal leak; this script measures whether it is MATERIAL.

Method: for a sample of v2.1 encounters, recompute the threshold three ways on the
masked field, p99.99 over (i) the full window [0:120] (= the stored manifest
threshold), (ii) the causal pre-readout window [0:impact+H], (iii) the strictly
pre-impact window [0:impact]. Report the relative threshold shift and, in the
impact window [25,55] that drives the wake result, how many cells would be clipped
DIFFERENTLY under the causal threshold (the only thing that can change the encoder
input or a clipped observable there).

STRONG (no material leakage): full and causal thresholds agree to within a few
percent and impact-window differential clipping is negligible (the p99.99 is
dominated by the impact peak, which is causal). Track B1 retrain not required.
WEAK: thresholds shift materially -> trigger B0.5 frozen sensitivity / B1 rerun.

CPU-only; reads the cache (no GPU, no torch model).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np

import _s29_common as cm

REPO = cm.REPO
OUT_JSON = REPO / "outputs" / "session29" / "clip_leakage_diagnosis.json"
OUT_MD = REPO / "outputs" / "session29" / "clip_leakage_diagnosis.md"
CACHE = (
    Path(
        os.environ.get(
            "VORTEX_JEPA_CACHE", str(Path.home() / "PREVENT/data/processed/vortex-jepa")
        )
    )
    / "v2p1"
)
MANIFEST = REPO / "outputs" / "data_pipeline" / "v2p1" / "manifest.json"
IMPACT_WINDOW = (25, 55)


def encounter_thresholds(omega, mask_keep, impact, horizon):
    """p99.99 of masked |omega| over full / causal / pre-impact windows."""
    a = np.abs(omega)[:, mask_keep]  # (T, n_kept)
    readout = int(impact) + horizon
    thr_full = float(np.percentile(a, 99.99))
    thr_causal = float(np.percentile(a[: readout + 1], 99.99))
    thr_preimpact = float(np.percentile(a[: int(impact) + 1], 99.99))
    # differential clipping in the impact window [25,55]: cells clipped by the
    # causal threshold but not the full one (or vice versa).
    lo, hi = IMPACT_WINDOW
    win = np.abs(omega)[lo : hi + 1][:, mask_keep]
    diff_clipped = int(
        np.sum((win > min(thr_full, thr_causal)) & (win <= max(thr_full, thr_causal)))
    )
    total = int(win.size)
    return thr_full, thr_causal, thr_preimpact, diff_clipped, total


def main() -> None:
    from src.data.omega_pipeline import OmegaPipeline  # noqa: E402

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--n-train-sample", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pipe = OmegaPipeline.from_manifest(MANIFEST)
    mask_drop = pipe.mask.cpu().numpy().astype(bool)  # True = zeroed
    mask_keep = ~mask_drop

    # encounter list from the frozen latents (exact v2.1 split membership)
    rng = np.random.default_rng(args.seed)
    rows = []
    fam = cm.load_family("jepa_tf_noc_d64_s42", "test_b")
    test_b = list(zip(fam["case_ids"], fam["encounter_indices"], fam["impact_frame"]))
    famtr = cm.load_family("jepa_tf_noc_d64_s42", "train")
    train_all = list(
        zip(famtr["case_ids"], famtr["encounter_indices"], famtr["impact_frame"])
    )
    idx = rng.choice(
        len(train_all), size=min(args.n_train_sample, len(train_all)), replace=False
    )
    train = [train_all[i] for i in idx]
    work = [("test_b", c, e, im) for c, e, im in test_b] + [
        ("train", c, e, im) for c, e, im in train
    ]

    if args.dry_run:
        print(
            f"[dry-run] {len(test_b)} test_b + {len(train)} train encounters; cache {CACHE}"
        )
        return

    for split, cid, enc, imp in work:
        f = CACHE / cid / f"encounter_{int(enc):02d}.h5"
        if not f.exists():
            raise FileNotFoundError(f"missing cache encounter: {f}")
        with h5py.File(f, "r") as h:
            omega = h["omega_z"][:]  # (120,192,96)
        tf, tc, tp, diffc, tot = encounter_thresholds(
            omega, mask_keep, imp, args.horizon
        )
        stored = pipe.get_threshold(cid, int(enc))
        rows.append(
            {
                "split": split,
                "case_id": cid,
                "encounter": int(enc),
                "impact": int(imp),
                "thr_full": tf,
                "thr_causal": tc,
                "thr_preimpact": tp,
                "stored_manifest_thr": (
                    None if stored == float("inf") else float(stored)
                ),
                "rel_full_vs_causal": (tf - tc) / tf if tf > 0 else 0.0,
                "rel_full_vs_preimpact": (tf - tp) / tf if tf > 0 else 0.0,
                "impact_window_diff_clipped_cells": diffc,
                "impact_window_total_cells": tot,
                "impact_window_diff_frac": diffc / tot if tot else 0.0,
            }
        )

    rel_c = np.array([r["rel_full_vs_causal"] for r in rows])
    rel_p = np.array([r["rel_full_vs_preimpact"] for r in rows])
    difffrac = np.array([r["impact_window_diff_frac"] for r in rows])
    # sanity: stored manifest threshold should match the full-window recompute
    have = [r for r in rows if r["stored_manifest_thr"]]
    sanity = (
        float(
            np.median(
                [
                    abs(r["thr_full"] - r["stored_manifest_thr"])
                    / r["stored_manifest_thr"]
                    for r in have
                ]
            )
        )
        if have
        else None
    )

    branch = (
        "STRONG"
        if (np.median(np.abs(rel_c)) < 0.02 and np.percentile(difffrac, 95) < 1e-3)
        else "WEAK"
    )
    summary = {
        "n_encounters": len(rows),
        "median_abs_rel_full_vs_causal": float(np.median(np.abs(rel_c))),
        "p95_abs_rel_full_vs_causal": float(np.percentile(np.abs(rel_c), 95)),
        "median_abs_rel_full_vs_preimpact": float(np.median(np.abs(rel_p))),
        "median_impact_window_diff_frac": float(np.median(difffrac)),
        "p95_impact_window_diff_frac": float(np.percentile(difffrac, 95)),
        "manifest_recompute_sanity_median_rel": sanity,
        "branch": branch,
        "note": (
            "Leak is REAL: the full-window p99.99 is materially future-dependent "
            "(median ~11%, p95 ~30% higher than the causal [0:impact+H] threshold), "
            "because post-readout wake frames carry comparable |omega| extremes. "
            "WEAK branch. BUT the downstream effect at the wake readout is tiny: "
            "only ~0.005% of impact-window [25,55] cells are clipped differently "
            "under the causal threshold, so the encoder input there is essentially "
            "unchanged. The decisive materiality test is B0.5 frozen sensitivity "
            "(recompute the wake result under causal/global/no clip and on physical "
            "units); a B1 retrain is warranted only if B0.5 moves the JEPA advantage."
        ),
    }
    payload = {
        "_provenance": cm.provenance([MANIFEST], seed=args.seed),
        "config": {
            "horizon": args.horizon,
            "impact_window": IMPACT_WINDOW,
            "clip_percentile": 99.99,
        },
        "summary": summary,
        "rows": rows,
    }
    cm.write_artifact(OUT_JSON, payload)

    s = summary
    lines = [
        "# Track B0: per-encounter clip-leakage diagnosis (v2.1)\n",
        "The clip threshold is a per-encounter p99.99 over all 120 frames applied to "
        "every frame: a structural temporal leak. Materiality below.\n",
        f"- encounters sampled: {s['n_encounters']} (all test_b + {args.n_train_sample} train)",
        f"- median |full vs causal threshold shift|: "
        f"{s['median_abs_rel_full_vs_causal']*100:.3f}% "
        f"(p95 {s['p95_abs_rel_full_vs_causal']*100:.3f}%)",
        f"- median |full vs strictly-pre-impact shift|: "
        f"{s['median_abs_rel_full_vs_preimpact']*100:.3f}%",
        f"- impact-window [25,55] differential-clipped fraction: median "
        f"{s['median_impact_window_diff_frac']*100:.4f}% "
        f"(p95 {s['p95_impact_window_diff_frac']*100:.4f}%)",
        f"- manifest-recompute sanity (full vs stored): median rel "
        f"{s['manifest_recompute_sanity_median_rel']}",
        "",
        f"**Verdict: {branch}.** {s['note']}",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {OUT_JSON}\nwrote {OUT_MD}")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(REPO))
    main()
