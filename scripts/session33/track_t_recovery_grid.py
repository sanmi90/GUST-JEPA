"""Track T (T1+T2) -- sensors traded for delays: (K, W) recovery grid on the pooled state.

SESSION_33_MANUSCRIPT_V3.md Section 11 items 10-11 (Tables T1/T2); HANDOFF D238/D239.

Reuses the Track O1 machinery verbatim (build_windows_seq, recover_oof, frozen
linear probes, case-clustered bootstrap; scripts/session32/track_o1_recovery.py).
Grid: K in {1, 2, 4, 8} x W in {1, 2, 4, 8, 16, 30} on qDEIM target-blind NESTED
tap prefixes (the addendum's "target-blind" wording; avoids the model-conditioned-
placement confound), plus ONE bridge cell (K=8, W=30) on the osp_per_model
jepa_pool taps to reconcile with the Track O1 headline (state R2 0.707) and the
frozen filter's sensing.

Scoring: coefficient-state R2 and the five recovered-observable R2 on the frozen
impact-union-relaxation window, reported for all OOF rows, train rows and test_b
rows (the paper tables cite test_b; the T2b (K_min, W_min) selection reads TRAIN
rows only; test_c is untouched here). Per-|G|-stratum recovery curves are stored
for Track T3's embedding-bound overlay.

Gates (case-clustered bootstrap, n_boot=2000):
  T1 strong: wake-circulation (circulation_neg, the O2 wall-blind direction)
     recovered R2 rises with W at K=8 AND the (W=30 - W=1) delta CI excludes 0.
     Weak: positive trend only.
  T2 strong: some cell with K<=2 and W>=8 matches the (K=8, W=1) coefficient-
     state recovery within CI (delta CI includes 0 or is entirely positive).
     Weak: monotone trade surface of the expected sign.

D-T1: delay stride = cache cadence (dt_tc = 0.05); the first-minimum lagged-MI
tau is reported as an Appendix B cross-check only (Fraser-Swinney 1986), never
used to build the windows.

Run (GPU for MLP/LSTM; ~2-3 h):
    taskset -c 0-15 python -m scripts.session33.track_t_recovery_grid --gpu 0
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

from scripts.session32.track_o1_recovery import (  # noqa: E402
    MAPPING_NAMES,
    OBSERVABLES,
    build_windows_seq,
    case_bootstrap_r2_delta,
    keys_of,
    load_cache,
    load_pressure,
    observable_readout,
    preimpact_mask,
    r2_from_se,
    score_observables,
    tss_row_fixed,
)

GRID_KS = (1, 2, 4, 8)
GRID_WS = (1, 2, 4, 8, 16, 30)
BLIND_OBS = "circulation_neg"  # O2 top-hidden direction (D230/D237)
G_BUCKETS = ("0", "0.25-0.5", "1", "1.5", "2", "3", "4")


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


# ----------------------------------------------------------------- |G| strata
def abs_g_of(case_id: np.ndarray) -> np.ndarray:
    out = np.zeros(case_id.shape[0], dtype=np.float64)
    for i, c in enumerate(case_id):
        m = re.match(r"^G([+-]?[0-9.]+)_", c)
        out[i] = abs(float(m.group(1))) if m else 0.0
    return out


def g_bucket_of(abs_g: np.ndarray) -> np.ndarray:
    lab = np.empty(abs_g.shape[0], dtype=object)
    for i, g in enumerate(abs_g):
        if g == 0.0:
            lab[i] = "0"
        elif 0.24 < g <= 0.5:
            lab[i] = "0.25-0.5"
        elif 0.9 < g <= 1.0:
            lab[i] = "1"
        elif 1.4 < g <= 1.5:
            lab[i] = "1.5"
        elif 1.9 < g <= 2.0:
            lab[i] = "2"
        elif 2.9 < g <= 3.0:
            lab[i] = "3"
        elif 3.9 < g <= 4.0:
            lab[i] = "4"
        else:
            lab[i] = f"other:{g}"
    return lab


# ------------------------------------------------------- MI stride cross-check
def mi_first_minimum(p_rows, keys, frame, taps, *, max_tau=40, bins=32) -> dict:
    """First local minimum of the time-lagged mutual information per tap
    (Fraser-Swinney), pooled over encounters. Cross-check for D-T1 only."""

    def mi_of(x, y):
        h, _, _ = np.histogram2d(x, y, bins=bins)
        pxy = h / h.sum()
        px = pxy.sum(axis=1, keepdims=True)
        py = pxy.sum(axis=0, keepdims=True)
        nz = pxy > 0
        return float((pxy[nz] * np.log(pxy[nz] / (px @ py)[nz])).sum())

    # per-encounter contiguous blocks, sorted by frame
    blocks = []
    for key in np.unique(keys):
        rows = np.where(keys == key)[0]
        blocks.append(p_rows[rows[np.argsort(frame[rows])]][:, taps])
    per_tap = {}
    for j, tap in enumerate(taps):
        mi = []
        for tau in range(1, max_tau + 1):
            xs = np.concatenate([b[:-tau, j] for b in blocks if b.shape[0] > tau])
            ys = np.concatenate([b[tau:, j] for b in blocks if b.shape[0] > tau])
            mi.append(mi_of(xs, ys))
        tau_min = None
        for t in range(1, len(mi) - 1):
            if mi[t] < mi[t - 1] and mi[t] <= mi[t + 1]:
                tau_min = t + 1  # taus are 1-indexed
                break
        per_tap[int(tap)] = {"tau_first_min": tau_min, "mi_curve": [round(v, 5) for v in mi]}
    taus = [v["tau_first_min"] for v in per_tap.values() if v["tau_first_min"] is not None]
    return {
        "per_tap": per_tap,
        "mean_tau_first_min": float(np.mean(taus)) if taus else None,
        "note": "cross-check only; windows are built at the cache cadence (D-T1)",
    }


# ------------------------------------------------------------------- one cell
def run_cell(
    p_rows, keys, frame, taps, w, z_pooled, targets, groups, masks,
    *, device, n_components, seed,
):
    """One (taps, W) cell: OOF recovery + frozen-probe readout + split/stratum scores.

    Returns (record, store) where store carries the row-level SE/yhat arrays the
    gates need.
    """
    from scripts.session32.track_o1_recovery import recover_oof

    Xseq = build_windows_seq(p_rows, keys, frame, list(taps), w)
    res = recover_oof(
        Xseq, z_pooled, groups, MAPPING_NAMES,
        device=device, n_components=n_components, seed=seed, gap_shape=None,
    )
    win, pre = masks["window"], masks["preimpact"]
    per_map = {
        m: {
            "state_r2_window": r2_from_se(res[m]["se_row"], z_pooled, win),
            "state_r2_preimpact": r2_from_se(res[m]["se_row"], z_pooled, pre),
        }
        for m in MAPPING_NAMES
    }
    cv_pick = max(MAPPING_NAMES, key=lambda mm: per_map[mm]["state_r2_window"])
    se_row = res[cv_pick]["se_row"]
    readout = observable_readout(res[cv_pick]["rgap"], z_pooled, targets, groups, seed=seed)

    rec = {
        "taps": [int(t) for t in taps],
        "W": int(w),
        "K": int(len(taps)),
        "per_mapping": per_map,
        "cv_pick": cv_pick,
        "state_r2": {},
        "obs": {},
        "per_G_window": {},
    }
    for scope, extra in (("all", None), ("train", masks["train"]), ("test_b", masks["test_b"])):
        mw = win if extra is None else (win & extra)
        mp = pre if extra is None else (pre & extra)
        rec["state_r2"][scope] = {
            "window": r2_from_se(se_row, z_pooled, mw),
            "preimpact": r2_from_se(se_row, z_pooled, mp),
        }
        rec["obs"][scope] = score_observables(readout, targets, mw)
    for b in G_BUCKETS:
        mb = win & masks["g_bucket"][b] if b in masks["g_bucket"] else None
        if mb is None or mb.sum() < 50:
            continue
        from src.evaluation.represent import r2_score_np

        rec["per_G_window"][b] = {
            "n_rows": int(mb.sum()),
            "state_r2": r2_from_se(se_row, z_pooled, mb),
            "per_observable_recovered_r2": {
                o: r2_score_np(targets[o][mb], readout[o][mb]) for o in OBSERVABLES
            },
        }
    store = {"se_row": se_row, "readout": readout}
    del res, Xseq
    return rec, store


# ------------------------------------------------------------------ gate math
def obs_se_tss(readout, targets, obs, mask):
    y = targets[obs].astype(np.float64)
    se = (readout[obs].astype(np.float64) - y) ** 2
    mu = y[mask].mean()
    tss = (y - mu) ** 2
    return se, tss


def main(argv=None):
    ap = argparse.ArgumentParser(description="Track T (K, W) recovery grid on jepa_pool")
    ap.add_argument("--cache-dir", default="outputs/session31/q1_latents")
    ap.add_argument("--qdeim-taps", default="outputs/session32/qdeim_taps_v2p2.json")
    ap.add_argument("--osp-taps", default="outputs/session32/osp_taps_v2p2.json")
    ap.add_argument("--windows", default="outputs/session31/windows_v2p2.json")
    ap.add_argument("--out", default="outputs/session33/track_t_recovery_grid.json")
    ap.add_argument("--ks", nargs="+", type=int, default=list(GRID_KS))
    ap.add_argument("--windows-grid", nargs="+", type=int, default=list(GRID_WS))
    ap.add_argument("--preimpact-lead", type=int, default=8)
    ap.add_argument("--n-components", type=int, default=400)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--no-bridge", action="store_true", help="skip the OSP bridge cell")
    args = ap.parse_args(argv)

    import torch  # noqa: F401

    from src.evaluation.rom_eval import load_windows
    from src.utils.device import require_rtx6000

    device = require_rtx6000(gpu_index=args.gpu)
    print(f"[track-t] device={device} ({torch.cuda.get_device_name(device.index)})", flush=True)

    cache_dir = _resolve(args.cache_dir)
    windows = load_windows(_resolve(args.windows))
    qdeim = json.loads(_resolve(args.qdeim_taps).read_text())
    osp = json.loads(_resolve(args.osp_taps).read_text())

    # ---- rows: train + test_b concatenated (O1 layout), split membership kept
    pooled = {s: load_cache(cache_dir, "jepa_pool", s) for s in ("train", "test_b")}
    pres = {s: load_pressure(cache_dir, s) for s in ("train", "test_b")}

    def cat(g):
        return np.concatenate([g("train"), g("test_b")], axis=0)

    case_id = cat(lambda s: pooled[s]["case_id"])
    enc = cat(lambda s: pooled[s]["encounter_index"])
    frame = cat(lambda s: pooled[s]["frame"])
    win_mask = cat(lambda s: pooled[s]["window_mask"])
    p_rows = cat(lambda s: pres[s]["p_wall"])
    assert np.array_equal(cat(lambda s: pres[s]["case_id"]), case_id)
    assert np.array_equal(cat(lambda s: pres[s]["frame"]), frame)
    keys = keys_of(case_id, enc)
    groups = keys
    n_train = pooled["train"]["z_gap"].shape[0]
    train_rows = np.zeros(len(case_id), dtype=bool)
    train_rows[:n_train] = True
    z_pooled = cat(lambda s: pooled[s]["z_gap"])
    targets = {o: cat(lambda s: pooled[s]["targets"][o]) for o in OBSERVABLES}
    abs_g = abs_g_of(case_id)
    bucket = g_bucket_of(abs_g)
    masks = {
        "window": win_mask,
        "preimpact": preimpact_mask(keys, frame, windows, args.preimpact_lead),
        "train": train_rows,
        "test_b": ~train_rows,
        "g_bucket": {b: bucket == b for b in G_BUCKETS if (bucket == b).any()},
    }

    # ---- nested target-blind taps: qDEIM QR-pivot prefixes (K1 = perm[:1])
    qdeim_full = qdeim["K16"]
    tap_sets = {int(k): [int(t) for t in qdeim_full[: int(k)]] for k in args.ks}
    if 8 in tap_sets:
        assert tap_sets[8] == [int(t) for t in qdeim["K8"]], "qDEIM prefix nesting violated"

    payload = {
        "task": "SESSION 33 Track T -- sensors traded for delays (T1+T2 grid)",
        "params": {
            "family": "jepa_pool (pooled d=32 coefficient state)",
            "grid_Ks": list(args.ks),
            "grid_Ws": list(args.windows_grid),
            "tap_policy": (
                "qDEIM target-blind nested prefixes (D238); "
                "bridge cell on osp_per_model jepa_pool"
            ),
            "tap_sets": {str(k): v for k, v in tap_sets.items()},
            "stride": "cache cadence dt_tc=0.05 (D-T1)",
            "recovery_mappings": list(MAPPING_NAMES),
            "recovery_selection": (
                "encounter-grouped 5-fold GroupKFold OOF, "
                "CV-pick by window state-R2 (O1 protocol)"
            ),
            "krr_n_components": args.n_components,
            "n_boot_case": args.n_boot,
            "seed": args.seed,
            "blind_observable": BLIND_OBS,
            "scoring": "frozen impact-union-relaxation window_mask; scopes all/train/test_b",
            "selection_discipline": (
                "T2b (K_min, W_min) selected on TRAIN rows only; "
                "test_b report-only; test_c untouched"
            ),
        },
        "mi_stride_crosscheck": None,
        "cells": {},
        "bridge_osp": None,
        "gates": {},
        "t2b_selection": None,
    }
    out_path = _resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- MI cross-check on the K8 target-blind taps (train rows only)
    payload["mi_stride_crosscheck"] = mi_first_minimum(
        p_rows[train_rows], keys[train_rows], frame[train_rows], tap_sets[max(args.ks)]
    )
    print(
        f"[track-t] MI stride cross-check: mean tau_first_min = "
        f"{payload['mi_stride_crosscheck']['mean_tau_first_min']} frames",
        flush=True,
    )

    # ---- grid
    stores = {}
    for k in args.ks:
        for w in args.windows_grid:
            name = f"K{k}_W{w}"
            rec, store = run_cell(
                p_rows, keys, frame, tap_sets[k], w, z_pooled, targets, groups, masks,
                device=device, n_components=args.n_components, seed=args.seed,
            )
            payload["cells"][name] = rec
            stores[name] = store
            print(
                f"[track-t] {name:8s} pick={rec['cv_pick']:4s} "
                f"stR2 win all/train/test_b="
                f"{rec['state_r2']['all']['window']:.3f}/"
                f"{rec['state_r2']['train']['window']:.3f}/"
                f"{rec['state_r2']['test_b']['window']:.3f} "
                f"obsR2 test_b={rec['obs']['test_b']['mean_recovered_r2']:.3f} "
                f"circ_neg test_b="
                f"{rec['obs']['test_b']['per_observable_recovered_r2'][BLIND_OBS]:.3f}",
                flush=True,
            )
            out_path.write_text(json.dumps(payload, indent=2))

    # ---- bridge cell: OSP jepa_pool K8, W=30 (reconciles with Track O1 headline)
    if not args.no_bridge:
        rec, _ = run_cell(
            p_rows, keys, frame, osp["jepa_pool"]["K8"], 30, z_pooled, targets, groups, masks,
            device=device, n_components=args.n_components, seed=args.seed,
        )
        rec["note"] = (
            "osp_per_model jepa_pool taps; compare Track O1 pooled K8 state_r2_window=0.707"
        )
        payload["bridge_osp"] = rec
        print(
            f"[track-t] bridge OSP K8_W30 stR2 all={rec['state_r2']['all']['window']:.3f} "
            f"(O1 headline 0.707)",
            flush=True,
        )

    # ---- Gate T1: circulation recovery vs W at K=8
    k8 = max(args.ks)
    ws = sorted(args.windows_grid)
    curve = {
        o: [
            payload["cells"][f"K{k8}_W{w}"]["obs"]["all"]["per_observable_recovered_r2"][o]
            for w in ws
        ]
        for o in OBSERVABLES
    }
    blind_curve = curve[BLIND_OBS]
    monotone = all(b >= a - 0.01 for a, b in zip(blind_curve, blind_curve[1:]))
    se_a, tss_a = obs_se_tss(
        stores[f"K{k8}_W{ws[-1]}"]["readout"], targets, BLIND_OBS, win_mask
    )
    se_b, tss_b = obs_se_tss(
        stores[f"K{k8}_W{ws[0]}"]["readout"], targets, BLIND_OBS, win_mask
    )
    t1_ci = case_bootstrap_r2_delta(
        se_a, tss_a, se_b, tss_b, case_id, win_mask, n_boot=args.n_boot, seed=args.seed
    )
    t1_ci_testb = case_bootstrap_r2_delta(
        se_a, tss_a, se_b, tss_b, case_id, win_mask & masks["test_b"],
        n_boot=args.n_boot, seed=args.seed,
    )
    t1_strong = bool(monotone and t1_ci["ci95"][0] > 0)
    t1_weak = bool(blind_curve[-1] > blind_curve[0])
    payload["gates"]["T1"] = {
        "observable": BLIND_OBS,
        "curve_W": ws,
        "curve_r2_all": blind_curve,
        "curve_r2_test_b": [
            payload["cells"][f"K{k8}_W{w}"]["obs"]["test_b"][
                "per_observable_recovered_r2"
            ][BLIND_OBS]
            for w in ws
        ],
        "all_observable_curves_all_rows": curve,
        "monotone_tol0.01": monotone,
        "delta_W30_minus_W1_all": t1_ci,
        "delta_W30_minus_W1_test_b": t1_ci_testb,
        "strong": t1_strong,
        "weak": t1_weak,
        "verdict": "STRONG" if t1_strong else ("WEAK" if t1_weak else "FAIL"),
    }
    print(
        f"[track-t] GATE T1 ({BLIND_OBS}): {payload['gates']['T1']['verdict']} "
        f"curve={['%.3f' % v for v in blind_curve]} "
        f"delta(all)={t1_ci['delta_r2']:+.3f} CI{[round(x, 3) for x in t1_ci['ci95']]}",
        flush=True,
    )

    # ---- Gate T2 + T2b selection (train rows only for the selection)
    ref_name = f"K{k8}_W{ws[0]}"  # (K=8, W=1)
    ref_train_r2 = payload["cells"][ref_name]["state_r2"]["train"]["window"]
    tss_state_win = tss_row_fixed(z_pooled, win_mask)
    tss_state_train = tss_row_fixed(z_pooled, win_mask & masks["train"])
    t2_cells = {}
    meets = []
    for k in args.ks:
        for w in args.windows_grid:
            name = f"K{k}_W{w}"
            if name == ref_name:
                continue
            d_all = case_bootstrap_r2_delta(
                stores[name]["se_row"], tss_state_win,
                stores[ref_name]["se_row"], tss_state_win,
                case_id, win_mask, n_boot=args.n_boot, seed=args.seed,
            )
            d_train = case_bootstrap_r2_delta(
                stores[name]["se_row"], tss_state_train,
                stores[ref_name]["se_row"], tss_state_train,
                case_id, win_mask & masks["train"], n_boot=args.n_boot, seed=args.seed,
            )
            t2_cells[name] = {"delta_vs_K8W1_all": d_all, "delta_vs_K8W1_train": d_train}
            if d_train["ci95"][1] >= 0:  # matches or beats the reference within CI (train)
                meets.append((k, w, d_train["delta_r2"]))
    strong_cells = [
        (k, w)
        for (k, w, _) in meets
        if k <= 2 and w >= 8
        and t2_cells[f"K{k}_W{w}"]["delta_vs_K8W1_all"]["ci95"][1] >= 0
    ]
    t2_strong = bool(strong_cells)
    # weak: recovery non-decreasing in W at every K and non-decreasing in K at every W
    surf = {
        (k, w): payload["cells"][f"K{k}_W{w}"]["state_r2"]["all"]["window"]
        for k in args.ks for w in args.windows_grid
    }
    ws_sorted = sorted(args.windows_grid)
    mono_w = all(
        surf[(k, b)] >= surf[(k, a)] - 0.02
        for k in args.ks for a, b in zip(ws_sorted, ws_sorted[1:])
    )
    mono_k = all(
        surf[(b, w)] >= surf[(a, w)] - 0.02
        for w in args.windows_grid for a, b in zip(sorted(args.ks), sorted(args.ks)[1:])
    )
    if meets:
        k_min, w_min, _ = sorted(meets, key=lambda t: (t[0], t[1]))[0]
    else:
        k_min = w_min = None
    payload["gates"]["T2"] = {
        "reference_cell": ref_name,
        "reference_train_state_r2": ref_train_r2,
        "cells": t2_cells,
        "strong_cells_K<=2_W>=8": [f"K{k}_W{w}" for k, w in strong_cells],
        "monotone_in_W_tol0.02": mono_w,
        "monotone_in_K_tol0.02": mono_k,
        "strong": t2_strong,
        "weak": bool(mono_w and mono_k),
        "verdict": "STRONG" if t2_strong else ("WEAK" if (mono_w and mono_k) else "FAIL"),
    }
    payload["t2b_selection"] = {
        "basis": "TRAIN rows only (selection discipline, D239)",
        "target": f"state_r2(train, window) of {ref_name} = {ref_train_r2:.4f}",
        "K_min": k_min,
        "W_min": w_min,
        "note": "smallest K, then smallest W, whose train delta CI upper >= 0",
    }
    print(
        f"[track-t] GATE T2: {payload['gates']['T2']['verdict']} "
        f"strong_cells={payload['gates']['T2']['strong_cells_K<=2_W>=8']} "
        f"T2b pick: K_min={k_min} W_min={w_min}",
        flush=True,
    )

    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[track-t] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
