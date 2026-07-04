"""Track O1 -- static wall-pressure -> state recovery (model-conditioned OSP).

REVISED per Carlos's two directives:
  D1  MODEL-CONDITIONED optimal sensor placement: each model selects its own K
      taps by TCSI greedy forward selection on the first PC of its pooled latent
      (scripts/session32/osp_select.py). The target-blind shared qDEIM set is kept
      as a comparison. No single array is reused across families.
  D2  Recovery mapping is CV-selected per model+K among {KRR (Nystroem-RBF+Ridge),
      MLP (torch), LSTM (PressureLSTM)}, encounter-grouped 5-fold GroupKFold
      (scripts/session32/recovery_maps.py).

Three state definitions per family remain: pooled z (the estimable coefficient
state), GAP of the spatial latent, and the flattened 9216-d spatial latent
(negative control). For the jepa family pooled = jepa_pool, spatial = jepa_wake;
POD and fukami are pooled-only references whose spatial latent is a broadcast, so
their gap/flat recovery equals the pooled recovery (short-circuited, flagged).

Windowed pressure (W frames) at the model's own OSP K taps -> state, scored on the
frozen impact-union-relaxation window and, separately, the strictly pre-impact
lead-in. The five observables are read through each RECOVERED state via per-fold
frozen linear probes. The previous shared-qDEIM + KRR-only numbers are embedded as
``baseline_qdeim_krr`` so the OSP + LSTM/MLP gain is visible.

Gate O (report, not enforced): strong = pooled recovery at K=8 exceeds flattened-
spatial by delta-R2 >= 0.2 (case-clustered CI), the family ordering (predictive
most recoverable, POD least) replicates on state-R2 AND observable-R2, and the
pre-impact variant preserves/sharpens it.

Run (GPU for MLP/LSTM; CPU for KRR/OSP):
    taskset -c 16-23 python -m scripts.session32.track_o1_recovery
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

OBSERVABLES = ("C_L", "C_D", "wake_enstrophy", "circulation_pos", "circulation_neg")
DEFAULT_KS = (2, 4, 8, 16)
MAPPING_NAMES = ("krr", "mlp", "lstm")

FAMILIES = {
    "jepa": {"pooled": "jepa_pool", "spatial": "jepa_wake", "role": "predictive"},
    "fukami": {"pooled": "fukami", "spatial": "fukami", "role": "reconstruction"},
    "POD": {"pooled": "pod", "spatial": "pod", "role": "linear_floor"},
}
# models that need a TCSI OSP tap set
OSP_MODELS = ("jepa_pool", "jepa_wake", "pod", "fukami")


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


# --------------------------------------------------------------------------- io
def load_cache(cache_dir: Path, model: str, split: str) -> dict:
    d = np.load(cache_dir / f"latents_{model}_{split}.npz", allow_pickle=True)
    return {
        "z_gap": np.asarray(d["z_gap"], dtype=np.float32),
        "z_spatial": np.asarray(d["z_spatial"], dtype=np.float32),
        "case_id": np.asarray(d["case_id"]).astype(str),
        "encounter_index": np.asarray(d["encounter_index"]).astype(int),
        "frame": np.asarray(d["frame"]).astype(int),
        "window_mask": np.asarray(d["window_mask"], dtype=bool),
        "latent_grid": tuple(int(x) for x in d["latent_grid"]),
        "targets": {o: np.asarray(d[f"target_{o}"], dtype=np.float64) for o in OBSERVABLES},
    }


def load_pressure(cache_dir: Path, split: str) -> dict:
    d = np.load(cache_dir / f"pressure_{split}.npz", allow_pickle=True)
    return {
        "p_wall": np.asarray(d["p_wall"], dtype=np.float32),
        "case_id": np.asarray(d["case_id"]).astype(str),
        "encounter_index": np.asarray(d["encounter_index"]).astype(int),
        "frame": np.asarray(d["frame"]).astype(int),
    }


def keys_of(case_id, enc):
    return np.array([f"{c}/{int(e):02d}" for c, e in zip(case_id, enc)])


# ------------------------------------------------------------------ features
def build_windows_seq(p_rows, keys, frame, taps, w) -> np.ndarray:
    """Windowed pressure sequence (N, W, K): taps over frames [f-w+1, f] (clamped)."""
    n, kk = p_rows.shape[0], len(taps)
    out = np.zeros((n, w, kk), dtype=np.float32)
    p_taps = np.ascontiguousarray(p_rows[:, taps])
    offs = np.arange(w)
    for key in np.unique(keys):
        rows = np.where(keys == key)[0]
        order = np.argsort(frame[rows])
        rows_s = rows[order]
        block = p_taps[rows_s]  # (T, K)
        t_local = np.arange(rows_s.shape[0])
        idx = np.clip(t_local[:, None] - (w - 1) + offs[None, :], 0, None)  # (T, w)
        out[rows_s] = block[idx]  # (T, w, K)
    return out


def preimpact_mask(keys, frame, windows, lead) -> np.ndarray:
    out = np.zeros(keys.shape[0], dtype=bool)
    for i, (k, f) in enumerate(zip(keys, frame)):
        w = windows.get(k)
        if w is None:
            continue
        ti = int(w["t_impact"])
        if (f >= ti - lead) and (f < ti):
            out[i] = True
    return out


# ------------------------------------------------------------------- recovery
def recover_oof(Xseq, state, groups, mappings, *, device, n_components, seed, gap_shape):
    """5-fold GroupKFold OOF for each mapping. Returns per-mapping se_row + GAP-of-
    recovered state (streamed; the 9216-d flat prediction is never stored whole)."""
    from sklearn.model_selection import GroupKFold

    from scripts.session32.recovery_maps import MAPPINGS

    n = Xseq.shape[0]
    w, kk = Xseq.shape[1], Xseq.shape[2]
    Xflat = Xseq.reshape(n, w * kk)
    d_gap = state.shape[1] if gap_shape is None else gap_shape[0]
    res = {
        m: {"se_row": np.zeros(n), "rgap": np.zeros((n, d_gap), dtype=np.float32)} for m in mappings
    }
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(Xseq, state, groups):
        for m in mappings:
            if m == "krr":
                pred_obj = MAPPINGS[m](
                    Xseq[tr], Xflat[tr], state[tr], groups[tr], n_components=n_components, seed=seed
                )
            else:
                pred_obj = MAPPINGS[m](
                    Xseq[tr], Xflat[tr], state[tr], groups[tr], device=device, seed=seed
                )
            pred = pred_obj.predict(Xseq[te], Xflat[te]).astype(np.float64)
            true = state[te].astype(np.float64)
            res[m]["se_row"][te] = ((pred - true) ** 2).sum(axis=1)
            if gap_shape is None:
                res[m]["rgap"][te] = pred.astype(np.float32)
            else:
                dd, hh, ww = gap_shape
                res[m]["rgap"][te] = (
                    pred.reshape(-1, dd, hh, ww).mean(axis=(2, 3)).astype(np.float32)
                )
            del pred_obj, pred, true
    return res


def r2_from_se(se_row, true_state, mask) -> float:
    if mask.sum() == 0:
        return float("nan")
    t = true_state[mask].astype(np.float64)
    mean = t.mean(axis=0, keepdims=True)
    tss = float(((t - mean) ** 2).sum())
    return float(1.0 - float(se_row[mask].sum()) / tss) if tss > 0 else float("nan")


def tss_row_fixed(true_state, mask) -> np.ndarray:
    mean = true_state[mask].astype(np.float64).mean(axis=0, keepdims=True)
    return ((true_state.astype(np.float64) - mean) ** 2).sum(axis=1)


def observable_readout(recovered_gap, true_gap, targets, groups, *, seed):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import GroupKFold

    n = recovered_gap.shape[0]
    gkf = GroupKFold(n_splits=5)
    out = {}
    for obs in OBSERVABLES:
        y = targets[obs]
        yhat_rec = np.zeros(n)
        for tr, te in gkf.split(recovered_gap, y, groups):
            reg = LinearRegression().fit(true_gap[tr], y[tr])
            yhat_rec[te] = reg.predict(recovered_gap[te])
        out[obs] = yhat_rec
    return out


def score_observables(readout, targets, mask) -> dict:
    from src.evaluation.represent import r2_score_np

    rec = {o: r2_score_np(targets[o][mask], readout[o][mask]) for o in OBSERVABLES}
    return {
        "per_observable_recovered_r2": rec,
        "mean_recovered_r2": float(np.mean(list(rec.values()))),
        "n": int(mask.sum()),
    }


def case_bootstrap_r2_delta(se_a, tss_a, se_b, tss_b, case_ids, mask, *, n_boot, seed):
    m = mask
    ci_case = case_ids[m]
    sa, ta, sb, tb = se_a[m], tss_a[m], se_b[m], tss_b[m]
    uc = np.array(sorted(set(ci_case.tolist())))
    by = {c: np.where(ci_case == c)[0] for c in uc}

    def r2(idx, se, tss):
        s, t = se[idx].sum(), tss[idx].sum()
        return 1.0 - s / t if t > 0 else np.nan

    all_idx = np.arange(int(m.sum()))
    point = float(r2(all_idx, sa, ta) - r2(all_idx, sb, tb))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        pick = uc[rng.integers(0, len(uc), size=len(uc))]
        idx = np.concatenate([by[c] for c in pick])
        boots[i] = r2(idx, sa, ta) - r2(idx, sb, tb)
    return {
        "delta_r2": point,
        "ci95": [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))],
        "n_cases": int(len(uc)),
        "n_rows": int(m.sum()),
    }


# ------------------------------------------------------------------- main
def run(
    cache_dir,
    osp_json,
    windows,
    *,
    ks,
    w,
    preimpact_lead,
    n_components,
    n_boot,
    seed,
    device,
    out_json,
    families,
    baseline,
):
    osp = json.loads(_resolve(osp_json).read_text())

    payload = {
        "task": "SESSION 32 Track O1 -- model-conditioned OSP + CV-selected recovery",
        "params": {
            "W_window": w,
            "Ks": list(ks),
            "osp_taps_source": str(osp_json),
            "sensor_placement": "per-model TCSI greedy (D1); qdeim_shared kept as comparison",
            "recovery_mappings": list(MAPPING_NAMES),
            "recovery_selection": "encounter-grouped 5-fold GroupKFold CV, best window state-R2 (D2)",
            "krr_n_components": n_components,
            "window_scoring": "frozen impact-union-relaxation window_mask",
            "preimpact_scoring": f"strictly pre-impact lead-in [t_impact-{preimpact_lead}, t_impact)",
            "observables": list(OBSERVABLES),
            "n_boot_case": n_boot,
            "seed": seed,
            "state_defs": {
                "pooled": "pooled d=32 coefficient state (own OSP taps)",
                "gap_spatial": "GAP of spatial latent (spatial model OSP taps)",
                "flat_spatial": "full 9216-d spatial latent (negative control)",
            },
        },
        "baseline_qdeim_krr": baseline,
        "families": {},
    }

    for fam in families:
        spec = FAMILIES[fam]
        broadcast = spec["pooled"] == spec["spatial"]
        print(
            f"\n[o1] ===== {fam} (pooled={spec['pooled']}, spatial={spec['spatial']}, "
            f"broadcast={broadcast}) =====",
            flush=True,
        )
        pooled = {s: load_cache(cache_dir, spec["pooled"], s) for s in ("train", "test_b")}
        spatial = (
            pooled
            if broadcast
            else {s: load_cache(cache_dir, spec["spatial"], s) for s in ("train", "test_b")}
        )
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
        pre_mask = preimpact_mask(keys, frame, windows, preimpact_lead)
        targets = {o: cat(lambda s: pooled[s]["targets"][o]) for o in OBSERVABLES}

        z_pooled = cat(lambda s: pooled[s]["z_gap"])
        z_gap_sp = cat(lambda s: spatial[s]["z_gap"])
        grid = spatial["train"]["latent_grid"]
        d = z_pooled.shape[1]
        z_flat = cat(lambda s: spatial[s]["z_spatial"]).reshape(len(case_id), -1)

        # state def -> (state array, gap_shape, osp_model)
        state_defs = {
            "pooled": (z_pooled, None, spec["pooled"]),
            "gap_spatial": (z_gap_sp, None, spec["spatial"]),
            "flat_spatial": (z_flat, (d, grid[0], grid[1]), spec["spatial"]),
        }
        active = ["pooled"] if broadcast else ["pooled", "gap_spatial", "flat_spatial"]

        fam_out = {
            "role": spec["role"],
            "pooled_model": spec["pooled"],
            "spatial_model": spec["spatial"],
            "spatial_is_broadcast": broadcast,
            "n_rows_total": int(len(case_id)),
            "n_window_rows": int(win_mask.sum()),
            "n_preimpact_rows": int(pre_mask.sum()),
            "by_K": {},
        }

        for k in ks:
            k_out, ci_store = {}, {}
            seq_cache = {}  # (osp_model, K) -> windows (gap & flat share spatial taps)
            for sname in active:
                state, gap_shape, osp_model = state_defs[sname]
                taps = osp[osp_model][f"K{int(k)}"]
                tkey = (osp_model, int(k))
                if tkey not in seq_cache:
                    seq_cache[tkey] = build_windows_seq(p_rows, keys, frame, taps, w)
                Xseq = seq_cache[tkey]
                res = recover_oof(
                    Xseq,
                    state,
                    groups,
                    MAPPING_NAMES,
                    device=device,
                    n_components=n_components,
                    seed=seed,
                    gap_shape=gap_shape,
                )
                per_map = {}
                for m in MAPPING_NAMES:
                    per_map[m] = {
                        "state_r2_window": r2_from_se(res[m]["se_row"], state, win_mask),
                        "state_r2_preimpact": r2_from_se(res[m]["se_row"], state, pre_mask),
                    }
                cv_pick = max(MAPPING_NAMES, key=lambda mm: per_map[mm]["state_r2_window"])
                readout = observable_readout(
                    res[cv_pick]["rgap"],
                    (z_pooled if sname == "pooled" else z_gap_sp),
                    targets,
                    groups,
                    seed=seed,
                )
                k_out[sname] = {
                    "osp_model": osp_model,
                    "taps": taps,
                    "per_mapping": per_map,
                    "cv_pick": cv_pick,
                    "state_r2_window": per_map[cv_pick]["state_r2_window"],
                    "state_r2_preimpact": per_map[cv_pick]["state_r2_preimpact"],
                    "obs_window": score_observables(readout, targets, win_mask),
                    "obs_preimpact": score_observables(readout, targets, pre_mask),
                    "state_dim": int(state.shape[1]),
                }
                if int(k) == 8:
                    ci_store[sname] = {
                        "se_row": res[cv_pick]["se_row"],
                        "tss_win": tss_row_fixed(state, win_mask),
                        "tss_pre": tss_row_fixed(state, pre_mask),
                    }
                maps_str = ",".join(
                    f"{m}={per_map[m]['state_r2_window']:.2f}" for m in MAPPING_NAMES
                )
                print(
                    f"[o1] {fam} K{k} {sname:12s} pick={cv_pick:4s} "
                    f"stR2 win/pre={k_out[sname]['state_r2_window']:.3f}/"
                    f"{k_out[sname]['state_r2_preimpact']:.3f} "
                    f"obsR2 win/pre={k_out[sname]['obs_window']['mean_recovered_r2']:.3f}/"
                    f"{k_out[sname]['obs_preimpact']['mean_recovered_r2']:.3f} "
                    f"(maps win: {maps_str})",
                    flush=True,
                )
                del res
            # broadcast references: gap/flat recovery equals pooled (spatial is tiled)
            if broadcast:
                k_out["gap_spatial"] = {
                    **k_out["pooled"],
                    "note": "spatial is broadcast of pooled; gap==pooled",
                }
                k_out["flat_spatial"] = {
                    **k_out["pooled"],
                    "note": "spatial is broadcast of pooled; flat R2==pooled",
                }
                if int(k) == 8:
                    ci_store["gap_spatial"] = ci_store["pooled"]
                    ci_store["flat_spatial"] = ci_store["pooled"]
            if int(k) == 8:
                gate = {}
                for ref in ("gap_spatial", "flat_spatial"):
                    gate[f"pooled_vs_{ref}"] = {
                        "delta_r2_window": case_bootstrap_r2_delta(
                            ci_store["pooled"]["se_row"],
                            ci_store["pooled"]["tss_win"],
                            ci_store[ref]["se_row"],
                            ci_store[ref]["tss_win"],
                            case_id,
                            win_mask,
                            n_boot=n_boot,
                            seed=seed,
                        ),
                        "delta_r2_preimpact": case_bootstrap_r2_delta(
                            ci_store["pooled"]["se_row"],
                            ci_store["pooled"]["tss_pre"],
                            ci_store[ref]["se_row"],
                            ci_store[ref]["tss_pre"],
                            case_id,
                            pre_mask,
                            n_boot=n_boot,
                            seed=seed,
                        ),
                    }
                k_out["gateO_K8"] = gate
                gw = gate["pooled_vs_flat_spatial"]["delta_r2_window"]
                print(
                    f"[o1] {fam} GATE-O K8 pooled-vs-flat deltaR2 win={gw['delta_r2']:+.3f} "
                    f"CI{[round(x, 3) for x in gw['ci95']]}",
                    flush=True,
                )
            fam_out["by_K"][str(int(k))] = k_out
        payload["families"][fam] = fam_out
        _resolve(out_json).write_text(json.dumps(payload, indent=2))
        del pooled, spatial, pres

    _gate_summary(payload)
    _resolve(out_json).write_text(json.dumps(payload, indent=2))
    print(f"\n[o1] wrote {_resolve(out_json)}", flush=True)
    return payload


def _gate_summary(payload):
    print("\n[o1] ===== GATE O SUMMARY (K=8, CV-selected recovery) =====", flush=True)
    st, ob = {}, {}
    for fam, f in payload["families"].items():
        k8 = f["by_K"]["8"]
        st[fam] = k8["pooled"]["state_r2_window"]
        ob[fam] = k8["pooled"]["obs_window"]["mean_recovered_r2"]
        g = k8.get("gateO_K8", {}).get("pooled_vs_flat_spatial", {})
        dw = g.get("delta_r2_window", {})
        dp = g.get("delta_r2_preimpact", {})
        base = payload["baseline_qdeim_krr"].get("K8", {}).get(fam, {})
        print(
            f"  {fam:8s} pick={k8['pooled']['cv_pick']:4s} pooled stR2={st[fam]:.3f} "
            f"obsR2={ob[fam]:.3f} | deltaR2(pooled-flat) win={dw.get('delta_r2', float('nan')):+.3f} "
            f"CI{[round(x, 3) for x in dw.get('ci95', [float('nan')] * 2)]} "
            f"pre={dp.get('delta_r2', float('nan')):+.3f} || base(qdeim+krr) "
            f"stR2={base.get('pooled_state_r2_window', float('nan')):.3f} "
            f"obsR2={base.get('pooled_obs_r2_window', float('nan')):.3f}",
            flush=True,
        )
    print(f"  state-R2 ranking:      {sorted(st, key=lambda k: st[k], reverse=True)}", flush=True)
    print(f"  observable-R2 ranking: {sorted(ob, key=lambda k: ob[k], reverse=True)}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Track O1 model-conditioned OSP recovery")
    ap.add_argument("--cache-dir", default="outputs/session31/q1_latents")
    ap.add_argument("--osp-taps", default="outputs/session32/osp_taps_v2p2.json")
    ap.add_argument("--qdeim-taps", default="outputs/session32/qdeim_taps_v2p2.json")
    ap.add_argument("--windows", default="outputs/session31/windows_v2p2.json")
    ap.add_argument("--baseline", default="outputs/session32/_baseline_block.json")
    ap.add_argument("--out", default="outputs/session32/track_o1_recovery.json")
    ap.add_argument("--ks", nargs="+", type=int, default=list(DEFAULT_KS))
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--preimpact-lead", type=int, default=8)
    ap.add_argument("--n-components", type=int, default=400)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--families", nargs="+", default=list(FAMILIES))
    ap.add_argument(
        "--mapping",
        choices=["krr", "mlp", "lstm", "lstm_tuned"],
        default=None,
        help="Force a single recovery estimator for every family (no CV select), "
        "e.g. --mapping lstm for a consistent LSTM wall-recovery table.",
    )
    ap.add_argument(
        "--build-osp",
        action="store_true",
        help="(Re)build osp_taps_v2p2.json from the caches before recovery.",
    )
    args = ap.parse_args(argv)

    if args.mapping is not None:
        global MAPPING_NAMES
        MAPPING_NAMES = (args.mapping,)
        print(f"[o1] forcing single recovery estimator: {args.mapping}", flush=True)

    import torch  # noqa: F401

    from src.evaluation.rom_eval import load_windows
    from src.utils.device import require_rtx6000

    device = require_rtx6000(gpu_index=args.gpu)
    gpu_name = torch.cuda.get_device_name(device.index)
    if "RTX" not in gpu_name or "6000" not in gpu_name:
        raise RuntimeError(f"hardware policy: gpu_name={gpu_name!r} not RTX 6000")
    print(f"[o1] device={device} ({gpu_name})", flush=True)

    windows = load_windows(_resolve(args.windows))
    cache_dir = _resolve(args.cache_dir)

    if args.build_osp or not _resolve(args.osp_taps).exists():
        from scripts.session32.osp_select import build_osp_taps, save_osp_taps

        qdeim = json.loads(_resolve(args.qdeim_taps).read_text())
        model_caches = {m: load_cache(cache_dir, m, "train") for m in OSP_MODELS}
        pres_train = load_pressure(cache_dir, "train")["p_wall"]
        osp_payload = build_osp_taps(
            model_caches,
            windows,
            pres_train,
            w=int(args.window),
            qdeim_taps=qdeim,
            seed=int(args.seed),
        )
        save_osp_taps(osp_payload, args.osp_taps)
        print(f"[o1] wrote {_resolve(args.osp_taps)}", flush=True)
        del model_caches

    baseline = (
        json.loads(_resolve(args.baseline).read_text()) if _resolve(args.baseline).exists() else {}
    )

    run(
        cache_dir,
        _resolve(args.osp_taps),
        windows,
        ks=tuple(int(k) for k in args.ks),
        w=int(args.window),
        preimpact_lead=int(args.preimpact_lead),
        n_components=int(args.n_components),
        n_boot=int(args.n_boot),
        seed=int(args.seed),
        device=device,
        out_json=_resolve(args.out),
        families=tuple(args.families),
        baseline=baseline,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
