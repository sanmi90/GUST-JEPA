"""Harvest every v3 paper-bound number into outputs/session33/numbers_parts/.

SESSION_33_MANUSCRIPT_V3.md Phase 4 (numbers freeze). Each part file follows the
session28 accretion schema ({"part": name, "numbers": {NAME: RECORD}});
scripts/session33/eval_all_v3.py merges + validates (incl. the v2.1 macro
collision check) and emit_macros_v3.py renders paper/macros_v3.tex.

Macros are assigned to the headline set the prose is certain to cite; table
cells carry values (and can gain macros in the L5 table pass without changing
any frozen value). Parts whose inputs do not exist yet (training-dependent:
dim plateau, min-d, seed band) are skipped with a warning and appear on re-run.

The final VERIFY block cross-checks the frozen anchors against the Session 32
report values (v3 doc tables); any mismatch exits nonzero and stops Phase 4.

Run (CPU):
    python -m scripts.session33.emit_numbers_parts
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PARTS = REPO / "outputs" / "session33" / "numbers_parts"
S31 = REPO / "outputs" / "session31"
S32 = REPO / "outputs" / "session32"
S33 = REPO / "outputs" / "session33"

GWORD = {"0": "Zero", "0.25-0.5": "Half", "1": "One", "1.5": "OneHalf",
         "2": "Two", "3": "Three", "4": "Four"}
KWORD = {1: "One", 2: "Two", 4: "Four", 8: "Eight"}
WWORD = {1: "One", 2: "Two", 4: "Four", 8: "Eight", 16: "Sixteen", 30: "Thirty"}


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def write_part(name: str, numbers: dict) -> None:
    PARTS.mkdir(parents=True, exist_ok=True)
    (PARTS / f"{name}.json").write_text(
        json.dumps({"part": name, "numbers": numbers}, indent=2, default=float)
    )
    n_mac = sum(1 for r in numbers.values() if r.get("macro"))
    print(f"[parts] {name}: {len(numbers)} numbers ({n_mac} macro-bound)")


def rec(value, macro=None, fmt="%.2f", **kw):
    r = {"value": value}
    if macro:
        r["macro"] = macro
    r["fmt"] = fmt
    r.update(kw)
    return r


# ------------------------------------------------------------------ constants
def part_constants():
    write_part("constants", {
        "train_std_v2p2": rec(3.5396, "TrainStd", "%.4f", source="v2p2 manifest"),
        "ssim_L_v2p2": rec(8.487, "SsimL", "%.3f", source="configs/ssim_data_range.json"),
        "lambda_sigreg": rec(0.02, "LambdaSigreg", "%.2f", source="D238/dda57b7"),
        "n_cases": rec(102, "NCases", "%d"),
        "n_encounters": rec(450, "NEncounters", "%d"),
        "n_train_cases": rec(84, "NTrainCases", "%d"),
        "n_testb_cases": rec(10, "NTestBCases", "%d"),
        "n_testc_cases": rec(8, "NTestCCases", "%d"),
        "n_train_encounters": rec(268, "NTrainEnc", "%d"),
        "n_val_encounters": rec(100, "NValEnc", "%d"),
        "n_testb_encounters": rec(42, "NTestBEnc", "%d"),
        "n_testc_encounters": rec(40, "NTestCEnc", "%d"),
        "n_testb_interior": rec(6, "NTestBInterior", "%d"),
        "n_testb_boundary": rec(4, "NTestBBoundary", "%d"),
        "latent_dim": rec(32, "LatentDim", "%d"),
        "enc_params_M": rec(6.7, "EncParams", "%.1f",
                            note="counted from jepa_pool_vec checkpoint_iter010000 (6.67M); "
                                 "encoder size unchanged vs the ResUNet-era flagship"),
        "pred_params_M": rec(10.7, "PredParams", "%.1f",
                             note="D250 native vector predictor: AutoregressivePredictor "
                                  "cond_dim=0, counted from jepa_pool_vec checkpoint (10.66M)"),
        "filter_taps": rec(8, "FilterTaps", "%d"),
        "filter_members": rec(64, "FilterMembers", "%d"),
        "assim_pre": rec(24, "AssimPre", "%d"),
        "assim_post": rec(48, "AssimPost", "%d"),
        "dt_tc": rec(0.05, "DtTc", "%.2f"),
    })


# ------------------------------------------------------------------ Table X
def part_table_x():
    q1p = _load(S32 / "q1_pool.json")
    q1a = _load(S31 / "q1_ablation.json")
    q1r = _load(S31 / "q1_reference.json")
    gates = _load(S32 / "track_p_gates.json")
    q2r = _load(S31 / "q2_reference.json")
    # D250: the flagship is jepa_pool_vec (native vector predictor). Its readability,
    # forecast merit and decode floor come from the vec eval, NOT the ResUNet-era
    # jepa_pool row of q1_ablation/track_p_gates.
    q1v = _load(S33 / "q1_vec.json")
    q2v = _load(S33 / "q2_vec.json")
    q2vn = _load(S33 / "q2_vec_native.json")
    if not all([q1p, q1a, q1r, gates, q1v, q2v]):
        print("[parts] table_x: missing inputs, skipped")
        return

    # D250 merit yardstick: the closure-table forecast merit is switched to a
    # matched autoregressive transformer (the flagship's training class) for EVERY
    # family, so no latent is probed off-class. Merged from the three merit runs.
    _tf_files = [
        _load(S33 / "q2_transformer_merit.json"),
        _load(S33 / "q2_transformer_merit_pool2.json"),
        _load(S33 / "q2_transformer_merit_ref.json"),
    ]
    _tf_models = {}
    for f in _tf_files:
        if f:
            _tf_models.update(f.get("models", {}))

    def tf_merit(model):
        """Matched-transformer merit (mean 5-obs MLP R2, H=8) or None if absent."""
        mv = _tf_models.get(model)
        if not mv or "transformer_matched" not in mv.get("predictors", {}):
            return None
        oc = mv["predictors"]["transformer_matched"]["observable_closure"]
        i = list(oc["C_L"]["horizons"]).index(8)
        ts = ("C_L", "C_D", "wake_enstrophy", "circulation_pos", "circulation_neg")
        return float(np.mean([oc[t]["model_mlp_r2"][i] for t in ts]))

    def wake(q1, m):
        return float(q1["models"][m]["probes"]["windowed"]["wake_enstrophy"]["linear_r2"])

    def vec_merit(q2, model, predictor):
        """Mean 5-obs MLP R2 at H=8 for a vec model under a named forecast operator."""
        oc = q2["models"][model]["predictors"][predictor]["observable_closure"]
        i = list(oc["C_L"]["horizons"]).index(8)
        ts = ("C_L", "C_D", "wake_enstrophy", "circulation_pos", "circulation_neg")
        return float(np.mean([oc[t]["model_mlp_r2"][i] for t in ts]))

    def vec_cell(q2, q2n, q1, model):
        """Flagship closure cell: matched-ResUNet merit (comparable to the baselines,
        all probed by the same off-class operator), the native as-built merit, field
        VRMSE (ResUNet operator), C_L closure and decode-floor SSIM, all on vec."""
        pr = q2["models"][model]["predictors"]["resunet_matched"]
        oc = pr["observable_closure"]
        i = list(oc["C_L"]["horizons"]).index(8)
        fv = pr["field_vrmse"]
        fi = list(fv["horizons"]).index(8)
        native = None
        if q2n and model in q2n.get("models", {}):
            native = vec_merit(q2n, model, "native")
        return {
            "merit_matched": vec_merit(q2, model, "resunet_matched"),
            "merit_native": native,
            "vrmse": float(fv["model"][fi]),
            "cl": float(oc["C_L"]["model_mlp_r2"][i]),
            "ssim": float(q1["models"][model]["decode_floor"]["windowed"]["ssim"]),
        }

    def q2_ref(m):
        """Reference merit (mean 5-obs mlp R2, h8) and field VRMSE (h8) on v2.2."""
        if not q2r or m not in q2r.get("models", {}):
            return None, None
        pred = q2r["models"][m]["predictors"]["resunet_matched"]
        oc = pred["observable_closure"]
        hs = list(oc["C_L"]["horizons"])
        i = hs.index(8)
        ts = ("C_L", "C_D", "wake_enstrophy", "circulation_pos", "circulation_neg")
        merit = float(np.mean([oc[t]["model_mlp_r2"][i] for t in ts]))
        fv = pred["field_vrmse"]
        fhs = list(fv["horizons"])
        vrmse = float(fv["model"][fhs.index(8)]) if 8 in fhs else None
        return merit, vrmse

    cells = gates["pooled_cells"]
    # rows: (paper name, wake source, cells key or None).
    # The flagship JepaWake reads the vec eval (readability from q1_vec, closure
    # cell from q2_vec); the baselines keep the frozen ResUNet-era pooled cells.
    rows = {
        "JepaWake": (wake(q1v, "jepa_pool_vec"), "VEC"),
        "SupOnly": (wake(q1p, "supervised_only_pool"), "supervised_only_pool"),
        "AeWake": (wake(q1p, "ae_wake_pool"), "ae_wake_pool"),
        "JepaNowake": (wake(q1p, "jepa_nowake_pool"), "jepa_nowake_pool"),
        "AeNowake": (wake(q1p, "ae_nowake_pool"), "ae_nowake_pool"),
        "RegAE": (wake(q1p, "regAE_pool"), "regAE_pool"),
        "Bvae": (wake(q1p, "bvae"), "bvae"),
        "Fukami": (wake(q1r, "fukami"), None),
        "FukamiWake": (wake(q1r, "fukami_wake"), None),
        "Pod": (wake(q1r, "pod"), None),
    }
    numbers = {}
    for name, (w, ck) in rows.items():
        numbers[f"x_wake_{name}"] = rec(
            w, f"Xwake{name}", "%.3f", split="test_b", observable="wake_enstrophy",
            probe="linear windowed",
        )
        # Merit column = the STABLE common ResUNet yardstick (a matched transformer
        # is unstable across off-class latents, e.g. bvae merit -1.375, so it is not
        # a usable common operator; D250 evidence). The flagship's own transformer
        # and native numbers are carried as separate evidence macros.
        if ck == "VEC":
            c = vec_cell(q2v, q2vn, q1v, "jepa_pool_vec")
            numbers[f"x_merit_{name}"] = rec(
                c["merit_matched"], f"Xmerit{name}", "%.3f", horizon=8,
                note="common matched-ResUNet yardstick; off-class for the transformer flagship")
            if c["merit_native"] is not None:
                numbers[f"x_merit_{name}_native"] = rec(
                    c["merit_native"], f"Xmerit{name}Native", "%.3f", horizon=8,
                    note="as-built ROM: the flagship's own co-trained vector predictor")
            tfm = tf_merit("jepa_pool_vec")
            if tfm is not None:
                numbers[f"x_merit_{name}_tf"] = rec(
                    tfm, f"Xmerit{name}Tf", "%.3f", horizon=8,
                    note="matched transformer of the training class on the frozen latent")
            numbers[f"x_vrmse_{name}"] = rec(c["vrmse"], f"Xvrmse{name}", "%.3f", horizon=8)
            numbers[f"x_ssim_{name}"] = rec(c["ssim"], f"Xssim{name}", "%.3f")
            numbers[f"x_cl_{name}"] = rec(c["cl"], f"Xcl{name}", "%.3f", horizon=8)
        elif ck and ck in cells:
            numbers[f"x_merit_{name}"] = rec(
                cells[ck]["merit_mean_obs_h8"], f"Xmerit{name}", "%.3f", horizon=8)
            numbers[f"x_vrmse_{name}"] = rec(
                cells[ck]["field_vrmse_model_h8"], f"Xvrmse{name}", "%.3f", horizon=8)
            numbers[f"x_ssim_{name}"] = rec(cells[ck]["floor_ssim"], f"Xssim{name}", "%.3f")
            numbers[f"x_cl_{name}"] = rec(
                cells[ck]["cl_closure_mlp_h8"], f"Xcl{name}", "%.3f", horizon=8)
    for name in ("Fukami", "FukamiWake", "Pod"):
        m = {"Fukami": "fukami", "FukamiWake": "fukami_wake", "Pod": "pod"}[name]
        mr, vr = q2_ref(m)
        if mr is not None:
            numbers[f"x_merit_{name}"] = rec(mr, f"Xmerit{name}", "%.3f", horizon=8)
        if vr is not None:
            numbers[f"x_vrmse_{name}"] = rec(vr, f"Xvrmse{name}", "%.3f", horizon=8)
        if q1r and m in q1r["models"]:
            numbers[f"x_ssim_{name}"] = rec(
                float(q1r["models"][m]["decode_floor"]["windowed"]["ssim"]),
                f"Xssim{name}", "%.3f")
    write_part("table_x", numbers)


# ------------------------------------------------------------------ P gates + Table Y
def part_gates_p():
    gates = _load(S32 / "track_p_gates.json")
    # D250: null-space mechanism (Table Y) re-derived on the vec flagship.
    p3 = _load(S33 / "track_p3_mechanism.json") or _load(S32 / "track_p3_mechanism.json")
    if not gates or not p3:
        print("[parts] gates_p: missing inputs, skipped")
        return
    numbers = {}
    p1 = gates.get("P1_readability", {})
    if "primary_wake_enstrophy" in p1:
        pw = p1["primary_wake_enstrophy"]
        numbers["p1_wake_delta"] = rec(
            pw["delta_a_minus_b"], "PoneWakeDelta", "%.3f",
            ci_lo=pw["ci95"][0], ci_hi=pw["ci95"][1],
            note="supervised_only - jepa_wake, windowed test_b",
        )
    # Table Y: condition numbers (empirical eigenvalues) + near-null fractions
    # (fractions live in comparisons.empirical: regAE = cand_frac, others =
    # comp_frac of their respective comparison rows).
    comp = p3.get("comparisons", {}).get("empirical", {})
    fracs = {}
    c_jw = comp.get("regAE_pool_vs_jepa_wake_pool", {})
    c_so = comp.get("regAE_pool_vs_supervised_only_pool", {})
    if c_jw:
        fracs["RegAE"] = c_jw["cand_frac"]
        fracs["JepaWake"] = c_jw["comp_frac"]
    if c_so:
        fracs["SupOnly"] = c_so["comp_frac"]
    for m, short in (("regAE_pool", "RegAE"), ("jepa_wake_pool", "JepaWake"),
                     ("supervised_only_pool", "SupOnly")):
        mm = p3["models"].get(m)
        if mm:
            est = mm["estimators"]["empirical"]
            ev = est.get("eigenvalues_ascending")
            cond = est.get("condition_number")
            if cond is None and ev:
                cond = float(ev[-1] / max(ev[0], 1e-12))
            numbers[f"y_cond_{short}"] = rec(cond, f"Ycond{short}", "%.0f")
        if short in fracs:
            numbers[f"y_nearnull_{short}"] = rec(fracs[short], f"Ynearnull{short}", "%.3f")
    for key, short in (("regAE_pool_vs_jepa_wake_pool", "JepaWake"),
                       ("regAE_pool_vs_supervised_only_pool", "SupOnly")):
        c = comp.get(key)
        if c:
            numbers[f"p3_delta_vs_{short}"] = rec(
                c["delta_cand_minus_comp"], f"PthreeDelta{short}", "%.3f",
                ci_lo=c["ci95"][0], ci_hi=c["ci95"][1])
    numbers["y_isotropic_null"] = rec(0.25, "YisoNull", "%.2f", note="k/d = 8/32")
    # P4 pooling cost (headline: jepa family)
    p4 = gates.get("P4_pooling_cost", {}).get("jepa_wake_pool")
    if p4 and "decode_ssim" in p4:
        numbers["p4_ssim_delta_jepa"] = rec(
            p4["decode_ssim"]["delta"], "PfourSsimDeltaJepa", "%.3f")
        numbers["p4_merit_delta_jepa"] = rec(
            p4["merit_h8"]["delta"], "PfourMeritDeltaJepa", "%.3f")
    write_part("gates_p_and_table_y", numbers)


# ------------------------------------------------------------------ Table Z (H_roll)
def part_table_z():
    # D250 flagship: H_roll ablation re-run on the vec pipeline (jepa_nowake_pool_vec
    # H=8 vs its H=1 twin). Falls back to the session32 ResUNet-era file if absent.
    hr = _load(S33 / "hroll_ablation.json") or _load(S32 / "hroll_ablation.json")
    if not hr:
        print("[parts] table_z: missing inputs, skipped")
        return
    a = hr["axis_a_forecast"]
    numbers = {}
    for hkey, suff in (("H_roll_1", "Hone"), ("H_roll_8", "Height")):
        blk = a[hkey]
        numbers[f"z_merit_{suff}"] = rec(
            blk["merit_mean_obs_mlp_h8"], f"Zmerit{suff}", "%.3f", horizon=8)
        numbers[f"z_cl_{suff}"] = rec(
            blk["cl_closure_mlp_h8"], f"Zcl{suff}", "%.3f", horizon=8)
        numbers[f"z_wake_{suff}"] = rec(
            blk["wake_enstrophy_closure_mlp_h8"], f"Zwake{suff}", "%.3f", horizon=8)
        numbers[f"z_vrmse_{suff}"] = rec(
            blk["field_vrmse_model_h8"], f"Zvrmse{suff}", "%.3f", horizon=8)
        numbers[f"z_drift_{suff}"] = rec(
            blk["drift_rel_l2_h16"], f"Zdrift{suff}", "%.3f", horizon=16)
        numbers[f"z_merit_hsixteen_{suff}"] = rec(
            blk["merit_curve"]["16"], f"ZmeritSixteen{suff}", "%.3f", horizon=16)
    write_part("table_z_hroll", numbers)


# ------------------------------------------------------------------ Table W + Gate O
def part_table_w():
    o1 = _load(S32 / "track_o1_recovery.json")
    # D250 flagship: the vec 3-seed O1 band replaces the single ResUNet-era "jepa"
    # recovery. The fukami / POD baselines stay from the frozen session32 O1.
    o1v = {
        0: _load(S33 / "track_o1_recovery_vec.json"),
        1: _load(S33 / "track_o1_recovery_vec_s1.json"),
        2: _load(S33 / "track_o1_recovery_vec_s2.json"),
    }
    if not o1 or not all(o1v.values()):
        print("[parts] table_w: missing inputs, skipped")
        return
    numbers = {}

    # ---- flagship recovery as a 3-seed band (window + preimpact + mean obs) ----
    vec_fams = {0: "jepa_vec", 1: "jepa_pool_vec_s1", 2: "jepa_pool_vec_s2"}
    st = [o1v[s]["families"][vec_fams[s]]["by_K"]["8"]["pooled"] for s in (0, 1, 2)]
    st_win = np.array([p["state_r2_window"] for p in st])
    st_obs = np.array([p["obs_window"]["mean_recovered_r2"] for p in st])
    numbers["w_state_Jepa"] = rec(
        float(st_win.mean()), "WstateJepa", "%.3f",
        note="3-seed mean, jepa_pool_vec O1 pooled state R2 (K8 window)")
    numbers["w_state_Jepa_sd"] = rec(float(st_win.std(ddof=1)), "WstateJepaSd", "%.3f")
    numbers["w_obs_Jepa"] = rec(float(st_obs.mean()), "WobsJepa", "%.3f")
    numbers["w_pick_Jepa"] = rec(st[0]["cv_pick"], "WpickJepa", "%s")

    # ---- baselines unchanged (frozen session32 O1) ----
    for fam, short in (("fukami", "Fukami"), ("POD", "Pod")):
        k8 = o1["families"][fam]["by_K"]["8"]["pooled"]
        numbers[f"w_state_{short}"] = rec(k8["state_r2_window"], f"Wstate{short}", "%.3f")
        numbers[f"w_obs_{short}"] = rec(
            k8["obs_window"]["mean_recovered_r2"], f"Wobs{short}", "%.3f")
        numbers[f"w_pick_{short}"] = rec(k8["cv_pick"], f"Wpick{short}", "%s")

    # Gate O (pooled d=32 vs the full flattened spatial latent) is a pooling-
    # losslessness argument. The vec flagship is natively pooled, so it has no
    # genuine spatial sibling to flatten; the gate is established on the original
    # spatial-latent family (pooled jepa_pool vs flat jepa_wake) and justifies the
    # d=32 pooling the vec flagship adopts. Kept from the frozen session32 O1.
    g = o1["families"]["jepa"]["by_K"]["8"]["gateO_K8"]["pooled_vs_flat_spatial"]
    d = g["delta_r2_window"]
    numbers["gate_o_delta"] = rec(
        d["delta_r2"], "GateODelta", "%.3f", ci_lo=d["ci95"][0], ci_hi=d["ci95"][1],
        note="pooled d=32 vs flat spatial latent, spatial-latent family; motivates d=32")

    # shared target-blind (qDEIM + KRR) placement-robustness baseline. The flagship
    # entry stays on the genuine spatial-latent family (jepa) for the same reason;
    # refs frozen.
    base = o1.get("baseline_qdeim_krr", {}).get("K8", {})
    for fam, s in (("jepa", "Jepa"), ("fukami", "Fukami"), ("POD", "Pod")):
        blk = base.get(fam, base.get(fam.lower(), {}))
        if "pooled_state_r2_window" in blk:
            numbers[f"wq_state_{s}"] = rec(
                blk["pooled_state_r2_window"], f"WqState{s}", "%.3f",
                note="shared qDEIM taps + KRR baseline")
            numbers[f"wq_obs_{s}"] = rec(blk["pooled_obs_r2_window"], f"WqObs{s}", "%.3f")
    write_part("table_w_gate_o", numbers)


# ------------------------------------------------------------------ Table V (envelope)
def part_table_v():
    # D250 flagship: the envelope is the frozen-filter run on jepa_pool_vec.
    env = _load(S33 / "envelope_vec.json")
    if not env:
        print("[parts] table_v: missing inputs, skipped")
        return
    m = env["models"]["jepa_pool_vec"]
    byg = m["aggregates"]["by_G"]
    numbers = {}
    for g, word in GWORD.items():
        if g not in byg or byg[g].get("n", 0) == 0:
            continue
        blk = byg[g]
        numbers[f"v_filt_cl_{word}"] = rec(
            blk["filter_CL_analysis_r2_impact"]["median"], f"VfiltCLG{word}", "%.2f",
            note="median analysis C_L R2, impact")
        numbers[f"v_div_{word}"] = rec(blk["div_rate"], f"VdivG{word}", "%.2f")
        numbers[f"v_rec_cl_{word}"] = rec(
            blk["recovery_CL_r2_impact"]["median"], f"VrecCLG{word}", "%.2f")
        numbers[f"v_nis_{word}"] = rec(
            blk["filter_mean_nis"]["mean"], f"VnisG{word}", "%.1f")
    thr = m["thresholds"]
    for dname, word in (("0.5", "DHalf"), ("1.0", "DOne"), ("1.5", "DOneHalf")):
        t = thr.get(dname, {})
        v = t.get("div_rate_exceeds_50pct_at_absG")
        if v is not None:
            numbers[f"v_divthresh_{word}"] = rec(v, f"VdivThresh{word}", "%.1f")
    write_part("table_v_envelope", numbers)


# ------------------------------------------------------------------ Track T grid (T1/T2)
def part_track_t():
    # D250 flagship: Track T grid re-run on jepa_pool_vec latents.
    grid = _load(S33 / "track_t_grid_vec.json")
    if not grid:
        print("[parts] track_t: missing inputs, skipped")
        return
    numbers = {}
    ws = sorted(grid["params"]["grid_Ws"])
    ks = sorted(grid["params"]["grid_Ks"])
    # T1 (K=8): per-observable test_b curves
    obs_short = {"C_L": "CL", "wake_enstrophy": "Ew",
                 "circulation_neg": "CircNeg", "circulation_pos": "CircPos",
                 "C_D": "CD"}
    for w in ws:
        cell = grid["cells"][f"K8_W{w}"]
        for obs, short in obs_short.items():
            v = cell["obs"]["test_b"]["per_observable_recovered_r2"][obs]
            numbers[f"t1_{short}_W{w}"] = rec(
                v, f"Tone{short}W{WWORD[w]}", "%.2f", split="test_b")
        numbers[f"t1_state_W{w}"] = rec(
            cell["state_r2"]["test_b"]["window"], f"ToneStateW{WWORD[w]}", "%.2f",
            split="test_b")
    g1 = grid["gates"]["T1"]["delta_W30_minus_W1_all"]
    numbers["t1_circ_delta"] = rec(
        g1["delta_r2"], "ToneCircDelta", "%.3f", ci_lo=g1["ci95"][0], ci_hi=g1["ci95"][1])
    numbers["t1_verdict"] = rec(grid["gates"]["T1"]["verdict"], "ToneVerdict", "%s")
    # T2 grid: state test_b matrix
    for k in ks:
        for w in ws:
            cell = grid["cells"][f"K{k}_W{w}"]
            numbers[f"t2_K{k}_W{w}"] = rec(
                cell["state_r2"]["test_b"]["window"],
                f"Ttwo{KWORD[k]}W{WWORD[w]}", "%.2f", split="test_b")
    sel = grid["t2b_selection"]
    numbers["t2_kmin"] = rec(sel["K_min"], "TtwoKmin", "%d")
    numbers["t2_wmin"] = rec(sel["W_min"], "TtwoWmin", "%d")
    numbers["t2_verdict"] = rec(grid["gates"]["T2"]["verdict"], "TtwoVerdict", "%s")
    br = grid.get("bridge_osp")
    if br:
        numbers["t_bridge_state"] = rec(
            br["state_r2"]["all"]["window"], "TbridgeState", "%.3f",
            note="OSP jepa_pool_vec K8 W30; reconciles with the vec O1 recovery band")
    mi = grid.get("mi_stride_crosscheck", {})
    if mi.get("mean_tau_first_min"):
        numbers["t_mi_tau"] = rec(mi["mean_tau_first_min"], "TmiTau", "%.0f",
                                  unit="frames")
    write_part("track_t_grid", numbers)


# ------------------------------------------------------------------ T2b filter
def part_t2b():
    t2b = _load(S33 / "t2b_reduced_filter.json")
    if not t2b:
        print("[parts] t2b: missing inputs, skipped")
        return
    numbers = {}
    for k in ("1", "2", "4"):
        blk = t2b["by_K"].get(k)
        if not blk:
            continue
        for g in ("1", "1.5", "2", "3", "4"):
            p = blk["paired_vs_frozen_K8"][g]["CL_analysis_r2_impact"]
            if p.get("n_pairs", 0):
                numbers[f"t2b_K{k}_d_{GWORD[g]}"] = rec(
                    p["mean_delta"], f"TtwobK{KWORD[int(k)]}DeltaG{GWORD[g]}", "%.2f",
                    ci_lo=p["ci95"][0], ci_hi=p["ci95"][1])
            agg = blk["aggregates"]["by_G"].get(g, {})
            med = agg.get("filter_CL_analysis_r2_impact", {}).get("median")
            if med is not None:
                numbers[f"t2b_K{k}_med_{GWORD[g]}"] = rec(
                    med, f"TtwobK{KWORD[int(k)]}MedG{GWORD[g]}", "%.2f")
    gate = t2b.get("gate_T2b", {})
    numbers["t2b_pass"] = rec("PASS" if gate.get("pass") else "FAIL",
                              "TtwobVerdict", "%s")
    write_part("t2b_filter", numbers)


# ------------------------------------------------------------------ Table T3
def part_table_t3():
    # D250 flagship: T3 effective-dimension analysis on jepa_pool_vec latents.
    t3 = _load(S33 / "track_t3_vec.json") or _load(S33 / "track_t3_effective_dimension.json")
    if not t3:
        print("[parts] table_t3: missing inputs, skipped")
        return
    numbers = {}
    for g, row in t3["table_T3"].items():
        word = GWORD.get(g)
        if not word:
            continue
        numbers[f"t3_deff_{word}"] = rec(
            row["d_eff"], f"TthreeDeffG{word}", "%.1f",
            ci_lo=row["d_eff_ci95"][0], ci_hi=row["d_eff_ci95"][1])
        numbers[f"t3_pr_{word}"] = rec(row["participation_ratio"],
                                       f"TthreePrG{word}", "%.1f")
        numbers[f"t3_npc_{word}"] = rec(row["n_pc_90"], f"TthreeNpcG{word}", "%d")
        if row.get("chi3d_wz_post_median") is not None:
            numbers[f"t3_chi_{word}"] = rec(
                row["chi3d_wz_post_median"], f"TthreeChiG{word}", "%.2f")
        if row.get("m_needed_abs0.5") is not None:
            numbers[f"t3_mneed_{word}"] = rec(
                row["m_needed_abs0.5"], f"TthreeMneedG{word}", "%d")
    write_part("table_t3", numbers)


# ------------------------------------------------------------------ paired stats
def part_paired_stats():
    ps = _load(S33 / "paired_stats_v3.json")
    if not ps:
        print("[parts] paired_stats: missing inputs, skipped")
        return
    numbers = {}
    word = {"CL_impact": "CLImp", "CL_relax": "CLRel",
            "Ew_impact": "EwImp", "Ew_relax": "EwRel"}
    for fam_key, suffix in (("F1_filter_vs_static", "St"),
                            ("F2_filter_vs_openloop", "Ol")):
        fam = ps["primary_test_b"][fam_key]
        holm = fam.get("holm", {}) or {}
        for t, v in fam["tests"].items():
            if "case_level" not in v:
                continue
            base = t.replace("_vs_static", "").replace("_vs_openloop", "")
            w = word[base] + suffix
            numbers[f"ps_{t}_med"] = rec(
                v["case_level"]["median_case_mean_delta"], f"Ps{w}Med", "%.2f",
                split="test_b")
            wp = v["case_level"]["wilcoxon_p_one_sided"]
            if isinstance(wp, float):
                numbers[f"ps_{t}_p"] = rec(wp, f"Ps{w}P", "%.3f")
            hp = holm.get("p_holm", {}).get(t)
            if hp is not None:
                numbers[f"ps_{t}_holm"] = rec(hp, f"Ps{w}Holm", "%.3f")
    tc = ps["annex_test_c"]["F1_filter_vs_static"]["tests"]
    for t, w in (("CL_impact_vs_static", "CLImpStTc"),
                 ("CL_relax_vs_static", "CLRelStTc")):
        v = tc.get(t, {})
        if "case_level" in v:
            numbers[f"stats_testc_{t}_med"] = rec(
                v["case_level"]["median_case_mean_delta"], f"Ps{w}Med", "%.2f",
                split="test_c", note="characterisation annex")
            numbers[f"stats_testc_{t}_p"] = rec(
                v["case_level"]["wilcoxon_p_one_sided"], f"Ps{w}P", "%.4f",
                split="test_c")
    write_part("paired_stats", numbers)


# ------------------------------------------------------------------ physics (4.6)
def part_physics():
    dmd = _load(S33 / "spectrum_dmd_v2p2.json")
    atlas = _load(S33 / "manifold_atlas_v2p2.json")
    wc = _load(S33 / "wake_code_v2p2.json")
    numbers = {}
    if dmd:
        for m, short in (("jepa_pool", "Jepa"), ("supervised_only_pool", "SupOnly"),
                         ("regAE_pool", "RegAE"), ("fukami", "Fukami"),
                         ("fukami_wake", "FukamiWake"), ("pod", "Pod")):
            dom = dmd["models"].get(m, {}).get("baseline", {}).get("dominant") or {}
            if "St" in dom:
                numbers[f"dmd_st_{short}"] = rec(dom["St"], f"DmdSt{short}", "%.3f")
                numbers[f"dmd_lam_{short}"] = rec(
                    dom["modulus"], f"DmdLam{short}", "%.3f")
    if atlas:
        jp = atlas["families"].get("jepa_pool", {}).get("probes", {})
        for axis in ("G", "D", "Y"):
            blk = jp.get(axis, {})
            if "test_b_r2" in blk:
                numbers[f"probe_{axis}_jepa"] = rec(
                    blk["test_b_r2"], f"Probe{axis}Jepa", "%.2f")
        ykrr = jp.get("Y", {}).get("krr_rbf", {}).get("test_b_r2")
        if ykrr is not None:
            numbers["probe_Y_krr_jepa"] = rec(ykrr, "ProbeYkrrJepa", "%.2f")
    if wc:
        for m, short in (("jepa_pool", "Jepa"), ("supervised_only_pool", "SupOnly"),
                         ("fukami_wake", "FukamiWake"), ("pod", "Pod"),
                         ("regAE_pool", "RegAE")):
            fb = wc["per_family"].get(m, {}).get("full_vs_best_coordinate", {})
            full = fb.get("full_r2_test_b")
            best = fb.get("best_r2_test_b", fb.get("best_coordinate_r2_test_b"))
            if best is None:
                best = next(
                    (v for k, v in fb.items()
                     if k.startswith("best") and k.endswith("_test_b")
                     and isinstance(v, float)),
                    None,
                )
            if full is not None and best is not None:
                numbers[f"wakecode_full_{short}"] = rec(full, f"WcFull{short}", "%.2f")
                numbers[f"wakecode_gap_{short}"] = rec(
                    full - best, f"WcGap{short}", "%.2f")
    topo = _load(S33 / "topology_v2p2.json")
    if topo and "summary" in topo:
        for m, s in (("jepa_pool", "Jepa"), ("supervised_only_pool", "SupOnly"),
                     ("regAE_pool", "RegAE"), ("pod", "Pod")):
            blk = topo["summary"].get(m, {}).get("by_metric", {}).get("stdz")
            if blk:
                numbers[f"topo_frac_{s}"] = rec(
                    blk["gusted_frac_single"], f"TopoFrac{s}", "%.2f",
                    note="stdz metric, 5pct floor, gusted test_b")
                numbers[f"topo_ctrl_{s}"] = rec(
                    blk["baseline_period_frac_single"], f"TopoCtrl{s}", "%.2f",
                    note="single-period no-gust control")
    if numbers:
        write_part("physics_latent", numbers)
    else:
        print("[parts] physics: no harvestable keys found (schema check needed)")


# ------------------------------------------------------------ audit (D247)
def part_family_filter():
    fam = _load(S33 / "envelope_family_audit.json")
    dcl = _load(S33 / "direct_pressure_cl_baseline.json")
    if not fam or not dcl:
        print("[parts] family_filter: missing inputs, skipped")
        return
    env = _load(S32 / "envelope_by_gust.json")
    numbers = {}
    short = {"ae_wake_pool": "AeWake", "fukami": "Fukami", "fukami_wake": "FukamiWake",
             "pod": "Pod"}
    for m, s in short.items():
        src = env if m == "pod" else fam
        byg = src["models"][m]["aggregates"]["by_G"]
        for g, word in GWORD.items():
            if g in ("0", "0.25-0.5") or byg.get(g, {}).get("n", 0) == 0:
                continue
            numbers[f"ff_cl_{m}_{word}"] = rec(
                byg[g]["filter_CL_analysis_r2_impact"]["median"],
                f"FfCL{s}G{word}", "%.2f")
            numbers[f"ff_div_{m}_{word}"] = rec(
                byg[g]["div_rate"], f"FfDiv{s}G{word}", "%.2f")
    for tapname, s in (("K8_qdeim", "Keight"), ("all192", "AllTaps")):
        for g, blk in dcl["direct"][tapname].items():
            word = GWORD.get(g)
            if word:
                numbers[f"dcl_{tapname}_{word}"] = rec(
                    blk["median_r2"], f"DirectCL{s}G{word}", "%.2f")
    write_part("family_filter_audit", numbers)


# ------------------------------------------------------------------ training-dependent
def part_training_dependent():
    dp = _load(S33 / "dim_plateau.json")
    md = _load(S33 / "min_d_panel.json")
    sb = _load(S33 / "seed_band_v3.json")
    if dp:
        numbers = {}
        for d, blk in dp["by_d"].items():
            word = {"16": "Sixteen", "32": "ThirtyTwo", "64": "SixtyFour"}[d]
            numbers[f"plateau_wake_d{d}"] = rec(
                blk["wake_linear_r2"], f"PlateauWakeD{word}", "%.3f",
                ci_lo=blk["wake_linear_ci"]["ci95"][0],
                ci_hi=blk["wake_linear_ci"]["ci95"][1])
            if blk.get("merit_mean_obs_h8") is not None:
                numbers[f"plateau_merit_d{d}"] = rec(
                    blk["merit_mean_obs_h8"], f"PlateauMeritD{word}", "%.3f")
        numbers["plateau_spread"] = rec(dp["plateau"]["spread_wake_r2"],
                                        "PlateauSpread", "%.3f")
        write_part("dim_plateau", numbers)
    else:
        print("[parts] dim_plateau: pending (training)")
    if md:
        numbers = {}
        for fam, short in (("jepa_pool", "Jepa"), ("fukami_wake", "FukamiWake"),
                           ("pod_truncated", "Pod")):
            v = md["panel"][fam]["smallest_d_r2_ge_0.5"]
            numbers[f"mind_{fam}"] = rec(
                v if v is not None else -1, f"MinD{short}",
                "%d", note="-1 = never reaches 0.5")
        write_part("min_d_panel", numbers)
    else:
        print("[parts] min_d_panel: pending (training)")
    if sb:
        numbers = {}
        for fam, short in (("jepa_wake_pool", "JepaWake"),
                           ("supervised_only_pool", "SupOnly"),
                           ("ae_wake_pool", "AeWake"),
                           ("jepa_wake_pool_vec", "JepaVec")):
            if fam not in sb["families"]:
                continue
            b = sb["families"][fam]["bands"]
            numbers[f"seed_wake_{short}"] = rec(
                b["wake_linear_r2"]["mean"], f"SeedWake{short}", "%.3f",
                seed_mean=b["wake_linear_r2"]["mean"], seed_sd=b["wake_linear_r2"]["sd"])
            numbers[f"seed_wake_sd_{short}"] = rec(
                b["wake_linear_r2"]["sd"], f"SeedWakeSd{short}", "%.3f")
            if b["merit_mean_obs_h8"]["mean"] is not None:
                numbers[f"seed_merit_{short}"] = rec(
                    b["merit_mean_obs_h8"]["mean"], f"SeedMerit{short}", "%.3f")
                numbers[f"seed_merit_sd_{short}"] = rec(
                    b["merit_mean_obs_h8"]["sd"], f"SeedMeritSd{short}", "%.3f")
        write_part("seed_band", numbers)
    else:
        print("[parts] seed_band: pending (training)")


# ------------------------------------------------------------------ verify anchors
# D250: flagship anchors updated to the jepa_pool_vec truth (native vector
# predictor). Baseline anchors (SupOnly, Fukami, near-null) are unchanged.
ANCHORS = [
    ("table_x", "x_wake_JepaWake", 0.751, 0.005),
    ("table_x", "x_wake_SupOnly", 0.792, 0.005),
    ("table_x", "x_merit_JepaWake", 0.591, 0.005),
    ("table_x", "x_merit_JepaWake_native", 0.755, 0.005),
    ("table_x", "x_merit_JepaWake_tf", 0.655, 0.005),
    ("table_x", "x_merit_SupOnly", 0.637, 0.005),
    ("table_w_gate_o", "w_state_Jepa", 0.659, 0.005),
    ("table_w_gate_o", "w_state_Fukami", 0.921, 0.005),
    ("table_v_envelope", "v_filt_cl_One", 0.74, 0.02),
    ("table_v_envelope", "v_filt_cl_Three", 0.63, 0.02),
    ("table_v_envelope", "v_filt_cl_Four", 0.84, 0.02),
    ("gates_p_and_table_y", "y_nearnull_RegAE", 0.100, 0.005),
    ("gates_p_and_table_y", "y_nearnull_JepaWake", 0.014, 0.005),
]


def verify() -> int:
    fails = 0
    for part, name, expect, tol in ANCHORS:
        p = PARTS / f"{part}.json"
        if not p.exists():
            print(f"[verify] MISSING part {part}")
            fails += 1
            continue
        numbers = json.loads(p.read_text())["numbers"]
        if name not in numbers:
            print(f"[verify] MISSING number {part}:{name}")
            fails += 1
            continue
        got = numbers[name]["value"]
        if not isinstance(got, (int, float)) or abs(got - expect) > tol:
            print(f"[verify] MISMATCH {part}:{name} = {got} (expected {expect} +- {tol})")
            fails += 1
    print(f"[verify] {'PASS' if fails == 0 else f'{fails} FAILURE(S)'} "
          f"({len(ANCHORS)} anchors)")
    return fails


def main() -> int:
    part_constants()
    part_table_x()
    part_gates_p()
    part_table_z()
    part_table_w()
    part_table_v()
    part_track_t()
    part_t2b()
    part_table_t3()
    part_paired_stats()
    part_physics()
    part_family_filter()
    part_training_dependent()
    return 1 if verify() else 0


if __name__ == "__main__":
    raise SystemExit(main())
