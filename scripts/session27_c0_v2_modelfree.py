#!/usr/bin/env python3
"""
session27_c0_v2_modelfree.py
============================

Model-free C0 SURD de-risking gate (Block A of run_causal_analysis.py), computed
on the EXACT v2 encounters the unconditioned model uses, sourced from the DNS
physical descriptors in the per_frame_targets file. This is deliberately
model-free: SURD of the future observable from {G, current wake enstrophy} uses
NO latent and NO predictor, so the synergy ratio is identical whether the model
is conditioned or unconditioned. We run it on the v2-matched encounters so the
gate is reported against the same data the unconditioned latent analysis uses,
and we cross-check it against the production v1 reproduction (outputs/session27/
causal_noc_tf/c0_v1/causal_results.json).

The estimators (quantise, joint_pmf, surd_discrete) are imported verbatim from
infotheory/, identical to scripts/run_causal_analysis.py. The gate criterion is
S[future wake | G + wake_now] / H >= ~2 x S[future lift | G + wake_now] / H
(HANDOFF D158). We report per-encounter (impact-frame) and per-frame-postimpact,
across bins {4,5,6,8}, exactly as scripts/session25_c0_robustness.py did.

Output: outputs/session27/causal_noc_tf/c0_v2_modelfree.{json,txt}. Nothing else
is written.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from infotheory.estimators import quantise, joint_pmf
from infotheory.surd import surd_discrete

# v2-matched DNS descriptors (encoder-independent), via the uncond per_frame_targets
# copy (physical columns identical to outputs/session16/exp2/per_frame_targets).
PF = REPO / "outputs/session27/causal_noc_tf/per_frame_targets/train.npz"
H = 16


def surd_ratio(target, s_g, s_wake, nb):
    """S/H for one target given 2 sources (G, wake_now) at nb bins."""
    t = quantise(target, nb)
    s1 = quantise(s_g, nb)
    s2 = quantise(s_wake, nb)
    p = joint_pmf(np.column_stack([t, s1, s2]), (nb, nb, nb))
    res = surd_discrete(p)
    h = res.h_target or 1.0
    S = sum(res.synergistic.values()) / h
    R = sum(res.redundant.values()) / h
    U_g = max((v for k, v in res.unique.items() if 1 in k), default=0.0) / h
    return {"S": float(S), "R": float(R), "U_G": float(U_g),
            "leak": float(res.info_leak), "H": float(res.h_target)}


def main():
    d = np.load(PF, allow_pickle=True)
    we = np.asarray(d["wake_enstrophy"], float)   # (n, 120)
    cl = np.asarray(d["C_L"], float)              # (n, 120)
    G_raw = np.asarray(d["G"], float)             # (n,) or (n, 120) broadcast
    G_per_enc = G_raw[:, 0] if G_raw.ndim == 2 else G_raw   # (n,) one G per encounter
    imp = np.asarray(d["impact_frame"], int)      # (n,)
    n, T = we.shape

    # --- per-encounter (impact-frame sources, impact+H targets) ---
    G_enc = G_per_enc
    wake_now = np.array([we[i, imp[i]] for i in range(n)])
    wake_fut = np.array([we[i, imp[i] + H] for i in range(n)])
    cl_fut = np.array([cl[i, imp[i] + H] for i in range(n)])

    # --- per-frame post-impact (pooled) ---
    rows = [(i, t) for i in range(n) for t in range(int(imp[i]), T - H)]
    ii = np.array([r[0] for r in rows]); tt = np.array([r[1] for r in rows])
    G_pf = G_per_enc[ii]
    wake_now_pf = we[ii, tt]
    wake_fut_pf = we[ii, tt + H]
    cl_fut_pf = cl[ii, tt + H]

    results = {"description": "Model-free C0 SURD gate on v2-matched encounters "
                              "(no model in the loop; identical for conditioned and "
                              "unconditioned). Gate: S_wake/H >= ~2 x S_lift/H.",
               "n_encounters": int(n), "n_pooled_postimpact": int(len(ii)),
               "v1_production_reproduction": {"per_encounter_bins6_ratio": 1.36,
                                              "S_wake": 0.205, "S_lift": 0.151},
               "per_encounter": {}, "per_frame_postimpact": {}}

    lines = ["Model-free C0 SURD gate (v2-matched encounters; no model in the loop)",
             "Gate criterion 1: S[future wake | G, wake_now]/H >= ~2 x S[future lift | ...]/H",
             "This is ENCODER-INDEPENDENT: SURD uses only DNS scalars (G, wake, C_L).",
             f"n encounters={n}, n pooled post-impact frames={len(ii)}", "",
             f"{'design':<22}{'bins':>5}{'S_wake/H':>10}{'S_lift/H':>10}{'ratio':>8}"
             f"{'U_G_lift':>10}{'leak_w':>8}{'leak_l':>8}"]

    for label, (gw, ww, fw, fl) in {
        "per_encounter": (G_enc, wake_now, wake_fut, cl_fut),
        "per_frame_postimpact": (G_pf, wake_now_pf, wake_fut_pf, cl_fut_pf),
    }.items():
        for nb in (4, 5, 6, 8):
            rw = surd_ratio(fw, gw, ww, nb)
            rl = surd_ratio(fl, gw, ww, nb)
            ratio = rw["S"] / rl["S"] if rl["S"] > 1e-9 else float("nan")
            results[label][f"bins{nb}"] = {
                "S_wake_over_H": rw["S"], "S_lift_over_H": rl["S"], "ratio": ratio,
                "U_G_lift_over_H": rl["U_G"], "leak_wake": rw["leak"], "leak_lift": rl["leak"]}
            lines.append(f"{label:<22}{nb:>5}{rw['S']:>10.3f}{rl['S']:>10.3f}"
                         f"{ratio:>8.2f}{rl['U_G']:>10.3f}{rw['leak']:>8.3f}{rl['leak']:>8.3f}")
        lines.append("")

    # gate readout
    pe = [results["per_encounter"][f"bins{b}"]["ratio"] for b in (4, 5, 6, 8)]
    pf = [results["per_frame_postimpact"][f"bins{b}"]["ratio"] for b in (4, 5, 6, 8)]
    results["gate"] = {
        "per_encounter_median_ratio": float(np.median(pe)),
        "per_encounter_range": [float(min(pe)), float(max(pe))],
        "per_frame_postimpact_median_ratio": float(np.median(pf)),
        "per_frame_postimpact_range": [float(min(pf)), float(max(pf))],
        "two_x_gate_passes": bool(np.median(pe) >= 2.0 or np.median(pf) >= 2.0),
    }
    lines += ["GATE READOUT (criterion 1: ratio >= ~2x)",
              f"  per_encounter        median ratio={np.median(pe):.2f} "
              f"range=[{min(pe):.2f},{max(pe):.2f}]",
              f"  per_frame_postimpact median ratio={np.median(pf):.2f} "
              f"range=[{min(pf):.2f},{max(pf):.2f}]",
              f"  2x gate passes: {results['gate']['two_x_gate_passes']}  "
              f"(production v1 per-encounter bins6 ratio = 1.36; gate FAILED there too)"]

    outj = REPO / "outputs/session27/causal_noc_tf/c0_v2_modelfree.json"
    outt = REPO / "outputs/session27/causal_noc_tf/c0_v2_modelfree.txt"
    outj.parent.mkdir(parents=True, exist_ok=True)
    outj.write_text(json.dumps(results, indent=2))
    outt.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[done] wrote {outj} and {outt}")


if __name__ == "__main__":
    main()
