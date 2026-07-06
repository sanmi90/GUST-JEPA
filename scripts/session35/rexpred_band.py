"""T1 (Session 35 P1): CLN-rexpred d=32 3-seed band, frozen Track C protocol.

Evaluates jepa_pool_ln_rexpred_s{0,1,2} through the byte-identical
trackc_lift_eval protocol (frozen RidgeCV linear probe + MLP probe fit on
train z_gap -> C_L, pooled peak-region R2 on test_b, half_width 8,
persistence floor frames [0, 25)). The s0 value must reproduce the
Session 34 single-seed 0.903; s1/s2 are the new seeds.

Gate (pre-registered, outputs/session35/p1_gates.md T1): the 3-seed band is
reported wherever 0.903 appears; if the band MEAN drops below the CLN probe
headline 0.862, the rexpred result moves to an appendix note.

Run (CPU, after trackc_encode of the two new runs):
    taskset -c 0-7 python -m scripts.session35.rexpred_band
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.session34.trackc_lift_eval import eval_run  # noqa: E402

CACHE = REPO_ROOT / "outputs/session34/trackc_latents"
RUNS = ("jepa_pool_ln_rexpred_s0", "jepa_pool_ln_rexpred_s1", "jepa_pool_ln_rexpred_s2")
CLN_PROBE_HEADLINE = 0.862  # gate threshold, pre-registered
S0_EXPECTED = 0.903         # Session 34 single-seed anchor (reproduction check)


def main() -> int:
    out: dict = {"protocol": "trackc_lift_eval frozen probes, pooled peak-region R2, test_b"}
    lin, mlp = [], []
    for i, run in enumerate(RUNS):
        res = eval_run(CACHE, run, seed=i)
        rec = {
            p: {k: v for k, v in res[p].items() if k != "encounters"}
            for p in res
        }
        out[run] = rec
        lin.append(rec["linear"]["pooled_peak_r2"])
        mlp.append(rec["mlp"]["pooled_peak_r2"])
        print(f"[t1] {run}: linear peak R2 = {lin[-1]:+.4f}  mlp = {mlp[-1]:+.4f}",
              flush=True)

    anchor_ok = abs(lin[0] - S0_EXPECTED) < 0.005
    band = {
        "linear_peak_r2_per_seed": lin,
        "mlp_peak_r2_per_seed": mlp,
        "seed_mean": float(np.mean(lin)),
        "seed_sd": float(np.std(lin, ddof=1)),
        "n": len(RUNS),
        "s0_reproduces_0p903": bool(anchor_ok),
        "gate_threshold_cln_headline": CLN_PROBE_HEADLINE,
        "gate_pass_mean_above_cln": bool(np.mean(lin) >= CLN_PROBE_HEADLINE),
    }
    out["band"] = band
    print(f"[t1] band: {band['seed_mean']:.4f} +- {band['seed_sd']:.4f} (n=3)  "
          f"s0-anchor {'OK' if anchor_ok else 'MISMATCH'}  "
          f"gate {'PASS' if band['gate_pass_mean_above_cln'] else 'FAIL -> appendix note'}",
          flush=True)
    dest = REPO_ROOT / "outputs/session35/rexpred_d32_band.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"[t1] wrote {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
