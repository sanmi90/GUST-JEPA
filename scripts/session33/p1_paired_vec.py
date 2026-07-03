"""Recompute the P1 paired wake-readability gap with the vec flagship (D250).

The readability attribution in Section 4.1 rests on a per-case paired bootstrap of
the windowed wake-enstrophy probe R2 between the objective-free supervised control
and the predictive flagship. The frozen track_p_gates ran this on the ResUNet-era
jepa_pool; here we rerun it with arm B pointed at jepa_pool_vec so the paired gap
and its CI are on the same encoder as every other flagship number.

Output: outputs/session33/p1_readability_vec.json (same structure part_gates_p
reads: P1_readability.primary_wake_enstrophy.{delta_a_minus_b, ci95}).

Run (CPU): taskset -c 16-23 python -m scripts.session33.p1_paired_vec
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

from scripts.session32.track_p_gates import paired_probe_delta  # noqa: E402

SUP = REPO / "outputs" / "session32" / "q1_pool_latents"
VEC = REPO / "outputs" / "session33" / "q1_vec_latents"


def main() -> int:
    a = {"train": SUP / "latents_supervised_only_pool_train.npz",
         "test_b": SUP / "latents_supervised_only_pool_test_b.npz"}
    b = {"train": VEC / "latents_jepa_pool_vec_train.npz",
         "test_b": VEC / "latents_jepa_pool_vec_test_b.npz"}
    res = paired_probe_delta(a, b, n_boot=10000, seed=0)
    pw = res["primary_wake_enstrophy"]
    payload = {
        "task": "SESSION 33 D250 -- P1 paired wake readability, supervised_only vs jepa_pool_vec",
        "P1_readability": {
            "primary_wake_enstrophy": pw,
            "note": "arm A = supervised_only_pool, arm B = jepa_pool_vec (native vector flagship)",
        },
    }
    out = REPO / "outputs" / "session33" / "p1_readability_vec.json"
    out.write_text(json.dumps(payload, indent=2, default=float))
    print(f"[p1-vec] delta(sup - vec) = {pw['delta_a_minus_b']:.4f}  "
          f"CI[{pw['ci95'][0]:.4f}, {pw['ci95'][1]:.4f}]  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
