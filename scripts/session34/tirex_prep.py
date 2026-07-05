"""Export latent trajectories + affine C_L probe for the TiRex experiments.

Runs in the PROJECT venv; writes a self-contained npz the TiRex scratch-venv
script (tirex_forecast.py) can consume with numpy only:
  Z_test (42, 120, d), CL_test (42, 120), wmask (42, 120),
  Z_train_ctx_stats, probe_w (d,), probe_b () -- the frozen RidgeCV probe
  collapsed to its exact affine form, G/D/Y (42,) episode parameters,
  plus the CLW/CLN reference forecast numbers for the comparison table.

Run: taskset -c 0-7 python -m scripts.session34.tirex_prep
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.session34.trackc_lift_eval import group_encounters, load_cache  # noqa: E402
from src.evaluation.represent import fit_linear_probe  # noqa: E402

CACHE = REPO_ROOT / "outputs/session34/trackc_latents"
RUN = "jepa_pool_vec"


def main() -> int:
    tr = load_cache(CACHE, RUN, "train")
    tb = load_cache(CACHE, RUN, "test_b")
    probe = fit_linear_probe(tr["z_gap"], tr["cl"])
    d = tr["z_gap"].shape[1]
    b = float(probe.predict(np.zeros((1, d)))[0])
    w = np.array([float(probe.predict(np.eye(d)[i][None])[0]) - b for i in range(d)])
    # exactness check of the affine collapse
    idx = np.random.default_rng(0).choice(len(tb["z_gap"]), 200, replace=False)
    err = np.abs(tb["z_gap"][idx] @ w + b - probe.predict(tb["z_gap"][idx])).max()
    assert err < 1e-6, f"probe not affine? max err {err}"

    split = json.loads((REPO_ROOT / "configs/splits/split_v2p2.json").read_text())["cases"]
    encs = group_encounters(tb)
    Z = np.stack([tb["z_gap"][e["rows"]] for e in encs])
    CL = np.stack([tb["cl"][e["rows"]] for e in encs])
    WM = np.stack([np.load(CACHE / f"latents_{RUN}_test_b.npz")["window_mask"][e["rows"]]
                   for e in encs])
    G = np.array([split[e["case_id"]]["G"] for e in encs])
    D = np.array([split[e["case_id"]]["D"] for e in encs])
    Y = np.array([split[e["case_id"]]["Y"] for e in encs])
    out = REPO_ROOT / "outputs/session34/tirex_input.npz"
    np.savez_compressed(out, Z_test=Z.astype(np.float32), CL_test=CL, wmask=WM,
                        probe_w=w, probe_b=b, G=G, D=D, Y=Y,
                        case_id=np.array([e["case_id"] for e in encs]),
                        encounter_index=np.array([e["encounter_index"] for e in encs]))
    print(f"[tirex-prep] wrote {out}: Z {Z.shape}, probe affine err {err:.1e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
