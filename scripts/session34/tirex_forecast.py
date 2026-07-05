"""Zero-shot TiRex latent-trajectory forecasting (Session 34, arXiv 2607.01204).

Runs inside the TiRex scratch venv (NOT the project venv). Consumes
outputs/session34/tirex_input.npz (from tirex_prep.py). Protocol mirrors
trackc_forecast: context frames [0, 25), horizon 40 through the gust impact,
latent R^2 (pooled, vs mean and vs persistence-of-last-context-frame) and
decoded C_L R^2 via the exported affine probe.

Modes:
  --model v1     tirex-ts (NX-AI/TiRex, ungated): channel-independent
                 univariate over the 32 latent dims, one batched call.
  --model v2     tirex-2 (NX-AI/TiRex-2, gated; needs HF token): 32-variate
                 with the covariate ladder: none -> +phase ramp -> +phase and
                 thermometer-encoded (G, D, Y) binary covariates (constant
                 float covariates are ERASED by instance norm; binary values
                 bypass it -- verified against the paper appendix).

Run (CPU): $SCRATCH/venv-tirex1/bin/python scripts/session34/tirex_forecast.py --model v1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CTX, H = 25, 40


def r2(t, p):
    return 1.0 - ((t - p) ** 2).sum() / max(((t - t.mean(0)) ** 2).sum(), 1e-12)


def thermometer(vals: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """(N,) -> (N, len(edges)) binary thermometer code (values in {0,1})."""
    return (vals[:, None] >= edges[None, :]).astype(np.float32)


def score(Z, CL, roll, w, b, tag, out):
    zt = Z[:, CTX:CTX + H]                                     # (42, H, d)
    lat = r2(zt.reshape(-1, zt.shape[-1]), roll.reshape(-1, roll.shape[-1]))
    pers = np.repeat(Z[:, CTX - 1][:, None], H, 1)
    lat_vs_pers = 1.0 - ((zt - roll) ** 2).sum() / ((zt - pers) ** 2).sum()
    ct = CL[:, CTX:CTX + H].ravel()
    cp = (roll.reshape(-1, roll.shape[-1]) @ w + b)
    cl = 1.0 - ((ct - cp) ** 2).sum() / ((ct - ct.mean()) ** 2).sum()
    print(f"[tirex] {tag}: latent R2={lat:+.3f} (vs persist {lat_vs_pers:+.3f})  "
          f"decoded C_L R2={cl:+.3f}", flush=True)
    out[tag] = {"latent_r2": float(lat), "latent_r2_vs_persistence": float(lat_vs_pers),
                "decoded_cl_r2": float(cl)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["v1", "v2"], required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    blob = np.load(REPO_ROOT / "outputs/session34/tirex_input.npz", allow_pickle=True)
    Z, CL = blob["Z_test"].astype(np.float64), blob["CL_test"]
    w, b = blob["probe_w"], float(blob["probe_b"])
    n_ep, T, d = Z.shape
    results: dict = {"protocol": {"ctx": CTX, "horizon": H, "episodes": n_ep, "d": d}}
    t0 = time.time()

    if args.model == "v1":
        import torch
        from tirex import load_model

        model = load_model("NX-AI/TiRex", device=args.device)
        ctx = torch.from_numpy(
            Z[:, :CTX].transpose(0, 2, 1).reshape(n_ep * d, CTX)).float()
        q, mean = model.forecast(context=ctx, prediction_length=H)
        arr = q if isinstance(q, np.ndarray) else q.numpy()
        med = arr[:, :, arr.shape[2] // 2] if arr.ndim == 3 and arr.shape[1] == H \
            else arr[:, arr.shape[1] // 2, :]
        # normalize to (n_ep*d, H); tirex-ts returns (B, H, Q) quantiles
        if med.shape[-1] != H:
            med = med.T
        roll = med.reshape(n_ep, d, H).transpose(0, 2, 1)      # (42, H, d)
        score(Z, CL, roll, w, b, "v1_channel_independent", results)
    else:
        import torch
        from tirex2 import TimeseriesType, load_model

        model = load_model("NX-AI/TiRex-2", device=args.device)
        phase = np.arange(T + H, dtype=np.float32) * 0.05      # time ramp, t/c
        gdy = np.concatenate([
            thermometer(blob["G"], np.array([-3, -2, -1, -0.5, 0.5, 1, 2, 3])),
            thermometer(blob["D"], np.array([0.75, 1.25])),
            thermometer(blob["Y"], np.array([-0.2, 0.0, 0.2])),
        ], axis=1)                                              # (42, 13) binary
        for tag, use_phase, use_gdy in (("v2_no_cov", False, False),
                                        ("v2_phase", True, False),
                                        ("v2_phase_gdy", True, True)):
            batch = []
            for i in range(n_ep):
                fcs = []
                if use_phase:
                    fcs.append(phase[:CTX + H])
                if use_gdy:
                    fcs += [np.full(CTX + H, v, dtype=np.float32) for v in gdy[i]]
                fcov = torch.tensor(np.stack(fcs)) if fcs else None
                batch.append(TimeseriesType(
                    target=torch.tensor(Z[i, :CTX].T, dtype=torch.float32),
                    past_covariates=None, future_covariates=fcov))
            fc = model.forecast(batch, prediction_length=H, output_type="numpy")
            roll = np.stack([f[:, f.shape[1] // 2, :].T for f in fc])  # median -> (42, H, d)
            score(Z, CL, roll, w, b, tag, results)

    results["wall_s"] = time.time() - t0
    out = Path(args.out or REPO_ROOT / f"outputs/session34/tirex_{args.model}_forecast.json")
    out.write_text(json.dumps(results, indent=1))
    print(f"[tirex] wrote {out} ({results['wall_s']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
