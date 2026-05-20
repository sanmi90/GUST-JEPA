"""Session 9 Step 3 A11: head-to-head evaluation of the Fukami AE against the JEPA.

After training, the Fukami baseline produces (z, omega_hat, CL_hat).
This script reports two comparable metrics against the JEPA's Section 5.5
production-cell numbers:

1. **Test B delta** = r2(z -> CL_future) - r2((c, t) -> CL_future)
   The same delta_test_b metric the JEPA bisection analysis reports
   (script `session9_bisection_analysis.py`). Computed by encoding
   Test A and Test B encounters with the Fukami encoder, fitting an
   MLP probe from z to CL_future on Test A, and evaluating on Test B.

2. **Reconstruction MSE + SSIM** on Test A / B / C, against the
   per-case-mean noise floor. Fukami's headline metric per the
   arXiv:2305.18394 supplementary material.

The two comparison points are:
- JEPA + visualisation decoder: Test B delta from the bisection
  winner; reconstruction MSE + SSIM from `scripts/session9_train_decoder.py`.
- Fukami AE: Test B delta from this script; reconstruction MSE + SSIM
  from `scripts/session9_train_fukami.py` final evaluation.

Usage:
    python scripts/session9_fukami_evaluation.py \\
        --fukami-checkpoint outputs/runs/session9/run_a11_fukami_ae/checkpoint_iter020000.pt \\
        --output-dir outputs/runs/session9/run_a11_fukami_ae \\
        --gpu 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.baselines.fukami_ae import FukamiAEWrapper  # noqa: E402
from src.utils.device import require_rtx6000  # noqa: E402


PREVENT = Path(os.environ.get("PREVENT_ROOT", "/home/carlos/PREVENT"))
CACHE = Path(os.environ.get("VORTEX_JEPA_CACHE", PREVENT / "data" / "processed" / "vortex-jepa"))

DELTAS = (8, 16, 24)


def load_fukami(ckpt_path: Path, device: torch.device) -> FukamiAEWrapper:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = blob["args"]
    n_deltas = len(args.get("observable_head_deltas", [8, 16, 24]))
    omega_pipeline = None
    manifest_path = args.get("omega_pipeline_manifest")
    if manifest_path is not None:
        from src.data.omega_pipeline import OmegaPipeline
        omega_pipeline = OmegaPipeline.from_manifest(manifest_path)
        print(f"[load_fukami] loaded OmegaPipeline from {manifest_path}", flush=True)
    wrapper = FukamiAEWrapper(
        latent_dim=int(args["d"]), n_deltas=n_deltas,
        omega_pipeline=omega_pipeline,
    )
    wrapper.load_state_dict(blob["wrapper_state_dict"])
    return wrapper.eval().to(device)


def gather_encounters(split: str) -> list[dict]:
    with open(REPO / "configs" / "splits" / "split_v1.json") as f:
        manifest = json.load(f)
    out = []
    for cid, case in manifest["cases"].items():
        if split == "test_a" and case["split"] == "train":
            ks = case["test_a_encounter_indices"]
        elif split == "test_b" and case["split"] == "test_b":
            ks = list(range(case["n_encounters_full"]))
        elif split == "test_c" and case["split"] == "test_c":
            ks = list(range(case["n_encounters_full"]))
        else:
            continue
        for k in ks:
            path = CACHE / "v1" / cid / f"encounter_{k:02d}.h5"
            if not path.exists():
                continue
            out.append({"case_id": cid, "k": int(k), "path": str(path),
                        "G": float(case["G"]),
                        "D": float(case["D"]),
                        "Y": float(case["Y"])})
    return out


def encode_split(wrapper, encs, device, d: int) -> np.ndarray:
    N, T = len(encs), 120
    out = np.empty((N, T, d), dtype=np.float32)
    pipe = getattr(wrapper, "omega_pipeline", None)
    with torch.no_grad():
        for i, e in enumerate(encs):
            with h5py.File(e["path"], "r") as f:
                omega = np.asarray(f["omega_z"], dtype=np.float32)
            if pipe is not None:
                omega = pipe.preprocess_raw(omega, e["case_id"], int(e["k"]))
            x = torch.from_numpy(omega).unsqueeze(0).unsqueeze(2).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                if pipe is not None:
                    x_norm = pipe.normalize(x)
                    z = wrapper.encoder(x_norm)
                else:
                    z = wrapper.encode(x)
            out[i] = z.squeeze(0).float().cpu().numpy()
    return out


def build_cl_future(encs):
    N, T = len(encs), 120
    out = np.empty((N, T, len(DELTAS)), dtype=np.float32)
    for i, e in enumerate(encs):
        with h5py.File(e["path"], "r") as f:
            cl = np.asarray(f["C_L"], dtype=np.float32)
        for j, d in enumerate(DELTAS):
            for t in range(T):
                src = t + d
                out[i, t, j] = cl[src] if src < cl.shape[0] else cl[-1]
    return out


def c_of(encs):
    return np.stack([[e["G"], e["D"], e["Y"]] for e in encs], axis=0).astype(np.float32)


def fit_mlp_oos(X_train, y_train, X_eval, y_eval, device, seed=0, hidden=64,
                epochs=400, lr=1e-2):
    torch.manual_seed(seed)
    X_tr = torch.from_numpy(X_train.astype(np.float32)).to(device)
    y_tr = torch.from_numpy(y_train.astype(np.float32)).to(device)
    X_ev = torch.from_numpy(X_eval.astype(np.float32)).to(device)
    y_ev = torch.from_numpy(y_eval.astype(np.float32)).to(device)
    model = nn.Sequential(
        nn.Linear(X_tr.shape[1], hidden), nn.GELU(),
        nn.Linear(hidden, y_tr.shape[1]),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        p = model(X_tr)
        loss = F.mse_loss(p, y_tr)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        p_ev = model(X_ev)
    ss_res = ((y_ev - p_ev) ** 2).sum(dim=0)
    ss_tot = ((y_ev - y_ev.mean(dim=0)) ** 2).sum(dim=0).clamp_min(1e-8)
    return float((1.0 - ss_res / ss_tot).mean().item())


def fit_mlp(X, y, device, seed=0, hidden=64, epochs=400, lr=1e-2):
    torch.manual_seed(seed)
    n = X.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    fit, ev = perm[: int(0.75 * n)], perm[int(0.75 * n):]
    return fit_mlp_oos(X[fit], y[fit], X[ev], y[ev], device, seed, hidden, epochs, lr)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Session 9 A11: Fukami evaluation")
    p.add_argument("--fukami-checkpoint", required=True, type=str)
    p.add_argument("--output-dir", required=True, type=str)
    p.add_argument("--gpu", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = require_rtx6000(gpu_index=args.gpu)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[fukami-eval] device={device}", flush=True)
    wrapper = load_fukami(Path(args.fukami_checkpoint), device)
    print("[fukami-eval] encoder loaded", flush=True)

    SPLITS = {s: gather_encounters(s) for s in ("test_a", "test_b", "test_c")}
    for s, encs in SPLITS.items():
        print(f"  {s}: {len(encs)} encs", flush=True)

    d = int(wrapper.latent_dim)
    print(f"[fukami-eval] latent_dim from checkpoint: {d}", flush=True)
    Z = {s: encode_split(wrapper, SPLITS[s], device, d=d) for s in SPLITS}
    CL_RAW = {s: build_cl_future(SPLITS[s]) for s in SPLITS}
    MASK = {s: np.isfinite(CL_RAW[s]).reshape(CL_RAW[s].shape[0], -1).all(axis=1) for s in SPLITS}
    C = {s: c_of(SPLITS[s]) for s in SPLITS}
    T = Z["test_a"].shape[1]

    # In-sample Test A first (for the r2(c, t) baseline + r2(z) probe)
    z_a = Z["test_a"][MASK["test_a"]]
    cl_a = CL_RAW["test_a"][MASK["test_a"]].reshape(-1, 3)
    c_a = C["test_a"][MASK["test_a"]]
    z_a_flat = z_a.reshape(-1, d)
    ct_a_flat = np.hstack([
        np.repeat(c_a, T, axis=0),
        np.tile(np.arange(T)[:, None], (z_a.shape[0], 1)).astype(np.float32),
    ])
    r2_cl_a = fit_mlp(z_a_flat, cl_a, device)
    r2_ct_a = fit_mlp(ct_a_flat, cl_a, device)
    delta_a = r2_cl_a - r2_ct_a
    print(f"[fukami-eval] Test A: r2(z)={r2_cl_a:.3f} r2(c,t)={r2_ct_a:.3f} delta={delta_a:+.3f}",
          flush=True)

    rows = [{"split": "test_a", "r2_z": r2_cl_a, "r2_ct": r2_ct_a, "delta": delta_a}]

    # OOS Test B / Test C
    for split in ("test_b", "test_c"):
        z_s = Z[split][MASK[split]]
        cl_s = CL_RAW[split][MASK[split]].reshape(-1, 3)
        c_s = C[split][MASK[split]]
        T_s = z_s.shape[1]
        z_s_flat = z_s.reshape(-1, d)
        ct_s_flat = np.hstack([
            np.repeat(c_s, T_s, axis=0),
            np.tile(np.arange(T_s)[:, None], (z_s.shape[0], 1)).astype(np.float32),
        ])
        r2_cl_s = fit_mlp_oos(z_a_flat, cl_a, z_s_flat, cl_s, device)
        r2_ct_s = fit_mlp_oos(ct_a_flat, cl_a, ct_s_flat, cl_s, device)
        delta_s = r2_cl_s - r2_ct_s
        rows.append({"split": split, "r2_z": r2_cl_s, "r2_ct": r2_ct_s, "delta": delta_s})
        print(f"[fukami-eval] {split}: r2(z)={r2_cl_s:.3f} r2(c,t)={r2_ct_s:.3f} "
              f"delta={delta_s:+.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "fukami_test_b_delta.csv", index=False)
    print(f"[fukami-eval] wrote {out_dir / 'fukami_test_b_delta.csv'}", flush=True)


if __name__ == "__main__":
    main()
