"""Session 8 Step 5: analyse the latent-dimension sweep at best (eta*, lambda*).

Loads the iter-20000 checkpoint of the three d-sweep runs (d=8 from
``run_d8_best``, d=16 from ``run_d16_best``, and d=32 from the Step 4
best-grid-point directory). For each, encodes Test A / Test B / Test C
and computes the same metric table.

Output:
- ``outputs/runs/session8/d_sweep.csv`` with one row per (d, split).
- ``outputs/runs/session8/fig_d_sweep.png`` delta_test_b vs d.

Usage:
    python scripts/session8_d_sweep_analysis.py \\
        --best-grid-run run_e5_or_whatever_step4_identifies

Wall-clock: ~3 min (three checkpoints * ~1 min each).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(8)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.models.encoder import HybridCNNViTEncoder  # noqa: E402
from src.training.diagnostics import linear_probe_r2, participation_ratio  # noqa: E402
from src.utils.device import require_rtx6000  # noqa: E402


PREVENT = Path(os.environ.get("PREVENT_ROOT", "/home/carlos/PREVENT"))
CACHE = Path(os.environ.get("VORTEX_JEPA_CACHE", PREVENT / "data" / "processed" / "vortex-jepa"))

DELTAS = (8, 16, 24)


def gather_encounters(split_key):
    with open(REPO / "configs" / "splits" / "split_v1.json") as f:
        manifest = json.load(f)
    encs = []
    for cid, case in manifest["cases"].items():
        if split_key == "test_a" and case["split"] == "train":
            ks = case["test_a_encounter_indices"]
        elif split_key == "test_b" and case["split"] == "test_b":
            ks = list(range(case["n_encounters_full"]))
        elif split_key == "test_c" and case["split"] == "test_c":
            ks = list(range(case["n_encounters_full"]))
        else:
            continue
        for k in ks:
            path = CACHE / "v1" / cid / f"encounter_{k:02d}.h5"
            with h5py.File(path, "r") as g:
                encs.append({
                    "case_id": cid, "k": int(k),
                    "omega_z": np.asarray(g["omega_z"], dtype=np.float32),
                    "C_L": np.asarray(g["C_L"], dtype=np.float32),
                    "G": float(g.attrs["G"]),
                    "D": float(g.attrs["D"]),
                    "Y": float(g.attrs["Y"]),
                })
    return encs


def encode(enc, encs, d, device):
    N, T = len(encs), encs[0]["omega_z"].shape[0]
    out = np.empty((N, T, d), dtype=np.float32)
    with torch.no_grad():
        for i, e in enumerate(encs):
            x = torch.from_numpy(e["omega_z"]).unsqueeze(1).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                z = enc(x.unsqueeze(0))
            out[i] = z.squeeze(0).float().cpu().numpy()
    return out


def build_cl_future(encs):
    N, T = len(encs), encs[0]["omega_z"].shape[0]
    out = np.empty((N, T, len(DELTAS)), dtype=np.float32)
    for i, e in enumerate(encs):
        cl = e["C_L"]
        for j, d in enumerate(DELTAS):
            for t in range(T):
                src = t + d
                out[i, t, j] = cl[src] if src < cl.shape[0] else cl[-1]
    return out


def case_of(encs): return np.array([e["case_id"] for e in encs])
def c_of(encs): return np.stack([[e["G"], e["D"], e["Y"]] for e in encs], axis=0).astype(np.float32)


def pr_of(arr_2d): return float(participation_ratio(torch.from_numpy(arr_2d.astype(np.float32))))


def probe(z, y, seed=0):
    n = z.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    nfit = max(2, int(0.75 * n))
    fit, ev = perm[:nfit], perm[nfit:]
    r = linear_probe_r2(
        torch.from_numpy(z.astype(np.float32)),
        torch.from_numpy(y.astype(np.float32)),
        torch.from_numpy(fit), torch.from_numpy(ev),
    )
    return float(r["r2_overall"]) if isinstance(r, dict) else float(r)


def split_static_dynamic(z_all, case_arr, d):
    cases = sorted(set(case_arr))
    means = np.stack([z_all[case_arr == cid].reshape(-1, d).mean(axis=0) for cid in cases])
    cid_to_idx = {cid: i for i, cid in enumerate(cases)}
    z_dyn = z_all.copy()
    for i in range(z_all.shape[0]):
        z_dyn[i] -= means[cid_to_idx[case_arr[i]]]
    return means, z_dyn


def fit_mlp_oos(X_train, y_train, X_eval, y_eval, device,
                seed=0, hidden=64, epochs=400, lr=1e-2):
    torch.manual_seed(seed)
    in_dim = X_train.shape[1]; out_dim = y_train.shape[1]
    X_tr = torch.from_numpy(X_train.astype(np.float32)).to(device)
    y_tr = torch.from_numpy(y_train.astype(np.float32)).to(device)
    X_ev = torch.from_numpy(X_eval.astype(np.float32)).to(device)
    y_ev = torch.from_numpy(y_eval.astype(np.float32)).to(device)
    model = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                          nn.Linear(hidden, out_dim)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        p = model(X_tr); loss = F.mse_loss(p, y_tr)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        p_ev = model(X_ev)
    ss_res = ((y_ev - p_ev) ** 2).sum(dim=0)
    ss_tot = ((y_ev - y_ev.mean(dim=0)) ** 2).sum(dim=0).clamp_min(1e-8)
    return float((1.0 - ss_res / ss_tot).mean().item())


def fit_mlp(X, y, device, seed=0, hidden=64, epochs=400, lr=1e-2):
    torch.manual_seed(seed)
    n, in_dim = X.shape; out_dim = y.shape[1]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    fit, ev = perm[: int(0.75 * n)], perm[int(0.75 * n):]
    X_t = torch.from_numpy(X.astype(np.float32)).to(device)
    y_t = torch.from_numpy(y.astype(np.float32)).to(device)
    model = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                          nn.Linear(hidden, out_dim)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        p = model(X_t[fit]); loss = F.mse_loss(p, y_t[fit])
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        p_ev = model(X_t[ev])
    y_ev = y_t[ev]
    ss_res = ((y_ev - p_ev) ** 2).sum(dim=0)
    ss_tot = ((y_ev - y_ev.mean(dim=0)) ** 2).sum(dim=0).clamp_min(1e-8)
    return float((1.0 - ss_res / ss_tot).mean().item())


def evaluate_run(run_dir: Path, iters: int, device: torch.device,
                 SPLITS, CL_FUTURE, MASK, case_split, c_split, T) -> list[dict]:
    ckpt = run_dir / f"checkpoint_iter{iters:06d}.pt"
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    args = blob["args"]
    d = int(args["d"])
    enc = HybridCNNViTEncoder(
        latent_dim=d,
        projection_norm=args.get("projection_norm", "batchnorm"),
    )
    state = {k.removeprefix("encoder."): v for k, v in blob["jepa_state_dict"].items()
             if k.startswith("encoder.")}
    enc.load_state_dict(state, strict=False)
    enc.eval().to(device)

    z = {s: encode(enc, SPLITS[s], d, device) for s in SPLITS}
    rows = []
    z_a_cl = z["test_a"][MASK["test_a"]]
    c_a_cl = c_split["test_a"][MASK["test_a"]]
    z_a_cl_flat = z_a_cl.reshape(-1, d)
    cl_a = CL_FUTURE["test_a"].reshape(-1, 3)
    ct_a = np.hstack([
        np.repeat(c_a_cl, T, axis=0),
        np.tile(np.arange(T)[:, None], (z_a_cl.shape[0], 1)).astype(np.float32),
    ])
    r2_cl_a = fit_mlp(z_a_cl_flat, cl_a, device)
    r2_ct_a = fit_mlp(ct_a, cl_a, device)
    z_a_flat = z["test_a"].reshape(-1, d)
    cm_a, z_dyn_a = split_static_dynamic(z["test_a"], case_split["test_a"], d)
    pr_a = pr_of(z_a_flat)
    pr_w_a = float(np.mean([
        pr_of(z_dyn_a[case_split["test_a"] == cid].reshape(-1, d))
        for cid in sorted(set(case_split["test_a"]))
    ]))
    r2_zc_a = probe(z_a_flat, np.repeat(c_split["test_a"], T, axis=0))
    rows.append({"d": d, "split": "test_a", "PR_all": pr_a, "PR_within": pr_w_a,
                 "r2_z_c": r2_zc_a, "r2_CL_future": r2_cl_a,
                 "r2_ct_baseline": r2_ct_a, "delta": r2_cl_a - r2_ct_a})

    for split in ("test_b", "test_c"):
        z_s = z[split]
        T_s = z_s.shape[1]
        z_s_cl = z_s[MASK[split]]
        c_s_cl = c_split[split][MASK[split]]
        z_s_cl_flat = z_s_cl.reshape(-1, d)
        cl_s = CL_FUTURE[split].reshape(-1, 3)
        ct_s = np.hstack([
            np.repeat(c_s_cl, T_s, axis=0),
            np.tile(np.arange(T_s)[:, None], (z_s_cl.shape[0], 1)).astype(np.float32),
        ])
        r2_cl_s = fit_mlp_oos(z_a_cl_flat, cl_a, z_s_cl_flat, cl_s, device)
        r2_ct_s = fit_mlp_oos(ct_a, cl_a, ct_s, cl_s, device)
        z_s_flat = z_s.reshape(-1, d)
        cm_s, z_dyn_s = split_static_dynamic(z_s, case_split[split], d)
        pr_s = pr_of(z_s_flat)
        pr_w_s = float(np.mean([
            pr_of(z_dyn_s[case_split[split] == cid].reshape(-1, d))
            for cid in sorted(set(case_split[split]))
        ]))
        r2_zc = probe(z_s_flat, np.repeat(c_split[split], T_s, axis=0))
        rows.append({"d": d, "split": split, "PR_all": pr_s, "PR_within": pr_w_s,
                     "r2_z_c": r2_zc, "r2_CL_future": r2_cl_s,
                     "r2_ct_baseline": r2_ct_s, "delta": r2_cl_s - r2_ct_s})
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Session 8 Step 5: d-sweep analysis")
    p.add_argument("--d32-run", type=str, required=True,
                   help="relative path to the d=32 best-grid-point run directory")
    p.add_argument("--d16-run", type=str, default="outputs/runs/session8/run_d16_best")
    p.add_argument("--d8-run", type=str, default="outputs/runs/session8/run_d8_best")
    p.add_argument("--iters", type=int, default=20000)
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--output", type=str, default="outputs/runs/session8/d_sweep.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = require_rtx6000(gpu_index=args.gpu)
    print(f"[d-sweep] device={device}", flush=True)

    SPLITS = {s: gather_encounters(s) for s in ("test_a", "test_b", "test_c")}
    CL_FUTURE_RAW = {s: build_cl_future(SPLITS[s]) for s in SPLITS}
    MASK = {s: np.isfinite(CL_FUTURE_RAW[s]).reshape(CL_FUTURE_RAW[s].shape[0], -1).all(axis=1)
            for s in SPLITS}
    CL_FUTURE = {s: CL_FUTURE_RAW[s][MASK[s]] for s in SPLITS}
    case_split = {s: case_of(SPLITS[s]) for s in SPLITS}
    c_split = {s: c_of(SPLITS[s]) for s in SPLITS}
    T = SPLITS["test_a"][0]["omega_z"].shape[0]

    all_rows = []
    for label, rel in [("d=8", args.d8_run), ("d=16", args.d16_run), ("d=32", args.d32_run)]:
        run_dir = REPO / rel
        ckpt = run_dir / f"checkpoint_iter{args.iters:06d}.pt"
        if not ckpt.exists():
            print(f"[d-sweep] MISSING {label} at {ckpt}", flush=True)
            continue
        print(f"[d-sweep] evaluating {label} from {run_dir}", flush=True)
        rows = evaluate_run(run_dir, args.iters, device,
                            SPLITS, CL_FUTURE, MASK, case_split, c_split, T)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_path = REPO / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[d-sweep] wrote {len(df)} rows to {out_path}", flush=True)
    print(df.round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
