"""Session 8 Step 3 verification: encode and compute Test B delta for R3 seed=42.

Quick check that R3 with seed=42 lands within the pass bracket [+0.05, +0.25].
Outside the bracket pauses Step 4 to investigate seed variance.

Encodes Test A and Test B from the R3-seed42 iter-20000 checkpoint, fits
MLPs as in the Session 7 pipeline, prints the delta_test_b. Wall-clock
budget ~2 min. Uses cuda:1 by default so the second card stays free for
the concurrent Step 4 chain.
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
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(8)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.models.encoder import HybridCNNViTEncoder  # noqa: E402
from src.utils.device import require_rtx6000  # noqa: E402


DELTAS = (8, 16, 24)
D = 32
DEFAULT_CKPT = "outputs/runs/session8/run_r3_seed42/checkpoint_iter020000.pt"

PREVENT = Path(os.environ.get("PREVENT_ROOT", "/home/carlos/PREVENT"))
CACHE = Path(os.environ.get("VORTEX_JEPA_CACHE", PREVENT / "data" / "processed" / "vortex-jepa"))


def gather_encounters(split_key: str) -> list[dict]:
    with open(REPO / "configs" / "splits" / "split_v1.json") as f:
        manifest = json.load(f)
    encs = []
    for cid, case in manifest["cases"].items():
        if split_key == "test_a" and case["split"] == "train":
            ks = case["test_a_encounter_indices"]
        elif split_key == "test_b" and case["split"] == "test_b":
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


def encode(enc, encs, device):
    N, T = len(encs), encs[0]["omega_z"].shape[0]
    out = np.empty((N, T, D), dtype=np.float32)
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


def c_of(encs):
    return np.stack([[e["G"], e["D"], e["Y"]] for e in encs], axis=0).astype(np.float32)


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
        p = model(X_tr)
        loss = F.mse_loss(p, y_tr)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        p_ev = model(X_ev)
    ss_res = ((y_ev - p_ev) ** 2).sum(dim=0)
    ss_tot = ((y_ev - y_ev.mean(dim=0)) ** 2).sum(dim=0).clamp_min(1e-8)
    return float((1.0 - ss_res / ss_tot).mean().item())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--bracket-min", type=float, default=0.05)
    p.add_argument("--bracket-max", type=float, default=0.25)
    p.add_argument("--output", type=str,
                   default="outputs/runs/session8/r3_seed42_eval.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = require_rtx6000(gpu_index=args.gpu)
    ckpt_path = REPO / args.checkpoint
    print(f"[eval-r3-seed42] device={device} ckpt={ckpt_path}", flush=True)

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    enc = HybridCNNViTEncoder(
        latent_dim=int(blob["args"]["d"]),
        projection_norm=blob["args"].get("projection_norm", "batchnorm"),
    )
    state = {k.removeprefix("encoder."): v for k, v in blob["jepa_state_dict"].items()
             if k.startswith("encoder.")}
    enc.load_state_dict(state, strict=False)
    enc.eval().to(device)

    SPLITS = {s: gather_encounters(s) for s in ("test_a", "test_b")}
    for s, encs in SPLITS.items():
        print(f"  {s}: {len(encs)} encs", flush=True)

    CL_FUTURE_RAW = {s: build_cl_future(SPLITS[s]) for s in SPLITS}
    MASK = {s: np.isfinite(CL_FUTURE_RAW[s]).reshape(CL_FUTURE_RAW[s].shape[0], -1).all(axis=1)
            for s in SPLITS}
    CL_FUTURE = {s: CL_FUTURE_RAW[s][MASK[s]] for s in SPLITS}

    z_a = encode(enc, SPLITS["test_a"], device)
    z_b = encode(enc, SPLITS["test_b"], device)
    T = z_a.shape[1]
    c_a = c_of(SPLITS["test_a"])
    c_b = c_of(SPLITS["test_b"])

    z_a_cl = z_a[MASK["test_a"]]; c_a_cl = c_a[MASK["test_a"]]
    z_b_cl = z_b[MASK["test_b"]]; c_b_cl = c_b[MASK["test_b"]]
    cl_a = CL_FUTURE["test_a"].reshape(-1, 3)
    cl_b = CL_FUTURE["test_b"].reshape(-1, 3)
    z_a_flat = z_a_cl.reshape(-1, D)
    z_b_flat = z_b_cl.reshape(-1, D)
    ct_a = np.hstack([np.repeat(c_a_cl, T, axis=0),
                      np.tile(np.arange(T)[:, None], (z_a_cl.shape[0], 1)).astype(np.float32)])
    ct_b = np.hstack([np.repeat(c_b_cl, T, axis=0),
                      np.tile(np.arange(T)[:, None], (z_b_cl.shape[0], 1)).astype(np.float32)])

    r2_cl = fit_mlp_oos(z_a_flat, cl_a, z_b_flat, cl_b, device)
    r2_ct = fit_mlp_oos(ct_a, cl_a, ct_b, cl_b, device)
    delta_b = r2_cl - r2_ct
    in_bracket = bool(args.bracket_min <= delta_b <= args.bracket_max)
    result = {
        "checkpoint": str(ckpt_path),
        "seed": int(blob["args"]["seed"]),
        "r2_cl_test_b": r2_cl,
        "r2_ct_test_b": r2_ct,
        "delta_test_b": delta_b,
        "bracket_min": args.bracket_min,
        "bracket_max": args.bracket_max,
        "in_bracket": in_bracket,
    }
    out_path = REPO / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)
    if not in_bracket:
        print(f"WARNING: R3 seed=42 Test B delta {delta_b:+.3f} is OUTSIDE "
              f"the pass bracket [{args.bracket_min:+.2f}, {args.bracket_max:+.2f}]. "
              f"Session 8 plan calls for pausing to investigate seed variance.",
              flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
