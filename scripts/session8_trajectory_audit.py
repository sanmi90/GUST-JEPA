"""Session 8 Step 1: trajectory audit for Session 7 R1/R2/R3 runs.

Loads checkpoints at iter 2000, 4000, ..., 20000 (10 per run, 30 total),
encodes every Test A and Test B encounter at each checkpoint, and computes
the full per-split metric table per checkpoint. Saves the result as
``outputs/runs/session8/trajectory_audit.csv``.

Sub-deliverable: an extra "within-Test-B" linear-probe r2 column. For
R2 (whose Session 7 final Test B delta is -0.85), this distinguishes
"latent itself is anti-generalising" from "distribution shift between
Test A and Test B is hurting the cross-split MLP fit."

Runs on the second RTX 6000 (``--gpu 1`` = ``cuda:3`` in torch view).
This frees the first RTX 6000 for the concurrent Step 3 R3-seed=42 run.

Wall-clock budget: ~10 minutes. 30 checkpoints x ~85 encounters x one
encoder forward each = ~2500 encoder passes on a Blackwell. Plus the
per-checkpoint MLP fits (cheap on Test A and Test B sizes).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Limit CPU threading to avoid contention with the concurrent training process
# on the other RTX 6000 (CPU is shared). 8 threads is plenty for the closed-
# form lstsq probes; the MLP fits move to GPU below.
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


SESSION7_ROOT = REPO / "outputs" / "runs" / "session7"
RUN_SPECS = {
    "R1 PLDM+OBS+BN": (SESSION7_ROOT / "run_r1_pldm_obs_bn", "pldm"),
    "R2 PLDM only BN": (SESSION7_ROOT / "run_r2_pldm_only_bn", "pldm"),
    "R3 SIGReg+OBS+BN": (SESSION7_ROOT / "run_r3_sigreg_obs_bn", "jepa"),
}
ITERS = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]
DELTAS = (8, 16, 24)
D = 32

PREVENT = Path(os.environ.get("PREVENT_ROOT", "/home/carlos/PREVENT"))
CACHE = Path(os.environ.get("VORTEX_JEPA_CACHE", PREVENT / "data" / "processed" / "vortex-jepa"))


def load_encoder(run_dir: Path, iters: int, kind: str, device: torch.device) -> HybridCNNViTEncoder:
    ckpt = run_dir / f"checkpoint_iter{iters:06d}.pt"
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    args = blob["args"]
    enc = HybridCNNViTEncoder(
        latent_dim=int(args["d"]),
        projection_norm=args.get("projection_norm", "batchnorm"),
    )
    state_key = "wrapper_state_dict" if kind == "pldm" else "jepa_state_dict"
    state = {
        k.removeprefix("encoder."): v
        for k, v in blob[state_key].items()
        if k.startswith("encoder.")
    }
    enc.load_state_dict(state, strict=False)
    return enc.eval().to(device)


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
                    "case_id": cid,
                    "k": int(k),
                    "omega_z": np.asarray(g["omega_z"], dtype=np.float32),
                    "C_L": np.asarray(g["C_L"], dtype=np.float32),
                    "G": float(g.attrs["G"]),
                    "D": float(g.attrs["D"]),
                    "Y": float(g.attrs["Y"]),
                })
    return encs


def encode_split(enc: HybridCNNViTEncoder, encs: list[dict], device: torch.device) -> np.ndarray:
    N = len(encs)
    T = encs[0]["omega_z"].shape[0]
    out = np.empty((N, T, D), dtype=np.float32)
    with torch.no_grad():
        for i, e in enumerate(encs):
            x = torch.from_numpy(e["omega_z"]).unsqueeze(1).to(device)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                z = enc(x.unsqueeze(0))
            out[i] = z.squeeze(0).float().cpu().numpy()
    return out


def build_cl_future(encs: list[dict]) -> np.ndarray:
    N = len(encs)
    T = encs[0]["omega_z"].shape[0]
    out = np.empty((N, T, len(DELTAS)), dtype=np.float32)
    for i, e in enumerate(encs):
        cl = e["C_L"]
        for j, d in enumerate(DELTAS):
            for t in range(T):
                src = t + d
                out[i, t, j] = cl[src] if src < cl.shape[0] else cl[-1]
    return out


def case_of(encs: list[dict]) -> np.ndarray:
    return np.array([e["case_id"] for e in encs])


def c_of(encs: list[dict]) -> np.ndarray:
    return np.stack([[e["G"], e["D"], e["Y"]] for e in encs], axis=0).astype(np.float32)


def pr_of(arr_2d: np.ndarray) -> float:
    return float(participation_ratio(torch.from_numpy(arr_2d.astype(np.float32))))


def probe(z: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    n = z.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    nfit = max(2, int(0.75 * n))
    fit, ev = perm[:nfit], perm[nfit:]
    r = linear_probe_r2(
        torch.from_numpy(z.astype(np.float32)),
        torch.from_numpy(y.astype(np.float32)),
        torch.from_numpy(fit),
        torch.from_numpy(ev),
    )
    return float(r["r2_overall"]) if isinstance(r, dict) else float(r)


def split_static_dynamic(z_all: np.ndarray, case_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cases = sorted(set(case_arr))
    means = np.stack([z_all[case_arr == cid].reshape(-1, D).mean(axis=0) for cid in cases])
    cid_to_idx = {cid: i for i, cid in enumerate(cases)}
    z_dyn = z_all.copy()
    for i in range(z_all.shape[0]):
        z_dyn[i] -= means[cid_to_idx[case_arr[i]]]
    return means, z_dyn


def fit_mlp(X: np.ndarray, y: np.ndarray, device: torch.device, seed: int = 0,
            hidden: int = 64, epochs: int = 400, lr: float = 1e-2) -> float:
    """75/25 fit/eval MLP on GPU for the in-split r2 estimate."""
    torch.manual_seed(seed)
    n, in_dim = X.shape
    out_dim = y.shape[1]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    fit, ev = perm[: int(0.75 * n)], perm[int(0.75 * n):]
    X_t = torch.from_numpy(X.astype(np.float32)).to(device)
    y_t = torch.from_numpy(y.astype(np.float32)).to(device)
    model = nn.Sequential(
        nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim)
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X_fit, y_fit = X_t[fit], y_t[fit]
    for _ in range(epochs):
        model.train()
        p = model(X_fit)
        loss = F.mse_loss(p, y_fit)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        p_ev = model(X_t[ev])
    y_ev = y_t[ev]
    ss_res = ((y_ev - p_ev) ** 2).sum(dim=0)
    ss_tot = ((y_ev - y_ev.mean(dim=0)) ** 2).sum(dim=0).clamp_min(1e-8)
    return float((1.0 - ss_res / ss_tot).mean().item())


def fit_mlp_oos(X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray, y_eval: np.ndarray,
                device: torch.device, seed: int = 0, hidden: int = 64,
                epochs: int = 400, lr: float = 1e-2) -> float:
    """Train on (X_train, y_train), evaluate on (X_eval, y_eval). GPU."""
    torch.manual_seed(seed)
    in_dim = X_train.shape[1]
    out_dim = y_train.shape[1]
    X_tr = torch.from_numpy(X_train.astype(np.float32)).to(device)
    y_tr = torch.from_numpy(y_train.astype(np.float32)).to(device)
    X_ev = torch.from_numpy(X_eval.astype(np.float32)).to(device)
    y_ev = torch.from_numpy(y_eval.astype(np.float32)).to(device)
    model = nn.Sequential(
        nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim)
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        p = model(X_tr)
        loss = F.mse_loss(p, y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        p_ev = model(X_ev)
    ss_res = ((y_ev - p_ev) ** 2).sum(dim=0)
    ss_tot = ((y_ev - y_ev.mean(dim=0)) ** 2).sum(dim=0).clamp_min(1e-8)
    return float((1.0 - ss_res / ss_tot).mean().item())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Session 8 Step 1: trajectory audit")
    p.add_argument(
        "--gpu",
        type=int,
        default=1,
        help="RTX 6000 index (default 1 = second card; cuda:0 expected busy with Step 3).",
    )
    p.add_argument(
        "--output",
        type=str,
        default="outputs/runs/session8/trajectory_audit.csv",
        help="Output CSV with per-(run, iter, split) metrics.",
    )
    p.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU (slow, ~30 min). Default uses --gpu 1.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = require_rtx6000(gpu_index=args.gpu)
    print(f"[trajectory] device={device}", flush=True)

    print("[trajectory] gathering Test A and Test B encounters", flush=True)
    SPLITS = {split: gather_encounters(split) for split in ("test_a", "test_b")}
    for split, encs in SPLITS.items():
        n_cases = len(set(e["case_id"] for e in encs))
        print(f"  {split}: {len(encs)} encs in {n_cases} cases", flush=True)

    print("[trajectory] building CL_future tables", flush=True)
    CL_FUTURE_RAW = {split: build_cl_future(SPLITS[split]) for split in SPLITS}
    CL_VALID_MASK = {}
    for split, cl in CL_FUTURE_RAW.items():
        per_enc_finite = np.isfinite(cl).reshape(cl.shape[0], -1).all(axis=1)
        CL_VALID_MASK[split] = per_enc_finite
        n_kept = int(per_enc_finite.sum())
        print(f"  cl_future {split}: kept {n_kept}/{cl.shape[0]} encounters", flush=True)
    CL_FUTURE = {split: CL_FUTURE_RAW[split][CL_VALID_MASK[split]] for split in SPLITS}

    case_a = case_of(SPLITS["test_a"])
    case_b = case_of(SPLITS["test_b"])
    c_a = c_of(SPLITS["test_a"])
    c_b = c_of(SPLITS["test_b"])

    rows: list[dict] = []
    for run_label, (run_dir, kind) in RUN_SPECS.items():
        for iters in ITERS:
            ckpt_path = run_dir / f"checkpoint_iter{iters:06d}.pt"
            if not ckpt_path.exists():
                print(f"  MISSING: {run_label} iter {iters}", flush=True)
                continue
            print(f"[trajectory] {run_label} iter {iters}: encoding", flush=True)
            enc = load_encoder(run_dir, iters, kind, device)
            z_a = encode_split(enc, SPLITS["test_a"], device)
            z_b = encode_split(enc, SPLITS["test_b"], device)
            print(f"[trajectory] {run_label} iter {iters}: computing metrics", flush=True)
            T_a = z_a.shape[1]
            T_b = z_b.shape[1]
            assert T_a == T_b == 120

            # PR + linear probes on z (use FULL split; no NaN filter for these)
            z_a_flat = z_a.reshape(-1, D)
            z_b_flat = z_b.reshape(-1, D)
            cm_a, z_dyn_a = split_static_dynamic(z_a, case_a)
            cm_b, z_dyn_b = split_static_dynamic(z_b, case_b)
            pr_a = pr_of(z_a_flat)
            pr_b = pr_of(z_b_flat)
            pr_w_a = float(np.mean([
                pr_of(z_dyn_a[case_a == cid].reshape(-1, D)) for cid in sorted(set(case_a))
            ]))
            pr_w_b = float(np.mean([
                pr_of(z_dyn_b[case_b == cid].reshape(-1, D)) for cid in sorted(set(case_b))
            ]))
            r2_zc_a = probe(z_a_flat, np.repeat(c_a, T_a, axis=0))
            r2_zc_b = probe(z_b_flat, np.repeat(c_b, T_b, axis=0))
            r2_dphase_a = probe(
                z_dyn_a.reshape(-1, D),
                np.tile(np.arange(T_a, dtype=np.float32), z_a.shape[0])[:, None],
            )
            r2_dphase_b = probe(
                z_dyn_b.reshape(-1, D),
                np.tile(np.arange(T_b, dtype=np.float32), z_b.shape[0])[:, None],
            )

            # CL future MLPs (CL-valid filter)
            mask_a = CL_VALID_MASK["test_a"]
            mask_b = CL_VALID_MASK["test_b"]
            z_a_cl = z_a[mask_a]
            z_b_cl = z_b[mask_b]
            c_a_cl = c_a[mask_a]
            c_b_cl = c_b[mask_b]
            cl_a = CL_FUTURE["test_a"].reshape(-1, 3)
            cl_b = CL_FUTURE["test_b"].reshape(-1, 3)
            z_a_cl_flat = z_a_cl.reshape(-1, D)
            z_b_cl_flat = z_b_cl.reshape(-1, D)
            ct_a_flat = np.hstack([
                np.repeat(c_a_cl, T_a, axis=0),
                np.tile(np.arange(T_a)[:, None], (z_a_cl.shape[0], 1)).astype(np.float32),
            ])
            ct_b_flat = np.hstack([
                np.repeat(c_b_cl, T_b, axis=0),
                np.tile(np.arange(T_b)[:, None], (z_b_cl.shape[0], 1)).astype(np.float32),
            ])

            # Test A in-sample r2 and (c,t) baseline
            r2_cl_a = fit_mlp(z_a_cl_flat, cl_a, device)
            r2_ct_a = fit_mlp(ct_a_flat, cl_a, device)
            delta_a = r2_cl_a - r2_ct_a

            # Test B OOS r2 (Session 7 method): fit on Test A, evaluate on Test B
            r2_cl_b_oos = fit_mlp_oos(z_a_cl_flat, cl_a, z_b_cl_flat, cl_b, device)
            r2_ct_b_oos = fit_mlp_oos(ct_a_flat, cl_a, ct_b_flat, cl_b, device)
            delta_b = r2_cl_b_oos - r2_ct_b_oos

            # Within-Test-B probe (R2 anomaly investigation): fit on random 75%
            # of Test B latents, evaluate on held-out 25%. Same for (c,t).
            r2_cl_b_within = fit_mlp(z_b_cl_flat, cl_b, device)
            r2_ct_b_within = fit_mlp(ct_b_flat, cl_b, device)
            delta_b_within = r2_cl_b_within - r2_ct_b_within

            rows.append({
                "run": run_label, "iter": iters,
                "PR_all_a": pr_a, "PR_within_a": pr_w_a, "r2_z_c_a": r2_zc_a,
                "r2_dyn_phase_a": r2_dphase_a,
                "r2_CL_future_a": r2_cl_a, "r2_ct_baseline_a": r2_ct_a, "delta_a": delta_a,
                "PR_all_b": pr_b, "PR_within_b": pr_w_b, "r2_z_c_b": r2_zc_b,
                "r2_dyn_phase_b": r2_dphase_b,
                "r2_CL_future_b_oos": r2_cl_b_oos, "r2_ct_baseline_b_oos": r2_ct_b_oos,
                "delta_b": delta_b,
                "r2_CL_future_b_within": r2_cl_b_within,
                "r2_ct_baseline_b_within": r2_ct_b_within,
                "delta_b_within": delta_b_within,
            })

            print(f"[trajectory] {run_label} iter {iters}: delta_a={delta_a:+.3f} "
                  f"delta_b={delta_b:+.3f} delta_b_within={delta_b_within:+.3f}", flush=True)
            # Free GPU memory for the next checkpoint
            del enc, z_a, z_b
            torch.cuda.empty_cache()

    out_path = REPO / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"[trajectory] wrote {len(df)} rows to {out_path}", flush=True)
    print(df.round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
