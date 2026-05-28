"""Session 8 Step 4: analyse the eta x lambda SIGReg grid + E10 PLDM reference.

Loads the iter-20000 checkpoint of every Session 8 grid run plus the
Session 7 R3 anchor (=E5, eta=0.01, lambda=0.1, the centre cell). For
each run, encodes Test A / Test B / Test C and computes the same metric
table as `notebooks/05_session7_full_evaluation.ipynb` Section 4.

Output:
- ``outputs/runs/session8/grid_analysis.csv`` with one row per
  (run_code, split) and the full per-split metrics.
- ``outputs/runs/session8/fig_grid_delta_b.png`` heatmap of
  delta_test_b across (eta, lambda).
- ``outputs/runs/session8/fig_grid_pr_all.png`` heatmap of PR_all.
- ``outputs/runs/session8/fig_grid_r2_z_c.png`` heatmap of r2(z->c).
- ``outputs/runs/session8/champion_table.csv`` comparing best SIGReg
  grid point, R3 Session 7 anchor, R1 PLDM default, E10 PLDM paper-tuned.

Runs on cuda:1 (RTX 6000 #2). After all grid + E10 runs land. Wall-clock
budget: ~10 min (10 checkpoints * ~1 min encoding + metric per checkpoint).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Limit CPU threads (avoids contention if a second job is on the other GPU)
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
D = 32

# (code, dir-relative, eta, lambda, kind, label-for-plot)
RUNS = [
    ("E1",  "outputs/runs/session8/run_e1_eta0p001_lam0p01", 0.001, 0.01, "jepa", "E1 eta=0.001 lam=0.01"),
    ("E2",  "outputs/runs/session8/run_e2_eta0p001_lam0p10", 0.001, 0.1,  "jepa", "E2 eta=0.001 lam=0.1"),
    ("E3",  "outputs/runs/session8/run_e3_eta0p001_lam1p00", 0.001, 1.0,  "jepa", "E3 eta=0.001 lam=1.0"),
    ("E4",  "outputs/runs/session8/run_e4_eta0p010_lam0p01", 0.01,  0.01, "jepa", "E4 eta=0.01 lam=0.01"),
    ("E5",  "outputs/runs/session7/run_r3_sigreg_obs_bn",    0.01,  0.1,  "jepa", "E5 = Session 7 R3 anchor"),
    ("E6",  "outputs/runs/session8/run_e6_eta0p010_lam1p00", 0.01,  1.0,  "jepa", "E6 eta=0.01 lam=1.0"),
    ("E7",  "outputs/runs/session8/run_e7_eta0p100_lam0p01", 0.1,   0.01, "jepa", "E7 eta=0.1 lam=0.01"),
    ("E8",  "outputs/runs/session8/run_e8_eta0p100_lam0p10", 0.1,   0.1,  "jepa", "E8 eta=0.1 lam=0.1"),
    ("E9",  "outputs/runs/session8/run_e9_eta0p100_lam1p00", 0.1,   1.0,  "jepa", "E9 eta=0.1 lam=1.0"),
    ("E10", "outputs/runs/session8/run_e10_pldm_paper_tuned",0.01,  None, "pldm", "E10 PLDM paper-tuned"),
    # Session 7 R1 PLDM (default unit weights) as a comparator for the champion table
    ("R1_S7","outputs/runs/session7/run_r1_pldm_obs_bn",     0.01,  None, "pldm", "R1 PLDM defaults (S7)"),
]

ETAS = [0.001, 0.01, 0.1]
LAMS = [0.01, 0.1, 1.0]


def load_encoder(run_dir: Path, kind: str, iters: int, device: torch.device) -> HybridCNNViTEncoder:
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
    with open(REPO / "configs" / "splits" / "split_v2.json") as f:
        manifest = json.load(f)
    encs = []
    for cid, case in manifest["cases"].items():
        if split_key == "test_a" and case["split"] == "train":
            ks = (case.get("val_encounter_indices") or case["test_a_encounter_indices"])
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
                    "case_id": cid,
                    "k": int(k),
                    "omega_z": np.asarray(g["omega_z"], dtype=np.float32),
                    "C_L": np.asarray(g["C_L"], dtype=np.float32),
                    "G": float(g.attrs["G"]),
                    "D": float(g.attrs["D"]),
                    "Y": float(g.attrs["Y"]),
                })
    return encs


def encode_split(enc, encs, device):
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


def split_static_dynamic(z_all, case_arr):
    cases = sorted(set(case_arr))
    means = np.stack([z_all[case_arr == cid].reshape(-1, D).mean(axis=0) for cid in cases])
    cid_to_idx = {cid: i for i, cid in enumerate(cases)}
    z_dyn = z_all.copy()
    for i in range(z_all.shape[0]):
        z_dyn[i] -= means[cid_to_idx[case_arr[i]]]
    return means, z_dyn


def fit_mlp(X, y, device, seed=0, hidden=64, epochs=400, lr=1e-2):
    torch.manual_seed(seed)
    n, in_dim = X.shape
    out_dim = y.shape[1]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    fit, ev = perm[: int(0.75 * n)], perm[int(0.75 * n):]
    X_t = torch.from_numpy(X.astype(np.float32)).to(device)
    y_t = torch.from_numpy(y.astype(np.float32)).to(device)
    model = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                          nn.Linear(hidden, out_dim)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        p = model(X_t[fit])
        loss = F.mse_loss(p, y_t[fit])
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        p_ev = model(X_t[ev])
    y_ev = y_t[ev]
    ss_res = ((y_ev - p_ev) ** 2).sum(dim=0)
    ss_tot = ((y_ev - y_ev.mean(dim=0)) ** 2).sum(dim=0).clamp_min(1e-8)
    return float((1.0 - ss_res / ss_tot).mean().item())


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
    p = argparse.ArgumentParser(description="Session 8 Step 4: grid analysis")
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument(
        "--iters",
        type=int,
        default=20000,
        help="Checkpoint iter to evaluate (default 20000).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="outputs/runs/session8",
    )
    p.add_argument(
        "--require-all",
        action="store_true",
        help="Fail if any run's checkpoint is missing (default: skip).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = require_rtx6000(gpu_index=args.gpu)
    print(f"[grid-analysis] device={device}", flush=True)

    print("[grid-analysis] gathering Test A / B / C encounters", flush=True)
    SPLITS = {s: gather_encounters(s) for s in ("test_a", "test_b", "test_c")}
    for s, encs in SPLITS.items():
        print(f"  {s}: {len(encs)} encs in {len(set(e['case_id'] for e in encs))} cases",
              flush=True)

    print("[grid-analysis] building CL_future tables", flush=True)
    CL_FUTURE_RAW = {s: build_cl_future(SPLITS[s]) for s in SPLITS}
    MASK = {s: np.isfinite(CL_FUTURE_RAW[s]).reshape(CL_FUTURE_RAW[s].shape[0], -1).all(axis=1)
            for s in SPLITS}
    CL_FUTURE = {s: CL_FUTURE_RAW[s][MASK[s]] for s in SPLITS}
    for s in SPLITS:
        n_drop = int((~MASK[s]).sum())
        if n_drop:
            print(f"  {s} CL: dropped {n_drop} non-finite encounters", flush=True)

    case_of_split = {s: case_of(SPLITS[s]) for s in SPLITS}
    c_of_split = {s: c_of(SPLITS[s]) for s in SPLITS}
    T = SPLITS["test_a"][0]["omega_z"].shape[0]

    rows = []
    for code, rel, eta, lam, kind, label in RUNS:
        run_dir = REPO / rel
        ckpt = run_dir / f"checkpoint_iter{args.iters:06d}.pt"
        if not ckpt.exists():
            msg = f"[grid-analysis] MISSING checkpoint for {code} at {ckpt}"
            if args.require_all:
                raise FileNotFoundError(msg)
            print(msg, flush=True)
            continue
        print(f"[grid-analysis] {code} {label}: loading + encoding", flush=True)
        enc = load_encoder(run_dir, kind, args.iters, device)
        z = {s: encode_split(enc, SPLITS[s], device) for s in SPLITS}

        # Test A in-sample
        z_a_flat = z["test_a"].reshape(-1, D)
        cm_a, z_dyn_a = split_static_dynamic(z["test_a"], case_of_split["test_a"])
        pr_a = pr_of(z_a_flat)
        pr_w_a = float(np.mean([
            pr_of(z_dyn_a[case_of_split["test_a"] == cid].reshape(-1, D))
            for cid in sorted(set(case_of_split["test_a"]))
        ]))
        r2_zc_a = probe(z_a_flat, np.repeat(c_of_split["test_a"], T, axis=0))
        r2_dyn_phase_a = probe(z_dyn_a.reshape(-1, D),
                               np.tile(np.arange(T, dtype=np.float32),
                                       z["test_a"].shape[0])[:, None])

        # Test A CL: filter
        z_a_cl = z["test_a"][MASK["test_a"]]
        c_a_cl = c_of_split["test_a"][MASK["test_a"]]
        z_a_cl_flat = z_a_cl.reshape(-1, D)
        cl_a = CL_FUTURE["test_a"].reshape(-1, 3)
        ct_a_flat = np.hstack([
            np.repeat(c_a_cl, T, axis=0),
            np.tile(np.arange(T)[:, None], (z_a_cl.shape[0], 1)).astype(np.float32),
        ])
        r2_cl_a = fit_mlp(z_a_cl_flat, cl_a, device)
        r2_ct_a = fit_mlp(ct_a_flat, cl_a, device)
        rows.append({
            "code": code, "eta": eta, "lambda": lam, "kind": kind,
            "label": label, "split": "test_a",
            "PR_all": pr_a, "PR_within": pr_w_a,
            "r2_z_c": r2_zc_a, "r2_dyn_phase": r2_dyn_phase_a,
            "r2_CL_future": r2_cl_a, "r2_ct_baseline": r2_ct_a,
            "delta": r2_cl_a - r2_ct_a,
        })

        # Test B / Test C OOS
        for split in ("test_b", "test_c"):
            z_s = z[split]
            T_s = z_s.shape[1]
            case_s = case_of_split[split]
            c_s = c_of_split[split]
            z_s_flat = z_s.reshape(-1, D)
            cm_s, z_dyn_s = split_static_dynamic(z_s, case_s)
            pr_s = pr_of(z_s_flat)
            pr_w_s = float(np.mean([
                pr_of(z_dyn_s[case_s == cid].reshape(-1, D))
                for cid in sorted(set(case_s))
            ]))
            r2_zc = probe(z_s_flat, np.repeat(c_s, T_s, axis=0))
            r2_dyn_phase = probe(z_dyn_s.reshape(-1, D),
                                 np.tile(np.arange(T_s, dtype=np.float32),
                                         z_s.shape[0])[:, None])

            z_s_cl = z_s[MASK[split]]
            c_s_cl = c_s[MASK[split]]
            z_s_cl_flat = z_s_cl.reshape(-1, D)
            cl_s = CL_FUTURE[split].reshape(-1, 3)
            ct_s_flat = np.hstack([
                np.repeat(c_s_cl, T_s, axis=0),
                np.tile(np.arange(T_s)[:, None], (z_s_cl.shape[0], 1)).astype(np.float32),
            ])
            r2_cl_s = fit_mlp_oos(z_a_cl_flat, cl_a, z_s_cl_flat, cl_s, device)
            r2_ct_s = fit_mlp_oos(ct_a_flat, cl_a, ct_s_flat, cl_s, device)
            rows.append({
                "code": code, "eta": eta, "lambda": lam, "kind": kind,
                "label": label, "split": split,
                "PR_all": pr_s, "PR_within": pr_w_s,
                "r2_z_c": r2_zc, "r2_dyn_phase": r2_dyn_phase,
                "r2_CL_future": r2_cl_s, "r2_ct_baseline": r2_ct_s,
                "delta": r2_cl_s - r2_ct_s,
            })

        del enc, z
        torch.cuda.empty_cache()

    out_dir = REPO / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "grid_analysis.csv", index=False)
    print(f"[grid-analysis] wrote {len(df)} rows to {out_dir / 'grid_analysis.csv'}",
          flush=True)

    # Heatmaps for SIGReg grid (E1-E9)
    grid_codes = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"]
    grid_df = df[df["code"].isin(grid_codes) & (df["split"] == "test_b")].copy()
    if len(grid_df) == 9:
        for metric, fname, title, cmap in [
            ("delta", "fig_grid_delta_b.png", "delta_test_b across (eta, lambda)", "RdBu_r"),
            ("PR_all", "fig_grid_pr_all.png", "PR_all on Test B across (eta, lambda)", "viridis"),
            ("r2_z_c", "fig_grid_r2_z_c.png", "r2(z->c) on Test B across (eta, lambda)", "viridis"),
        ]:
            mat = np.zeros((len(ETAS), len(LAMS)))
            for _, r in grid_df.iterrows():
                i_eta = ETAS.index(r["eta"])
                j_lam = LAMS.index(r["lambda"])
                mat[i_eta, j_lam] = r[metric]
            fig, ax = plt.subplots(figsize=(6, 5))
            vmin, vmax = (mat.min(), mat.max())
            if metric == "delta":
                vmax = max(abs(vmin), abs(vmax))
                vmin = -vmax
            im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
            for i in range(len(ETAS)):
                for j in range(len(LAMS)):
                    ax.text(j, i, f"{mat[i, j]:+.3f}", ha="center", va="center",
                            color="black" if abs(mat[i, j]) < 0.5 * (vmax or 1.0) else "white",
                            fontsize=10)
            ax.set_xticks(range(len(LAMS)))
            ax.set_yticks(range(len(ETAS)))
            ax.set_xticklabels([f"lam={L}" for L in LAMS])
            ax.set_yticklabels([f"eta={E}" for E in ETAS])
            ax.set_title(title)
            fig.colorbar(im, ax=ax, label=metric)
            fig.tight_layout()
            fig.savefig(out_dir / fname, dpi=130, bbox_inches="tight")
            plt.close(fig)
            print(f"[grid-analysis] saved {fname}", flush=True)

        # Identify best (eta*, lambda*) by max delta_test_b
        idx_best = grid_df["delta"].idxmax()
        best = grid_df.loc[idx_best]
        best_dir = next((rel for code, rel, _, _, _, _ in RUNS if code == best["code"]), None)
        print(f"[grid-analysis] BEST SIGReg grid point: {best['code']} "
              f"eta={best['eta']} lambda={best['lambda']} delta_b={best['delta']:+.3f} "
              f"dir={best_dir}",
              flush=True)
        with open(out_dir / "best_grid_point.json", "w") as f:
            json.dump({
                "code": best["code"],
                "eta": float(best["eta"]),
                "lambda": float(best["lambda"]),
                "delta_b": float(best["delta"]),
                "run_dir": best_dir,
            }, f, indent=2)

    # Champion table (Test B comparison)
    champ_codes = []
    if (grid_df["delta"].notna()).any():
        idx_best = grid_df["delta"].idxmax()
        champ_codes.append(grid_df.loc[idx_best, "code"])
    champ_codes += ["E5", "E10", "R1_S7"]
    champ = df[(df["code"].isin(champ_codes)) & (df["split"] == "test_b")].copy()
    champ = champ.drop_duplicates(subset=["code"])
    champ = champ[["code", "label", "eta", "lambda", "PR_all", "r2_z_c",
                   "r2_CL_future", "r2_ct_baseline", "delta"]]
    champ.to_csv(out_dir / "champion_table.csv", index=False)
    print(f"[grid-analysis] champion table:", flush=True)
    print(champ.round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
