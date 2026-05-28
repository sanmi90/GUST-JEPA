"""Session 9 Step 1: analyse the lambda bisection at the production point.

Loads the iter-20000 checkpoint of every Session 9 bisection run plus the
two anchors:
  - E4 (Session 8, eta=0.01, lambda=0.01) -- anchor in the bisection table
  - E5 (= Session 7 R3, eta=0.01, lambda=0.1) -- anchor in the bisection table

Evaluates Test A / B / C with the same per-split metric table as
``session8_grid_analysis.py``. Identifies lambda* = argmax delta_test_b
across {F1, F2, E4, F3, E5} at seed=0 and writes the best lambda to
``outputs/runs/session9/best_lambda_star.json`` for the F4/F5 launchers.

If F4 (seed=42) and F5 (seed=123) checkpoints are also on disk at lambda*,
the seed-variance bound is computed against the seed=0 result and reported.

Runs on cuda:1 (RTX 6000 #2 by default). Wall-clock budget: ~10 min for
the seed=0 table (5 checkpoints x ~1 min each) + ~5 min for any F4/F5.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Limit CPU threads (avoid contention if a second job is on the other GPU)
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

# (code, dir-relative, lambda, seed, label)
# E4 and E5 anchors stay seed=0; F1/F2/F3 are the new seed=0 bisection points.
# F4/F5 are added at lambda* once Step 1 identifies it.
BISECTION_SEED0 = [
    ("F1", "outputs/runs/session9/run_f1_lam0p001_seed0", 0.001, 0, "F1 lam=0.001 (new)"),
    ("F2", "outputs/runs/session9/run_f2_lam0p003_seed0", 0.003, 0, "F2 lam=0.003 (new)"),
    ("E4", "outputs/runs/session8/run_e4_eta0p010_lam0p01", 0.01, 0, "E4 lam=0.01 (S8 anchor)"),
    ("F3", "outputs/runs/session9/run_f3_lam0p030_seed0", 0.03, 0, "F3 lam=0.03 (new)"),
    ("E5", "outputs/runs/session7/run_r3_sigreg_obs_bn", 0.1, 0, "E5 lam=0.1 (S7 R3 anchor)"),
]


def load_encoder(run_dir: Path, iters: int, device: torch.device) -> HybridCNNViTEncoder:
    ckpt = run_dir / f"checkpoint_iter{iters:06d}.pt"
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    args = blob["args"]
    enc = HybridCNNViTEncoder(
        latent_dim=int(args["d"]),
        projection_norm=args.get("projection_norm", "batchnorm"),
    )
    state = {
        k.removeprefix("encoder."): v
        for k, v in blob["jepa_state_dict"].items()
        if k.startswith("encoder.")
    }
    enc.load_state_dict(state, strict=False)
    enc = enc.eval().to(device)
    # Attach OmegaPipeline if the checkpoint was trained with one. Used by
    # ``encode_split`` to preprocess inputs identically to training.
    manifest_rel = args.get("omega_pipeline_manifest")
    if manifest_rel:
        from src.data.omega_pipeline import OmegaPipeline
        manifest = Path(manifest_rel)
        if not manifest.is_absolute():
            manifest = REPO / manifest
        enc._omega_pipeline = OmegaPipeline.from_manifest(manifest)
    return enc


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
    pipe = getattr(enc, "_omega_pipeline", None)
    with torch.no_grad():
        for i, e in enumerate(encs):
            omega = e["omega_z"]
            if pipe is not None:
                omega = pipe.preprocess_raw(omega, e["case_id"], int(e["k"]))
            x = torch.from_numpy(omega).unsqueeze(1).to(device)
            if pipe is not None:
                x = pipe.normalize(x)
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


def evaluate_one(run_dir: Path, code: str, lam: float, seed: int, label: str,
                 splits: dict, cl_future: dict, mask: dict, case_of_split: dict,
                 c_of_split: dict, T: int, iters: int, device) -> list[dict]:
    """Evaluate one checkpoint across all three splits. Returns the new rows."""
    enc = load_encoder(run_dir, iters, device)
    z = {s: encode_split(enc, splits[s], device) for s in splits}

    # Test A in-sample
    z_a_flat = z["test_a"].reshape(-1, D)
    cm_a, z_dyn_a = split_static_dynamic(z["test_a"], case_of_split["test_a"])
    pr_a = pr_of(z_a_flat)
    pr_w_a = float(np.mean([
        pr_of(z_dyn_a[case_of_split["test_a"] == cid].reshape(-1, D))
        for cid in sorted(set(case_of_split["test_a"]))
    ]))
    r2_zc_a = probe(z_a_flat, np.repeat(c_of_split["test_a"], T, axis=0))
    r2_dyn_phase_a = probe(
        z_dyn_a.reshape(-1, D),
        np.tile(np.arange(T, dtype=np.float32), z["test_a"].shape[0])[:, None],
    )

    z_a_cl = z["test_a"][mask["test_a"]]
    c_a_cl = c_of_split["test_a"][mask["test_a"]]
    z_a_cl_flat = z_a_cl.reshape(-1, D)
    cl_a = cl_future["test_a"].reshape(-1, 3)
    ct_a_flat = np.hstack([
        np.repeat(c_a_cl, T, axis=0),
        np.tile(np.arange(T)[:, None], (z_a_cl.shape[0], 1)).astype(np.float32),
    ])
    r2_cl_a = fit_mlp(z_a_cl_flat, cl_a, device)
    r2_ct_a = fit_mlp(ct_a_flat, cl_a, device)

    rows = [{
        "code": code, "lambda": lam, "seed": seed, "label": label, "split": "test_a",
        "PR_all": pr_a, "PR_within": pr_w_a,
        "r2_z_c": r2_zc_a, "r2_dyn_phase": r2_dyn_phase_a,
        "r2_CL_future": r2_cl_a, "r2_ct_baseline": r2_ct_a,
        "delta": r2_cl_a - r2_ct_a,
    }]

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
        r2_dyn_phase = probe(
            z_dyn_s.reshape(-1, D),
            np.tile(np.arange(T_s, dtype=np.float32), z_s.shape[0])[:, None],
        )

        z_s_cl = z_s[mask[split]]
        c_s_cl = c_s[mask[split]]
        z_s_cl_flat = z_s_cl.reshape(-1, D)
        cl_s = cl_future[split].reshape(-1, 3)
        ct_s_flat = np.hstack([
            np.repeat(c_s_cl, T_s, axis=0),
            np.tile(np.arange(T_s)[:, None], (z_s_cl.shape[0], 1)).astype(np.float32),
        ])
        r2_cl_s = fit_mlp_oos(z_a_cl_flat, cl_a, z_s_cl_flat, cl_s, device)
        r2_ct_s = fit_mlp_oos(ct_a_flat, cl_a, ct_s_flat, cl_s, device)
        rows.append({
            "code": code, "lambda": lam, "seed": seed, "label": label, "split": split,
            "PR_all": pr_s, "PR_within": pr_w_s,
            "r2_z_c": r2_zc, "r2_dyn_phase": r2_dyn_phase,
            "r2_CL_future": r2_cl_s, "r2_ct_baseline": r2_ct_s,
            "delta": r2_cl_s - r2_ct_s,
        })

    del enc, z
    torch.cuda.empty_cache()
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Session 9 Step 1: lambda bisection analysis")
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--iters", type=int, default=20000)
    p.add_argument("--output-dir", type=str, default="outputs/runs/session9")
    p.add_argument(
        "--seed-variance",
        action="store_true",
        help=(
            "Also include F4 (seed=42) and F5 (seed=123) checkpoints if present, "
            "computing seed-variance bound vs the seed=0 result at lambda*."
        ),
    )
    p.add_argument(
        "--require-all-seed0",
        action="store_true",
        help="Fail if any seed=0 bisection checkpoint is missing.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = require_rtx6000(gpu_index=args.gpu)
    print(f"[bisection-analysis] device={device}", flush=True)

    print("[bisection-analysis] gathering Test A / B / C encounters", flush=True)
    SPLITS = {s: gather_encounters(s) for s in ("test_a", "test_b", "test_c")}
    for s, encs in SPLITS.items():
        print(f"  {s}: {len(encs)} encs in {len(set(e['case_id'] for e in encs))} cases",
              flush=True)

    print("[bisection-analysis] building CL_future tables", flush=True)
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
    for code, rel, lam, seed, label in BISECTION_SEED0:
        run_dir = REPO / rel
        ckpt = run_dir / f"checkpoint_iter{args.iters:06d}.pt"
        if not ckpt.exists():
            msg = f"[bisection-analysis] MISSING checkpoint for {code} at {ckpt}"
            if args.require_all_seed0:
                raise FileNotFoundError(msg)
            print(msg, flush=True)
            continue
        print(f"[bisection-analysis] {code} {label}: loading + encoding", flush=True)
        rows.extend(evaluate_one(
            run_dir, code, lam, seed, label,
            SPLITS, CL_FUTURE, MASK, case_of_split, c_of_split, T,
            args.iters, device,
        ))

    out_dir = REPO / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Identify lambda* from seed=0 set so far. We need all 5 codes available
    # to lock the bisection winner.
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "bisection_seed0.csv", index=False)
    print(f"[bisection-analysis] wrote {len(df)} rows to {out_dir / 'bisection_seed0.csv'}",
          flush=True)

    seed0_b = df[(df["split"] == "test_b") & (df["seed"] == 0)].copy()
    seed0_b = seed0_b.sort_values("lambda").reset_index(drop=True)
    print("[bisection-analysis] seed=0 Test B summary:", flush=True)
    print(seed0_b[["code", "lambda", "PR_all", "r2_z_c", "r2_CL_future",
                   "r2_ct_baseline", "delta"]].round(3).to_string(index=False),
          flush=True)

    best_codes = {"F1", "F2", "E4", "F3", "E5"}
    present = set(seed0_b["code"].tolist())
    if not best_codes.issubset(present):
        missing = best_codes - present
        print(f"[bisection-analysis] WARNING: missing seed=0 codes {sorted(missing)};",
              " writing partial best_lambda_star.json (re-run after they land).",
              flush=True)

    if not seed0_b.empty:
        idx_best = seed0_b["delta"].idxmax()
        best = seed0_b.loc[idx_best]
        best_lambda = float(best["lambda"])
        best_delta = float(best["delta"])
        best_code = str(best["code"])
        # The run_dir for the F4/F5 launcher.
        run_dir = next(rel for code, rel, _, _, _ in BISECTION_SEED0 if code == best_code)
        out_json = {
            "best_code": best_code,
            "best_lambda": best_lambda,
            "best_delta_test_b": best_delta,
            "best_run_dir": run_dir,
            "complete": best_codes.issubset(present),
        }
        with open(out_dir / "best_lambda_star.json", "w") as f:
            json.dump(out_json, f, indent=2)
        print(f"[bisection-analysis] BEST seed=0: {best_code} lambda={best_lambda} "
              f"delta_b={best_delta:+.3f}", flush=True)
        print(f"[bisection-analysis] wrote {out_dir / 'best_lambda_star.json'}", flush=True)

    # Bisection curve plot
    if not seed0_b.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.semilogx(seed0_b["lambda"], seed0_b["delta"], marker="o", linewidth=2,
                    color="tab:blue")
        for _, r in seed0_b.iterrows():
            ax.annotate(r["code"], xy=(r["lambda"], r["delta"]),
                        xytext=(5, 5), textcoords="offset points", fontsize=9)
        ax.axhline(0.0, color="black", linewidth=0.5, linestyle="--")
        ax.set_xlabel("lambda (SIGReg weight)")
        ax.set_ylabel("delta_test_b = r2(z) - r2(c, t)")
        ax.set_title("Session 9 lambda bisection at (d=32, eta=0.01), seed=0")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig_path = out_dir / "fig_bisection_curve_seed0.png"
        fig.savefig(fig_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"[bisection-analysis] saved {fig_path}", flush=True)

    # Optional: seed-variance evaluation at lambda*.
    if args.seed_variance:
        with open(out_dir / "best_lambda_star.json") as f:
            best = json.load(f)
        best_lam = float(best["best_lambda"])
        lam_tag = str(best_lam).replace(".", "p")
        variance_rows = []
        for code, seed in (("F4", 42), ("F5", 123)):
            run_dir = REPO / f"outputs/runs/session9/run_{code.lower()}_lam{lam_tag}_seed{seed}"
            ckpt = run_dir / f"checkpoint_iter{args.iters:06d}.pt"
            if not ckpt.exists():
                print(f"[bisection-analysis] {code} checkpoint not yet at {ckpt}", flush=True)
                continue
            print(f"[bisection-analysis] {code} lam={best_lam} seed={seed}: encoding",
                  flush=True)
            label = f"{code} lam={best_lam} seed={seed}"
            variance_rows.extend(evaluate_one(
                run_dir, code, best_lam, seed, label,
                SPLITS, CL_FUTURE, MASK, case_of_split, c_of_split, T,
                args.iters, device,
            ))
        if variance_rows:
            vdf = pd.DataFrame(variance_rows)
            vdf.to_csv(out_dir / "bisection_seed_variance.csv", index=False)
            seed0_at_star = seed0_b[seed0_b["lambda"] == best_lam]["delta"].iloc[0] \
                if (seed0_b["lambda"] == best_lam).any() else None
            print("[bisection-analysis] seed-variance Test B (seed=0 delta=" +
                  (f"{seed0_at_star:+.3f}" if seed0_at_star is not None else "n/a") + "):",
                  flush=True)
            print(vdf[vdf["split"] == "test_b"][
                ["code", "seed", "lambda", "delta"]
            ].round(3).to_string(index=False), flush=True)
            if seed0_at_star is not None:
                for _, r in vdf[vdf["split"] == "test_b"].iterrows():
                    diff = r["delta"] - seed0_at_star
                    flag = "PASS" if abs(diff) <= 0.03 else "FAIL"
                    print(f"  {r['code']} seed={r['seed']}: delta_b={r['delta']:+.3f} "
                          f"diff vs seed=0={diff:+.3f} ({flag})", flush=True)


if __name__ == "__main__":
    main()
