"""Session 16, Experiment 1, Part (a-ter): three user-prompted follow-ups
to the D118-bis nonlinear-recovery finding.

(a) PER-SEED KernelRidge(RBF) on the 4 production + Thrust-6 seed retrains.
    Question: does the nonlinear recoverability of Y also vary across seeds,
    or is it a stable property of the JEPA architecture + training recipe?

(b) PROPERLY-REGULARIZED MLP probe (weight decay 1e-2, early stopping on
    test_a). Question: can a generalising MLP match KernelRidge on Y? If
    yes, the Exp 2 finding 'MLP fails on Y' was a regularization artefact,
    not an architectural limit.

(c) ISOMAP n_components in (3, 5, 8, 12) + Ridge. Question: how many
    nonlinear coordinates are needed before Isomap matches Ridge / KRR?
    If d=8 catches up, the curved manifold has ~8 useful nonlinear
    coordinates; if even d=12 lags, the curvature is not what's limiting
    linear PLS/PCA-3.

Output: outputs/session16/exp1/exp1a_ter_followups.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.manifold import Isomap
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.data.omega_pipeline import OmegaPipeline  # noqa: E402
from src.models.encoder import HybridCNNViTEncoder  # noqa: E402
from src.utils.device import require_rtx6000  # noqa: E402

LATENTS = REPO / "outputs" / "session14" / "latents" / "S12_E_d64"
OMEGA_MANIFEST = REPO / "outputs" / "data_pipeline" / "v1" / "manifest.json"
SPLIT_MANIFEST = REPO / "configs" / "splits" / "split_v2.json"
OUT = REPO / "outputs" / "session16" / "exp1"
PARTITION = "v1"
DEFAULT_IMPACT_FRAME = 40

import os
CACHE_ROOT = Path(
    os.environ.get(
        "VORTEX_JEPA_CACHE",
        str(Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
            / "data" / "processed" / "vortex-jepa"),
    )
)

SEED_ENCODERS = {
    "production": REPO / "outputs" / "runs" / "session12" / "S12_E_d64" / "encoder" / "checkpoint_iter020000.pt",
    "seed0": REPO / "outputs" / "runs" / "session14" / "thrust6" / "jepa_d64_seed0" / "encoder" / "checkpoint_iter020000.pt",
    "seed1": REPO / "outputs" / "runs" / "session14" / "thrust6" / "jepa_d64_seed1" / "encoder" / "checkpoint_iter020000.pt",
    "seed2": REPO / "outputs" / "runs" / "session14" / "thrust6" / "jepa_d64_seed2" / "encoder" / "checkpoint_iter020000.pt",
}


def load_production_split(name: str) -> dict:
    d = np.load(LATENTS / f"{name}.npz", allow_pickle=True)
    return {
        "z": d["z"].astype(np.float64),
        "G": d["G"].astype(np.float64),
        "D": d["D"].astype(np.float64),
        "Y": d["Y"].astype(np.float64),
    }


def gather_encounters(split: str, cache_root: Path) -> list[dict]:
    with open(SPLIT_MANIFEST) as f:
        manifest = json.load(f)
    out: list[dict] = []
    for cid, case in manifest["cases"].items():
        if split == "train" and case["split"] == "train":
            ks = case["train_encounter_indices"]
        elif split == "test_a" and case["split"] == "train":
            ks = (case.get("val_encounter_indices") or case["test_a_encounter_indices"])
        elif split == "test_b" and case["split"] == "test_b":
            ks = list(range(int(case["n_encounters_full"])))
        elif split == "test_c" and case["split"] == "test_c":
            ks = list(range(int(case["n_encounters_full"])))
        else:
            continue
        for k in ks:
            path = cache_root / cid / f"encounter_{int(k):02d}.h5"
            if path.exists():
                out.append({"case_id": cid, "k": int(k), "G": float(case["G"]),
                            "D": float(case["D"]), "Y": float(case["Y"]), "path": path})
    return out


def load_encoder(ckpt_path: Path, device: torch.device):
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = blob["args"]
    enc = HybridCNNViTEncoder(
        latent_dim=int(args["d"]),
        projection_norm=args.get("projection_norm", "batchnorm"),
    )
    state = {
        k.removeprefix("encoder."): v
        for k, v in blob["jepa_state_dict"].items() if k.startswith("encoder.")
    }
    enc.load_state_dict(state, strict=False)
    enc.eval().to(device)
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


@torch.no_grad()
def encode_impact_only(encs, enc, pipeline, device):
    z_imp_rows = []
    g_col, d_col, y_col = [], [], []
    for rec in encs:
        path = rec["path"]
        if not path.exists():
            continue
        with h5py.File(path, "r") as f:
            omega = np.asarray(f["omega_z"], dtype=np.float32)
            impact = int(f.attrs.get("impact_frame_estimate", DEFAULT_IMPACT_FRAME))
        omega = pipeline.preprocess_raw(omega, rec["case_id"], int(rec["k"]))
        ot = torch.from_numpy(omega)
        ot = pipeline.normalize(ot).unsqueeze(0).unsqueeze(2).to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            zt = enc(ot)
        z_arr = zt.squeeze(0).float().cpu().numpy()
        imp = min(impact, z_arr.shape[0] - 1)
        z_imp_rows.append(z_arr[imp])
        g_col.append(rec["G"])
        d_col.append(rec["D"])
        y_col.append(rec["Y"])
    return {
        "z": np.stack(z_imp_rows).astype(np.float64),
        "G": np.array(g_col), "D": np.array(d_col), "Y": np.array(y_col),
    }


def per_param_r2(y_true: dict, y_pred: dict) -> dict:
    out = {n: float(r2_score(y_true[n], y_pred[n])) for n in ("G", "D", "Y")}
    out["mean"] = float(np.mean(list(out.values())))
    return out


def fit_kernel_ridge(X_train, targets_train, alpha=0.1, gammas=(0.05, 0.01, 0.05)):
    """Per-param KernelRidge(RBF). gammas tuple is per (G, D, Y) -- the CV-best
    from exp1a_bis_cv."""
    models = {}
    for n, g in zip(("G", "D", "Y"), gammas):
        m = KernelRidge(kernel="rbf", gamma=g, alpha=alpha)
        m.fit(X_train, targets_train[n])
        models[n] = m
    return models


def predict_per_param(models, X):
    return {n: m.predict(X) for n, m in models.items()}


# ---- (b) Regularized MLP probe ----

class MLPProbe(torch.nn.Module):
    def __init__(self, in_dim=64, hidden=256, out_dim=1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_regularized_mlp(
    X_train, y_train, X_val, y_val, X_test_dict, y_test_dict_per_split,
    device, weight_decay=1e-2, lr=3e-4, batch=64, max_iters=4000, patience=400,
    seed=0,
):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mean = float(np.mean(y_train))
    std = float(np.std(y_train)) or 1.0
    y_train_n = (y_train - mean) / std
    y_val_n = (y_val - mean) / std
    in_dim = X_train.shape[1]
    probe = MLPProbe(in_dim=in_dim).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)

    X_tr_t = torch.from_numpy(X_train.astype(np.float32)).to(device)
    y_tr_t = torch.from_numpy(y_train_n.astype(np.float32)).to(device)
    X_val_t = torch.from_numpy(X_val.astype(np.float32)).to(device)
    y_val_t = torch.from_numpy(y_val_n.astype(np.float32)).to(device)

    best_val_loss = float("inf")
    best_state = None
    best_iter = 0
    for it in range(max_iters):
        idx = rng.choice(X_tr_t.shape[0], size=min(batch, X_tr_t.shape[0]), replace=False)
        idx_t = torch.from_numpy(idx).to(device)
        z = X_tr_t[idx_t]
        y = y_tr_t[idx_t]
        probe.train()
        opt.zero_grad()
        loss = ((probe(z) - y) ** 2).mean()
        loss.backward()
        opt.step()
        if (it + 1) % 50 == 0:
            probe.eval()
            with torch.no_grad():
                vl = ((probe(X_val_t) - y_val_t) ** 2).mean().item()
            if vl < best_val_loss:
                best_val_loss = vl
                best_state = {k: v.clone() for k, v in probe.state_dict().items()}
                best_iter = it + 1
            elif it + 1 - best_iter > patience:
                break
    if best_state is not None:
        probe.load_state_dict(best_state)
    probe.eval()
    results = {}
    with torch.no_grad():
        for sp, X_test in X_test_dict.items():
            X_test_t = torch.from_numpy(X_test.astype(np.float32)).to(device)
            y_pred_n = probe(X_test_t).cpu().numpy()
            y_pred = y_pred_n * std + mean
            y_true = y_test_dict_per_split[sp]
            results[sp] = float(r2_score(y_true, y_pred))
    results["best_iter"] = best_iter
    return results


# ---- (c) Isomap d sweep ----

def isomap_sweep(X_train, X_test_dict, targets_train, targets_per_split, ds=(3, 5, 8, 12)):
    out = {}
    for d in ds:
        try:
            iso = Isomap(n_components=d, n_neighbors=10)
            Xtr = iso.fit_transform(X_train)
        except Exception as e:
            out[f"d={d}"] = {"error": str(e)}
            continue
        models = {n: Ridge(alpha=1.0).fit(Xtr, targets_train[n]) for n in ("G", "D", "Y")}
        per_split = {}
        for sp, X_test in X_test_dict.items():
            try:
                Xt_emb = iso.transform(X_test)
            except Exception as e:
                per_split[sp] = {"error": str(e)}
                continue
            preds = {n: m.predict(Xt_emb) for n, m in models.items()}
            per_split[sp] = per_param_r2(targets_per_split[sp], preds)
        out[f"d={d}"] = per_split
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = require_rtx6000(gpu_index=args.gpu)
    print(f"[a-ter] device={device}")

    pipeline = OmegaPipeline.from_manifest(OMEGA_MANIFEST)

    splits_prod = {n: load_production_split(n) for n in ("train", "test_a", "test_b", "test_c")}
    train_prod = splits_prod["train"]

    results: dict = {}

    # ===== (a) Per-seed KernelRidge =====
    print("\n[a-ter] (a) Per-seed KernelRidge(RBF) on 4 seed retrains")
    cache_root = CACHE_ROOT / PARTITION
    encs_by_split = {s: gather_encounters(s, cache_root) for s in ("train", "test_a", "test_b", "test_c")}
    per_seed_block = {}
    for seed_name, ckpt in SEED_ENCODERS.items():
        if not ckpt.exists():
            print(f"[a-ter] missing {ckpt}, skipping {seed_name}")
            continue
        t0 = time.time()
        enc = load_encoder(ckpt, device)
        seed_data = {sp: encode_impact_only(encs_by_split[sp], enc, pipeline, device)
                     for sp in encs_by_split}
        del enc
        torch.cuda.empty_cache()
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(seed_data["train"]["z"])
        X_dict = {sp: scaler.transform(seed_data[sp]["z"]) for sp in seed_data}
        targets_train = {n: seed_data["train"][n] for n in ("G", "D", "Y")}
        targets_per_split = {sp: {n: seed_data[sp][n] for n in ("G", "D", "Y")}
                             for sp in seed_data}
        models = fit_kernel_ridge(X_tr, targets_train, alpha=0.1, gammas=(0.05, 0.01, 0.05))
        per_split = {}
        for sp in seed_data:
            preds = predict_per_param(models, X_dict[sp])
            per_split[sp] = per_param_r2(targets_per_split[sp], preds)
        per_seed_block[seed_name] = per_split
        elapsed = time.time() - t0
        print(f"  {seed_name:<12s} ({elapsed:.1f}s)  test_b: "
              f"G={per_split['test_b']['G']:+.3f} D={per_split['test_b']['D']:+.3f} "
              f"Y={per_split['test_b']['Y']:+.3f} mean={per_split['test_b']['mean']:+.3f}")
    results["per_seed_kernel_ridge"] = per_seed_block

    # Aggregate per-seed
    print("\n[a-ter] (a) per-seed Test B R^2 summary:")
    print(f"  {'seed':<12s} {'G':>7s} {'D':>7s} {'Y':>7s} {'mean':>7s}")
    for sn, ps in per_seed_block.items():
        r = ps["test_b"]
        print(f"  {sn:<12s} {r['G']:>+7.3f} {r['D']:>+7.3f} {r['Y']:>+7.3f} {r['mean']:>+7.3f}")
    seeds = list(per_seed_block.keys())
    for n in ("G", "D", "Y", "mean"):
        vals = np.array([per_seed_block[s]["test_b"][n] for s in seeds])
        print(f"    {n}: mean={vals.mean():.3f}  std={vals.std(ddof=1):.3f}  range=[{vals.min():.3f}, {vals.max():.3f}]")

    # ===== (b) Regularized MLP probe on production =====
    print("\n[a-ter] (b) Properly-regularized MLP probe on production encoder")
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train_prod["z"])
    X_dict = {sp: scaler.transform(splits_prod[sp]["z"]) for sp in splits_prod}
    mlp_block = {}
    for target in ("G", "D", "Y"):
        y_train = train_prod[target]
        y_val = splits_prod["test_a"][target]
        per_split_targets = {sp: splits_prod[sp][target] for sp in splits_prod}
        res = train_regularized_mlp(
            X_tr, y_train, X_dict["test_a"], y_val,
            X_dict, per_split_targets, device,
            weight_decay=1e-2, lr=3e-4, batch=64, max_iters=4000, patience=400,
        )
        mlp_block[target] = res
        print(f"  target={target}  test_b R^2={res['test_b']:+.3f}  test_c R^2={res['test_c']:+.3f}  "
              f"best_iter={res['best_iter']}")
    results["regularized_mlp_test_b_r2"] = {n: mlp_block[n]["test_b"] for n in ("G", "D", "Y")}
    results["regularized_mlp_test_c_r2"] = {n: mlp_block[n]["test_c"] for n in ("G", "D", "Y")}
    results["regularized_mlp_test_a_r2"] = {n: mlp_block[n]["test_a"] for n in ("G", "D", "Y")}
    results["regularized_mlp_train_r2"] = {n: mlp_block[n]["train"] for n in ("G", "D", "Y")}
    results["regularized_mlp_best_iter"] = {n: mlp_block[n]["best_iter"] for n in ("G", "D", "Y")}

    # ===== (c) Isomap d sweep =====
    print("\n[a-ter] (c) Isomap d sweep + Ridge on production encoder")
    targets_train_prod = {n: train_prod[n] for n in ("G", "D", "Y")}
    targets_per_split_prod = {sp: {n: splits_prod[sp][n] for n in ("G", "D", "Y")}
                               for sp in splits_prod}
    iso_block = isomap_sweep(X_tr, X_dict, targets_train_prod, targets_per_split_prod,
                             ds=(3, 5, 8, 12))
    results["isomap_d_sweep"] = iso_block
    print(f"  {'d':>4s} {'split':<8s} {'G':>7s} {'D':>7s} {'Y':>7s} {'mean':>7s}")
    for d_key, per_split in iso_block.items():
        for sp in ("test_b", "test_c"):
            if "error" in per_split:
                print(f"  {d_key:>4s} ERROR")
                continue
            r = per_split[sp]
            if "error" in r:
                print(f"  {d_key:>4s} {sp:<8s} ERROR {r['error']}")
                continue
            print(f"  {d_key:>4s} {sp:<8s} {r['G']:>+7.3f} {r['D']:>+7.3f} {r['Y']:>+7.3f} {r['mean']:>+7.3f}")

    save = OUT / "exp1a_ter_followups.json"
    save.write_text(json.dumps(results, indent=2))
    print(f"\n[a-ter] wrote {save.relative_to(REPO)}")


if __name__ == "__main__":
    main()
