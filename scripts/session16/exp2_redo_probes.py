"""Session 16, Experiment 2 REDO: 14-target probe sweep with three probe families.

Per the D118-ter implication: the original Exp 2 used an unregularized
3-layer MLP that overfit Y (test_b R^2 = -0.21 despite train 0.91). We
redo the sweep adding two probes per target:

    MLP_unreg : original Exp 2 recipe (weight_decay 1e-4, no early stop)
    MLP_reg   : weight_decay 1e-2, early stopping on test_a (patience 400)
    KernelRidge(RBF) : CV-selected (alpha, gamma) per target

Output:
    outputs/session16/exp2/probe_sweep_redo.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.utils.device import require_rtx6000  # noqa: E402

TARGETS_DIR = REPO / "outputs" / "session16" / "exp2" / "per_frame_targets"
OUT = REPO / "outputs" / "session16" / "exp2"

TARGET_NAMES = (
    "G", "D", "Y", "C_L", "C_D",
    "peak_pos_omega", "peak_neg_omega",
    "centroid_x", "centroid_y",
    "circulation_pos", "circulation_neg",
    "wake_length", "wake_thickness", "wake_enstrophy",
)


class MLPProbe(nn.Module):
    def __init__(self, in_dim: int = 64, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)


def load_split(name: str) -> dict:
    d = np.load(TARGETS_DIR / f"{name}.npz", allow_pickle=True)
    return {
        "z_full": d["z_full"].astype(np.float32),
        **{k: d[k].astype(np.float32) for k in TARGET_NAMES},
    }


def train_mlp(
    train: dict, test_a: dict, test_b: dict, test_c: dict,
    target: str, device, *,
    weight_decay: float, lr: float, batch: int, max_iters: int,
    early_stop: bool, patience: int = 400, seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    z_full = train["z_full"]
    n_enc, T_max, d = z_full.shape
    y_train_all = train[target]
    finite_mask = np.isfinite(y_train_all)
    y_finite = y_train_all[finite_mask]
    mean = float(y_finite.mean())
    std = float(y_finite.std()) or 1.0
    y_norm = (y_train_all - mean) / std

    candidate_encs = np.where(finite_mask.any(axis=1))[0]
    probe = MLPProbe(in_dim=d).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)

    # If early stopping, build a flat val set from test_a (all frames).
    if early_stop:
        za = test_a["z_full"].reshape(-1, d).astype(np.float32)
        ya = test_a[target].reshape(-1).astype(np.float32)
        mask_a = np.isfinite(ya)
        za = za[mask_a]
        ya_norm = (ya[mask_a] - mean) / std
        z_val = torch.from_numpy(za).to(device)
        y_val = torch.from_numpy(ya_norm).to(device)
        best_val = float("inf")
        best_state = None
        best_iter = 0

    losses = np.zeros(max_iters, dtype=np.float32)
    for it in range(max_iters):
        chosen = rng.choice(candidate_encs, size=min(batch, candidate_encs.size), replace=False)
        z_batch = np.empty((chosen.size, d), dtype=np.float32)
        y_batch = np.empty(chosen.size, dtype=np.float32)
        for j, enc_idx in enumerate(chosen):
            ok = np.where(finite_mask[enc_idx])[0]
            t = int(rng.choice(ok))
            z_batch[j] = z_full[enc_idx, t]
            y_batch[j] = y_norm[enc_idx, t]
        probe.train()
        z_t = torch.from_numpy(z_batch).to(device)
        y_t = torch.from_numpy(y_batch).to(device)
        opt.zero_grad()
        loss = ((probe(z_t) - y_t) ** 2).mean()
        loss.backward()
        opt.step()
        losses[it] = float(loss.item())
        if early_stop and (it + 1) % 50 == 0:
            probe.eval()
            with torch.no_grad():
                vl = ((probe(z_val) - y_val) ** 2).mean().item()
            if vl < best_val:
                best_val = vl
                best_state = {k: v.clone() for k, v in probe.state_dict().items()}
                best_iter = it + 1
            elif it + 1 - best_iter > patience:
                break
    if early_stop and best_state is not None:
        probe.load_state_dict(best_state)
    probe.eval()

    out = {"final_train_loss_normed": float(losses[max(0, it-50):it+1].mean())}
    if early_stop:
        out["best_iter"] = best_iter
    for sp, data in (("train", train), ("test_a", test_a), ("test_b", test_b), ("test_c", test_c)):
        z = data["z_full"].reshape(-1, d).astype(np.float32)
        y = data[target].reshape(-1).astype(np.float32)
        mask = np.isfinite(y)
        z = z[mask]
        y_true = y[mask]
        with torch.no_grad():
            y_pred_norm = probe(torch.from_numpy(z).to(device)).cpu().numpy()
        y_pred = y_pred_norm * std + mean
        out[sp] = {"r2": float(r2_score(y_true, y_pred)),
                   "n_samples": int(y_true.size)}
    return out


def train_kernel_ridge(
    train: dict, test_a: dict, test_b: dict, test_c: dict,
    target: str,
) -> dict:
    """Per-target KernelRidge(RBF), CV-selected over (alpha, gamma).
    For probes, we average over impact-frame slices (1 per encounter)
    rather than all frames to keep the CV honest under IID.
    """
    z_full_train = train["z_full"]
    n_enc, T_max, d = z_full_train.shape
    y_train_all = train[target]
    rng = np.random.default_rng(0)
    finite_mask = np.isfinite(y_train_all)
    Xs = []
    ys = []
    for i in range(n_enc):
        ok = np.where(finite_mask[i])[0]
        if ok.size == 0:
            continue
        t = int(rng.choice(ok))
        Xs.append(z_full_train[i, t])
        ys.append(y_train_all[i, t])
    Xtr = np.stack(Xs)
    ytr = np.array(ys, dtype=np.float64)
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    gs = GridSearchCV(
        KernelRidge(kernel="rbf"),
        {"alpha": [0.1, 1.0, 10.0], "gamma": [0.01, 0.05, 0.1, 0.3]},
        cv=KFold(n_splits=5, shuffle=True, random_state=0), scoring="r2", n_jobs=-1,
    )
    gs.fit(Xtr_s, ytr)
    model = gs.best_estimator_

    out = {
        "best_alpha": float(gs.best_params_["alpha"]),
        "best_gamma": float(gs.best_params_["gamma"]),
        "best_cv_r2": float(gs.best_score_),
    }
    for sp, data in (("train", train), ("test_a", test_a), ("test_b", test_b), ("test_c", test_c)):
        z = data["z_full"].reshape(-1, d).astype(np.float32)
        y = data[target].reshape(-1).astype(np.float32)
        mask = np.isfinite(y)
        z = z[mask]
        y_true = y[mask]
        Xs_eval = scaler.transform(z)
        y_pred = model.predict(Xs_eval)
        out[sp] = {"r2": float(r2_score(y_true, y_pred)),
                   "n_samples": int(y_true.size)}
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = require_rtx6000(gpu_index=args.gpu)
    train = load_split("train")
    test_a = load_split("test_a")
    test_b = load_split("test_b")
    test_c = load_split("test_c")
    print(f"[exp2-redo] device={device} z_full shape={train['z_full'].shape}")

    summary: dict = {"results_by_target_and_probe": {}}
    print(f"\n[exp2-redo] {'target':<20s} {'probe':<10s} "
          f"{'train R^2':>10s} {'test_a R^2':>11s} {'test_b R^2':>11s} {'test_c R^2':>11s}")

    t0 = time.time()
    for target in TARGET_NAMES:
        per_target: dict = {}

        # MLP unregularized (original Exp 2 recipe)
        res_unreg = train_mlp(train, test_a, test_b, test_c, target, device,
                              weight_decay=1e-4, lr=3e-4, batch=64, max_iters=2000,
                              early_stop=False)
        per_target["mlp_unregularized"] = res_unreg
        print(f"  {target:<20s} {'MLP_unreg':<10s} "
              f"{res_unreg['train']['r2']:>+10.3f} {res_unreg['test_a']['r2']:>+11.3f} "
              f"{res_unreg['test_b']['r2']:>+11.3f} {res_unreg['test_c']['r2']:>+11.3f}")

        # MLP regularized (weight_decay 1e-2, early stopping on test_a)
        res_reg = train_mlp(train, test_a, test_b, test_c, target, device,
                            weight_decay=1e-2, lr=3e-4, batch=64, max_iters=4000,
                            early_stop=True, patience=400)
        per_target["mlp_regularized"] = res_reg
        print(f"  {target:<20s} {'MLP_reg':<10s} "
              f"{res_reg['train']['r2']:>+10.3f} {res_reg['test_a']['r2']:>+11.3f} "
              f"{res_reg['test_b']['r2']:>+11.3f} {res_reg['test_c']['r2']:>+11.3f}  "
              f"(best_iter={res_reg.get('best_iter')})")

        # KernelRidge(RBF) CV
        res_krr = train_kernel_ridge(train, test_a, test_b, test_c, target)
        per_target["kernel_ridge_rbf"] = res_krr
        print(f"  {target:<20s} {'KRR_RBF':<10s} "
              f"{res_krr['train']['r2']:>+10.3f} {res_krr['test_a']['r2']:>+11.3f} "
              f"{res_krr['test_b']['r2']:>+11.3f} {res_krr['test_c']['r2']:>+11.3f}  "
              f"(alpha={res_krr['best_alpha']}, gamma={res_krr['best_gamma']})")

        summary["results_by_target_and_probe"][target] = per_target

    # Rank by test_b R^2 per probe
    print("\n[exp2-redo] Test B R^2 ranking by best-probe-per-target:")
    print(f"  {'target':<20s} {'MLP_unreg':>10s} {'MLP_reg':>10s} {'KRR_RBF':>10s} {'best':>10s}")
    best_table = []
    for target in TARGET_NAMES:
        per = summary["results_by_target_and_probe"][target]
        u = per["mlp_unregularized"]["test_b"]["r2"]
        r = per["mlp_regularized"]["test_b"]["r2"]
        k = per["kernel_ridge_rbf"]["test_b"]["r2"]
        best = max(u, r, k)
        best_table.append((target, u, r, k, best))
    best_table.sort(key=lambda x: -x[4])
    for target, u, r, k, best in best_table:
        print(f"  {target:<20s} {u:>+10.3f} {r:>+10.3f} {k:>+10.3f} {best:>+10.3f}")

    save = OUT / "probe_sweep_redo.json"
    save.write_text(json.dumps(summary, indent=2))
    print(f"\n[exp2-redo] total wall: {time.time()-t0:.1f}s")
    print(f"[exp2-redo] wrote {save.relative_to(REPO)}")


if __name__ == "__main__":
    main()
