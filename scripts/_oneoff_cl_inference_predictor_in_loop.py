"""Compare THREE pipelines for inferring C_L at impact from pre-impact pressure:

  (A) Direct:               pressure(τ window) → C_L(impact)         [KRR]
  (B) Via baseline:         pressure(τ window) → ẑ(impact) → C_L     [pressure→z KRR, then train-fit z→C_L probe]
  (C) Predictor-in-loop:    pressure(τ window) → ẑ(t_end)
                            → predictor.rollout(τ steps) → ẑ(impact)
                            → C_L probe → Ĉ_L(impact)

The question: does the baseline's trained TRANSFORMER predictor add value
beyond impact-state-estimation? Pipeline (C) routes through the learned
dynamics; (B) skips the predictor (state-estimation only).

At each baseline (jepa_d64, jepa_d32, fukami_d64, pod_d64) and lead time
τ ∈ {30, 20, 10, 5, 2}, we compute test_b R²(C_L at impact).

Output:
    outputs/session18/exp_b1_test3/cl_inference_predictor_in_loop.csv
    outputs/session18/exp_b1_test3/cl_inference_predictor_in_loop_figure.png
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import warnings; warnings.filterwarnings("ignore")
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler

REPO = Path("/home/carlos/GUST-JEPA")
sys.path.insert(0, str(REPO))

from src.models.predictor import AutoregressivePredictor  # noqa: E402
from src.utils.device import require_rtx6000  # noqa: E402
from torch import nn  # noqa: E402

CACHE = Path("/home/carlos/PREVENT/data/processed/vortex-jepa/v1")

LEAD_TAUS = [30, 20, 10, 5, 2]
WINDOW = 10
K_SENSORS = 8
N_PRESSURE_FULL = 192

# (tag, latents_subdir, predictor_subdir, d). All d=64 except jepa_d32 included for the d=32 view.
BASELINES = [
    ("jepa_d64",   "latents_jepa_d64",   "predictor_jepa_d64_test1_noBN", 64),
    ("jepa_d32",   "latents_jepa_d32",   "predictor_jepa_d32_noBN",       32),
    ("fukami_d64", "latents_fukami_d64", "predictor_fukami_d64_noBN",     64),
    ("pod_d64",    "latents_pod_d64",    "predictor_pod_d64_noBN",        64),
]


def r2(yhat, y, eps=1e-9):
    ss_res = ((yhat - y) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0)
    return 1.0 - ss_res / np.maximum(ss_tot, eps)


def select_sensors_evenly(K: int) -> list[int]:
    return list(np.linspace(0, N_PRESSURE_FULL - 1, K, dtype=int).tolist())


def load_pressure_and_cl(case_id, k, impact, lead_tau, window):
    """Returns (pressure_window (W, 192), C_L_at_impact)."""
    p = CACHE / case_id / f"encounter_{int(k):02d}.h5"
    with h5py.File(p, "r") as f:
        p_wall = np.asarray(f["p_wall"], dtype=np.float32)
        c_l = np.asarray(f["C_L"], dtype=np.float32)
    t_end = impact - lead_tau
    t_start = max(0, t_end - window)
    win = p_wall[t_start:t_end]
    if win.shape[0] < window:
        pad = np.zeros((window - win.shape[0], N_PRESSURE_FULL), dtype=np.float32)
        win = np.concatenate([pad, win], axis=0)
    return win, float(c_l[min(impact, len(c_l) - 1)])


def get_cid_ei(npz):
    cid_key = "case_id" if "case_id" in npz.files else "case_ids"
    ei_key  = "encounter_index" if "encounter_index" in npz.files else "encounter_indices"
    return npz[cid_key].astype(str), npz[ei_key].astype(int)


def get_impact(npz):
    if "impact_frame" in npz.files:
        return npz["impact_frame"].astype(int)
    cids, _ = get_cid_ei(npz)
    return np.full(len(cids), 40)


def get_z_at_frame(npz, frame_idx: np.ndarray) -> np.ndarray:
    """Returns z at the provided frame indices (n, d). frame_idx is an array of ints."""
    z_full = npz["z_full"]  # (n, T_total, d)
    n = z_full.shape[0]
    out = np.zeros((n, z_full.shape[-1]), dtype=np.float32)
    T_total = z_full.shape[1]
    for i in range(n):
        t = int(np.clip(frame_idx[i], 0, T_total - 1))
        out[i] = z_full[i, t]
    return out


def get_z_impact(npz):
    """Returns impact-frame z (n, d)."""
    if "z" in npz.files:
        return npz["z"]
    return get_z_at_frame(npz, get_impact(npz))


def gather_pressure_and_cl(latents_npz, lead_tau, window, sensor_idx):
    """Returns (X_pressure (n, window*K), cl_target (n,))."""
    cids, eis = get_cid_ei(latents_npz)
    impact = get_impact(latents_npz)
    n = len(cids)
    X_pressure = np.zeros((n, window, len(sensor_idx)), dtype=np.float32)
    cl_target = np.zeros(n, dtype=np.float32)
    for i, (cid, ei, imp) in enumerate(zip(cids, eis, impact)):
        win, cl = load_pressure_and_cl(cid, int(ei), int(imp), lead_tau, window)
        X_pressure[i] = win[:, sensor_idx]
        cl_target[i] = cl
    return X_pressure.reshape(n, -1), cl_target


def load_predictor(ckpt_path: Path, d: int, device: torch.device):
    """Load the trained baseline predictor with --no-output-bn and latent norm stats."""
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = blob.get("run_config", {})
    pcfg = cfg.get("predictor_config", {})
    pred = AutoregressivePredictor(
        latent_dim=d,
        cond_dim=int(pcfg.get("cond_dim", 3)),
        hidden_dim=int(pcfg.get("hidden_dim", 384)),
        depth=int(pcfg.get("depth", 6)),
        heads=int(pcfg.get("heads", 16)),
        mlp_ratio=float(pcfg.get("mlp_ratio", 4.0)),
        dropout=float(pcfg.get("dropout", 0.1)),
        max_seq_len=int(pcfg.get("max_seq_len", 32)),
    ).to(device)
    state = blob["predictor_state_dict"]
    if "out_proj.1.weight" not in state and "out_proj.1.running_mean" not in state:
        out_lin = pred.out_proj[0]
        pred.out_proj = nn.Sequential(out_lin, nn.Identity()).to(device)
    pred.load_state_dict(state)
    pred.eval()
    for p in pred.parameters():
        p.requires_grad_(False)
    mean = blob["latent_mean"]
    std = blob["latent_std"]
    mean_t = torch.tensor(np.asarray(mean), dtype=torch.float32, device=device)
    std_t = torch.tensor(np.asarray(std), dtype=torch.float32, device=device)
    return pred, mean_t, std_t


@torch.no_grad()
def rollout_steps(
    pred: AutoregressivePredictor,
    z_seed_norm: torch.Tensor,
    cond: torch.Tensor,
    steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Autoregressive open-loop rollout from a single normalised seed.

    z_seed_norm: (B, 1, d) normalised latents at t_end
    cond: (B, cond_dim)
    Returns: (B, d) — predicted normalised latent after `steps` autoregressive steps.
    """
    max_seq = int(pred.max_seq_len)
    z_full = z_seed_norm.clone()
    for _ in range(steps):
        ctx = z_full[:, -max_seq:, :]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            z_hat = pred(ctx, cond)
        z_full = torch.cat([z_full, z_hat[:, -1:, :].float()], dim=1)
    return z_full[:, -1, :]  # (B, d)


def per_dim_krr_press_to_z(X_tr, z_tr, X_te_list):
    """Pressure → normalised ẑ via per-dim KRR-RBF (same recipe as the existing script).

    Returns predictions for each X_te in X_te_list (each shape (n_te, d)).
    The output z is in the SAME space as z_tr (i.e. inputs/outputs use one
    StandardScaler each, but the predicted z is returned in the input-z space).
    """
    sx = StandardScaler().fit(X_tr)
    sz = StandardScaler().fit(z_tr)
    m = KernelRidge(alpha=0.1, kernel="rbf", gamma=0.01)
    m.fit(sx.transform(X_tr), sz.transform(z_tr))
    outs = []
    for X_te in X_te_list:
        z_hat = sz.inverse_transform(m.predict(sx.transform(X_te)))
        outs.append(z_hat)
    return outs


def fit_predict_cl_direct(X_tr, y_tr, X_te_list):
    """Direct pressure → C_L via KRR-RBF (same recipe as the existing script)."""
    sx = StandardScaler().fit(X_tr)
    m = KernelRidge(alpha=0.1, kernel="rbf", gamma=0.01)
    m.fit(sx.transform(X_tr), y_tr)
    return [m.predict(sx.transform(X_te)) for X_te in X_te_list]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--predictor-root",
        type=Path,
        default=REPO / "outputs/session18/exp_b1_test3",
        help="Directory containing predictor_{tag}_noBN/checkpoint_iter020000.pt",
    )
    parser.add_argument(
        "--latents-root",
        type=Path,
        default=REPO / "outputs/session18/exp_b1",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "outputs/session18/exp_b1_test3",
    )
    args = parser.parse_args()

    device = require_rtx6000(gpu_index=args.gpu)
    print(f"[device] using {device} ({torch.cuda.get_device_name(device.index)})")

    sensor_idx = select_sensors_evenly(K_SENSORS)
    rows = []

    for tag, lat_subdir, pred_subdir, d in BASELINES:
        lat_dir = args.latents_root / lat_subdir
        ckpt = args.predictor_root / pred_subdir / "checkpoint_iter020000.pt"
        if not (lat_dir / "train.npz").exists():
            print(f"SKIP {tag}: train.npz missing"); continue
        if not ckpt.exists():
            print(f"SKIP {tag}: predictor checkpoint missing at {ckpt}"); continue

        try:
            train_npz = np.load(lat_dir / "train.npz", allow_pickle=True)
            test_b_npz = np.load(lat_dir / "test_b.npz", allow_pickle=True)
        except Exception as e:
            print(f"SKIP {tag}: {e}"); continue

        # Impact-frame z for the z→C_L probe + via_baseline pipeline
        z_tr_imp = get_z_impact(train_npz).astype(np.float32)
        z_tb_imp = get_z_impact(test_b_npz).astype(np.float32)

        # DNS C_L at impact
        cids_tr, eis_tr = get_cid_ei(train_npz); imp_tr = get_impact(train_npz)
        cids_tb, eis_tb = get_cid_ei(test_b_npz); imp_tb = get_impact(test_b_npz)
        cl_tr = np.array([load_pressure_and_cl(c, e, im, 0, 1)[1]
                          for c, e, im in zip(cids_tr, eis_tr, imp_tr)], dtype=np.float32)
        cl_tb = np.array([load_pressure_and_cl(c, e, im, 0, 1)[1]
                          for c, e, im in zip(cids_tb, eis_tb, imp_tb)], dtype=np.float32)

        # Train-only z→C_L probe (KRR-RBF; same as existing script)
        sx_z = StandardScaler().fit(z_tr_imp)
        probe = KernelRidge(alpha=0.1, kernel="rbf", gamma=0.05)
        probe.fit(sx_z.transform(z_tr_imp), cl_tr)
        cl_hat_tb_oracle = probe.predict(sx_z.transform(z_tb_imp))
        r2_tb_oracle = float(r2(cl_hat_tb_oracle[:, None], cl_tb[:, None])[0])

        # Load predictor
        pred, mean_t, std_t = load_predictor(ckpt, d, device)
        mean_np = mean_t.cpu().numpy()
        std_np = std_t.cpu().numpy()
        n_params = sum(p.numel() for p in pred.parameters())

        # Conditioning vectors
        cond_tr = np.stack([train_npz["G"], train_npz["D"], train_npz["Y"]], axis=1).astype(np.float32)
        cond_tb = np.stack([test_b_npz["G"], test_b_npz["D"], test_b_npz["Y"]], axis=1).astype(np.float32)

        print(f"\n=== {tag} (d={d}; predictor params={n_params/1e6:.2f}M) ===")
        print(f"  z(true)→C_L probe (upper bound): test_b R²={r2_tb_oracle:+.3f}")

        for tau in LEAD_TAUS:
            # ----- Gather pressure inputs and target-frame z at t_end = impact - tau -----
            try:
                X_tr_p, _ = gather_pressure_and_cl(train_npz, tau, WINDOW, sensor_idx)
                X_tb_p, _ = gather_pressure_and_cl(test_b_npz, tau, WINDOW, sensor_idx)
            except Exception as e:
                print(f"  τ=-{tau}: FAIL gather: {e}"); continue

            # Target frames at t_end = impact - tau
            t_end_tr = (imp_tr - tau).astype(np.int64)
            t_end_tb = (imp_tb - tau).astype(np.int64)
            z_tr_at_tend = get_z_at_frame(train_npz, t_end_tr).astype(np.float32)

            # ===== Pipeline (A) Direct pressure → C_L =====
            cl_hat_tb_A, = fit_predict_cl_direct(X_tr_p, cl_tr, [X_tb_p])
            r2_tb_A = float(r2(cl_hat_tb_A[:, None], cl_tb[:, None])[0])

            # ===== Pipeline (B) Via baseline: pressure → ẑ(impact) → z→C_L probe =====
            # (uses impact-frame z as the regression target)
            z_hat_tb_imp, = per_dim_krr_press_to_z(X_tr_p, z_tr_imp, [X_tb_p])
            cl_hat_tb_B = probe.predict(sx_z.transform(z_hat_tb_imp))
            r2_tb_B = float(r2(cl_hat_tb_B[:, None], cl_tb[:, None])[0])

            # ===== Pipeline (C) Predictor-in-loop =====
            # Stage 1: pressure → ẑ(t_end) — train pressure→z KRR with target = z at t_end.
            z_hat_tb_at_tend, = per_dim_krr_press_to_z(X_tr_p, z_tr_at_tend, [X_tb_p])

            # Stage 2: normalise ẑ(t_end) using the predictor's training stats,
            # rollout `tau` autoregressive steps, then un-normalise to get ẑ(impact).
            z_seed_norm_np = (z_hat_tb_at_tend - mean_np) / std_np
            z_seed_norm = torch.from_numpy(z_seed_norm_np.astype(np.float32))[:, None, :].to(device)
            cond_t = torch.from_numpy(cond_tb).to(device)

            # Roll out in chunks of e.g. 16 encounters to keep memory tame.
            batch = 16
            preds_norm = []
            for i in range(0, z_seed_norm.shape[0], batch):
                z_chunk = z_seed_norm[i : i + batch]
                c_chunk = cond_t[i : i + batch]
                z_end = rollout_steps(pred, z_chunk, c_chunk, steps=int(tau), device=device)
                preds_norm.append(z_end.cpu().numpy())
            z_hat_tb_at_imp_norm = np.concatenate(preds_norm, axis=0)
            z_hat_tb_at_imp_C = z_hat_tb_at_imp_norm * std_np + mean_np

            # Stage 3: apply train-fit z→C_L probe
            cl_hat_tb_C = probe.predict(sx_z.transform(z_hat_tb_at_imp_C))
            r2_tb_C = float(r2(cl_hat_tb_C[:, None], cl_tb[:, None])[0])

            row = {
                "tag": tag, "d": d, "lead_tau": tau,
                "test_b_R2_oracle":              r2_tb_oracle,
                "test_b_R2_direct":              r2_tb_A,
                "test_b_R2_via_baseline":        r2_tb_B,
                "test_b_R2_predictor_in_loop":   r2_tb_C,
            }
            rows.append(row)
            print(f"  τ=-{tau:2d}  direct={r2_tb_A:+.3f}  "
                  f"via_baseline={r2_tb_B:+.3f}  predictor_in_loop={r2_tb_C:+.3f}  "
                  f"(oracle={r2_tb_oracle:+.3f})")

        # Free predictor
        del pred
        torch.cuda.empty_cache()

    out_csv = args.out_dir / "cl_inference_predictor_in_loop.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out_csv}")

    # ---------- Figure ----------
    if rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tags = [t for t, *_ in BASELINES if any(r["tag"] == t for r in rows)]
        n_baselines = len(tags)
        fig, axes = plt.subplots(1, n_baselines, figsize=(4.0 * n_baselines, 3.8),
                                 sharey=True, squeeze=False)
        for ax, tag in zip(axes[0], tags):
            sub = [r for r in rows if r["tag"] == tag]
            sub.sort(key=lambda r: -int(r["lead_tau"]))  # τ=30 first → τ=2 last (chronological)
            taus = [int(r["lead_tau"]) for r in sub]
            d_A = [r["test_b_R2_direct"] for r in sub]
            d_B = [r["test_b_R2_via_baseline"] for r in sub]
            d_C = [r["test_b_R2_predictor_in_loop"] for r in sub]
            oracle = sub[0]["test_b_R2_oracle"]

            ax.plot(taus, d_A, marker="o", label="(A) Direct", color="#1f77b4")
            ax.plot(taus, d_B, marker="s", label="(B) Via baseline", color="#2ca02c")
            ax.plot(taus, d_C, marker="^", label="(C) Predictor-in-loop", color="#d62728")
            ax.axhline(oracle, color="black", linestyle="--", alpha=0.5, label=f"Oracle z→C_L ({oracle:+.2f})")
            ax.axhline(0.0, color="grey", linestyle=":", alpha=0.4)
            d = sub[0]["d"]
            ax.set_title(f"{tag} (d={d})")
            ax.set_xlabel(r"lead time $\tau$ (frames before impact)")
            ax.invert_xaxis()  # τ=30 on the left (earlier), τ=2 on the right (closer)
            ax.set_ylim(-0.6, 0.85)
            if ax is axes[0, 0]:
                ax.set_ylabel(r"test_b $R^2(C_L \text{ at impact})$")
        axes[0, -1].legend(loc="lower left", fontsize=8, framealpha=0.9)
        fig.suptitle("Pre-impact pressure → C_L(impact): three pipelines (test_b)", y=1.02)
        fig.tight_layout()
        fig_path = args.out_dir / "cl_inference_predictor_in_loop_figure.png"
        fig.savefig(fig_path, dpi=160, bbox_inches="tight")
        print(f"Wrote figure to {fig_path}")


if __name__ == "__main__":
    main()
