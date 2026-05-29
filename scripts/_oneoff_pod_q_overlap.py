"""Q-criterion overlap analysis for POD d=16 modes (Section 5 of the paper).

Question: do POD's pressure-recoverable modes correspond to Q-criterion vortex
regions?

Existing finding (outputs/session18/exp_b1_test3/baseline_pressure_observability.csv):
POD d=16 recovers (G, D) from K=8 wall-pressure sensors better than JEPA,
even though POD's overall R^2(z) recovery is far below JEPA's. Hypothesis:
POD's pressure-recoverable coefficients correspond to dominant Q-criterion
vortex structures, hence the wall-pressure alignment.

Pipeline:
  1. Compute Q = 0.5 * (Omega^2 - S^2) on test_b mid-plane (z-index 16) at
     impact frame, using velocity field /u from raw HDF5.
  2. For each POD mode phi_i in pod_d16/pod_basis.npz, compute the spatial
     overlap of |phi_i|^2 with the Q > tau mask, averaged across test_b
     encounters at impact frame.
  3. Recompute per-mode pressure-recoverability: K=8 evenly-spaced sensors
     -> z_train (KRR-RBF), eval per-dim R^2(z_i) on test_b.
  4. Scatter: x = R^2_i (per mode), y = Q-overlap_i. Pearson + Spearman.

Outputs:
  outputs/session18/exp_b1_test3/pod_q_overlap_pressure.png
  outputs/session18/exp_b1_test3/pod_q_overlap_pressure.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

REPO = Path("/home/carlos/GUST-JEPA")
sys.path.insert(0, str(REPO))

PREVENT = Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
CACHE = Path(
    os.environ.get(
        "VORTEX_JEPA_CACHE",
        str(PREVENT / "data" / "processed" / "vortex-jepa"),
    )
) / "v1"

from src.data.omega_pipeline import OmegaPipeline  # noqa: E402

# Constants
SPLIT_PATH = REPO / "configs" / "splits" / "split_v2.json"
POD_BASIS = REPO / "outputs/session18/exp_b1/pod_d16/pod_basis.npz"
LATENT_DIR = REPO / "outputs/session18/exp_b1/latents_pod_d16"
OUT_DIR = REPO / "outputs/session18/exp_b1_test3"
OUT_PNG = OUT_DIR / "pod_q_overlap_pressure.png"
OUT_JSON = OUT_DIR / "pod_q_overlap_pressure.json"
PIPELINE_MANIFEST = REPO / "outputs/data_pipeline/v1/manifest.json"

H, W = 192, 96
MID_Z = 16
K_SENSORS = 8
WINDOW = 30
N_PRESSURE = 192


def r2(yhat, y):
    ss_res = ((yhat - y) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-9)


def load_q_at_impact(raw_path: Path, frame_start: int, impact_frame_local: int,
                     dx: float, dy: float) -> np.ndarray:
    """Load mid-plane velocity at the global frame corresponding to impact and
    compute the 2D Q-criterion on the (H=192, W=96) mid-plane.

    Q = 0.5 * (||Omega||^2 - ||S||^2)
    where for 2D mid-plane velocity (u_x, u_y):
        S_xx = du/dx, S_yy = dv/dy, S_xy = 0.5*(du/dy + dv/dx)
        Omega_xy = 0.5*(dv/dx - du/dy)
        ||S||^2 = S_xx^2 + S_yy^2 + 2 S_xy^2
        ||Omega||^2 = 2 Omega_xy^2

    Returns Q on the (H, W) mid-plane with NaN cells (inside the airfoil) set
    to 0.
    """
    global_frame = frame_start + impact_frame_local
    with h5py.File(raw_path, "r") as f:
        u_mid = np.asarray(f["u"][global_frame, :, :, MID_Z, :], dtype=np.float32)
    u_x = np.nan_to_num(u_mid[..., 0], nan=0.0)
    u_y = np.nan_to_num(u_mid[..., 1], nan=0.0)
    # Centered gradients with edge order 2; np.gradient handles uniform spacing
    du_dx, du_dy = np.gradient(u_x, dx, dy)
    dv_dx, dv_dy = np.gradient(u_y, dx, dy)
    S_xx = du_dx
    S_yy = dv_dy
    S_xy = 0.5 * (du_dy + dv_dx)
    Omega_xy = 0.5 * (dv_dx - du_dy)
    S_sq = S_xx ** 2 + S_yy ** 2 + 2.0 * S_xy ** 2
    Omega_sq = 2.0 * Omega_xy ** 2
    Q = 0.5 * (Omega_sq - S_sq)
    return Q.astype(np.float32)


def collect_test_b_q(pipeline: OmegaPipeline) -> tuple[np.ndarray, np.ndarray]:
    """For every test_b encounter, compute mid-plane Q at the impact frame and
    return both the raw Q stack and a normalized "Q > tau" indicator stack."""
    with open(SPLIT_PATH) as f:
        m = json.load(f)

    # grid spacing from one of the files
    sample_path = None
    for cid, case in m["cases"].items():
        if case["split"] == "test_b":
            sample_path = PREVENT / case["relative_path"]
            break
    with h5py.File(sample_path, "r") as f:
        x = np.asarray(f["x"])
        y = np.asarray(f["y"])
    dx = float(np.mean(np.diff(x)))
    dy = float(np.mean(np.diff(y)))
    print(f"[Q] grid dx={dx:.5f}, dy={dy:.5f}")

    # Mask: True for cells to zero (inside-solid + adjacent)
    spatial_mask = pipeline.mask.cpu().numpy()  # (H, W) bool

    q_stack: list[np.ndarray] = []
    meta: list[dict] = []
    for cid, case in m["cases"].items():
        if case["split"] != "test_b":
            continue
        raw_path = PREVENT / case["relative_path"]
        n_full = case["n_encounters_full"]
        for k in range(n_full):
            cache_path = CACHE / cid / f"encounter_{k:02d}.h5"
            if not cache_path.exists():
                continue
            with h5py.File(cache_path, "r") as f:
                frame_start = int(f.attrs["frame_start"])
                impact_local = int(f.attrs.get("impact_frame_estimate", 40))
            Q = load_q_at_impact(raw_path, frame_start, impact_local, dx, dy)
            # Zero airfoil-adjacent cells so they don't dominate Q overlap stats
            Q[spatial_mask] = 0.0
            q_stack.append(Q)
            meta.append({"case_id": cid, "k": k})
    q_arr = np.stack(q_stack, axis=0)  # (n_enc, H, W)
    print(f"[Q] computed Q for {q_arr.shape[0]} test_b encounters at impact frame")
    print(f"[Q] Q stats: min={q_arr.min():.3f}, max={q_arr.max():.3f}, "
          f"mean={q_arr.mean():.3f}, p90={np.percentile(q_arr, 90):.3f}, "
          f"p99={np.percentile(q_arr, 99):.3f}")
    return q_arr, np.array([(m_["case_id"], m_["k"]) for m_ in meta])


def compute_mode_q_overlap(phi: np.ndarray, q_pos_mean: np.ndarray) -> float:
    """Mode-energy-weighted mean of (Q > tau) indicator.

    phi: (H, W) POD mode (single mode reshaped)
    q_pos_mean: (H, W) average Q>tau indicator across encounters in [0, 1].

    Returns the |phi|^2-weighted mean of q_pos_mean, i.e., fraction of mode
    energy that lives in vortex-positive regions on average.
    """
    energy = phi.astype(np.float64) ** 2
    energy = energy / np.maximum(energy.sum(), 1e-12)
    return float((energy * q_pos_mean.astype(np.float64)).sum())


def select_sensors_evenly(K: int) -> list[int]:
    return list((np.linspace(0, N_PRESSURE - 1, K, dtype=int)).tolist())


def load_pressure_window(case_id: str, k: int, impact_frame_local: int) -> np.ndarray:
    p = CACHE / case_id / f"encounter_{int(k):02d}.h5"
    with h5py.File(p, "r") as f:
        p_wall = np.asarray(f["p_wall"], dtype=np.float32)  # (120, 192)
    t_start = max(0, impact_frame_local - WINDOW)
    window = p_wall[t_start:impact_frame_local]
    if window.shape[0] < WINDOW:
        pad = np.zeros((WINDOW - window.shape[0], N_PRESSURE), dtype=np.float32)
        window = np.concatenate([pad, window], axis=0)
    return window


def gather_pressure_and_z(latents_npz) -> tuple[np.ndarray, np.ndarray]:
    cids = latents_npz["case_ids"].astype(str)
    eis = latents_npz["encounter_indices"].astype(int)
    impact = (latents_npz["impact_frame"].astype(int)
              if "impact_frame" in latents_npz.files
              else np.full(len(cids), 40))
    z_full = latents_npz["z_full"]
    z = np.stack([z_full[i, int(impact[i])] for i in range(len(cids))], axis=0)
    X = np.zeros((len(cids), WINDOW, N_PRESSURE), dtype=np.float32)
    for i, (cid, ei, imp) in enumerate(zip(cids, eis, impact)):
        X[i] = load_pressure_window(cid, int(ei), int(imp))
    return X, z


def fit_per_mode_pressure_r2(K: int) -> np.ndarray:
    """KRR-RBF: pressure_window -> z_impact for K sensors. Returns per-dim
    R^2 on test_b for the d=16 POD coefficients (length-16 vector)."""
    train_npz = np.load(LATENT_DIR / "train.npz", allow_pickle=True)
    test_b_npz = np.load(LATENT_DIR / "test_b.npz", allow_pickle=True)

    X_tr, z_tr = gather_pressure_and_z(train_npz)
    X_tb, z_tb = gather_pressure_and_z(test_b_npz)

    sensors = select_sensors_evenly(K)
    X_tr_K = X_tr[:, :, sensors].reshape(len(X_tr), -1)
    X_tb_K = X_tb[:, :, sensors].reshape(len(X_tb), -1)

    sx = StandardScaler().fit(X_tr_K)
    sy = StandardScaler().fit(z_tr)
    m = KernelRidge(alpha=0.1, kernel="rbf", gamma=0.01)
    m.fit(sx.transform(X_tr_K), sy.transform(z_tr))
    yhat = sy.inverse_transform(m.predict(sx.transform(X_tb_K)))
    r2_per_dim = r2(yhat, z_tb)  # (d,)
    return r2_per_dim, z_tr


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pipeline = OmegaPipeline.from_manifest(PIPELINE_MANIFEST)
    print(f"[setup] pipeline loaded; train_std={pipeline.train_stats.std:.4f}")

    # 1. Load POD basis (Phi: (H*W, d=16))
    blob = np.load(POD_BASIS)
    Phi = blob["Phi"].astype(np.float32)  # (18432, 16)
    S = blob["S"].astype(np.float32)  # singular values (16,)
    d = int(blob["d"])
    print(f"[POD] loaded basis Phi: {Phi.shape}, d={d}")

    phi_2d = Phi.reshape(H, W, d).transpose(2, 0, 1)  # (d, H, W)

    # 2. Compute Q at impact frame across test_b
    q_arr, meta = collect_test_b_q(pipeline)
    n_enc = q_arr.shape[0]

    # Per-encounter normalize Q before averaging so high-G cases don't
    # dominate. Use per-encounter p90 of Q>0 as a soft amplitude.
    q_pos = np.clip(q_arr, 0.0, None)
    q_amp = np.percentile(q_pos.reshape(n_enc, -1), 90, axis=1)
    q_amp = np.maximum(q_amp, 1e-6)
    # Soft indicator: tanh(Q / amp) -- bounded in [0, 1] for Q>0
    q_indicator = np.tanh(q_pos / q_amp[:, None, None]).astype(np.float32)
    q_pos_mean = q_indicator.mean(axis=0)  # (H, W) in [0, 1]
    print(f"[Q] q_pos_mean stats: min={q_pos_mean.min():.3f}, "
          f"max={q_pos_mean.max():.3f}, mean={q_pos_mean.mean():.3f}")

    # Hard threshold for comparison / robustness
    # Use the median of per-encounter p99 of |Q| as a global threshold
    q_thresh = float(np.median(np.percentile(np.abs(q_arr).reshape(n_enc, -1),
                                              99, axis=1)))
    print(f"[Q] hard threshold tau={q_thresh:.3f}")
    q_hard = (q_arr > q_thresh).astype(np.float32).mean(axis=0)  # (H, W) in [0, 1]
    print(f"[Q] hard q_mask mean fraction: {q_hard.mean():.4f}")

    # 3. Per-mode Q-overlap
    overlap_soft = np.zeros(d, dtype=np.float32)
    overlap_hard = np.zeros(d, dtype=np.float32)
    for i in range(d):
        overlap_soft[i] = compute_mode_q_overlap(phi_2d[i], q_pos_mean)
        overlap_hard[i] = compute_mode_q_overlap(phi_2d[i], q_hard)

    # 4. Per-mode pressure-recoverability
    r2_per_dim, _ = fit_per_mode_pressure_r2(K_SENSORS)
    print(f"[pressure] per-dim R^2 (K={K_SENSORS}): "
          f"mean={r2_per_dim.mean():.3f}, "
          f"min={r2_per_dim.min():.3f}, max={r2_per_dim.max():.3f}")

    # 5. Correlation
    r_soft_p, p_soft_p = pearsonr(r2_per_dim, overlap_soft)
    r_soft_s, p_soft_s = spearmanr(r2_per_dim, overlap_soft)
    r_hard_p, p_hard_p = pearsonr(r2_per_dim, overlap_hard)
    r_hard_s, p_hard_s = spearmanr(r2_per_dim, overlap_hard)
    print(f"[corr] soft  Pearson r={r_soft_p:+.3f} (p={p_soft_p:.3g}), "
          f"Spearman r={r_soft_s:+.3f} (p={p_soft_s:.3g})")
    print(f"[corr] hard  Pearson r={r_hard_p:+.3f} (p={p_hard_p:.3g}), "
          f"Spearman r={r_hard_s:+.3f} (p={p_hard_s:.3g})")

    # 6. Scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, ov, label, r_p, p_p in [
        (axes[0], overlap_soft, "soft Q indicator (tanh)", r_soft_p, p_soft_p),
        (axes[1], overlap_hard, f"hard Q > tau ({q_thresh:.0f})", r_hard_p, p_hard_p),
    ]:
        ax.scatter(r2_per_dim, ov, s=70, c=np.arange(d), cmap="viridis",
                   edgecolor="k", linewidth=0.5, zorder=3)
        for i, (xi, yi) in enumerate(zip(r2_per_dim, ov)):
            ax.annotate(str(i), (xi, yi), xytext=(4, 4),
                        textcoords="offset points", fontsize=8, alpha=0.8)
        # Linear fit overlay
        if np.isfinite(r_p):
            zline = np.polyfit(r2_per_dim, ov, 1)
            xs = np.linspace(r2_per_dim.min(), r2_per_dim.max(), 50)
            ax.plot(xs, np.polyval(zline, xs), "r--", alpha=0.6,
                    label=f"linear fit (r={r_p:+.2f}, p={p_p:.2g})")
        ax.set_xlabel("per-mode pressure recoverability $R^2(z_i)$\n"
                      f"(KRR-RBF, K={K_SENSORS}, test_b)")
        ax.set_ylabel("$|\\phi_i|^2$-weighted Q-overlap")
        ax.set_title(label)
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.axvline(0, color="grey", linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
    fig.suptitle("POD d=16: pressure-recoverability vs Q-criterion overlap "
                 "(test_b at impact frame)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=160)
    print(f"[save] wrote {OUT_PNG}")

    # 7. Save stats
    stats = {
        "K_sensors": K_SENSORS,
        "n_test_b_encounters": int(n_enc),
        "n_modes": int(d),
        "q_threshold_hard": float(q_thresh),
        "q_pos_mean_fraction": float(q_pos_mean.mean()),
        "q_hard_mean_fraction": float(q_hard.mean()),
        "r2_per_mode": [float(v) for v in r2_per_dim],
        "q_overlap_soft": [float(v) for v in overlap_soft],
        "q_overlap_hard": [float(v) for v in overlap_hard],
        "pearson_soft": {"r": float(r_soft_p), "p": float(p_soft_p)},
        "spearman_soft": {"r": float(r_soft_s), "p": float(p_soft_s)},
        "pearson_hard": {"r": float(r_hard_p), "p": float(p_hard_p)},
        "spearman_hard": {"r": float(r_hard_s), "p": float(p_hard_s)},
        "hypothesis_supported_pearson_soft": bool(r_soft_p > 0.6),
        "hypothesis_supported_pearson_hard": bool(r_hard_p > 0.6),
        "method": {
            "Q": ("2D Q-criterion = 0.5*(|Omega|^2 - |S|^2) computed on "
                  "mid-plane (z=16) velocity at impact frame, NaN -> 0, "
                  "airfoil-mask cells zeroed."),
            "soft_indicator": ("tanh(max(Q,0) / p90(max(Q,0))) per encounter, "
                               "then averaged across encounters."),
            "hard_indicator": ("indicator Q > tau where tau = median over "
                               "encounters of p99(|Q|), then averaged "
                               "across encounters."),
            "overlap": ("sum_xy |phi_i(x,y)|^2 * Q_indicator_mean(x,y) "
                        "/ sum_xy |phi_i(x,y)|^2."),
            "pressure_recoverability": (f"KRR-RBF (alpha=0.1, gamma=0.01) on "
                                         f"(WINDOW=30, K={K_SENSORS}) "
                                         f"pre-impact wall-pressure window "
                                         f"-> z_impact; per-mode R^2 on test_b."),
        },
    }
    with open(OUT_JSON, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[save] wrote {OUT_JSON}")
    print(f"[DONE] hypothesis supported (Pearson > 0.6)? "
          f"soft={stats['hypothesis_supported_pearson_soft']}, "
          f"hard={stats['hypothesis_supported_pearson_hard']}")


if __name__ == "__main__":
    main()
