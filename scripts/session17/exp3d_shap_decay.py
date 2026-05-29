"""Session 17, Experiment 3, Part (d): SHAP-on-Y attribution decay vs tau.

For 5 representative Test B encounters, compute integrated-gradient SHAP for
Y at frame offsets tau in {-10, -5, 0, +5, +10}. Each tau has its own probe
trained on z(t_impact + tau) -> Y.

Spatial concentration metric:
  conc(tau) = sum(|attr|[disk around LE]) / sum(|attr|[full field])
where the LE disk has radius 0.5 c (16 pixels) centered at (pixel_x=48,
pixel_y=48). Hypothesis: conc(tau) peaks at tau=0 and decays by |tau|=10.

Outputs:
    outputs/session17/exp3/shap_decay.npz
    outputs/session17/exp3/shap_decay_summary.json
    outputs/session17/figures/exp3_shap_decay_panels.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.data.omega_pipeline import OmegaPipeline  # noqa: E402
from src.utils.device import require_rtx6000  # noqa: E402
from scripts.session16.exp3_shap import (  # noqa: E402
    load_encoder,
    OMEGA_MANIFEST,
    DEFAULT_IMPACT_FRAME,
)


SEED_LATENTS = REPO / "outputs" / "session17" / "seed_latents"
EXP3 = REPO / "outputs" / "session17" / "exp3"
FIGS = REPO / "outputs" / "session17" / "figures"
SPLIT_MANIFEST = REPO / "configs" / "splits" / "split_v2.json"
T_IMPACT = 40
TAUS = (-10, -5, 0, +5, +10)
N_STEPS = 32

# Physical extent: x in [-1.5, +4.5] (192 pixels), y in [-1.5, +1.5] (96 pixels).
# LE at physical (0, 0) -> pixel (48, 48) in the (H=192, W=96) omega array.
LE_PX = (48, 48)  # (x_pixel, y_pixel)
DISK_RADIUS_PX = 16  # 0.5 c at 0.03125 c/pixel
H_GRID, W_GRID = 192, 96

# Representative encounters (subset of 10 reps from Exp 1(b)).
REP_5 = [
    {"case_id": "G+1.00_D1.00_Y+0.10", "k": 0, "Y": +0.10, "label": "G+1.00 Y+0.10"},
    {"case_id": "G+0.50_D1.00_Y+0.20", "k": 0, "Y": +0.20, "label": "G+0.50 Y+0.20"},
    {"case_id": "G+3.00_D1.00_Y-0.20", "k": 0, "Y": -0.20, "label": "G+3.00 Y-0.20"},
    {"case_id": "G-1.50_D0.50_Y-0.20", "k": 0, "Y": -0.20, "label": "G-1.50 Y-0.20"},
    {"case_id": "G-2.00_D1.00_Y+0.00", "k": 0, "Y": +0.00, "label": "G-2.00 Y+0.00"},
]


def le_disk_mask() -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(W_GRID), np.arange(H_GRID), indexing="ij")
    xx = xx.T
    yy = yy.T
    dist = np.sqrt((xx - LE_PX[0]) ** 2 + (yy - LE_PX[1]) ** 2)
    return dist <= DISK_RADIUS_PX


class MLPProbe(nn.Module):
    def __init__(self, in_dim=64, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_train_test_a_for_tau(tau: int) -> tuple[dict, dict]:
    """Use session14 pre-extracted train/test_a z_full to grab z(t_impact+tau)."""
    train = np.load(
        REPO / "outputs" / "session14" / "latents" / "S12_E_d64" / "train.npz",
        allow_pickle=True,
    )
    test_a = np.load(
        REPO / "outputs" / "session14" / "latents" / "S12_E_d64" / "test_a.npz",
        allow_pickle=True,
    )
    t = T_IMPACT + tau
    return (
        {
            "z": train["z_full"].astype(np.float32)[:, t, :],
            "Y": train["Y"].astype(np.float32),
        },
        {
            "z": test_a["z_full"].astype(np.float32)[:, t, :],
            "Y": test_a["Y"].astype(np.float32),
        },
    )


def train_y_probe(
    train: dict, test_a: dict, device, *,
    weight_decay=1e-2, lr=3e-4, batch=64, max_iters=4000, patience=400, seed=0,
) -> tuple[MLPProbe, float, float, int, float]:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    z_train = train["z"].astype(np.float32)
    y_train = train["Y"].astype(np.float32)
    z_val = test_a["z"].astype(np.float32)
    y_val = test_a["Y"].astype(np.float32)
    d = z_train.shape[1]
    mean = float(y_train.mean())
    std = float(y_train.std()) or 1.0
    y_norm = (y_train - mean) / std
    yv_norm = (y_val - mean) / std

    probe = MLPProbe(in_dim=d).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    z_val_t = torch.from_numpy(z_val).to(device)
    y_val_t = torch.from_numpy(yv_norm).to(device)

    best_val = float("inf")
    best_state = None
    best_iter = 0
    for it in range(max_iters):
        idx = rng.choice(z_train.shape[0], size=min(batch, z_train.shape[0]), replace=False)
        z_batch = z_train[idx]
        y_batch = y_norm[idx]
        probe.train()
        z_t = torch.from_numpy(z_batch).to(device)
        y_t = torch.from_numpy(y_batch).to(device)
        opt.zero_grad()
        loss = ((probe(z_t) - y_t) ** 2).mean()
        loss.backward()
        opt.step()
        if (it + 1) % 50 == 0:
            probe.eval()
            with torch.no_grad():
                vl = ((probe(z_val_t) - y_val_t) ** 2).mean().item()
            if vl < best_val:
                best_val = vl
                best_state = {k: v.clone() for k, v in probe.state_dict().items()}
                best_iter = it + 1
            elif it + 1 - best_iter > patience:
                break
    if best_state is not None:
        probe.load_state_dict(best_state)
    probe.eval()
    for p in probe.parameters():
        p.requires_grad_(False)
    return probe, mean, std, best_iter, best_val


def integrated_gradients(
    enc, probe, omega_baseline: Tensor, omega_input: Tensor,
    n_steps: int, device: torch.device, frame_t_in_seq: int,
) -> Tensor:
    """SHAP for probe(enc(omega)[:, frame_t_in_seq, :]) wrt omega.

    omega_baseline, omega_input: (H, W) normalised. Returns (H, W) attribution.
    For Exp 3(d), the encoder is fed a single frame (T=1), so frame_t_in_seq=0.
    """
    alphas = torch.linspace(0.0, 1.0, n_steps, device=device)
    diff = (omega_input - omega_baseline).to(device)
    grads_accum = torch.zeros_like(diff)
    for alpha in alphas:
        x = omega_baseline + alpha * diff
        x = x.detach().requires_grad_(True)
        x_in = x.view(1, 1, 1, x.shape[0], x.shape[1])
        z = enc(x_in)
        z_pick = z[:, frame_t_in_seq, :]
        y = probe(z_pick)
        y.sum().backward()
        grads_accum = grads_accum + x.grad.detach()
    avg_grad = grads_accum / n_steps
    return (avg_grad * diff).detach()


def resolve_omega_path(cid: str, k: int) -> Path:
    import os
    cache = os.environ.get("VORTEX_JEPA_CACHE")
    if cache:
        return Path(cache) / "v1" / cid / f"encounter_{int(k):02d}.h5"
    return (
        Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
        / "data" / "processed" / "vortex-jepa" / "v1"
        / cid / f"encounter_{int(k):02d}.h5"
    )


def gather_baseline_paths() -> list[dict]:
    """Baseline encounters: case 'Baseline', k in 0..3 (train-side per CLAUDE.md)."""
    return [
        {"case_id": "Baseline", "k": k, "path": resolve_omega_path("Baseline", k)}
        for k in range(4)
    ]


def load_omega_normalized_(path: Path, pipeline: OmegaPipeline, cid: str, k: int) -> tuple[np.ndarray, int]:
    with h5py.File(path, "r") as f:
        omega_raw = np.asarray(f["omega_z"], dtype=np.float32)
        impact = int(f.attrs.get("impact_frame_estimate", DEFAULT_IMPACT_FRAME))
    omega_clean = pipeline.preprocess_raw(omega_raw, cid, int(k))
    omega_norm = pipeline.normalize(omega_clean).astype(np.float32)
    return omega_norm, impact


def render_figure_from_cache() -> bool:
    """Re-render the decay figure from the cached NPZ + summary JSON.

    Returns True if both artefacts existed and the figure was written.
    Used when the user only wants to refresh plot styling without
    re-running the SHAP integration on GPU.
    """
    npz_path = EXP3 / "shap_decay.npz"
    summary_path = EXP3 / "shap_decay_summary.json"
    if not npz_path.exists() or not summary_path.exists():
        return False
    data = np.load(npz_path, allow_pickle=True)
    summary = json.loads(summary_path.read_text())

    taus = list(data["taus"].tolist())
    # Recover per-encounter attribution stacks.
    keys_list = []
    attr_by_key: dict = {}
    j = 0
    while f"attr_{j}" in data.files:
        a = data[f"attr_{j}"]  # (n_tau, H, W) signed
        cid = str(data[f"case_id_{j}"])
        k = int(data[f"k_{j}"])
        key = (cid, k)
        keys_list.append(key)
        attr_by_key[key] = {tau: a[i] for i, tau in enumerate(taus)}
        j += 1

    # Per-encounter normalised |attr|, then mean across encounters per tau.
    mean_attr_per_tau = []
    for t in taus:
        attrs = np.stack([attr_by_key[key][t] for key in keys_list], axis=0)
        attrs_abs = np.abs(attrs)
        max_per_enc = attrs_abs.reshape(attrs.shape[0], -1).max(axis=1)
        attrs_norm = attrs_abs / np.maximum(max_per_enc[:, None, None], 1e-12)
        mean_attr_per_tau.append(attrs_norm.mean(axis=0))

    mean_per_tau = np.array(summary["aggregate_concentration_LE"]["mean_per_tau"])
    _render_decay_figure(mean_attr_per_tau, mean_per_tau, summary, taus)
    return True


def _render_decay_figure(
    mean_attr_per_tau: list, mean_per_tau: np.ndarray,
    summary: dict, taus: list,
) -> None:
    """Render the SHAP-decay figure with paper-grade typography."""
    from matplotlib.patches import Circle

    X_EXT = (-1.5, 4.5)
    Y_EXT = (-1.5, 1.5)
    # Use 99th-percentile clip across the pooled mean-attr fields so the
    # low-magnitude structure is visible; raw max is dominated by a few
    # bright pixels near the LE.
    pooled = np.concatenate([m.ravel() for m in mean_attr_per_tau])
    vmax = float(np.percentile(pooled, 99.0))
    vmin = 0.0
    cmap = "magma"

    # 5 SHAP panels + colorbar slot + LE-concentration line plot.
    # Outer 2-column split: SHAP block (wide) + line plot (square).
    # Inside the SHAP block: 5 imshow panels + a thin colorbar.
    fig = plt.figure(figsize=(26, 4.6))
    outer = fig.add_gridspec(
        1, 2, width_ratios=[5.5, 1.6], wspace=0.18,
    )
    shap_block = outer[0, 0].subgridspec(
        1, 6,
        width_ratios=[1.0, 1.0, 1.0, 1.0, 1.0, 0.08],
        wspace=0.18,
    )
    panel_axes = [fig.add_subplot(shap_block[0, i]) for i in range(5)]
    cax = fig.add_subplot(shap_block[0, 5])
    line_ax = fig.add_subplot(outer[0, 1])

    im = None
    for col, (t, mattr) in enumerate(zip(taus, mean_attr_per_tau)):
        ax = panel_axes[col]
        im = ax.imshow(
            mattr.T, origin="lower", extent=(*X_EXT, *Y_EXT),
            cmap=cmap, vmin=vmin, vmax=vmax,
        )
        # LE disk overlay (radius 0.5 c around physical (0, 0)).
        ax.add_patch(Circle((0, 0), 0.5, fill=False, color="white", lw=1.4))
        # Airfoil chord.
        ax.plot([0, 1], [0, 0], "w-", lw=2.0, alpha=0.85)
        conc = summary["aggregate_concentration_LE"]["mean_per_tau"][col]
        ax.set_title(
            rf"$\tau = {t:+d}$    conc={conc:.3f}",
            fontsize=13, pad=6,
        )
        ax.set_xlim(*X_EXT)
        ax.set_ylim(*Y_EXT)
        ax.set_aspect("equal")
        ax.tick_params(axis="both", labelsize=10)
        if col == 0:
            ax.set_ylabel("y / c", fontsize=12)
        ax.set_xlabel("x / c", fontsize=12)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(
        "mean per-encounter normalised |attr|",
        fontsize=12,
    )
    cbar.ax.tick_params(labelsize=10)

    line_ax.plot(taus, mean_per_tau, "o-", color="C0", lw=2.0,
                 markersize=8, label="mean")
    line_ax.fill_between(
        taus,
        np.array(summary["aggregate_concentration_LE"]["min_per_tau"]),
        np.array(summary["aggregate_concentration_LE"]["max_per_tau"]),
        alpha=0.25, color="C0", label="min/max",
    )
    line_ax.axvline(0, color="k", lw=1, alpha=0.6)
    line_ax.set_xlabel(r"$\tau$ (frames from impact)", fontsize=12)
    line_ax.set_ylabel(r"frac |attr| within LE disk ($r = 0.5\,c$)",
                       fontsize=12)
    line_ax.set_title("LE concentration vs " + r"$\tau$", fontsize=13)
    line_ax.legend(fontsize=11, loc="best")
    line_ax.grid(alpha=0.3)
    line_ax.tick_params(axis="both", labelsize=11)

    fig.suptitle(
        "SHAP-on-Y attribution decay vs frame offset from impact",
        fontsize=15, y=1.02,
    )
    fig.savefig(
        FIGS / "exp3_shap_decay_panels.png",
        dpi=160, bbox_inches="tight",
    )
    fig.savefig(
        FIGS / "exp3_shap_decay_panels.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"[exp3d] wrote {FIGS / 'exp3_shap_decay_panels.png'}")
    print(f"[exp3d] wrote {FIGS / 'exp3_shap_decay_panels.pdf'}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n-steps", type=int, default=N_STEPS)
    p.add_argument(
        "--plot-only", action="store_true",
        help="Re-render the figure from the cached NPZ + summary JSON; "
        "skip probe training and SHAP integration.",
    )
    args = p.parse_args()

    if args.plot_only:
        ok = render_figure_from_cache()
        if not ok:
            raise SystemExit(
                "[exp3d] --plot-only requested but cache is missing: "
                f"{EXP3 / 'shap_decay.npz'} or {EXP3 / 'shap_decay_summary.json'}"
            )
        return

    device = require_rtx6000(gpu_index=args.gpu)
    print(f"[exp3d] device={device}")

    enc = load_encoder(device)
    pipeline = OmegaPipeline.from_manifest(OMEGA_MANIFEST)
    disk = le_disk_mask()
    print(
        f"[exp3d] LE disk: {disk.sum()} pixels of {disk.size}  "
        f"({100*disk.sum()/disk.size:.2f}% of field)"
    )

    # Train one probe per tau.
    probes = {}
    probe_info = {}
    for tau in TAUS:
        t0 = time.time()
        train, test_a = load_train_test_a_for_tau(tau)
        probe, mean, std, best_iter, best_val = train_y_probe(train, test_a, device)
        probes[tau] = (probe, mean, std)
        probe_info[tau] = {
            "best_iter": best_iter,
            "best_val_loss": best_val,
            "y_mean": mean,
            "y_std": std,
            "train_time_s": time.time() - t0,
        }
        print(
            f"[exp3d] probe tau={tau:+3d} trained in {time.time() - t0:.1f}s  "
            f"best_iter={best_iter} val_loss={best_val:.4f}"
        )

    # Phase-matched baseline (mean of Baseline encs 0..3).
    baselines = gather_baseline_paths()
    base_arrs = []
    for b in baselines:
        if not b["path"].exists():
            continue
        o, _imp = load_omega_normalized_(b["path"], pipeline, b["case_id"], b["k"])
        base_arrs.append(o)
    base_mean = np.stack(base_arrs, axis=0).mean(axis=0)
    print(f"[exp3d] baseline mean per-frame shape: {base_mean.shape}")

    # Compute SHAP attribution for each (encounter, tau).
    results = {}
    for r in REP_5:
        cid, k = r["case_id"], r["k"]
        path = resolve_omega_path(cid, k)
        if not path.exists():
            print(f"[exp3d] missing {path}, skip")
            continue
        omega_norm, impact = load_omega_normalized_(path, pipeline, cid, k)
        if impact != T_IMPACT:
            print(
                f"[exp3d] WARN {cid} k={k}: impact={impact}, expected {T_IMPACT}; "
                f"using cache value"
            )
        per_tau = {}
        for tau in TAUS:
            t = T_IMPACT + tau
            if t < 0 or t >= omega_norm.shape[0]:
                continue
            probe, mean, std = probes[tau]
            omega_input = torch.from_numpy(omega_norm[t]).to(device)
            omega_baseline = torch.from_numpy(base_mean[t]).to(device)
            attr = integrated_gradients(
                enc, probe, omega_baseline, omega_input,
                n_steps=args.n_steps, device=device, frame_t_in_seq=0,
            )
            attr_np = attr.cpu().numpy()
            abs_attr = np.abs(attr_np)
            disk_sum = abs_attr[disk].sum()
            total_sum = abs_attr.sum()
            conc = float(disk_sum / max(total_sum, 1e-12))
            per_tau[tau] = {
                "attr_l1": float(total_sum),
                "concentration_LE": conc,
                "max_attr": float(np.abs(attr_np).max()),
            }
            results.setdefault("attr", {}).setdefault((cid, k), {})[tau] = attr_np
        results.setdefault("per_tau", {})[(cid, k)] = per_tau
        results.setdefault("meta", {})[(cid, k)] = {
            "Y": float(r["Y"]),
            "label": r["label"],
            "impact": int(impact),
        }
        print(f"[exp3d] {r['label']:20s}  conc(tau): "
              + "  ".join(f"{tau:+3d}={per_tau[tau]['concentration_LE']:.3f}" for tau in TAUS))

    # Save NPZ.
    save: dict = {
        "taus": np.array(TAUS),
        "disk_mask": disk,
        "le_pixel": np.array(LE_PX),
        "disk_radius_px": DISK_RADIUS_PX,
        "n_integration_steps": args.n_steps,
    }
    keys_list = list(results.get("attr", {}).keys())
    for j, key in enumerate(keys_list):
        save[f"attr_{j}"] = np.stack(
            [results["attr"][key][tau] for tau in TAUS], axis=0
        )  # (n_tau, H, W)
        save[f"label_{j}"] = results["meta"][key]["label"]
        save[f"case_id_{j}"] = key[0]
        save[f"k_{j}"] = key[1]
        save[f"Y_{j}"] = results["meta"][key]["Y"]
    np.savez_compressed(EXP3 / "shap_decay.npz", **save)
    print(f"[exp3d] wrote {EXP3 / 'shap_decay.npz'}")

    summary = {
        "taus": list(TAUS),
        "disk_radius_px": DISK_RADIUS_PX,
        "disk_area_frac": float(disk.sum() / disk.size),
        "probes": {str(t): probe_info[t] for t in TAUS},
        "per_encounter": [
            {
                "case_id": key[0],
                "k": int(key[1]),
                "Y": results["meta"][key]["Y"],
                "label": results["meta"][key]["label"],
                "per_tau": {
                    str(t): results["per_tau"][key][t] for t in TAUS
                    if t in results["per_tau"][key]
                },
            }
            for key in keys_list
        ],
    }
    # Aggregate across encounters
    concs = np.zeros((len(keys_list), len(TAUS)))
    for i, key in enumerate(keys_list):
        for j, t in enumerate(TAUS):
            if t in results["per_tau"][key]:
                concs[i, j] = results["per_tau"][key][t]["concentration_LE"]
    summary["aggregate_concentration_LE"] = {
        "mean_per_tau": [float(v) for v in concs.mean(axis=0)],
        "std_per_tau": [float(v) for v in concs.std(axis=0)],
        "min_per_tau": [float(v) for v in concs.min(axis=0)],
        "max_per_tau": [float(v) for v in concs.max(axis=0)],
    }
    (EXP3 / "shap_decay_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[exp3d] wrote {EXP3 / 'shap_decay_summary.json'}")

    # Aggregate gate: peak at tau=0, drop to <0.5 of peak by |tau|=10.
    mean_per_tau = np.array(summary["aggregate_concentration_LE"]["mean_per_tau"])
    tau_arr = np.array(TAUS)
    peak_idx = int(np.argmax(mean_per_tau))
    peak_tau = int(tau_arr[peak_idx])
    peak_val = float(mean_per_tau[peak_idx])
    # halve condition at |tau| = 10
    minus10_idx = list(tau_arr).index(-10)
    plus10_idx = list(tau_arr).index(+10)
    halved_left = mean_per_tau[minus10_idx] < 0.5 * peak_val
    halved_right = mean_per_tau[plus10_idx] < 0.5 * peak_val
    print(
        f"\n[exp3d] gate: peak_tau={peak_tau} (val={peak_val:.3f})  "
        f"halved_left={halved_left}  halved_right={halved_right}"
    )

    # Figure: 5 SHAP panels + colorbar + LE-concentration line plot.
    mean_attr_per_tau = []
    for j, t in enumerate(TAUS):
        attrs = np.stack(
            [results["attr"][key][t] for key in keys_list], axis=0
        )  # (n_enc, H, W)
        # Per-encounter normalise to its own max |attr| so cross-encounter
        # averaging is fair.
        attrs_abs = np.abs(attrs)
        max_per_enc = attrs_abs.reshape(attrs.shape[0], -1).max(axis=1)
        attrs_norm = attrs_abs / np.maximum(max_per_enc[:, None, None], 1e-12)
        mean_attr = attrs_norm.mean(axis=0)  # (H, W)
        mean_attr_per_tau.append(mean_attr)

    _render_decay_figure(mean_attr_per_tau, mean_per_tau, summary, list(TAUS))


if __name__ == "__main__":
    main()
