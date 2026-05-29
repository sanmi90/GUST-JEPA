"""Diagnostic: does rolled-out latent drift off the encoded-latent manifold?

This is the OOD-decoder check we discussed — the SL decoder was trained on
encoded z_t (from DNS frames), so if rolled-out predicted ẑ drift away from
that distribution, the decoder is being asked to reconstruct from OOD inputs.

For each baseline (JEPA d=64, JEPA d=32, POD d=64, Fukami d=64), we compute:
  1. Per-frame ||z_dns − z_predicted||₂ (L2 drift)
  2. Mahalanobis distance of z_predicted from the encoded-z train distribution
  3. PCA-2D projection scatter: encoded test_b vs Markov-predicted ẑ over time

Output: outputs/session18/exp_b1_test3/latent_drift_diagnostic.png + JSON summary.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA

REPO = Path("/home/carlos/GUST-JEPA")
ROOT = REPO / "outputs/session18/exp_b1_test3"
LAT  = REPO / "outputs/session18/exp_b1"

BASELINES = {
    "JEPA d=64":   ("jepa_d64_test1_noBN", "latents_jepa_d64", "#1f77b4"),
    "JEPA d=32":   ("jepa_d32_noBN",       "latents_jepa_d32", "#17becf"),
    "POD d=64":    ("pod_d64_noBN",        "latents_pod_d64",  "#d62728"),
    "Fukami d=64": ("fukami_d64_noBN",     "latents_fukami_d64","#2ca02c"),
}


def load_rollouts(tag):
    return np.load(ROOT / f"rollouts_{tag}" / "test_b.npz", allow_pickle=True)


def load_train_z(latents_subdir):
    f = LAT / latents_subdir / "train.npz"
    d = np.load(f, allow_pickle=True)
    z = d["z_full"]  # (n, T, d)
    return z.reshape(-1, z.shape[-1])  # (n*T, d)


def main():
    print("Computing latent drift diagnostics...")
    summary = {}

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    horizons = list(range(0, 120))

    for col, (label, (tag, lat_dir, color)) in enumerate(BASELINES.items()):
        ro = load_rollouts(tag)
        z_dns = ro["z_dns"]      # (42, 120, d)  encoded ground truth
        z_markov = ro["z_markov"]  # (42, 120, d)  Markov-only rollout from z_impact
        z_full   = ro["z_full"]    # (42, 120, d)  full-context rollout
        d = z_dns.shape[-1]
        n_enc = z_dns.shape[0]

        z_train = load_train_z(lat_dir)  # (n_train_frames, d)
        print(f"\n[{label}] tag={tag}  d={d}  test_b encs={n_enc}  train_frames={len(z_train)}")

        # ---- (a) per-frame L2 drift ‖z_dns − z_predicted‖₂ ----
        l2_markov = np.linalg.norm(z_markov - z_dns, axis=-1)  # (42, 120)
        l2_full   = np.linalg.norm(z_full   - z_dns, axis=-1)
        # normalize by typical ‖z_dns‖
        l2_dns    = np.linalg.norm(z_dns, axis=-1)
        rel_markov = (l2_markov / np.maximum(l2_dns, 1e-9))
        rel_full   = (l2_full   / np.maximum(l2_dns, 1e-9))

        # ---- (b) Mahalanobis distance from train distribution ----
        mu = z_train.mean(0)
        cov = np.cov(z_train.T) + 1e-6 * np.eye(d)
        cov_inv = np.linalg.inv(cov)
        def mahal(z):  # z: (..., d)
            shape = z.shape[:-1]
            zf = z.reshape(-1, d) - mu
            md = np.sqrt(np.sum((zf @ cov_inv) * zf, axis=-1))
            return md.reshape(shape)

        md_dns    = mahal(z_dns)
        md_markov = mahal(z_markov)

        # Per-horizon mean across encounters
        ax = axes[0, col]
        ax.plot(horizons, rel_markov.mean(0), color="#d62728", linewidth=2, label="Markov ẑ")
        ax.fill_between(horizons,
                        rel_markov.mean(0) - rel_markov.std(0),
                        rel_markov.mean(0) + rel_markov.std(0),
                        color="#d62728", alpha=0.15)
        ax.plot(horizons, rel_full.mean(0), color="#1f77b4", linewidth=2, label="Full-context ẑ")
        ax.fill_between(horizons,
                        rel_full.mean(0) - rel_full.std(0),
                        rel_full.mean(0) + rel_full.std(0),
                        color="#1f77b4", alpha=0.15)
        ax.axvline(40, color="k", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.text(40, 0.05, "impact", rotation=90, ha="right", va="bottom", fontsize=8)
        ax.set_title(f"{label}\nrel. ‖z_dns − ẑ‖ / ‖z_dns‖")
        ax.set_xlabel("frame")
        ax.set_ylabel("relative drift")
        ax.set_ylim(0, max(rel_markov.mean(0).max(), rel_full.mean(0).max()) * 1.2 + 0.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")

        # Mahalanobis panel below
        ax2 = axes[1, col]
        ax2.plot(horizons, md_dns.mean(0),    color="black",   linewidth=2, label="z_dns (encoded)")
        ax2.plot(horizons, md_markov.mean(0), color="#d62728", linewidth=2, label="Markov ẑ")
        ax2.fill_between(horizons,
                         md_markov.mean(0) - md_markov.std(0),
                         md_markov.mean(0) + md_markov.std(0),
                         color="#d62728", alpha=0.15)
        ax2.axvline(40, color="k", linestyle="--", alpha=0.5, linewidth=0.8)
        ax2.set_title(f"Mahalanobis from train z dist.")
        ax2.set_xlabel("frame")
        ax2.set_ylabel("Mahalanobis dist")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8, loc="best")

        summary[label] = {
            "tag": tag, "d": int(d), "n_encounters": int(n_enc),
            "rel_drift_markov_at_h0":  float(rel_markov[:, 40].mean()),  # at impact
            "rel_drift_markov_at_h16": float(rel_markov[:, 56].mean()) if 56 < 120 else None,  # impact+16
            "rel_drift_markov_at_h32": float(rel_markov[:, 72].mean()) if 72 < 120 else None,
            "rel_drift_full_at_h16":   float(rel_full[:, 56].mean()) if 56 < 120 else None,
            "rel_drift_full_at_h32":   float(rel_full[:, 72].mean()) if 72 < 120 else None,
            "mahal_dns_mean":     float(md_dns.mean()),
            "mahal_markov_mean":  float(md_markov.mean()),
            "mahal_ratio_markov_over_dns": float(md_markov.mean() / md_dns.mean()),
        }

    plt.suptitle("Latent drift: rolled-out ẑ vs encoded z (test_b) — JEPA vs POD vs Fukami",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_png = ROOT / "latent_drift_diagnostic.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"\nFigure saved: {out_png}")

    with open(ROOT / "latent_drift_diagnostic.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary ===")
    print(f"{'Baseline':<14}  rel drift Markov     rel drift Full       Mahal ratio (Markov/DNS)")
    for label, s in summary.items():
        r_m_h0  = s["rel_drift_markov_at_h0"]
        r_m_h16 = s["rel_drift_markov_at_h16"]
        r_m_h32 = s["rel_drift_markov_at_h32"]
        r_f_h16 = s["rel_drift_full_at_h16"]
        r_f_h32 = s["rel_drift_full_at_h32"]
        mr = s["mahal_ratio_markov_over_dns"]
        print(f"  {label:<12}  h0={r_m_h0:.3f}  h16={r_m_h16:.3f}  h32={r_m_h32:.3f}  |  "
              f"h16={r_f_h16:.3f}  h32={r_f_h32:.3f}  |  {mr:.2f}x")


if __name__ == "__main__":
    main()
