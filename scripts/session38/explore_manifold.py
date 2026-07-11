"""Exploratory: Fukami-style manifold view of the JEPA latent trajectories.

Nature-Communications-2023-style check WITHOUT retraining: PCA-3 projection
(fit on train latents) of (a) the d=32 pooled flagship and (b) the d=4
variant; baseline limit cycle + representative gusted encounters coloured by
|G|. Companion numbers: explained variance of the projection and the
Vietoris-Rips H1 counts already in outputs/session33/topology_v2p2.json.
Exploratory only; not wired into the manuscript.

Run (CPU): taskset -c 0-15 python -m scripts.session38.explore_manifold
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "session21"))

import matplotlib.pyplot as plt  # noqa: E402
from figstyle import TEXTWIDTH_IN, use_style  # noqa: E402

CACHE = REPO_ROOT / "outputs/session34/trackc_latents"
OUT = REPO_ROOT / "outputs/session38"

ENCOUNTERS = [  # (case_id, enc, |G|, split)
    ("G+0.50_D1.50_Y+0.00", 0, 0.5, "test_b"),
    ("G+2.00_D0.50_Y+0.10", 1, 2.0, "test_b"),
    ("G-3.00_D1.50_Y-0.10", 0, 3.0, "test_b"),
]
G_COLORS = {0.5: "#2166ac", 2.0: "#e08214", 3.0: "#b2182b"}
IMPACT = 40


def load(run, split):
    z = np.load(CACHE / f"latents_{run}_{split}.npz", allow_pickle=True)
    return z


def enc_rows(z, case_id, k):
    m = (z["case_id"] == case_id) & (z["encounter_index"] == k)
    idx = np.where(m)[0]
    return idx[np.argsort(z["frame"][idx])]


def main() -> int:
    use_style()
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 0.48 * TEXTWIDTH_IN))
    for panel, run in enumerate(("jepa_pool_vec", "jepa_pool_vec_d4")):
        tr = load(run, "train")
        tb = load(run, "test_b")
        Z = tr["z_gap"].astype(np.float64)
        mu = Z.mean(0)
        U, s, Vt = np.linalg.svd(Z - mu, full_matrices=False)
        ev = (s**2 / (s**2).sum())[:3].sum()
        P = Vt[:3].T

        ax = fig.add_subplot(1, 2, panel + 1, projection="3d")
        rows = enc_rows(tr, "Baseline", 0)
        B = (tr["z_gap"][rows].astype(np.float64) - mu) @ P
        ax.plot(B[:, 0], B[:, 1], B[:, 2], color="0.25", lw=1.0,
                label="baseline limit cycle")
        for case_id, k, g, split in ENCOUNTERS:
            src = tb if split == "test_b" else tr
            rows = enc_rows(src, case_id, k)
            T = (src["z_gap"][rows].astype(np.float64) - mu) @ P
            ax.plot(T[:, 0], T[:, 1], T[:, 2], color=G_COLORS[g], lw=1.0,
                    alpha=0.9, label=f"$|G| = {g:g}$")
            ax.scatter(*T[IMPACT], color=G_COLORS[g], s=14, marker="o",
                       edgecolors="k", linewidths=0.4, zorder=5)
            ax.scatter(*T[0], color=G_COLORS[g], s=10, marker="^",
                       edgecolors="k", linewidths=0.3, zorder=5)
            ax.scatter(*T[-1], color=G_COLORS[g], s=10, marker="s",
                       edgecolors="k", linewidths=0.3, zorder=5)
        d = tr["z_gap"].shape[1]
        ax.set_title(f"({'ab'[panel]}) $d = {d}$ latent, PCA-3 "
                     f"({100 * ev:.0f} per cent variance)", fontsize=7.5)
        for axis in ("x", "y", "z"):
            getattr(ax, f"set_{axis}label")(f"PC{'xyz'.index(axis) + 1}",
                                            fontsize=6.5, labelpad=-4)
            getattr(ax, f"{axis}axis").set_ticklabels([])
        ax.view_init(elev=18, azim=-60)
        if panel == 0:
            ax.legend(fontsize=6, loc="upper left", frameon=False)
        # zoomed view of the baseline neighbourhood: the shedding cycle is
        # SMALL against the gust excursions (that scale separation is the
        # observation, not an artefact)
        span = np.abs(B).max() * 1.6
        ax.text2D(0.02, 0.02,
                  f"baseline cycle spans {span / np.abs(P.T @ (0*mu) - 0).max() if False else 0:.0f}",
                  transform=ax.transAxes, fontsize=5) if False else None
        print(f"[manifold] {run}: d={d}, PCA-3 variance {ev:.3f}")
    fig.suptitle("JEPA latent trajectories: baseline limit cycle + gusted "
                 "excursions (impact marked)", fontsize=8, y=1.02)
    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        path = OUT / f"explore_manifold_jepa.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
