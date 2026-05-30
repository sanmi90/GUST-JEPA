"""Session 20 Track F: Gaussian scale decomposition of decoded vorticity fields.

READ-ONLY analysis (CPU numpy/scipy; no training, no GPU).

Goal
----
Tie the JEPA wake-observable advantage (its held-out wake-enstrophy closure is the
only positive one; Fukami and POD fall below the predict-the-mean floor) to
PHYSICAL-SPACE structure. We decompose every decoded omega_z field into a
large-scale part u_L (the leading-edge vortex and shear layer that carry the lift)
and a small-scale part u_S, then ask how faithfully each reconstruction's
LARGE-SCALE wake enstrophy tracks the DNS large-scale wake enstrophy.

References
----------
- Odaka, Lopez-Doriga, Taira, J. Fluid Mech. 1031, R3 (2026).
- Motoori & Goto (2019), Gaussian scale decomposition: u_L = G_sigma * u, u_S = u - u_L.

Method (per CLAUDE.md grid conventions)
---------------------------------------
Decoded fields live in pipeline-normalised 3-sigma vorticity space, shape
(n, 7, 192, 96). The array axes are (x, y): axis 0 (192) is the streamwise
direction over x in (-1.5, 4.5); axis 1 (96) is the cross-stream direction over
y in (-1.5, 1.5). The grid is isotropic at 32 pixels/chord (dx = dy = 1/32).

1. Gaussian scale decomposition: u_L = gaussian_filter(field, sigma) with
   sigma/c = 0.05  ->  sigma = 0.05 * 32 = 1.6 px. u_S = field - u_L.
2. Wake region: the downstream half x > LE, i.e. streamwise pixel index >= 48
   (x = 0 maps to index 48), with the frozen airfoil-adjacent mask (140 cells,
   inside-solid + 1-cell-neighbour) removed so the body itself does not enter the
   enstrophy. Enstrophy(field) = 0.5 * sum_{wake px} u_L^2 (large-scale field).
3. Per (method, frame-offset, encounter) we record the large-scale wake enstrophy,
   plus the small-scale wake enstrophy for the energy split.
4. Tracking metric: per method and split, the Pearson correlation and the relative
   error of large-scale wake enstrophy vs DNS at impact (offset 0) and impact+16,
   pooled over held-out encounters.
5. Staged-encounter curves: mean large-scale wake enstrophy across the 7 offsets
   for DNS / JEPA / Fukami / POD / JEPA-d32.

Offsets are [-8, 0, 8, 16, 24, 32, 40] relative to impact; index 1 is impact.

Output
------
outputs/session20/scale_decomp/scale_decomp.json  (all numbers)
outputs/session20/scale_decomp/scale_decomp.png   (figure)
outputs/session20/scale_decomp/scale_decomp.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

REPO = Path(__file__).resolve().parents[2]
DECODED_ROOT = REPO / "outputs" / "session20" / "decoded"
MASK_PATH = REPO / "outputs" / "data_pipeline" / "v1" / "airfoil_adjacent_mask.npy"
OUT_DIR = REPO / "outputs" / "session20" / "scale_decomp"

# Grid conventions (CLAUDE.md): array (192, 96) = (x, y), 32 px/chord.
PX_PER_CHORD = 32
SIGMA_OVER_C = 0.05
SIGMA_PX = SIGMA_OVER_C * PX_PER_CHORD  # = 1.6 px
# x = 0 (leading edge) maps to streamwise pixel index 48 in extent (-1.5, 4.5).
LE_X_INDEX = int(round((0.0 - (-1.5)) / (6.0 / 192)))  # = 48

SPLITS = ("test_a", "test_b", "test_c")
# (key in npz, label) ; target handled separately as the reference.
METHODS = [
    ("jepa_norm", "jepa_d64"),
    ("jepa_d32_norm", "jepa_d32"),
    ("fukami_norm", "fukami"),
    ("pod_norm", "pod"),
]
OFFSETS = [-8, 0, 8, 16, 24, 32, 40]  # index 1 = impact (offset 0)
IMPACT_IDX = 1
PLUS16_IDX = 3  # offset +16


def load_wake_mask() -> np.ndarray:
    """Boolean (192, 96) selecting wake-region pixels (x >= LE, body removed)."""
    airfoil = np.load(MASK_PATH).astype(bool)  # 140 inside-solid + adjacent cells
    wake = np.zeros((192, 96), dtype=bool)
    wake[LE_X_INDEX:, :] = True  # downstream half (streamwise index >= 48)
    wake &= ~airfoil  # remove the airfoil body and its 1-cell neighbourhood
    return wake


def scale_decompose(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian scale decomposition on the last two (spatial) axes.

    field: (..., 192, 96). Returns (u_L, u_S) with u_S = field - u_L.
    gaussian_filter is applied only over the spatial axes (sigma 0 on the rest).
    """
    sig = [0.0] * (field.ndim - 2) + [SIGMA_PX, SIGMA_PX]
    u_L = gaussian_filter(field, sigma=sig, mode="nearest")
    return u_L, field - u_L


def wake_enstrophy(field_L: np.ndarray, wake: np.ndarray) -> np.ndarray:
    """0.5 * sum over wake pixels of field_L^2.

    field_L: (n, 7, 192, 96). Returns (n, 7).
    """
    masked = field_L * wake[None, None, :, :]
    return 0.5 * (masked**2).sum(axis=(2, 3))


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rel_error(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean relative error |pred - true| / |true|, pooled over encounters."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    denom = np.abs(true)
    denom = np.where(denom < 1e-12, np.nan, denom)
    return float(np.nanmean(np.abs(pred - true) / denom))


def boot_ci(fn, x, y, rng, n=2000, ci=0.95):
    m = len(x)
    if m < 3:
        return float("nan"), float("nan")
    stats = []
    for _ in range(n):
        idx = rng.integers(0, m, size=m)
        stats.append(fn(x[idx], y[idx]))
    a = (1 - ci) / 2
    return float(np.nanquantile(stats, a)), float(np.nanquantile(stats, 1 - a))


def process_split(split: str, wake: np.ndarray, rng) -> dict:
    blob = np.load(DECODED_ROOT / f"{split}.npz", allow_pickle=True)
    target = blob["target_norm"].astype(np.float32)
    n_enc = target.shape[0]

    # Large-scale wake enstrophy and the small/large energy split per method.
    tgt_L, tgt_S = scale_decompose(target)
    ens_L = {"dns": wake_enstrophy(tgt_L, wake)}
    ens_S = {"dns": wake_enstrophy(tgt_S, wake)}
    for key, label in METHODS:
        fL, fS = scale_decompose(blob[key].astype(np.float32))
        ens_L[label] = wake_enstrophy(fL, wake)
        ens_S[label] = wake_enstrophy(fS, wake)

    out = {"n_enc": int(n_enc), "tracking": {}, "staged": {}, "energy_split": {}}

    # --- tracking metrics at impact and +16 (large-scale wake enstrophy vs DNS) ---
    for stage_name, idx in (("impact", IMPACT_IDX), ("impact+16", PLUS16_IDX)):
        dns_vals = ens_L["dns"][:, idx]
        out["tracking"][stage_name] = {}
        for label in [m[1] for m in METHODS]:
            pv = ens_L[label][:, idx]
            r = pearson(pv, dns_vals)
            re = rel_error(pv, dns_vals)
            rlo, rhi = boot_ci(pearson, pv, dns_vals, rng)
            relo, rehi = boot_ci(rel_error, pv, dns_vals, rng)
            out["tracking"][stage_name][label] = {
                "corr": r,
                "corr_ci": [rlo, rhi],
                "rel_error": re,
                "rel_error_ci": [relo, rehi],
                "mean_pred": float(pv.mean()),
                "mean_dns": float(dns_vals.mean()),
            }

    # --- staged curves: mean (+/- std) large-scale wake enstrophy per offset ---
    for label in ["dns"] + [m[1] for m in METHODS]:
        out["staged"][label] = {
            "mean": ens_L[label].mean(axis=0).tolist(),
            "std": ens_L[label].std(axis=0).tolist(),
        }
    out["offsets"] = OFFSETS

    # --- small-scale vs large-scale energy split (mean over enc, at impact & +16) ---
    for stage_name, idx in (("impact", IMPACT_IDX), ("impact+16", PLUS16_IDX)):
        out["energy_split"][stage_name] = {}
        for label in ["dns"] + [m[1] for m in METHODS]:
            L = float(ens_L[label][:, idx].mean())
            S = float(ens_S[label][:, idx].mean())
            tot = L + S
            out["energy_split"][stage_name][label] = {
                "large_scale": L,
                "small_scale": S,
                "small_frac": float(S / tot) if tot > 0 else float("nan"),
            }

    # keep raw arrays for the figure (only need staged DNS/jepa/fukami already in 'staged')
    out["_ens_L"] = {k: v for k, v in ens_L.items()}  # not serialised; popped before json
    out["_target_L"] = tgt_L  # for figure panels
    out["_recon_L"] = {
        label: scale_decompose(blob[key].astype(np.float32))[0] for key, label in METHODS
    }
    out["_case_ids"] = blob["case_ids"]
    out["_G"] = blob["G"]
    return out


def _load_airfoil_xy() -> "np.ndarray":
    """NACA 0012 outline in physical (x/c, y/c) coords from Baseline.h5, closed."""
    import os

    import h5py

    raw = (Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
           / "data" / "raw" / "periodic" / "Baseline.h5")
    with h5py.File(raw, "r") as f:
        xy = np.asarray(f["airfoil_xy"], dtype=np.float64)
    if not np.allclose(xy[0], xy[-1]):
        xy = np.vstack([xy, xy[0:1]])
    return xy


def make_figure(results: dict, wake: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    airfoil_xy = _load_airfoil_xy()  # physical coords, matches imshow extent

    rb = results["test_b"]
    # ---- representative large-scale fields at impact+16: DNS vs JEPA vs Fukami ----
    # pick the encounter whose DNS large-scale wake enstrophy is the median at +16.
    dns_p16 = rb["_ens_L"]["dns"][:, PLUS16_IDX]
    pick = int(np.argsort(dns_p16)[len(dns_p16) // 2])
    cid = str(rb["_case_ids"][pick])

    panel_fields = [
        ("DNS (large-scale)", rb["_target_L"][pick, PLUS16_IDX]),
        ("JEPA d64 (large-scale)", rb["_recon_L"]["jepa_d64"][pick, PLUS16_IDX]),
        ("Fukami d64 (large-scale)", rb["_recon_L"]["fukami"][pick, PLUS16_IDX]),
    ]

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.1], hspace=0.35, wspace=0.25)

    extent = [-1.5, 4.5, -1.5, 1.5]
    for j, (title, field) in enumerate(panel_fields):
        ax = fig.add_subplot(gs[0, j])
        # field is (192, 96) = (x, y); transpose to (y, x) for imshow with x horizontal.
        im = ax.imshow(
            field.T, origin="lower", extent=extent, cmap="RdBu_r", vmin=-3, vmax=3, aspect="equal"
        )
        ax.add_patch(Polygon(airfoil_xy, closed=True, facecolor="black",
                             edgecolor="black", linewidth=0.6, zorder=10))
        ax.axvline(0.0, color="k", lw=0.6, ls=":")  # leading edge / wake boundary
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x/c")
        if j == 0:
            ax.set_ylabel("y/c")
        ax.set_xlim(-0.5, 4.5)
    cax = fig.add_axes([0.92, 0.56, 0.012, 0.30])
    fig.colorbar(im, cax=cax, label=r"$\omega_z$ (norm.)")
    fig.text(
        0.5, 0.945, f"test_b large-scale fields at impact+16 (case {cid})", ha="center", fontsize=11
    )

    # ---- staged large-scale wake-enstrophy curves for the three splits ----
    colors = {"dns": "k", "jepa_d64": "C0", "jepa_d32": "C2", "fukami": "C3", "pod": "C1"}
    styles = {"dns": "-", "jepa_d64": "-", "jepa_d32": "--", "fukami": "-", "pod": ":"}
    for j, split in enumerate(SPLITS):
        ax = fig.add_subplot(gs[1, j])
        st = results[split]["staged"]
        for label in ["dns", "jepa_d64", "jepa_d32", "fukami", "pod"]:
            ax.plot(
                OFFSETS,
                st[label]["mean"],
                styles[label],
                color=colors[label],
                marker="o",
                ms=3,
                label=label,
                lw=1.8 if label == "dns" else 1.2,
            )
        ax.axvline(0.0, color="gray", lw=0.6, ls=":")
        ax.set_title(f"{split} (n={results[split]['n_enc']})", fontsize=10)
        ax.set_xlabel("frame offset from impact")
        if j == 0:
            ax.set_ylabel("large-scale wake enstrophy")
        if split == "test_c":
            ax.text(
                0.5,
                0.95,
                "|G|=4: flow is 3-D\n(mid-plane decomp. incomplete)",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=7.5,
                color="firebrick",
            )
        if j == 0:
            ax.legend(fontsize=7.5, loc="upper left")
    fig.text(
        0.5, 0.50, "Staged large-scale wake enstrophy vs frame offset", ha="center", fontsize=11
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"scale_decomp.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[scale_decomp] wrote figure to {OUT_DIR/'scale_decomp.png'}")


def strip_arrays(d: dict) -> dict:
    """Remove non-serialisable numpy arrays (keys starting with '_')."""
    return {k: v for k, v in d.items() if not k.startswith("_")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    wake = load_wake_mask()
    print(
        f"[scale_decomp] sigma = {SIGMA_PX:.2f} px (sigma/c = {SIGMA_OVER_C}); "
        f"LE x-index = {LE_X_INDEX}; wake pixels = {int(wake.sum())} "
        f"({100*wake.sum()/wake.size:.1f}% of grid)"
    )

    results = {}
    for split in SPLITS:
        results[split] = process_split(split, wake, rng)
        # concise stdout summary
        print(
            f"\n[{split}] n={results[split]['n_enc']}  large-scale wake-enstrophy tracking vs DNS"
        )
        for stage in ("impact", "impact+16"):
            print(f"  @{stage}:")
            for label in [m[1] for m in METHODS]:
                t = results[split]["tracking"][stage][label]
                print(
                    f"    {label:9s} corr={t['corr']:+.3f} "
                    f"rel_err={t['rel_error']:.3f} "
                    f"(mean pred/dns = {t['mean_pred']:.1f}/{t['mean_dns']:.1f})"
                )

    make_figure(results, wake)

    serial = {
        "config": {
            "sigma_px": SIGMA_PX,
            "sigma_over_c": SIGMA_OVER_C,
            "px_per_chord": PX_PER_CHORD,
            "le_x_index": LE_X_INDEX,
            "wake_pixels": int(wake.sum()),
            "wake_definition": "streamwise index >= 48 (x >= LE), airfoil-adjacent "
            "mask (140 cells) removed",
            "enstrophy_definition": "0.5 * sum_{wake px} u_L^2 on the large-scale field",
            "offsets": OFFSETS,
            "impact_index": IMPACT_IDX,
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
        },
        "splits": {s: strip_arrays(results[s]) for s in SPLITS},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "scale_decomp.json", "w") as f:
        json.dump(serial, f, indent=2)
    print(f"\n[scale_decomp] wrote {OUT_DIR/'scale_decomp.json'}")


if __name__ == "__main__":
    main()
