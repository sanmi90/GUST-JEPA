"""Session 17, Experiment 4: from SHAP maps to coherent structures.

Reads Session 16 SHAP attribution maps for 4 targets (centroid_x,
circulation_pos, peak_neg_omega, Y) and:
  (a) Connected-component extraction at 98th-percentile threshold.
  (b) Threshold sensitivity sweep over {95, 97.5, 98, 99, 99.5}.
  (c) Q-criterion comparison: compute Q = 0.5*(||Omega||^2 - ||S||^2) at the
      impact frame from cached velocity (u, v, w) fields and check overlap
      with SHAP structures.
  (d) Y sign analysis: confirm the strongest SHAP_Y structure flips spatial
      position with sign(Y).

Outputs:
    outputs/session17/exp4/structure_catalog.csv
    outputs/session17/exp4/threshold_sensitivity.json
    outputs/session17/exp4/q_overlap.csv
    outputs/session17/exp4/Y_sign_flip.json
    outputs/session17/figures/exp4_structures_4target_panel.png
    outputs/session17/figures/exp4_q_overlap_summary.png
    outputs/session17/figures/exp4_Y_sign_flip.png
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label as cc_label


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


EXP3_S16 = REPO / "outputs" / "session16" / "exp3"
EXP4 = REPO / "outputs" / "session17" / "exp4"
EXP4.mkdir(parents=True, exist_ok=True)
FIGS = REPO / "outputs" / "session17" / "figures"
PARTITION = "v1"
DEFAULT_IMPACT_FRAME = 40
CACHE_ROOT = Path(
    os.environ.get(
        "VORTEX_JEPA_CACHE",
        str(Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
            / "data" / "processed" / "vortex-jepa"),
    )
)
PREVENT = Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))

DX = 6.0 / 192
DY = 3.0 / 96
LE_PX = (48, 48)
TARGETS = ("centroid_x", "circulation_pos", "peak_neg_omega", "Y")
PERCENTILES = (95.0, 97.5, 98.0, 99.0, 99.5)
MIN_AREA_PX = 10
SPLITS = ("test_b", "test_c")
AIRFOIL_MASK_PATH = REPO / "outputs" / "data_pipeline" / "v1" / "airfoil_adjacent_mask.npy"


def load_shap_attr(split: str) -> dict[str, np.ndarray]:
    """Load attribution maps + stability flags for all 4 targets."""
    d3 = np.load(EXP3_S16 / "shap_attribution.npz", allow_pickle=True)
    dY = np.load(EXP3_S16 / "shap_Y_attribution.npz", allow_pickle=True)
    dboot = np.load(EXP3_S16 / "shap_bootstrap.npz", allow_pickle=True)
    out = {
        "attr": {
            "centroid_x": d3[f"{split}_centroid_x_attr"],
            "circulation_pos": d3[f"{split}_circulation_pos_attr"],
            "peak_neg_omega": d3[f"{split}_peak_neg_omega_attr"],
            "Y": dY[f"{split}_Y_attr"],
        },
        "stable": {
            "centroid_x": dboot[f"{split}_centroid_x_stable"],
            "circulation_pos": dboot[f"{split}_circulation_pos_stable"],
            "peak_neg_omega": dboot[f"{split}_peak_neg_omega_stable"],
            "Y": dY[f"{split}_Y_bootstrap_stable"],
        },
        "case_id": dY[f"{split}_case_id"].astype(str),
        "encounter_index": dY[f"{split}_encounter_index"].astype(int),
        "G": dY[f"{split}_G"].astype(np.float32),
        "D": dY[f"{split}_D"].astype(np.float32),
        "Y": dY[f"{split}_Y"].astype(np.float32),
        "impact_frame": dY[f"{split}_impact_frame"].astype(int),
    }
    return out


def airfoil_mask() -> np.ndarray:
    """Load 192x96 airfoil-adjacent mask (True = inside body or 1-cell adjacent)."""
    if AIRFOIL_MASK_PATH.exists():
        return np.load(AIRFOIL_MASK_PATH).astype(bool)
    # Fallback: small disk around LE; not ideal.
    print(f"[exp4] WARN: airfoil mask not found at {AIRFOIL_MASK_PATH}")
    return np.zeros((192, 96), dtype=bool)


def load_omega_at_impact(case_id: str, k: int) -> tuple[np.ndarray, int]:
    """Load raw omega_z at the impact frame from cache."""
    path = CACHE_ROOT / PARTITION / case_id / f"encounter_{int(k):02d}.h5"
    with h5py.File(path, "r") as f:
        impact = int(f.attrs.get("impact_frame_estimate", DEFAULT_IMPACT_FRAME))
        omega = np.asarray(f["omega_z"][impact], dtype=np.float32)
    return omega, impact


def load_velocity_at_impact_path(case_id: str) -> Path:
    """Locate the raw HDF5 with velocity data /u for Q-criterion."""
    # Try several locations
    candidates = [
        PREVENT / "data" / "raw" / "periodic" / "run3" / f"*{case_id}*.h5",
        PREVENT / "data" / "raw" / "periodic" / f"*{case_id}*.h5",
    ]
    for pattern in candidates:
        matches = list(pattern.parent.glob(pattern.name))
        if matches:
            return matches[0]
    return None


def compute_Q_at_impact(case_id: str, impact: int) -> np.ndarray | None:
    """Compute Q-criterion field at mid-span at the impact frame.

    Q = 0.5*(||Omega||^2 - ||S||^2)
    where S is symmetric part of grad(u), Omega is antisymmetric.

    Velocity /u shape: (T, 192, 96, 32, 3). Mid-span = z-index 16.
    """
    # Look up the case file from inventory
    inv = json.load(
        open(REPO / "data_manifest" / "raw_cases_inventory.yaml")
    ) if (REPO / "data_manifest" / "raw_cases_inventory.yaml").suffix == ".json" else None
    if inv is None:
        # Use yaml
        import yaml
        with open(REPO / "data_manifest" / "raw_cases_inventory.yaml") as f:
            inv = yaml.safe_load(f)
    # Find matching case
    file_path = None
    for case in inv.get("cases", []):
        if case.get("case_id") == case_id:
            rel = case.get("relative_path")
            file_path = PREVENT / rel
            break
    if file_path is None or not file_path.exists():
        return None
    with h5py.File(file_path, "r") as f:
        if "u" not in f:
            return None
        # Read just one time slice to save memory
        u_full = f["u"][impact]  # shape (192, 96, 32, 3)
    u = u_full[..., 16, 0]  # mid-span u_x: shape (192, 96)
    v = u_full[..., 16, 1]
    w = u_full[..., 16, 2]
    # Compute 2D Q from in-plane gradients
    # du/dx, du/dy with central differences
    du_dx = np.zeros_like(u)
    du_dy = np.zeros_like(u)
    dv_dx = np.zeros_like(v)
    dv_dy = np.zeros_like(v)
    du_dx[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * DX)
    du_dy[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * DY)
    dv_dx[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2 * DX)
    dv_dy[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * DY)
    # In-plane S and Omega
    S11 = du_dx
    S22 = dv_dy
    S12 = 0.5 * (du_dy + dv_dx)
    Omega12 = 0.5 * (dv_dx - du_dy)
    S2 = S11**2 + S22**2 + 2.0 * S12**2
    Omega2 = 2.0 * Omega12**2
    Q = 0.5 * (Omega2 - S2)
    return Q


def extract_components(
    attr: np.ndarray, percentile: float, mask_exclude: np.ndarray,
    min_area: int = MIN_AREA_PX,
) -> list[dict]:
    """Return list of components for an attribution map."""
    abs_attr = np.abs(attr)
    thr = np.percentile(abs_attr, percentile)
    binary = (abs_attr > thr) & (~mask_exclude)
    labels, n = cc_label(binary)
    components = []
    for c in range(1, n + 1):
        mask = labels == c
        area = int(mask.sum())
        if area < min_area:
            continue
        ys, xs = np.where(mask)
        x_c = float(xs.mean())  # pixel coords (axis=1, W=96 -> Y dir)
        y_c = float(ys.mean())  # pixel coords (axis=0, H=192 -> X dir)
        # Map to physical
        # axis 0 (192) is x in [-1.5, 4.5], axis 1 (96) is y in [-1.5, 1.5]
        x_phys = (y_c / 192) * 6.0 - 1.5  # ys is along H/x axis
        y_phys = (x_c / 96) * 3.0 - 1.5
        mean_shap = float(abs_attr[mask].mean())
        sign_shap = float(attr[mask].sum() / max(area, 1))
        # Distance from LE in physical coords (LE at (0, 0))
        dist_le = float(np.sqrt(x_phys**2 + y_phys**2))
        components.append({
            "area": area,
            "x_pixel": y_c,
            "y_pixel": x_c,
            "x_phys": x_phys,
            "y_phys": y_phys,
            "dist_le": dist_le,
            "mean_abs_shap": mean_shap,
            "mean_signed_shap": sign_shap,
            "mask_indices": (np.where(mask)),
        })
    return components


def compute_circulation(omega: np.ndarray, mask: np.ndarray) -> float:
    return float((omega * mask).sum() * DX * DY)


def main() -> None:
    mask_excl = airfoil_mask()
    print(f"[exp4] airfoil mask covers {mask_excl.sum()}/{mask_excl.size} pixels "
          f"({100*mask_excl.sum()/mask_excl.size:.2f}%)")

    catalog_rows = []
    thresh_sens = {}
    q_rows = []
    Y_signs = {"positive": [], "negative": [], "zero": []}

    for split in SPLITS:
        d = load_shap_attr(split)
        for target in TARGETS:
            attrs = d["attr"][target]  # (n, 192, 96)
            stable = d["stable"][target]
            # (a) Connected components at 98th percentile for stable encounters.
            comps_at_98 = []
            for i in range(attrs.shape[0]):
                if not stable[i]:
                    continue
                comps = extract_components(attrs[i], 98.0, mask_excl)
                if not comps:
                    continue
                # Sort by mean_abs_shap descending; report top 3 per encounter.
                comps.sort(key=lambda c: c["mean_abs_shap"], reverse=True)
                for rank, c in enumerate(comps[:3]):
                    # Compute omega circulation inside this component on DNS.
                    try:
                        omega, impact = load_omega_at_impact(
                            d["case_id"][i], int(d["encounter_index"][i])
                        )
                        mask = np.zeros_like(omega, dtype=bool)
                        mask[c["mask_indices"]] = True
                        circ = compute_circulation(omega, mask)
                        omega_mean = float((omega * mask).sum() / max(c["area"], 1))
                    except Exception:
                        circ = float("nan")
                        omega_mean = float("nan")
                    catalog_rows.append({
                        "split": split,
                        "target": target,
                        "case_id": str(d["case_id"][i]),
                        "encounter_index": int(d["encounter_index"][i]),
                        "G": float(d["G"][i]),
                        "D": float(d["D"][i]),
                        "Y": float(d["Y"][i]),
                        "rank": rank,
                        "area_px": c["area"],
                        "x_phys": c["x_phys"],
                        "y_phys": c["y_phys"],
                        "dist_le": c["dist_le"],
                        "mean_abs_shap": c["mean_abs_shap"],
                        "mean_signed_shap": c["mean_signed_shap"],
                        "circulation": circ,
                        "omega_mean_in_component": omega_mean,
                    })
                    comps_at_98.append((i, rank, c))
                    # Y sign analysis
                    if target == "Y" and rank == 0:
                        Yv = float(d["Y"][i])
                        if Yv > 0.05:
                            Y_signs["positive"].append(c)
                        elif Yv < -0.05:
                            Y_signs["negative"].append(c)
                        else:
                            Y_signs["zero"].append(c)
            # (b) Threshold sensitivity: for each percentile, count stable
            # structures (centroid shift < 5 px and area change < 50%
            # vs the 98th-percentile baseline component).
            sens = {}
            for pct in PERCENTILES:
                stable_count = 0
                total_count = 0
                for i in range(attrs.shape[0]):
                    if not stable[i]:
                        continue
                    base_comps = extract_components(attrs[i], 98.0, mask_excl)
                    pct_comps = extract_components(attrs[i], pct, mask_excl)
                    if not base_comps or not pct_comps:
                        continue
                    base_top = sorted(
                        base_comps, key=lambda c: c["mean_abs_shap"], reverse=True
                    )[0]
                    pct_top = sorted(
                        pct_comps, key=lambda c: c["mean_abs_shap"], reverse=True
                    )[0]
                    cd = np.sqrt(
                        (base_top["x_pixel"] - pct_top["x_pixel"]) ** 2
                        + (base_top["y_pixel"] - pct_top["y_pixel"]) ** 2
                    )
                    area_change = abs(base_top["area"] - pct_top["area"]) / max(
                        base_top["area"], 1
                    )
                    if cd < 5 and area_change < 0.5:
                        stable_count += 1
                    total_count += 1
                sens[str(pct)] = {
                    "stable_count": stable_count,
                    "total_stable_enc": total_count,
                    "stable_frac": (
                        stable_count / total_count if total_count > 0 else None
                    ),
                }
            thresh_sens.setdefault(split, {})[target] = sens

    # (c) Q-criterion comparison for top SHAP structures per encounter
    print(f"\n[exp4] computing Q-criterion overlap (samples 5 per target)...")
    for split in SPLITS:
        d = load_shap_attr(split)
        for target in TARGETS:
            attrs = d["attr"][target]
            stable = d["stable"][target]
            stable_idx = np.where(stable)[0]
            sample = stable_idx[:5]  # first 5 stable encounters per (split, target)
            for i in sample:
                cid = str(d["case_id"][i])
                k = int(d["encounter_index"][i])
                impact = int(d["impact_frame"][i])
                Q = compute_Q_at_impact(cid, impact)
                if Q is None:
                    continue
                # Extract SHAP top component
                comps = extract_components(attrs[i], 98.0, mask_excl)
                if not comps:
                    continue
                top_shap = sorted(comps, key=lambda c: c["mean_abs_shap"], reverse=True)[0]
                shap_mask = np.zeros_like(Q, dtype=bool)
                shap_mask[top_shap["mask_indices"]] = True
                # Q > 0 connected components
                q_pos = Q > 0.0
                q_labels, q_n = cc_label(q_pos)
                # For each SHAP component, find nearest Q-structure (by centroid distance)
                if q_n == 0:
                    nearest = None
                    iou = 0.0
                    centroid_dist = float("nan")
                    overlap_ratio = 0.0
                else:
                    sx = top_shap["x_pixel"]
                    sy = top_shap["y_pixel"]
                    best_dist = float("inf")
                    best_cid = -1
                    for c in range(1, q_n + 1):
                        qmask = q_labels == c
                        if qmask.sum() < 10:
                            continue
                        qys, qxs = np.where(qmask)
                        qcd = np.sqrt(
                            (sy - qxs.mean()) ** 2 + (sx - qys.mean()) ** 2
                        )
                        if qcd < best_dist:
                            best_dist = qcd
                            best_cid = c
                    if best_cid > 0:
                        qmask_best = q_labels == best_cid
                        intersection = (shap_mask & qmask_best).sum()
                        union = (shap_mask | qmask_best).sum()
                        iou = float(intersection / max(union, 1))
                        overlap_ratio = float(intersection / max(shap_mask.sum(), 1))
                        centroid_dist = float(best_dist)
                    else:
                        iou = 0.0
                        overlap_ratio = 0.0
                        centroid_dist = float("inf")
                q_rows.append({
                    "split": split,
                    "target": target,
                    "case_id": cid,
                    "encounter_index": k,
                    "Y": float(d["Y"][i]),
                    "G": float(d["G"][i]),
                    "shap_area": top_shap["area"],
                    "shap_x_phys": top_shap["x_phys"],
                    "shap_y_phys": top_shap["y_phys"],
                    "iou_with_nearest_Q": iou,
                    "overlap_fraction": overlap_ratio,
                    "centroid_dist_px": centroid_dist,
                })

    # Save catalog (without mask_indices to keep CSV simple)
    catalog_rows_simple = []
    for r in catalog_rows:
        r2 = {k: v for k, v in r.items() if k != "mask_indices"}
        catalog_rows_simple.append(r2)
    with (EXP4 / "structure_catalog.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(catalog_rows_simple[0].keys()))
        w.writeheader()
        w.writerows(catalog_rows_simple)
    print(f"[exp4] wrote {EXP4 / 'structure_catalog.csv'}  ({len(catalog_rows)} rows)")

    (EXP4 / "threshold_sensitivity.json").write_text(json.dumps(thresh_sens, indent=2))
    print(f"[exp4] wrote {EXP4 / 'threshold_sensitivity.json'}")

    with (EXP4 / "q_overlap.csv").open("w", newline="") as f:
        if q_rows:
            w = csv.DictWriter(f, fieldnames=list(q_rows[0].keys()))
            w.writeheader()
            w.writerows(q_rows)
            print(f"[exp4] wrote {EXP4 / 'q_overlap.csv'}  ({len(q_rows)} rows)")

    # Y sign-flip analysis
    Y_summary = {}
    for sign_name, comps in Y_signs.items():
        if not comps:
            Y_summary[sign_name] = {"n": 0}
            continue
        xs = np.array([c["x_phys"] for c in comps])
        ys = np.array([c["y_phys"] for c in comps])
        Y_summary[sign_name] = {
            "n": len(comps),
            "mean_x": float(xs.mean()),
            "std_x": float(xs.std()),
            "mean_y": float(ys.mean()),
            "std_y": float(ys.std()),
        }
    # Bootstrap CIs on mean centroid per sign group
    rng = np.random.default_rng(0)
    for sign_name in ("positive", "negative"):
        if Y_summary.get(sign_name, {}).get("n", 0) < 3:
            continue
        comps = Y_signs[sign_name]
        xs = np.array([c["x_phys"] for c in comps])
        ys = np.array([c["y_phys"] for c in comps])
        bs_x, bs_y = [], []
        for _ in range(2000):
            idx = rng.integers(0, xs.size, xs.size)
            bs_x.append(xs[idx].mean())
            bs_y.append(ys[idx].mean())
        Y_summary[sign_name]["ci_x"] = [
            float(np.quantile(bs_x, 0.025)), float(np.quantile(bs_x, 0.975))
        ]
        Y_summary[sign_name]["ci_y"] = [
            float(np.quantile(bs_y, 0.025)), float(np.quantile(bs_y, 0.975))
        ]
    (EXP4 / "Y_sign_flip.json").write_text(json.dumps(Y_summary, indent=2))
    print(f"[exp4] wrote {EXP4 / 'Y_sign_flip.json'}")
    print("\n[exp4] Y sign-flip summary:")
    for sign_name, s in Y_summary.items():
        print(f"  {sign_name:>10s}: {s}")

    # Figure 1: 4-target panel (3 representative encounters per target).
    d_b = load_shap_attr("test_b")
    d_c = load_shap_attr("test_c")
    fig, axes = plt.subplots(4, 3, figsize=(14, 16))
    for row, target in enumerate(TARGETS):
        # 2 test_b + 1 test_c
        chosen = []
        for d_split, split_name in ((d_b, "test_b"), (d_c, "test_c")):
            stable_idx = np.where(d_split["stable"][target])[0]
            n_pick = 2 if split_name == "test_b" else 1
            for ii in stable_idx[:n_pick]:
                chosen.append((d_split, split_name, ii))
        for col, (d_split, split_name, i) in enumerate(chosen[:3]):
            ax = axes[row, col]
            attr = d_split["attr"][target][i]
            comps = extract_components(attr, 98.0, mask_excl)
            mean_attr = np.abs(attr)
            ax.imshow(
                mean_attr.T, origin="lower", extent=(-1.5, 4.5, -1.5, 1.5),
                cmap="viridis", vmax=np.percentile(mean_attr, 99.5),
            )
            ax.plot([0, 1], [0, 0], "w-", lw=2, alpha=0.6)
            # Draw component centroids
            for c in comps:
                ax.scatter([c["x_phys"]], [c["y_phys"]], s=c["area"]/2,
                           facecolor="none", edgecolor="red", lw=1.0)
            ax.set_xlim(-1.5, 4.5)
            ax.set_ylim(-1.5, 1.5)
            ax.set_aspect("equal")
            G = float(d_split["G"][i]); Yv = float(d_split["Y"][i])
            ax.set_title(
                f"{target}  ({split_name}, G={G:+.2f}, Y={Yv:+.2f})",
                fontsize=9,
            )
    fig.suptitle("SHAP attribution + 98th-percentile connected components")
    fig.tight_layout()
    fig.savefig(FIGS / "exp4_structures_4target_panel.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[exp4] wrote {FIGS / 'exp4_structures_4target_panel.png'}")

    # Figure 2: Q overlap summary - IoU and overlap fraction distributions per target
    if q_rows:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        target_data = {t: {"iou": [], "overlap": []} for t in TARGETS}
        for r in q_rows:
            target_data[r["target"]]["iou"].append(r["iou_with_nearest_Q"])
            target_data[r["target"]]["overlap"].append(r["overlap_fraction"])
        positions = np.arange(len(TARGETS))
        for j, (key, ylabel) in enumerate([("iou", "IoU"), ("overlap", "Overlap fraction")]):
            vals = [target_data[t][key] for t in TARGETS]
            axes[j].boxplot(vals, positions=positions, widths=0.6)
            axes[j].set_xticks(positions)
            axes[j].set_xticklabels(TARGETS, rotation=15, ha="right")
            axes[j].set_ylabel(ylabel)
            axes[j].grid(alpha=0.3)
            axes[j].set_title(f"{ylabel} of top SHAP structure with nearest Q>0 component")
        fig.tight_layout()
        fig.savefig(FIGS / "exp4_q_overlap_summary.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"[exp4] wrote {FIGS / 'exp4_q_overlap_summary.png'}")

    # Figure 3: Y sign-flip
    if Y_signs["positive"] and Y_signs["negative"]:
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        for sign_name, color in (("positive", "tab:red"), ("negative", "tab:blue"),
                                 ("zero", "gray")):
            comps = Y_signs[sign_name]
            if not comps:
                continue
            xs = [c["x_phys"] for c in comps]
            ys = [c["y_phys"] for c in comps]
            sizes = [c["area"] for c in comps]
            ax.scatter(xs, ys, s=sizes, color=color, alpha=0.5,
                       label=f"Y {sign_name} (n={len(comps)})")
            # Mark mean with cross
            mx, my = np.mean(xs), np.mean(ys)
            ax.plot(mx, my, "+", color=color, markersize=20, markeredgewidth=3)
        # Airfoil
        ax.plot([0, 1], [0, 0], "k-", lw=3, alpha=0.7)
        ax.set_xlim(-1.0, 2.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_xlabel("x/c")
        ax.set_ylabel("y/c")
        ax.set_title("Y SHAP top-structure centroid by sign(Y)")
        ax.set_aspect("equal")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGS / "exp4_Y_sign_flip.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"[exp4] wrote {FIGS / 'exp4_Y_sign_flip.png'}")


if __name__ == "__main__":
    main()
