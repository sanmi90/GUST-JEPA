"""Session 18 / B1 Test 3 sanity check: does the d=32 JEPA latent retain
the wake-aware structure encoded by the patch_signed_spectrum auxiliary
target as well as the d=64 production encoder?

The production d=64 JEPA was trained with
``--wake-observable-type patch_signed_spectrum --lambda-wake 1.00`` (an
80-dim auxiliary wake target supervises the encoder). The d=32 add-on
used the IDENTICAL recipe. Matched-capacity question: does halving the
latent dimension cost us any patch_signed_spectrum band, and if so,
which band(s)?

Method
------
1. Load impact-frame z from
   ``outputs/session14/latents/S12_E_d{32,64}/{train,test_b}.npz``.
2. Load wake_observables ``patch_signed_spectrum`` (80 dim) per encounter
   from ``${VORTEX_JEPA_CACHE}/v1/wake_observables/{case_id}/encounter_{k:02d}.h5``,
   sliced at the encounter's impact frame.
3. Standardize the wake observable using ``_train_stats.json`` (per-dim
   mean/std).
4. Fit ``KernelRidge(alpha=0.1, kernel='rbf', gamma=0.05)`` per-dim
   on the train split (impact frames), evaluate on test_b.
5. Report mean R^2 across the 80 dims for d=32 and d=64, plus the per-dim
   breakdown so we can see which spectrum bins suffered at d=32 (if any).

The 80 dims are (LeWM convention from
``src.data.wake_observables.patch_signed_spectrum_target``):

    dims [0:32]   patch energies, positive omega (8x4 grid)
    dims [32:64]  patch energies, negative omega (8x4 grid)
    dims [64:80]  16-bin radial spectrum (low-k -> high-k)

Run::

    source .venv/bin/activate
    export PREVENT_ROOT=$HOME/PREVENT
    python scripts/_oneoff_wake_observable_d32_check.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score


REPO = Path(__file__).resolve().parents[1]
PREVENT = Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
CACHE = Path(
    os.environ.get("VORTEX_JEPA_CACHE", str(PREVENT / "data" / "processed" / "vortex-jepa"))
)
WAKE_ROOT = CACHE / "v1" / "wake_observables"
STATS_PATH = WAKE_ROOT / "_train_stats.json"

LATENT_BASE = REPO / "outputs" / "session14" / "latents"
LATENT_DIRS = {64: LATENT_BASE / "S12_E_d64", 32: LATENT_BASE / "S12_E_d32"}

OUT_DIR = REPO / "outputs" / "session18" / "exp_b1_test3"
CSV_PATH = OUT_DIR / "wake_observable_d32_vs_d64.csv"
PNG_PATH = OUT_DIR / "wake_observable_d32_vs_d64.png"

MODE = "patch_signed_spectrum"
KRR_ALPHA = 0.1
KRR_GAMMA = 0.05


def load_split_latents(d: int, split: str) -> dict:
    """Return dict with impact-frame z and per-encounter metadata."""
    path = LATENT_DIRS[d] / f"{split}.npz"
    f = np.load(path, allow_pickle=True)
    out = {
        "z": np.asarray(f["z"], dtype=np.float64),  # (N, d) at impact frame
        "case_id": [str(c) for c in f["case_id"]],
        "encounter_index": np.asarray(f["encounter_index"], dtype=int),
        "impact_frame": np.asarray(f["impact_frame"], dtype=int),
        "G": np.asarray(f["G"], dtype=np.float32),
        "D": np.asarray(f["D"], dtype=np.float32),
        "Y": np.asarray(f["Y"], dtype=np.float32),
    }
    return out


def load_wake_targets_at_impact(
    case_ids: list[str],
    enc_idx: np.ndarray,
    imp_frame: np.ndarray,
) -> np.ndarray:
    """Read patch_signed_spectrum[impact_frame] for each (case, encounter).

    Returns (N, 80) float64 array of UN-standardized targets.
    """
    n = len(case_ids)
    out = np.empty((n, 80), dtype=np.float64)
    for i in range(n):
        path = WAKE_ROOT / case_ids[i] / f"encounter_{int(enc_idx[i]):02d}.h5"
        if not path.exists():
            raise FileNotFoundError(f"missing wake obs file: {path}")
        with h5py.File(path, "r") as g:
            arr = g[MODE][...]  # (120, 80)
        out[i] = arr[int(imp_frame[i])].astype(np.float64)
    return out


def standardize(y: np.ndarray, stats: dict) -> np.ndarray:
    mean = np.asarray(stats["mean"], dtype=np.float64)
    std = np.asarray(stats["std"], dtype=np.float64)
    eps = float(stats.get("eps", 1e-6))
    return (y - mean) / (std + eps)


def fit_and_score_krr(
    z_train: np.ndarray, y_train: np.ndarray,
    z_test: np.ndarray, y_test: np.ndarray,
) -> tuple[float, np.ndarray]:
    """KRR-RBF (alpha=0.1, gamma=0.05) per-dim z -> y.

    Returns (mean_r2, per_dim_r2 of shape (80,)).

    We fit a single multi-output KernelRidge model (predicts all 80 dims at
    once -- the kernel matrix is shared across outputs, only the dual coefs
    differ per dim). Then compute R^2 per-output on the test split.
    """
    model = KernelRidge(alpha=KRR_ALPHA, kernel="rbf", gamma=KRR_GAMMA)
    model.fit(z_train, y_train)
    y_pred = model.predict(z_test)
    per_dim = np.empty(y_test.shape[1], dtype=np.float64)
    for j in range(y_test.shape[1]):
        per_dim[j] = r2_score(y_test[:, j], y_pred[:, j])
    return float(per_dim.mean()), per_dim


def main() -> None:
    if not STATS_PATH.exists():
        print(f"ERROR: stats not found at {STATS_PATH}", file=sys.stderr)
        sys.exit(1)
    with open(STATS_PATH) as f:
        all_stats = json.load(f)
    if MODE not in all_stats:
        print(f"ERROR: {MODE} not in stats keys {list(all_stats)}", file=sys.stderr)
        sys.exit(1)
    stats = all_stats[MODE]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load impact-frame z + matched wake targets for train and test_b for both d
    results = {}
    for d in (64, 32):
        print(f"[d={d}] loading latents...")
        train_l = load_split_latents(d, "train")
        test_l = load_split_latents(d, "test_b")
        print(f"[d={d}] train z={train_l['z'].shape}  test_b z={test_l['z'].shape}")

        print(f"[d={d}] loading wake targets (impact frames) ...")
        y_train_raw = load_wake_targets_at_impact(
            train_l["case_id"], train_l["encounter_index"], train_l["impact_frame"]
        )
        y_test_raw = load_wake_targets_at_impact(
            test_l["case_id"], test_l["encounter_index"], test_l["impact_frame"]
        )
        # Standardize using TRAIN stats so dimensions are comparable
        y_train = standardize(y_train_raw, stats)
        y_test = standardize(y_test_raw, stats)
        print(f"[d={d}] y train={y_train.shape}  test_b={y_test.shape}")

        print(f"[d={d}] fitting KRR-RBF (alpha={KRR_ALPHA}, gamma={KRR_GAMMA}) ...")
        mean_r2, per_dim = fit_and_score_krr(
            train_l["z"], y_train, test_l["z"], y_test
        )
        results[d] = {
            "mean_r2": mean_r2,
            "per_dim": per_dim,
        }
        # Sub-block summaries
        pos_block = per_dim[0:32]
        neg_block = per_dim[32:64]
        spec_block = per_dim[64:80]
        print(
            f"[d={d}] mean_r2={mean_r2:.4f}  "
            f"pos_patch={pos_block.mean():.4f}  "
            f"neg_patch={neg_block.mean():.4f}  "
            f"radial_spec={spec_block.mean():.4f}"
        )

    # Save CSV: per-dim breakdown + sub-block summaries
    print(f"\nWriting CSV -> {CSV_PATH}")
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "dim_index", "block",
                "r2_d64", "r2_d32",
                "delta_d32_minus_d64",
            ]
        )
        per_d64 = results[64]["per_dim"]
        per_d32 = results[32]["per_dim"]
        for j in range(80):
            if j < 32:
                block = "patch_pos"
            elif j < 64:
                block = "patch_neg"
            else:
                block = f"radial_spec_bin_{j - 64:02d}"
            w.writerow(
                [
                    j, block,
                    f"{per_d64[j]:.6f}",
                    f"{per_d32[j]:.6f}",
                    f"{per_d32[j] - per_d64[j]:+.6f}",
                ]
            )
        # Block-mean summary rows
        for label, idx in [
            ("MEAN_ALL_80", slice(None)),
            ("MEAN_patch_pos_32", slice(0, 32)),
            ("MEAN_patch_neg_32", slice(32, 64)),
            ("MEAN_radial_spec_16", slice(64, 80)),
        ]:
            r64 = per_d64[idx].mean()
            r32 = per_d32[idx].mean()
            w.writerow([label, "", f"{r64:.6f}", f"{r32:.6f}", f"{r32 - r64:+.6f}"])

    # Save comparison plot: 2 panels
    print(f"Writing PNG -> {PNG_PATH}")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    # Panel 1: per-dim R^2 d=64 vs d=32 grouped by block
    ax = axes[0]
    dims = np.arange(80)
    ax.plot(dims, results[64]["per_dim"], "o-", color="#238b45",
            label=f"d=64 (mean={results[64]['mean_r2']:.3f})", ms=3.5, lw=1.0)
    ax.plot(dims, results[32]["per_dim"], "s-", color="#cb181d",
            label=f"d=32 (mean={results[32]['mean_r2']:.3f})", ms=3.5, lw=1.0)
    ax.axvline(31.5, color="grey", ls="--", lw=0.5)
    ax.axvline(63.5, color="grey", ls="--", lw=0.5)
    ax.text(15.5, ax.get_ylim()[0] + 0.02, "patch +",
            ha="center", va="bottom", color="grey", fontsize=8)
    ax.text(47.5, ax.get_ylim()[0] + 0.02, "patch -",
            ha="center", va="bottom", color="grey", fontsize=8)
    ax.text(71.5, ax.get_ylim()[0] + 0.02, "radial spec",
            ha="center", va="bottom", color="grey", fontsize=8)
    ax.set_xlabel("wake observable dim (0-79)")
    ax.set_ylabel(r"per-dim KRR-RBF $R^2$ (test_b)")
    ax.set_title("Per-dim R^2: d=32 vs d=64 (impact frame, patch_signed_spectrum)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", lw=0.5)

    # Panel 2: delta per dim (d32 - d64) -- negative means d=32 worse
    ax = axes[1]
    delta = results[32]["per_dim"] - results[64]["per_dim"]
    colors = np.where(delta < 0, "#cb181d", "#238b45")
    ax.bar(dims, delta, color=colors, width=0.9)
    ax.axvline(31.5, color="grey", ls="--", lw=0.5)
    ax.axvline(63.5, color="grey", ls="--", lw=0.5)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("wake observable dim (0-79)")
    ax.set_ylabel(r"$R^2_{d=32} - R^2_{d=64}$")
    ax.set_title(f"Per-dim delta (mean delta = {delta.mean():+.4f})")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Wake-observable retention: d=32 vs d=64 JEPA latents "
        f"({MODE}, KRR-RBF alpha={KRR_ALPHA}, gamma={KRR_GAMMA})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=150)
    plt.close(fig)

    # Print summary
    print("\n=== SUMMARY ===")
    print(f"Mean R^2 across 80 wake obs dims (test_b):")
    print(f"  d=64: {results[64]['mean_r2']:.4f}")
    print(f"  d=32: {results[32]['mean_r2']:.4f}")
    print(f"  delta: {results[32]['mean_r2'] - results[64]['mean_r2']:+.4f}")
    for label, idx in [
        ("patch_pos (dims 0-31)", slice(0, 32)),
        ("patch_neg (dims 32-63)", slice(32, 64)),
        ("radial_spec (dims 64-79)", slice(64, 80)),
    ]:
        r64 = results[64]["per_dim"][idx].mean()
        r32 = results[32]["per_dim"][idx].mean()
        print(f"  {label}: d64={r64:.4f}  d32={r32:.4f}  delta={r32 - r64:+.4f}")


if __name__ == "__main__":
    main()
