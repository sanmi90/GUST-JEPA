"""Session 16, Experiment 1, Part (c): seed-variance bounds on the PLS-3 and
PCA-3 bases.

For each of the 3 Thrust-6 seed retrains plus the production encoder:
    1. Encode impact-frame latents for train, test_a, test_b, test_c.
    2. Fit PLS-3 on train; report R^2 per parameter on test_a, test_b, test_c.
    3. Fit PCA-3 on train; report PCA-3 score variance per axis on each split.
    4. Compute principal-angle subspace overlap with P_base (production PLS-3
       and PCA-3 subspaces).

Output:
    outputs/session16/exp1/exp1c_seed_variance.json

This script encodes only the impact-frame latents (one forward per encounter
with T=1), which keeps the runtime under ~5 min on a single RTX 6000.

Note: per session priority 2 (Honesty over headline), the variance summary
is reported INCLUDING the seeds even though the Part (a) gate failed at the
production seed. The point of (c) is to bound how much the failure varies
across seeds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.data.omega_pipeline import OmegaPipeline  # noqa: E402
from src.models.encoder import HybridCNNViTEncoder  # noqa: E402
from src.utils.device import require_rtx6000  # noqa: E402

OMEGA_MANIFEST = REPO / "outputs" / "data_pipeline" / "v1" / "manifest.json"
SPLIT_MANIFEST = REPO / "configs" / "splits" / "split_v1.json"
DEFAULT_IMPACT_FRAME = 40
PARTITION = "v1"
SPLITS = ("train", "test_a", "test_b", "test_c")
OUT = REPO / "outputs" / "session16" / "exp1"

SEED_ENCODERS = {
    "production": REPO / "outputs" / "runs" / "session12" / "S12_E_d64" / "encoder" / "checkpoint_iter020000.pt",
    "seed0": REPO / "outputs" / "runs" / "session14" / "thrust6" / "jepa_d64_seed0" / "encoder" / "checkpoint_iter020000.pt",
    "seed1": REPO / "outputs" / "runs" / "session14" / "thrust6" / "jepa_d64_seed1" / "encoder" / "checkpoint_iter020000.pt",
    "seed2": REPO / "outputs" / "runs" / "session14" / "thrust6" / "jepa_d64_seed2" / "encoder" / "checkpoint_iter020000.pt",
}


def resolve_cache_root() -> Path:
    env_cache = os.environ.get("VORTEX_JEPA_CACHE")
    if env_cache:
        return Path(env_cache) / PARTITION
    prevent_root = Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
    return prevent_root / "data" / "processed" / "vortex-jepa" / PARTITION


def gather_encounters(split: str, cache_root: Path) -> list[dict]:
    with open(SPLIT_MANIFEST) as f:
        manifest = json.load(f)
    out: list[dict] = []
    for cid, case in manifest["cases"].items():
        if split == "train" and case["split"] == "train":
            ks = list(case["train_encounter_indices"])
        elif split == "test_a" and case["split"] == "train":
            ks = list(case["test_a_encounter_indices"])
        elif split == "test_b" and case["split"] == "test_b":
            ks = list(range(int(case["n_encounters_full"])))
        elif split == "test_c" and case["split"] == "test_c":
            ks = list(range(int(case["n_encounters_full"])))
        else:
            continue
        for k in ks:
            path = cache_root / cid / f"encounter_{int(k):02d}.h5"
            out.append({
                "case_id": cid,
                "k": int(k),
                "G": float(case["G"]),
                "D": float(case["D"]),
                "Y": float(case["Y"]),
                "path": path,
            })
    return out


@torch.no_grad()
def encode_impact_only(
    encs: list[dict],
    encoder: HybridCNNViTEncoder,
    pipeline: OmegaPipeline,
    device: torch.device,
    latent_dim: int,
) -> dict:
    z_imp_rows: list[np.ndarray] = []
    g_col: list[float] = []
    d_col: list[float] = []
    y_col: list[float] = []
    case_col: list[str] = []
    k_col: list[int] = []
    for rec in encs:
        path: Path = rec["path"]
        if not path.exists():
            continue
        with h5py.File(path, "r") as f:
            omega_np = np.asarray(f["omega_z"], dtype=np.float32)
            impact_frame = int(f.attrs.get("impact_frame_estimate", DEFAULT_IMPACT_FRAME))
        omega_np = pipeline.preprocess_raw(omega_np, rec["case_id"], int(rec["k"]))
        omega_t = torch.from_numpy(omega_np)
        omega_t = pipeline.normalize(omega_t)
        omega_t = omega_t.unsqueeze(0).unsqueeze(2).to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z_t = encoder(omega_t)
        z_arr = z_t.squeeze(0).float().cpu().numpy()
        imp = min(impact_frame, z_arr.shape[0] - 1)
        z_imp_rows.append(z_arr[imp])
        g_col.append(rec["G"])
        d_col.append(rec["D"])
        y_col.append(rec["Y"])
        case_col.append(rec["case_id"])
        k_col.append(int(rec["k"]))
    return {
        "z": np.stack(z_imp_rows, axis=0).astype(np.float64),
        "G": np.asarray(g_col, dtype=np.float64),
        "D": np.asarray(d_col, dtype=np.float64),
        "Y": np.asarray(y_col, dtype=np.float64),
        "case_id": np.asarray(case_col, dtype=object),
        "encounter_index": np.asarray(k_col, dtype=np.int32),
    }


def load_encoder(ckpt_path: Path, device: torch.device) -> tuple[HybridCNNViTEncoder, int]:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = blob["args"]
    latent_dim = int(args["d"])
    proj_norm = args.get("projection_norm", "batchnorm")
    enc = HybridCNNViTEncoder(latent_dim=latent_dim, projection_norm=proj_norm)
    state = {
        k.removeprefix("encoder."): v
        for k, v in blob["jepa_state_dict"].items()
        if k.startswith("encoder.")
    }
    enc.load_state_dict(state, strict=False)
    enc.eval().to(device)
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc, latent_dim


def stack_targets(split_data: dict) -> np.ndarray:
    return np.stack([split_data["G"], split_data["D"], split_data["Y"]], axis=1)


def per_param_r2(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    names = ("G", "D", "Y")
    return {n: float(r2_score(y_true[:, i], y_pred[:, i])) for i, n in enumerate(names)}


def orthonormalize(M: np.ndarray) -> np.ndarray:
    """Orthonormalize rows of M via QR; M shape (K, d)."""
    Q, _ = np.linalg.qr(M.T)
    return Q.T  # (K, d), rows orthonormal


def subspace_overlap(U1: np.ndarray, U2: np.ndarray) -> dict:
    """Principal-angle-based overlap between two K-dim subspaces.

    U1, U2 are (K, d) with orthonormal rows (passed through orthonormalize).
    Returns mean cos^2 of principal angles and the per-angle list.
    """
    U1_orth = orthonormalize(U1)
    U2_orth = orthonormalize(U2)
    M = U1_orth @ U2_orth.T  # (K, K)
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    cos2 = (s ** 2).tolist()
    return {
        "cos_principal_angles": [float(x) for x in s],
        "cos2_principal_angles": [float(x) for x in cos2],
        "mean_cos2": float(np.mean(cos2)),
        "min_cos2": float(np.min(cos2)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = require_rtx6000(gpu_index=args.gpu)
    print(f"[exp1c] device={device} ({torch.cuda.get_device_name(device.index)})")

    pipeline = OmegaPipeline.from_manifest(OMEGA_MANIFEST)
    cache_root = resolve_cache_root()

    encs_by_split: dict[str, list[dict]] = {s: gather_encounters(s, cache_root) for s in SPLITS}
    for s, encs in encs_by_split.items():
        print(f"[exp1c] {s}: {len(encs)} encounters")

    # Encode each seed's impact-frame latents
    seed_latents: dict[str, dict[str, dict]] = {}
    for seed_name, ckpt_path in SEED_ENCODERS.items():
        if not ckpt_path.exists():
            print(f"[exp1c] WARNING: missing {ckpt_path}, skipping {seed_name}")
            continue
        t0 = time.time()
        enc, latent_dim = load_encoder(ckpt_path, device)
        print(f"[exp1c] {seed_name}: encoder loaded, d={latent_dim}")
        per_split: dict[str, dict] = {}
        for split in SPLITS:
            per_split[split] = encode_impact_only(
                encs_by_split[split], enc, pipeline, device, latent_dim
            )
        seed_latents[seed_name] = per_split
        del enc
        torch.cuda.empty_cache()
        print(f"[exp1c] {seed_name}: encoded in {time.time() - t0:.1f}s")

    # Build P_base = production PLS-3 and PCA-3 from production train z
    prod_z_train = seed_latents["production"]["train"]["z"]
    prod_Y_train = stack_targets(seed_latents["production"]["train"])
    pls_base = PLSRegression(n_components=3, scale=True)
    pls_base.fit(prod_z_train, prod_Y_train)
    pls_base_basis = pls_base.x_rotations_.T  # (3, 64)
    pca_base = PCA(n_components=3, svd_solver="full")
    pca_base.fit(prod_z_train)
    pca_base_basis = pca_base.components_  # (3, 64)

    out: dict = {
        "per_seed_pls3": {},
        "per_seed_pca3": {},
        "subspace_overlap_with_production": {},
        "r2_summary_across_seeds": {},
    }

    print("\n[exp1c] PLS-3 per-seed evaluation:")
    print(f"  {'seed':<12s} {'split':<8s} {'G':>8s} {'D':>8s} {'Y':>8s} {'mean':>8s}")
    for seed_name, per_split in seed_latents.items():
        z_train = per_split["train"]["z"]
        Y_train = stack_targets(per_split["train"])
        pls = PLSRegression(n_components=3, scale=True)
        pls.fit(z_train, Y_train)
        per_seed: dict = {"pls_basis": pls.x_rotations_.T.tolist()}
        for split in SPLITS:
            split_data = per_split[split]
            y_true = stack_targets(split_data)
            y_pred = pls.predict(split_data["z"])
            r2 = per_param_r2(y_true, y_pred)
            r2["mean"] = float(np.mean(list(r2.values())))
            per_seed[split] = r2
            print(
                f"  {seed_name:<12s} {split:<8s} "
                f"{r2['G']:>+8.3f} {r2['D']:>+8.3f} {r2['Y']:>+8.3f} {r2['mean']:>+8.3f}"
            )
        out["per_seed_pls3"][seed_name] = per_seed
        out["subspace_overlap_with_production"][seed_name] = {
            "PLS3_vs_prod_PLS3": subspace_overlap(np.array(per_seed["pls_basis"]), pls_base_basis),
        }

    print("\n[exp1c] PCA-3 per-seed: variance ratio captured by top-3 PCs:")
    for seed_name, per_split in seed_latents.items():
        z_train = per_split["train"]["z"]
        pca_seed = PCA(n_components=8, svd_solver="full")
        pca_seed.fit(z_train)
        cumvar = np.cumsum(pca_seed.explained_variance_ratio_)
        out["per_seed_pca3"][seed_name] = {
            "explained_variance_ratio": pca_seed.explained_variance_ratio_.tolist(),
            "cumvar_first_8": cumvar.tolist(),
            "pca3_basis": pca_seed.components_[:3].tolist(),
        }
        out["subspace_overlap_with_production"][seed_name]["PCA3_vs_prod_PCA3"] = (
            subspace_overlap(pca_seed.components_[:3], pca_base_basis)
        )
        print(
            f"  {seed_name:<12s} PC1={pca_seed.explained_variance_ratio_[0]:.3f} "
            f"PC1-3={cumvar[2]:.3f} PC1-8={cumvar[7]:.3f}"
        )

    # Cross-seed mean and 95% CI on test_b / test_c PLS-3 R^2 mean
    for split in SPLITS:
        r2_means_across_seeds = np.array(
            [out["per_seed_pls3"][seed][split]["mean"] for seed in seed_latents]
        )
        r2_g = np.array([out["per_seed_pls3"][seed][split]["G"] for seed in seed_latents])
        r2_d = np.array([out["per_seed_pls3"][seed][split]["D"] for seed in seed_latents])
        r2_y = np.array([out["per_seed_pls3"][seed][split]["Y"] for seed in seed_latents])
        out["r2_summary_across_seeds"][split] = {
            "n_seeds": int(r2_means_across_seeds.shape[0]),
            "r2_mean_across_seeds": float(r2_means_across_seeds.mean()),
            "r2_mean_std_across_seeds": float(r2_means_across_seeds.std(ddof=1)),
            "r2_G_mean": float(r2_g.mean()),
            "r2_G_std": float(r2_g.std(ddof=1)),
            "r2_D_mean": float(r2_d.mean()),
            "r2_D_std": float(r2_d.std(ddof=1)),
            "r2_Y_mean": float(r2_y.mean()),
            "r2_Y_std": float(r2_y.std(ddof=1)),
        }

    print("\n[exp1c] Subspace overlap (mean cos^2 of 3 principal angles vs production):")
    print(f"  {'seed':<12s}  {'PLS3':>8s}  {'PCA3':>8s}")
    for seed_name in seed_latents:
        ov = out["subspace_overlap_with_production"][seed_name]
        print(
            f"  {seed_name:<12s}  {ov['PLS3_vs_prod_PLS3']['mean_cos2']:>8.3f}  "
            f"{ov['PCA3_vs_prod_PCA3']['mean_cos2']:>8.3f}"
        )

    save = OUT / "exp1c_seed_variance.json"
    save.write_text(json.dumps(out, indent=2))
    print(f"\n[exp1c] wrote {save.relative_to(REPO)}")


if __name__ == "__main__":
    main()
