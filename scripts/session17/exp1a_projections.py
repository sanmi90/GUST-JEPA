"""Session 17, Experiment 1, Part (a): build three candidate 3-D projections of
the per-frame latent trajectories from the production E d=64 encoder.

Three projections (SESSION17_PLAN.md):
    P1. PCA on impact-frame latents only (180 train encounters, 64-D z).
    P2. PCA on pooled per-frame latents (180 * 120 train frames).
    P3. Supervised PLS on per-frame latents against
        (G, D, Y, sin(2*pi*phi), cos(2*pi*phi)) where
        phi = (t - t_impact) / 40 is the impact-relative phase.

Test B uses split_v1p5 (28 v1 + 28 supplement = 56 encounters).
Per-frame trajectories live in z_full (n_enc, 120, 64) from
outputs/session14/latents/S12_E_d64/.

Reports the variance explained by the first 3 components of each projection
(or first 3 latent components, in the PLS case). Saves projection matrices
and per-frame 3-D scores so Parts (b) and (c) can reuse them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA


REPO = Path(__file__).resolve().parents[2]
LATENTS = REPO / "outputs" / "session14" / "latents" / "S12_E_d64"
OUT = REPO / "outputs" / "session17" / "exp1"
OUT.mkdir(parents=True, exist_ok=True)

T_ENC = 120
T_IMPACT = 40
PHASE_SCALE = 40.0  # half-encounter; phi in roughly [-1, +1.975]


def load_split(name: str, supplement: str | None = None) -> dict:
    d = np.load(LATENTS / f"{name}.npz", allow_pickle=True)
    out = {
        "z": d["z"].astype(np.float32),
        "z_full": d["z_full"].astype(np.float32),
        "G": d["G"].astype(np.float32),
        "D": d["D"].astype(np.float32),
        "Y": d["Y"].astype(np.float32),
        "case_id": np.asarray(d["case_id"]).astype(object),
        "encounter_index": d["encounter_index"].astype(np.int32),
        "impact_frame": d["impact_frame"].astype(np.int32),
    }
    if supplement is not None:
        ds = np.load(LATENTS / f"{supplement}.npz", allow_pickle=True)
        out["z"] = np.concatenate([out["z"], ds["z"].astype(np.float32)], axis=0)
        out["z_full"] = np.concatenate(
            [out["z_full"], ds["z_full"].astype(np.float32)], axis=0
        )
        for k in ("G", "D", "Y"):
            out[k] = np.concatenate([out[k], ds[k].astype(np.float32)], axis=0)
        out["case_id"] = np.concatenate(
            [out["case_id"], np.asarray(ds["case_id"]).astype(object)], axis=0
        )
        out["encounter_index"] = np.concatenate(
            [out["encounter_index"], ds["encounter_index"].astype(np.int32)], axis=0
        )
        out["impact_frame"] = np.concatenate(
            [out["impact_frame"], ds["impact_frame"].astype(np.int32)], axis=0
        )
    return out


def phase_features(t_imp: np.ndarray) -> np.ndarray:
    """Per-frame phase feature: shape (n_enc, T_ENC, 2) -> sin, cos of 2*pi*phi."""
    t = np.arange(T_ENC, dtype=np.float32)
    phi = (t[None, :] - t_imp[:, None].astype(np.float32)) / PHASE_SCALE
    return np.stack(
        [np.sin(2.0 * np.pi * phi), np.cos(2.0 * np.pi * phi)], axis=-1
    )


def main() -> None:
    splits = {
        "train": load_split("train"),
        "test_a": load_split("test_a"),
        "test_b": load_split("test_b", supplement="test_b_v1p5_supplement"),
        "test_c": load_split("test_c"),
    }
    for n, s in splits.items():
        print(
            f"[exp1a] {n:8s} z={s['z'].shape} z_full={s['z_full'].shape} "
            f"impact_frame=[{s['impact_frame'].min()}, {s['impact_frame'].max()}]"
        )

    # P1: PCA on impact-frame latents.
    Z_imp = splits["train"]["z"]
    pca_imp = PCA(n_components=3).fit(Z_imp)
    print(
        f"[exp1a] P1 PCA(impact) var_ratio_3={pca_imp.explained_variance_ratio_} "
        f"cum={pca_imp.explained_variance_ratio_.sum():.4f}"
    )
    # full spectrum for context
    pca_imp_full = PCA(n_components=min(Z_imp.shape)).fit(Z_imp)

    # P2: PCA on pooled per-frame latents.
    Z_pool = splits["train"]["z_full"].reshape(-1, splits["train"]["z_full"].shape[-1])
    pca_pool = PCA(n_components=3).fit(Z_pool)
    print(
        f"[exp1a] P2 PCA(pool)   var_ratio_3={pca_pool.explained_variance_ratio_} "
        f"cum={pca_pool.explained_variance_ratio_.sum():.4f}"
    )
    pca_pool_full = PCA(n_components=64).fit(Z_pool)

    # P3: supervised PLS-3 on per-frame z vs (G, D, Y, sin(2pi phi), cos(2pi phi)).
    G = splits["train"]["G"]
    D = splits["train"]["D"]
    Yp = splits["train"]["Y"]
    t_imp = splits["train"]["impact_frame"]
    phase_feat = phase_features(t_imp)  # (n, T, 2)
    targets_frame = np.zeros((G.size, T_ENC, 5), dtype=np.float32)
    targets_frame[..., 0] = G[:, None]
    targets_frame[..., 1] = D[:, None]
    targets_frame[..., 2] = Yp[:, None]
    targets_frame[..., 3:5] = phase_feat
    Y_pls = targets_frame.reshape(-1, 5)
    print(f"[exp1a] P3 PLS data: X={Z_pool.shape} Y={Y_pls.shape}")
    pls = PLSRegression(n_components=3, scale=True)
    pls.fit(Z_pool, Y_pls)

    # PLS variance explained: ratio of variance captured by X-scores wrt total X variance.
    Xc = (Z_pool - Z_pool.mean(axis=0)) / Z_pool.std(axis=0, ddof=0).clip(min=1e-12)
    total_var = (Xc**2).sum() / Xc.shape[0]
    Xs = pls.x_scores_
    # Approximate variance explained by reconstructing X via X_pls = X_scores @ x_loadings.T
    X_approx = Xs @ pls.x_loadings_.T
    var_per_comp = []
    for k in range(3):
        Xk = Xs[:, : k + 1] @ pls.x_loadings_[:, : k + 1].T
        var_per_comp.append(float(1 - ((Xc - Xk) ** 2).sum() / (Xc**2).sum()))
    var_ratio_3 = np.array(
        [var_per_comp[0]] + [var_per_comp[k] - var_per_comp[k - 1] for k in range(1, 3)]
    )
    print(
        f"[exp1a] P3 PLS    var_ratio_3 (X)={var_ratio_3} "
        f"cum={var_per_comp[-1]:.4f}"
    )

    # Project per-frame latents through each basis for every split. Save coords.
    proj_per_frame: dict[str, dict[str, np.ndarray]] = {}
    for name, split in splits.items():
        zf = split["z_full"]
        n_enc, T, d = zf.shape
        zf_flat = zf.reshape(-1, d)
        proj_per_frame[name] = {
            "pca_impact": pca_imp.transform(zf_flat).reshape(n_enc, T, 3),
            "pca_pool": pca_pool.transform(zf_flat).reshape(n_enc, T, 3),
            "pls_supervised": pls.transform(zf_flat).reshape(n_enc, T, 3),
        }

    # Save everything.
    save = {
        "pca_impact_mean": pca_imp.mean_,
        "pca_impact_components": pca_imp.components_,  # (3, 64)
        "pca_impact_var_ratio_full": pca_imp_full.explained_variance_ratio_,
        "pca_pool_mean": pca_pool.mean_,
        "pca_pool_components": pca_pool.components_,
        "pca_pool_var_ratio_full": pca_pool_full.explained_variance_ratio_,
        "pls_x_mean": pls._x_mean,
        "pls_x_std": pls._x_std,
        "pls_y_mean": pls._y_mean,
        "pls_y_std": pls._y_std,
        "pls_x_rotations": pls.x_rotations_,  # (64, 3)
        "pls_x_loadings": pls.x_loadings_,
        "pls_y_loadings": pls.y_loadings_,
        "pls_var_ratio_X_3": var_ratio_3,
        "T_ENC": T_ENC,
        "T_IMPACT": T_IMPACT,
        "PHASE_SCALE": PHASE_SCALE,
    }
    for name in splits:
        save[f"scores_pca_impact_{name}"] = proj_per_frame[name]["pca_impact"]
        save[f"scores_pca_pool_{name}"] = proj_per_frame[name]["pca_pool"]
        save[f"scores_pls_{name}"] = proj_per_frame[name]["pls_supervised"]
        save[f"G_{name}"] = splits[name]["G"]
        save[f"D_{name}"] = splits[name]["D"]
        save[f"Y_{name}"] = splits[name]["Y"]
        save[f"impact_frame_{name}"] = splits[name]["impact_frame"]
        save[f"case_id_{name}"] = splits[name]["case_id"].astype(str)
        save[f"encounter_index_{name}"] = splits[name]["encounter_index"]

    np.savez_compressed(OUT / "projections.npz", **save)
    print(f"[exp1a] wrote {OUT / 'projections.npz'}")

    summary = {
        "encoder_ckpt": "outputs/runs/session12/S12_E_d64/encoder/checkpoint_iter020000.pt",
        "split_summary": {
            n: {
                "n_enc": int(s["z"].shape[0]),
                "n_frames_total": int(s["z_full"].shape[0] * s["z_full"].shape[1]),
            }
            for n, s in splits.items()
        },
        "projection_variance_explained_3comp": {
            "pca_impact": [
                float(v) for v in pca_imp.explained_variance_ratio_
            ],
            "pca_impact_cum_3": float(pca_imp.explained_variance_ratio_.sum()),
            "pca_pool": [
                float(v) for v in pca_pool.explained_variance_ratio_
            ],
            "pca_pool_cum_3": float(pca_pool.explained_variance_ratio_.sum()),
            "pls_X_var_ratio": [float(v) for v in var_ratio_3],
            "pls_X_cum_3": float(var_per_comp[-1]),
        },
        "phase": {
            "definition": "phi = (t - t_impact) / 40, feat=(sin(2pi phi), cos(2pi phi))",
            "t_impact": T_IMPACT,
            "T_enc": T_ENC,
        },
    }
    (OUT / "projection_variance.json").write_text(json.dumps(summary, indent=2))
    print(f"[exp1a] wrote {OUT / 'projection_variance.json'}")


if __name__ == "__main__":
    main()
