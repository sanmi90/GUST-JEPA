"""Session 16, Experiment 1, Part (a) -- alternative PCA-3 basis.

Per pivot_decision.json: the PLS-3 acceptance gate failed because the encoder
organises its latent variance hierarchically by physical impact magnitude, not
by (G, D, Y) parameter slot. The PCA-3 basis is the encoder's natural variance
basis and is carried forward as a candidate physical-axis basis alongside
the recipe-locked PLS-3 artefact.

Saves:
    outputs/session16/exp1/pca_base.npz
        components (3, 64): top-3 PCA components in the train impact-frame z
        components_full (16, 64): top-16 for completeness
        mean (64,): train z mean (centering vector)
        explained_variance, explained_variance_ratio
        z_scores_{split} (n, 3): per-split projections onto the top-3 PCs
        case_id_{split}, encounter_index_{split}
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


REPO = Path(__file__).resolve().parents[2]
LATENTS = REPO / "outputs" / "session14" / "latents" / "S12_E_d64"
OUT = REPO / "outputs" / "session16" / "exp1"
OUT.mkdir(parents=True, exist_ok=True)


def load_split(name: str) -> dict:
    d = np.load(LATENTS / f"{name}.npz", allow_pickle=True)
    return {
        "z": d["z"].astype(np.float64),
        "case_id": d["case_id"],
        "encounter_index": d["encounter_index"].astype(np.int32),
    }


def main() -> None:
    train = load_split("train")
    pca_full = PCA(n_components=16, svd_solver="full")
    pca_full.fit(train["z"])

    artefacts = {
        "components": pca_full.components_[:3],
        "components_full": pca_full.components_,
        "mean": pca_full.mean_,
        "explained_variance": pca_full.explained_variance_,
        "explained_variance_ratio": pca_full.explained_variance_ratio_,
    }
    for sp in ("train", "test_a", "test_b", "test_c"):
        s = load_split(sp)
        z_centered = s["z"] - pca_full.mean_
        scores = z_centered @ pca_full.components_[:3].T
        artefacts[f"z_scores_{sp}"] = scores
        artefacts[f"case_id_{sp}"] = s["case_id"]
        artefacts[f"encounter_index_{sp}"] = s["encounter_index"]

    save = OUT / "pca_base.npz"
    np.savez(save, **artefacts)
    print(f"[pca_base] wrote {save.relative_to(REPO)}")
    print(
        f"[pca_base] PC1-3 cumulative variance ratio: "
        f"{np.sum(pca_full.explained_variance_ratio_[:3]):.3f}"
    )
    print(
        f"[pca_base] PC1-8 cumulative variance ratio: "
        f"{np.sum(pca_full.explained_variance_ratio_[:8]):.3f}"
    )


if __name__ == "__main__":
    main()
