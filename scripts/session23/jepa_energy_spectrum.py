"""Can JEPA modes be sorted by energy like POD? A diagnostic.

POD modes come with an intrinsic energy ordering (singular values). JEPA latent
axes do not, so we impose one two ways and compare to POD:
  (1) latent-PCA variance: PCA the JEPA latents; fraction of latent variance per PC.
  (2) decoded enstrophy: perturb the mean latent along each PC by its std, decode
      with the visualisation decoder, and measure the enstrophy of the field
      perturbation -> a physical energy per JEPA mode (the closest analogue to POD).
POD energy spectrum is S_i^2 / sum(S_i^2) from the saved basis.
Runs the decoder on the RTX 6000.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/carlos/GUST-JEPA")
for p in ("scripts/session21", "", "scripts", "scripts/session20"):
    sys.path.insert(0, str(REPO / p))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import figstyle as fs  # noqa: E402
import figG_flow_recovery as fg  # noqa: E402

LAT = REPO / "outputs/session18/exp_b1/latents_jepa_d64_test1_noBN"
PODF = REPO / "outputs/session18/exp_b1/pod_d64/pod_basis.npz"
OUT = REPO / "outputs/session23/jepa_energy_spectrum.png"


def main() -> None:
    fs.use_style()
    tr = np.load(LAT / "train.npz", allow_pickle=True)
    Zf = tr["z_full"].astype(np.float64).reshape(-1, 64)   # (n*120, d) all snapshots
    mu = Zf.mean(0)
    Zc = Zf - mu
    # PCA via SVD of the snapshot matrix
    _, S, Vt = np.linalg.svd(Zc, full_matrices=False)       # Vt: (d, d) PCs
    lat_var = S ** 2
    lat_frac = lat_var / lat_var.sum()
    sig = S / np.sqrt(len(Zc) - 1)                          # std along each PC

    # decode each PC direction, measure enstrophy of the field perturbation
    device = fg.require_rtx6000(gpu_index=0)
    decode = fg.load_decoder(device)
    base = decode(mu[None].astype(np.float32))[0]
    enstr = np.zeros(64)
    for i in range(64):
        dpert = decode((mu + sig[i] * Vt[i])[None].astype(np.float32))[0] - base
        enstr[i] = float(np.sum(dpert ** 2))
    enstr_frac = np.sort(enstr / enstr.sum())[::-1]         # sorted descending

    # POD energy spectrum
    pod = np.load(PODF, allow_pickle=True)
    Spod = pod["S"].astype(np.float64)
    pod_frac = (Spod ** 2) / (Spod ** 2).sum()

    k = np.arange(1, 65)
    fig, ax = plt.subplots(figsize=fs.figure_size(0.85, 0.62))
    ax.semilogy(k, np.sort(lat_frac)[::-1], "o-", ms=2.5, lw=1.0,
                color=fs.FAMILY_COLOR["jepa"], label="JEPA latent-PCA variance")
    ax.semilogy(k, enstr_frac, "s--", ms=2.5, lw=1.0, color="#7b3294",
                label="JEPA decoded enstrophy")
    ax.semilogy(k, pod_frac, "^-", ms=2.5, lw=1.0, color=fs.FAMILY_COLOR["pod"],
                label="POD energy")
    ax.set_xlabel("mode index (sorted)")
    ax.set_ylabel("fraction (log scale)")
    ax.set_ylim(1e-5, 1)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)

    def cum_to(frac, t):
        c = np.cumsum(np.sort(frac)[::-1])
        return int(np.searchsorted(c, t) + 1)
    print(f"PC1 latent-variance fraction: {np.sort(lat_frac)[::-1][0]:.3f}")
    print(f"modes for 90% latent variance: {cum_to(lat_frac, 0.9)} / 64")
    print(f"modes for 90% decoded enstrophy: {cum_to(enstr/enstr.sum(), 0.9)} / 64")
    print(f"modes for 90% POD energy: {cum_to(pod_frac, 0.9)} / 64")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
