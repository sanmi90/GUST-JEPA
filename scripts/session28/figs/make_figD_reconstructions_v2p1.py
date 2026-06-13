"""Reconstruction figure (figD_reconstructions_v2p1.pdf): decoded vorticity at the
impact frame on held-out encounters from each split, for the three families vs DNS.
v2.1.

Matches the v2.1 caption (fig:recon, section_4_results.tex): decoded vorticity at
the impact frame on held-out encounters from the splits (rows) for the simulation
and the predictive, reconstructive, and POD decodes (columns). The predictive
decoder is blurry by design but localises the impingement; the reconstructive
autoencoder collapses toward a near-uniform field on the held-out interpolation
condition; the linear basis is amplitude-accurate. All families degrade in the
|G| = 4 regime.

DATA NOTE (reported, not hidden): the v2.1 decode campaign
(decode_operating_points.py) decoded only test_b and test_c (decode/README.md:
"Two endpoints, on test_b and test_c"); the no-gust/trained-gust test_a row the
caption lists was NOT decoded this campaign (a test_a decode needs a GPU pass,
forbidden here). This figure therefore shows the two decoded splits, the
interpolation test_b row and the |G| = 4 test_c row; the missing test_a row is
flagged in the regeneration report.

Decoded fields are RAW (pipeline-unnormalized) omega; both decode and DNS truth are
displayed pipeline-NORMALISED (divide by 3*train_std) in the locked Fig-3
convention (RdBu_r, vmin/vmax = -3/+3, black airfoil) via figstyle.vort_panel. DNS
truth is the v2p1 cache omega_z through the same pipeline (mask + clip), per the
decode README dns_truth_ref.

Representative-not-worst (CLAUDE.md): each row's encounter is the one whose
predictive decode is closest to the per-split MEDIAN impact-frame field MSE against
the DNS, a typical encounter rather than the best or worst.

CPU only; numpy + h5py for the cache. Reads the decode NPZs + the v2p1 cache;
writes only the PDF.

Output: paper/sections/figures/results/figD_reconstructions_v2p1.pdf
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import _common as cm  # noqa: E402

fs = cm.fs

DECODE_ROOT = cm.SESSION28 / "decode"
OMEGA_MANIFEST = cm.REPO / "outputs" / "data_pipeline" / "v2p1" / "manifest.json"

ROWS = [("test_b", "test B\n(interpolation)"), ("test_c", r"test C ($|G|{=}4$)")]
# (decode-subdir, column label) in family order: sim is row-shared from the cache.
COLS = [("tf_noc", "predictive"), ("fukami", "reconstructive"), ("pod", "POD")]
IMPACT_OFFSET = 0  # the impact frame itself


def cache_root() -> Path:
    base = os.environ.get(
        "VORTEX_JEPA_CACHE",
        str(
            Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
            / "data"
            / "processed"
            / "vortex-jepa"
        ),
    )
    return Path(base) / "v2p1"


def load_pipeline():
    from src.data.omega_pipeline import OmegaPipeline

    return OmegaPipeline.from_manifest(OMEGA_MANIFEST)


def dns_impact_field(root: Path, cid: str, ei: int, frame: int, pipe) -> np.ndarray:
    import h5py

    with h5py.File(root / cid / f"encounter_{ei:02d}.h5", "r") as f:
        omega = f["omega_z"][:].astype(np.float64)
    return np.asarray(pipe.preprocess_raw(omega, cid, ei))[frame]


def horizon0_index(blob) -> int:
    return list(blob["horizon_offsets"]).index(IMPACT_OFFSET)


def main() -> None:
    fs.use_style()
    pipe = load_pipeline()
    root = cache_root()
    norm_div = 3.0 * pipe.train_stats.std

    fig, axes = plt.subplots(len(ROWS), 4, figsize=fs.figure_size(1.0, aspect=0.40))
    if len(ROWS) == 1:
        axes = axes[None, :]
    im = None
    picks = {}
    for r, (split, rlab) in enumerate(ROWS):
        # load all three family decodes for this split + the impact-frame index
        decs = {
            sub: np.load(DECODE_ROOT / sub / f"recon_{split}.npz", allow_pickle=True)
            for sub, _ in COLS
        }
        ref = decs["tf_noc"]
        h0 = horizon0_index(ref)
        cids = np.array([str(c) for c in ref["case_ids"]])
        eis = ref["encounter_indices"].astype(int)
        impact = ref["impact_frame"].astype(int)

        # DNS impact fields for all encounters in this split
        dns_fields = np.stack(
            [
                dns_impact_field(root, cids[i], int(eis[i]), int(impact[i]), pipe)
                for i in range(len(cids))
            ]
        )
        jepa_h0 = ref["omega_decoded_horizon"][:, h0].astype(np.float64)
        mse = ((jepa_h0 - dns_fields) ** 2).mean(axis=(1, 2))
        rep = int(np.argmin(np.abs(mse - np.median(mse))))  # median-difficulty
        picks[split] = (cids[rep], int(eis[rep]))

        # column 0: simulation (DNS); columns 1-3: family decodes, all normalised.
        fields = [("simulation", dns_fields[rep])]
        fields += [
            (
                lab,
                decs[sub]["omega_decoded_horizon"][rep, horizon0_index(decs[sub])].astype(
                    np.float64
                ),
            )
            for sub, lab in COLS
        ]
        for c, (clab, raw) in enumerate(fields):
            ax = axes[r, c]
            im = fs.vort_panel(ax, np.asarray(raw, dtype=np.float64) / norm_div)
            if r == 0:
                ax.set_title(clab, fontsize=7.5)
            if c == 0:
                ax.text(
                    -0.06,
                    0.5,
                    rlab,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="right",
                    va="center",
                    fontsize=7,
                )

    fig.subplots_adjust(left=0.07, right=0.9, top=0.90, bottom=0.03, wspace=0.06, hspace=0.10)
    cax = fig.add_axes([0.915, 0.2, 0.013, 0.55])
    fig.colorbar(im, cax=cax, label=r"$\omega_z$ (norm.)")

    out = cm.out_path("figD_reconstructions")
    fig.savefig(out)
    print(f"wrote {out}")
    for split, (cid, ei) in picks.items():
        print(f"  {split}: representative {cid} enc {ei}")
    print(
        "  NOTE: test_a was NOT decoded this campaign (decode/ has only test_b, "
        "test_c); the caption's test_a row needs a GPU decode pass."
    )


if __name__ == "__main__":
    main()
