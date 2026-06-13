"""Rollout-traces figure (figA_traces_v2p1.pdf): predicted observables through
representative held-out encounters, full-context rollout. v2.1.

Matches the v2.1 caption (fig:traces, section_4_results.tex): predicted observables
through representative held-out encounters (rows) for the lift coefficient, the wake
enstrophy, and the negative circulation (columns). The simulation is the bold
reference; the rollout of the predictor started from the pre-impact context
(z_full) is the coloured line; the dashed line is the reconstruction from the
encoded latent (z_dns). Numbered glyphs mark the four encounter stages (baseline,
impact, peak load, recovery). Reported as qualitative tracking, not a closure R^2.

Representative-not-worst rule (CLAUDE.md memory): the three rows are picked as
MEDIAN-difficulty encounters, one near the median |G| of each of three regimes
(weak gust, strong +G, strong -G), so the panel shows a typical encounter rather
than the best or the worst.

Probe: a single fixed linear (ridge) probe per observable, fit over all pooled
TRAIN frames of the predictive (jepa_tf_noc) latents, exactly as the closure
protocol (closure_matrix.fit_ridge / apply_probe). The same probe is applied to the
full-context rollout (z_full) and to the encoded latent (z_dns). Read-only on the
rollouts + DNS metrics; writes only the PDF. CPU only; numpy.

Output: paper/sections/figures/results/figA_traces_v2p1.pdf
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import _common as cm  # noqa: E402
import closure_matrix as clo  # noqa: E402  (fit_ridge / apply_probe / match_index)

fs = cm.fs

ROLLOUT_NPZ = cm.SESSION28 / "rollouts" / "jepa_d64" / "test_b.npz"
TRAIN_LATENTS = cm.SESSION28 / "latents" / "jepa_tf_noc_d64_s42" / "train.npz"
DNS_NPZ = cm.SESSION28 / "exp2" / "dns_physical_metrics.npz"

OBS = [
    ("C_L", r"$C_L$"),
    ("wake_enstrophy", r"wake enstrophy $\Omega_w$"),
    ("circulation_neg", r"signed circulation $\Gamma^{-}$"),
]
STAGES = [-8, 0, 16, 32]  # frames relative to impact; glyphs 1..4
WIN = (-10, 40)
SPLIT = "test_b"


def fit_train_probes(dns) -> dict:
    """One fixed ridge probe per observable, fit over all pooled TRAIN frames."""
    tr = np.load(TRAIN_LATENTS, allow_pickle=True)
    z = tr["z_full"].astype(np.float64)
    di = clo.match_index(
        tr["case_ids"], tr["encounter_indices"], dns["train_case_id"], dns["train_encounter_index"]
    )
    keep = di >= 0
    z, di = z[keep], di[keep]
    n, t_len, d = z.shape
    Zf = z.reshape(n * t_len, d)
    probes = {}
    for metric, _ in OBS:
        y = dns[f"train_{metric}"][di].reshape(n * t_len).astype(np.float64)
        probes[metric] = clo.fit_ridge(Zf, y)
    return probes


def pick_representative(G: np.ndarray) -> dict[str, int]:
    """Three distinct MEDIAN-difficulty encounters spanning the gust regimes.

    Representative-not-worst (CLAUDE.md): a weak gust (smallest |G|), a typical
    positive gust (median of the positive arm), and a typical negative gust (median
    of the negative arm). Distinct indices are guaranteed by construction.
    """

    def nearest(idx: np.ndarray, target: float, exclude: set[int]) -> int:
        order = idx[np.argsort(np.abs(G[idx] - target))]
        for j in order:
            if int(j) not in exclude:
                return int(j)
        return int(order[0])

    pos = np.flatnonzero(G > 0.1)
    neg = np.flatnonzero(G < -0.1)
    used: set[int] = set()
    weak = int(np.argmin(np.abs(G)))
    used.add(weak)
    p = nearest(pos, float(np.median(G[pos])), used) if pos.size else int(np.argmax(G))
    used.add(p)
    n = nearest(neg, float(np.median(G[neg])), used) if neg.size else int(np.argmin(G))
    return {"weak gust": weak, r"typical $+G$": p, r"typical $-G$": n}


def main() -> None:
    fs.use_style()
    dns = np.load(DNS_NPZ, allow_pickle=True)
    roll = np.load(ROLLOUT_NPZ, allow_pickle=True)
    probes = fit_train_probes(dns)

    cids = np.array([str(c) for c in roll["case_ids"]])
    eis = roll["encounter_indices"].astype(int)
    G = roll["G"].astype(float)
    di = clo.match_index(cids, eis, dns[f"{SPLIT}_case_id"], dns[f"{SPLIT}_encounter_index"])
    pick = pick_representative(G)

    fig, axes = plt.subplots(
        3, 3, figsize=fs.figure_size(1.0, aspect=0.85), sharex=True, layout="constrained"
    )
    for r, (title, ridx) in enumerate(pick.items()):
        impact = int(roll["impact_frame"][ridx])
        ddi = int(di[ridx])
        frames = np.arange(impact + WIN[0], impact + WIN[1] + 1)
        trel = frames - impact
        z_roll = roll["z_full"][ridx]  # full-context rollout
        z_enc = roll["z_dns"][ridx]  # encoded (reconstruction) latent
        for c, (metric, mlab) in enumerate(OBS):
            ax = axes[r, c]
            yt = dns[f"{SPLIT}_{metric}"][ddi, frames]
            ax.plot(trel, yt, color="0.15", lw=1.6, zorder=5, label="simulation")
            # rollout from impact onward (coloured); reconstruction (dashed)
            fr = np.arange(impact, min(impact + WIN[1] + 1, z_roll.shape[0]))
            yp_roll = clo.apply_probe(z_roll[fr], probes[metric])
            yp_enc = clo.apply_probe(z_enc[fr], probes[metric])
            ax.plot(
                fr - impact,
                yp_roll,
                color=fs.FAMILY_COLOR["jepa"],
                lw=1.1,
                zorder=4,
                label="predictive rollout",
            )
            ax.plot(
                fr - impact,
                yp_enc,
                color=fs.FAMILY_COLOR["jepa"],
                lw=0.9,
                ls=(0, (4, 2)),
                alpha=0.8,
                zorder=3,
                label="encoded (recon.)",
            )
            ax.axvline(0, color="0.85", lw=0.6, zorder=0)
            for n, s in enumerate(STAGES, start=1):
                if frames[0] <= impact + s <= frames[-1]:
                    ys = dns[f"{SPLIT}_{metric}"][ddi, impact + s]
                    fs.stage_glyph(ax, s, ys, n, color="0.30", fontsize=4.5, s=34)
            if r == 0:
                ax.set_title(mlab)
            if r == 2 and c == 1:
                ax.set_xlabel("frames relative to impact")
            if c == 0:
                ax.set_ylabel(title)

    handles = [
        plt.Line2D([], [], color="0.15", lw=1.6, label="simulation"),
        plt.Line2D(
            [], [], color=fs.FAMILY_COLOR["jepa"], lw=1.1, label="predictive rollout (from context)"
        ),
        plt.Line2D(
            [],
            [],
            color=fs.FAMILY_COLOR["jepa"],
            lw=0.9,
            ls=(0, (4, 2)),
            label="reconstruction (encoded latent)",
        ),
    ]
    fig.legend(
        handles=handles, loc="outside lower center", ncol=3, columnspacing=1.3, handletextpad=0.4
    )
    out = cm.out_path("figA_traces")
    fig.savefig(out)
    print(f"wrote {out}")
    for title, ridx in pick.items():
        print(f"  {title}: {cids[ridx]} enc {eis[ridx]} (G={G[ridx]:+.2f})")


if __name__ == "__main__":
    main()
