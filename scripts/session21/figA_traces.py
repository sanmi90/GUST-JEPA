"""SPEC 2 - NEW FIG A: predicted observables through an encounter.

One idea: under rollout the predictive trace tracks the simulation through the
LEV peak and into recovery, while the reconstructive trace flattens or diverges.
Three representative test_b encounters (weak, strong-positive, strong-negative
gust) x three observables (C_L, wake enstrophy, signed circulation). Simulation
is the bold reference; family rollouts are the fixed colour key; numbered stage
glyphs 1..4 mark the staged encounter (same glyphs as NEW FIG C).

Probe and rollout machinery are the verified Session 20 closure pipeline.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "session20"))
from exp_closure_r2 import (  # noqa: E402
    DNS_METRICS_PATH, LATENTS_ROOT, ROLLOUTS_ROOT, apply_probe, fit_probes,
    match_index,
)

OUT_PDF = REPO / "paper/sections/figures/results/figA_traces.pdf"
OUT_PNG = REPO / "outputs/session21/figs/figA_traces.png"

FAMS = [("jepa", "jepa_d64_test1_noBN"), ("fukami", "fukami_d64_noBN"),
        ("pod", "pod_d64_noBN")]
OBS = [("C_L", r"$C_L$"), ("wake_enstrophy", r"wake enstrophy $\Omega_w$"),
       ("circulation_neg", r"signed circulation $\Gamma^{-}$")]
STAGES = [-8, 0, 16, 32]   # frames relative to impact; glyphs 1..4
WIN = (-10, 40)
SPLIT = "test_b"
GRE = re.compile(r"G([+-]?\d+\.\d+)_D([\d.]+)_Y([+-]?\d+\.\d+)")


def parse_G(cid: str) -> float:
    m = GRE.search(cid)
    return float(m.group(1)) if m else np.nan


def main() -> None:
    fs.use_style()
    dns = np.load(DNS_METRICS_PATH, allow_pickle=True)
    probes, rolls = {}, {}
    for fam, tag in FAMS:
        probes[fam] = fit_probes(LATENTS_ROOT / f"latents_{tag}", dns)
        rolls[fam] = np.load(ROLLOUTS_ROOT / f"rollouts_{tag}" / f"{SPLIT}.npz",
                             allow_pickle=True)

    # pick 3 encounters by gust strength from the JEPA rollout index
    jb = rolls["jepa"]
    jcid = jb["case_ids"] if "case_ids" in jb.files else jb["case_id"]
    jei = jb["encounter_indices"] if "encounter_indices" in jb.files else jb["encounter_index"]
    G = np.array([parse_G(str(c)) for c in jcid])
    pick = {"weak gust": int(np.argmin(np.abs(G))),
            "strong $+G$": int(np.argmax(G)),
            "strong $-G$": int(np.argmin(G))}

    di_dns = match_index(jcid, jei, dns[f"{SPLIT}_case_id"], dns[f"{SPLIT}_encounter_index"])

    fig, axes = plt.subplots(3, 3, figsize=fs.figure_size(1.0, aspect=0.85),
                             sharex=True)
    for r, (title, ridx) in enumerate(pick.items()):
        cid, ei = str(jcid[ridx]), int(jei[ridx])
        impact = int(jb["impact_frame"][ridx])
        ddi = di_dns[ridx]
        frames = np.arange(impact + WIN[0], impact + WIN[1] + 1)
        trel = frames - impact
        for c, (metric, mlab) in enumerate(OBS):
            ax = axes[r, c]
            # simulation reference
            yt = dns[f"{SPLIT}_{metric}"][ddi, frames]
            ax.plot(trel, yt, color="0.15", lw=1.6, zorder=5)
            # family rollouts from impact onward
            for fam, tag in FAMS:
                b = rolls[fam]
                bc = b["case_ids"] if "case_ids" in b.files else b["case_id"]
                be = b["encounter_indices"] if "encounter_indices" in b.files else b["encounter_index"]
                idx = next((k for k in range(len(bc))
                            if str(bc[k]) == cid and int(be[k]) == ei), None)
                if idx is None:
                    continue
                z = b["z_markov"][idx]
                imp = int(b["impact_frame"][idx])
                pr = probes[fam][metric]
                fr = np.arange(imp, min(imp + WIN[1] + 1, z.shape[0]))
                yp = apply_probe(z[fr], pr)
                ax.plot(fr - imp, yp, color=fs.FAMILY_COLOR[fam], lw=1.0,
                        zorder=3)
            ax.axvline(0, color="0.85", lw=0.6, zorder=0)
            # stage glyphs on the simulation reference
            for n, s in enumerate(STAGES, start=1):
                if frames[0] <= impact + s <= frames[-1]:
                    ys = dns[f"{SPLIT}_{metric}"][ddi, impact + s]
                    fs.stage_glyph(ax, s, ys, n, color="0.30", fontsize=4.5,
                                   s=34)
            if r == 0:
                ax.set_title(mlab)
            if r == 2:
                ax.set_xlabel("frames relative to impact")
            if c == 0:
                ax.set_ylabel(f"{title}")

    handles = [plt.Line2D([], [], color="0.15", lw=1.6, label="simulation")]
    handles += [plt.Line2D([], [], color=fs.FAMILY_COLOR[f], lw=1.0,
                           label=fs.FAMILY_LABEL[f]) for f, _ in FAMS]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02), columnspacing=1.3, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=200)
    print(f"wrote {OUT_PDF}\nwrote {OUT_PNG}")


if __name__ == "__main__":
    main()
