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

# SESSION27 UNCONDITIONED VARIANT of scripts/session21/figA_traces.py.
# Same probe + rollout machinery (exp_closure_r2.fit_probes / apply_probe), but the
# predictive (JEPA) family is fit/rolled from the fully-unconditioned (no-c) model:
#   probe fit on outputs/session27/latents_own_jepa_noc_lstm/train.npz,
#   rollouts from outputs/session27/rollouts_noc_lstm/{split}.npz.
# The reconstructive (Fukami) and POD reference families are kept identical to
# production (LATENTS_ROOT / ROLLOUTS_ROOT under session18).
REPO = Path(__file__).resolve().parents[1]  # session27 copy lives directly under scripts/
sys.path.insert(0, str(REPO / "scripts" / "session21"))
sys.path.insert(0, str(REPO / "scripts" / "session20"))
import figstyle as fs  # noqa: E402
from exp_closure_r2 import (  # noqa: E402
    DNS_METRICS_PATH, LATENTS_ROOT, ROLLOUTS_ROOT, apply_probe, fit_probes,
    match_index,
)

OUT_PDF = REPO / "outputs/session27/figs_uncond_lstm/figA_traces.pdf"
OUT_PNG = REPO / "outputs/session27/figs_uncond_lstm/figA_traces.png"

SESSION27 = REPO / "outputs/session27"
# Per-family (latents_dir, rollouts_dir): JEPA -> unconditioned session27 artifacts;
# Fukami/POD -> production roots. These dirs hold train.npz and {split}.npz resp.
FAM_DIRS = {
    "jepa": (SESSION27 / "latents_own_jepa_noc_lstm", SESSION27 / "rollouts_noc_lstm"),
    "fukami": (LATENTS_ROOT / "latents_fukami_d64_noBN", ROLLOUTS_ROOT / "rollouts_fukami_d64_noBN"),
    "pod": (LATENTS_ROOT / "latents_pod_d64_noBN", ROLLOUTS_ROOT / "rollouts_pod_d64_noBN"),
}
FAMS = [("jepa", "noc_lstm"), ("fukami", "fukami_d64_noBN"),
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
        lat_dir, roll_dir = FAM_DIRS[fam]
        probes[fam] = fit_probes(lat_dir, dns)
        rolls[fam] = np.load(roll_dir / f"{SPLIT}.npz", allow_pickle=True)

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
                             sharex=True, layout="constrained")
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
            if r == 2 and c == 1:
                ax.set_xlabel("frames relative to impact")
            if c == 0:
                ax.set_ylabel(f"{title}")

    handles = [plt.Line2D([], [], color="0.15", lw=1.6, label="simulation")]
    handles += [plt.Line2D([], [], color=fs.FAMILY_COLOR[f], lw=1.0,
                           label=fs.FAMILY_LABEL[f]) for f, _ in FAMS]
    fig.legend(handles=handles, loc="outside lower center", ncol=4,
               columnspacing=1.3, handletextpad=0.4)
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=200)
    print(f"wrote {OUT_PDF}\nwrote {OUT_PNG}")


if __name__ == "__main__":
    main()
