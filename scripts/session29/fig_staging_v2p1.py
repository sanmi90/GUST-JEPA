"""DNS physical-staging figure (fig_staging_v2p1): the vortex-gust encounter as a
wake-reorganisation event.

Session 29.3 section 3.3, Fukami-JFM-style "first results" figure. Built BEFORE any
model result so the reader understands why wake enstrophy at the H = 16 readout is the
primary forecasting endpoint.

Layout: 4 rows (representative cases spanning the gust envelope) by 4 vorticity columns
(encounter stages) plus a slim 5th column of force/wake traces per row.

Rows (low-error TYPICAL cases, interior of the envelope, D ~ 1.0; not the hardest case):
  * Baseline             no gust (G = 0)
  * G+0.50_D1.00_Y+0.10  weak gust   (|G| ~ 0.5)
  * G+1.50_D1.00_Y+0.00  medium gust (|G| ~ 1.5)
  * G-3.00_D1.00_Y+0.10  strong gust (|G| ~ 3.0, D ~ 1.0 interior)

Columns (fixed across rows so the grid reads as one shared timeline; impact frame ~ 40):
  1 pre-impact  frame ~ 25  (impact - 15)
  2 LEV roll-up frame ~ 37  (impact -  3)
  3 peak load   frame ~ 44  (impact +  4)
  4 readout     frame ~ 56  (impact + 16 = H = 16 endpoint)
  5 traces      C_L and wake enstrophy E_w vs t/c, vertical line at the H = 16 readout

Vorticity convention (MANDATORY): figstyle.vort_panel -- RdBu_r, vmin/vmax = -3/+3,
filled-black NACA 0012 airfoil, physical extent (-1.5, 4.5) x (-1.5, 1.5). Fields are
displayed pipeline-NORMALISED (mask + per-encounter clip + 3-sigma scale) via the v2.1
omega manifest, the locked Fig-3 scale.

Data: omega_z, C_L from the v2p1 cache encounter_00.h5; per-frame wake enstrophy from
outputs/session28/exp2/dns_physical_metrics.npz (wake box x in [0.5, 4.0], y in
[-1.0, 1.0]). dt_tc = 0.05.

CPU only (GPU is busy with training). No W&B, no model, no git side effects.

Output: paper/sections/figures/results/fig_staging_v2p1.pdf
        outputs/session29/narrative/fig_staging_v2p1.png (preview)
        outputs/session29/narrative/fig_staging_v2p1.json (provenance sidecar)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "session21"))

import figstyle as fs  # noqa: E402

OMEGA_MANIFEST = REPO / "outputs" / "data_pipeline" / "v2p1" / "manifest.json"
METRICS_NPZ = REPO / "outputs" / "session28" / "exp2" / "dns_physical_metrics.npz"
PDF_OUT = REPO / "paper" / "sections" / "figures" / "results" / "fig_staging_v2p1.pdf"
PNG_OUT = REPO / "outputs" / "session29" / "narrative" / "fig_staging_v2p1.png"
JSON_OUT = REPO / "outputs" / "session29" / "narrative" / "fig_staging_v2p1.json"

DT_TC = 0.05
IMPACT_FRAME = 40
READOUT_H = 16  # wake-enstrophy readout horizon in frames (impact + 16)

# (case_id, short label, gust-tier description) in increasing gust strength.
DEFAULT_CASES = [
    ("Baseline", "no gust", r"$G = 0$"),
    ("G+0.50_D1.00_Y+0.10", "weak", r"$G = +0.5,\ D = 1.0$"),
    ("G+1.50_D1.00_Y+0.00", "medium", r"$G = +1.5,\ D = 1.0$"),
    ("G-3.00_D1.00_Y+0.10", "strong", r"$G = -3.0,\ D = 1.0$"),
]

# (frame, stage label). Fixed across rows so the columns read as one timeline.
DEFAULT_STAGES = [
    (IMPACT_FRAME - 15, "pre-impact"),
    (IMPACT_FRAME - 3, "LEV roll-up"),
    (IMPACT_FRAME + 4, "peak load"),
    (IMPACT_FRAME + READOUT_H, "readout ($H{=}16$)"),
]

ENCOUNTER = 0  # encounter_00 for every row (no-gust baseline + first gust release)


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


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

    if not OMEGA_MANIFEST.exists():
        raise FileNotFoundError(f"omega manifest not found: {OMEGA_MANIFEST}")
    return OmegaPipeline.from_manifest(OMEGA_MANIFEST)


def load_encounter(root: Path, case_id: str, ei: int):
    """Return (omega_z (T, 192, 96) raw, C_L (T,)) and fail loudly if missing."""
    import h5py

    path = root / case_id / f"encounter_{ei:02d}.h5"
    if not path.exists():
        raise FileNotFoundError(
            f"cache encounter missing: {path}\n"
            f"check VORTEX_JEPA_CACHE points at the v2.1 cache and the case_id exists."
        )
    with h5py.File(path, "r") as f:
        omega = f["omega_z"][:].astype(np.float64)
        cl = np.asarray(f["C_L"]).astype(np.float64)
    return omega, cl


def lookup_wake_enstrophy(case_id: str, ei: int) -> np.ndarray:
    """Per-frame wake enstrophy E_w (T,) from the Session 28 DNS metrics NPZ."""
    if not METRICS_NPZ.exists():
        raise FileNotFoundError(f"DNS metrics NPZ not found: {METRICS_NPZ}")
    d = np.load(METRICS_NPZ, allow_pickle=True)
    for split in ("train", "test_a", "test_b", "test_c"):
        cids = d[f"{split}_case_id"]
        eis = d[f"{split}_encounter_index"]
        idx = np.where((cids == case_id) & (eis == ei))[0]
        if len(idx):
            return d[f"{split}_wake_enstrophy"][int(idx[0])].astype(np.float64)
    raise KeyError(f"wake enstrophy not found for case {case_id} encounter {ei} in {METRICS_NPZ}")


def build_figure(cases, stages, root: Path, pipe):
    fs.use_style()
    n_rows = len(cases)
    n_vcols = len(stages)
    # 4 vorticity columns + 1 wide trace column. The vorticity panels are equal-aspect
    # 2:1 (192 x 96), so the figure aspect is tuned so the gridspec cells match the
    # panel shape and no vertical whitespace opens inside a row.
    fig = plt.figure(figsize=fs.figure_size(1.0, aspect=0.66))
    gs = fig.add_gridspec(
        n_rows,
        n_vcols + 1,
        width_ratios=[1.0] * n_vcols + [1.6],
        left=0.075,
        right=0.965,
        top=0.91,
        bottom=0.115,
        wspace=0.07,
        hspace=0.20,
    )

    im = None
    readout_tc = (IMPACT_FRAME + READOUT_H) * DT_TC
    picks = []
    for r, (cid, short, desc) in enumerate(cases):
        omega_raw, cl = load_encounter(root, cid, ENCOUNTER)
        omega_norm = np.asarray(pipe(omega_raw, cid, ENCOUNTER), dtype=np.float64)
        ew = lookup_wake_enstrophy(cid, ENCOUNTER)
        n_t = omega_norm.shape[0]
        picks.append({"case_id": cid, "tier": short, "encounter": ENCOUNTER})

        for c, (frame, _slab) in enumerate(stages):
            if frame < 0 or frame >= n_t:
                raise ValueError(f"stage frame {frame} out of range [0, {n_t}) for {cid}")
            ax = fig.add_subplot(gs[r, c])
            im = fs.vort_panel(ax, omega_norm[frame])
            if r == 0:
                ax.set_title(_slab, fontsize=7.5, pad=3)
            if c == 0:
                ax.text(
                    -0.10,
                    0.5,
                    f"{short}\n{desc}",
                    transform=ax.transAxes,
                    rotation=90,
                    ha="right",
                    va="center",
                    fontsize=6.8,
                )

        # trace column: C_L (left axis) and wake enstrophy E_w (right axis) vs t/c.
        axt = fig.add_subplot(gs[r, n_vcols])
        t = np.arange(n_t) * DT_TC
        c_cl = fs.FAMILY_COLOR["fukami"]
        c_ew = fs.FAMILY_COLOR["pod"]
        axt.plot(t, cl, color=c_cl, lw=1.0)
        axt.axvline(readout_tc, color="0.35", lw=0.8, ls=(0, (3, 2)), zorder=1)
        axt.set_ylabel(r"$C_L$", color=c_cl, fontsize=7, labelpad=1)
        axt.tick_params(axis="y", labelcolor=c_cl, labelsize=6)
        axt.tick_params(axis="x", labelsize=6)
        axt.spines["right"].set_visible(True)

        axe = axt.twinx()
        axe.plot(t, ew, color=c_ew, lw=1.0, ls="-")
        axe.set_ylabel(r"$\Omega_w$", color=c_ew, fontsize=7, labelpad=1, rotation=270, va="bottom")
        axe.tick_params(axis="y", labelcolor=c_ew, labelsize=6)
        axe.spines["top"].set_visible(False)
        axe.spines["left"].set_visible(False)

        if r == 0:
            axt.set_title(r"$C_L,\ \Omega_w$ vs $t/c$", fontsize=7.5, pad=3)
        if r == n_rows - 1:
            axt.set_xlabel(r"$t/c$", fontsize=7, labelpad=1)
        else:
            axt.set_xticklabels([])
        # annotate the readout line once, on the top row.
        if r == 0:
            axt.text(
                readout_tc,
                axt.get_ylim()[1],
                "  readout",
                fontsize=5.5,
                color="0.35",
                ha="left",
                va="top",
                rotation=90,
            )

    # one shared vorticity colourbar spanning the vorticity columns.
    cax = fig.add_axes([0.075, 0.035, 0.46, 0.018])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label(r"$\omega_z$ (3-$\sigma$ normalised)", fontsize=7, labelpad=1)
    cb.ax.tick_params(labelsize=6)

    fig.suptitle(
        "Vortex-gust impingement as a wake-reorganisation event (DNS)",
        fontsize=8.5,
        y=0.975,
    )
    return fig, picks


def parse_stage(spec: str):
    """`--stages 25:pre-impact 37:roll-up ...` -> [(frame, label), ...]."""
    frame_s, _, label = spec.partition(":")
    return int(frame_s), (label or f"f{frame_s}")


def parse_case(spec: str):
    """`--cases CID:tier:desc` (desc optional). Falls back to a bare case_id."""
    parts = spec.split(":")
    cid = parts[0]
    tier = parts[1] if len(parts) > 1 else cid
    desc = parts[2] if len(parts) > 2 else ""
    return (cid, tier, desc)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="case specs CID[:tier[:desc]]; default is the 4 envelope-spanning rows.",
    )
    p.add_argument(
        "--stages",
        nargs="+",
        default=None,
        help="stage specs FRAME[:label]; default pre-impact/roll-up/peak/readout.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="validate cache + metrics + frames and print the plan; do not render.",
    )
    args = p.parse_args(argv)

    cases = [parse_case(s) for s in args.cases] if args.cases else DEFAULT_CASES
    stages = [parse_stage(s) for s in args.stages] if args.stages else DEFAULT_STAGES

    cmd = "python " + " ".join([sys.argv[0]] + (argv if argv is not None else sys.argv[1:]))
    sha = git_sha()
    utc = dt.datetime.now(dt.timezone.utc).isoformat()
    print("=" * 78)
    print("fig_staging_v2p1: DNS physical-staging figure (Session 29.3 section 3.3)")
    print(f"  git sha : {sha}")
    print(f"  cmd     : {cmd}")
    print(f"  utc     : {utc}")
    print(f"  cache   : {cache_root()}")
    print(f"  metrics : {METRICS_NPZ}")
    print(f"  rows    : {[c[0] for c in cases]}")
    print(f"  stages  : {[(f, lab) for f, lab in stages]}")
    print("=" * 78)

    root = cache_root()
    pipe = load_pipeline()

    # Validate every case and stage frame up front (fail loudly).
    for cid, _short, _desc in cases:
        omega_raw, cl = load_encounter(root, cid, ENCOUNTER)
        ew = lookup_wake_enstrophy(cid, ENCOUNTER)
        n_t = omega_raw.shape[0]
        for frame, lab in stages:
            if frame < 0 or frame >= n_t:
                raise ValueError(f"stage '{lab}' frame {frame} out of range [0,{n_t}) for {cid}")
        clpk = int(np.argmax(np.abs(cl[25:60])) + 25)
        rf = IMPACT_FRAME + READOUT_H
        print(
            f"  OK {cid:22s} omega{omega_raw.shape} CLpeak@frame {clpk} "
            f"(C_L={cl[clpk]:+.2f}) E_w[readout {rf}]={ew[rf]:.1f}"
        )

    if args.dry_run:
        print("dry-run OK: all cases/stages validated; no figure rendered.")
        return 0

    fig, picks = build_figure(cases, stages, root, pipe)

    PDF_OUT.parent.mkdir(parents=True, exist_ok=True)
    PNG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF_OUT)
    fig.savefig(PNG_OUT, dpi=200)
    plt.close(fig)

    sidecar = {
        "figure": "fig_staging_v2p1",
        "section": "Session 29.3 section 3.3 (DNS physical-staging)",
        "git_sha": sha,
        "cmd": cmd,
        "utc": utc,
        "pdf": str(PDF_OUT),
        "png": str(PNG_OUT),
        "omega_manifest": str(OMEGA_MANIFEST),
        "metrics_npz": str(METRICS_NPZ),
        "dt_tc": DT_TC,
        "impact_frame": IMPACT_FRAME,
        "readout_horizon_frames": READOUT_H,
        "encounter": ENCOUNTER,
        "rows": picks,
        "stages": [{"frame": f, "label": lab} for f, lab in stages],
        "wake_box": {"x": [0.5, 4.0], "y": [-1.0, 1.0], "threshold": 1.0},
        "convention": "figstyle.vort_panel RdBu_r vmin/vmax=-3/+3 black NACA0012",
    }
    JSON_OUT.write_text(json.dumps(sidecar, indent=2))

    print(f"wrote {PDF_OUT}")
    print(f"wrote {PNG_OUT}")
    print(f"wrote {JSON_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
