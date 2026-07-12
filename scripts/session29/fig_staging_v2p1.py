"""DNS physical-staging figure (fig_staging_v2p1): a gust-strength sweep at matched
core diameter and offset, in the Fukami, Nakao and Taira Figure-5 format.

Session 29.3 section 3.3, Fukami-JFM-style "first results" figure. Built BEFORE any
model result so the reader sees the raw vortex-gust forcing across the gust-strength
envelope before reading a single forecast.

Layout (mirrors Fukami Fig 5):
  Panel (a), top, full width: lift coefficient C_L vs convective time t/c for all four
    cases on one axis. The undisturbed baseline (G = 0) is drawn in grey; the three gust
    cases use a blue gradient that darkens with G. Each curve is labelled inline (no
    legend box), in Fukami's by-parameter shading.
  Panel (b), below: a 4 row x 4 column grid of mid-plane spanwise vorticity omega_z, one
    ROW per case (same order as panel a) and one COLUMN per fixed time instant (the SAME
    frames for every row). The case is labelled at the left of each row, the time value
    under each column.
  One shared horizontal colorbar at the bottom.

Cases (gust-strength sweep at matched D = 0.5, Y = -0.1; encounter_00 for every row):
  * Baseline             no gust      (G = 0)
  * G+0.50_D0.50_Y-0.10  weak gust    (G = 0.5)
  * G+1.50_D0.50_Y-0.10  medium gust  (G = 1.5)
  * G+4.00_D0.50_Y-0.10  strong gust  (G = 4)

Time instants (fixed across rows so the grid reads as one shared timeline; t/c = 0 at
gust release at frame 0, impact at frame 40 = t/c 2.0):
  frame 20  t/c = 1.0   pre-impact
  frame 40  t/c = 2.0   impact
  frame 60  t/c = 3.0   post-impact
  frame 80  t/c = 4.0   wake

Vorticity convention (MANDATORY): figstyle.vort_panel -- RdBu_r, vmin/vmax = -3/+3,
filled-black NACA 0012 airfoil, physical extent (-1.5, 4.5) x (-1.5, 1.5). Fields are
displayed pipeline-NORMALISED (mask + per-encounter clip + 3-sigma scale) via the v2.1
omega manifest, the locked Fig-3 scale.

Data: omega_z, C_L from the v2p1 cache encounter_00.h5. dt_tc = 0.05.

CPU only (GPU is busy with training). No W&B, no model, no git side effects.

Output: paper/sections/figures/results/fig_staging_v2p1.pdf
        outputs/session29/narrative/fig_staging_v2p1.png (preview)
        outputs/session29/narrative/fig_staging_v2p1.json (provenance sidecar)
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
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
PDF_OUT = REPO / "paper" / "sections" / "figures" / "results" / "fig_staging_v2p1.pdf"
PNG_OUT = REPO / "outputs" / "session29" / "narrative" / "fig_staging_v2p1.png"
JSON_OUT = REPO / "outputs" / "session29" / "narrative" / "fig_staging_v2p1.json"

DT_TC = 0.05
IMPACT_FRAME = 40  # gust impact (t/c = 2.0); gust release is at frame 0 (t/c = 0)

ENCOUNTER = 0  # encounter_00 for every row (no-gust baseline + first gust release)

# (case_id, row label, inline C_L curve label, gust value G). Ordered by increasing
# gust strength so both panels read top-to-bottom / light-to-dark as G grows.
# Displayed G uses the paper's sign convention (G = -s, opposite the dataset's
# solver sign in the case id); the 4th tuple field keeps the MAGNITUDE for the
# colour gradient and label-anchor logic.
DEFAULT_CASES = [
    ("Baseline", r"$G = 0$", r"$G = 0$", 0.0),
    ("G+0.50_D0.50_Y-0.10", r"$G = -0.5$", r"$G = -0.5$", 0.5),
    ("G+1.50_D0.50_Y-0.10", r"$G = -1.5$", r"$G = -1.5$", 1.5),
    ("G+4.00_D0.50_Y-0.10", r"$G = -4$", r"$G = -4$", 4.0),
]

# Inline-label anchor (frame, vertical-align) per gust value G. Each label sits in a band
# that is empty of every other curve, so no label collides with a curve or another label:
#   G = 4   : above its deep impact dip (band from its dip up to the G = 1.5 dip is clear)
#   G = 1.5 : above its post-impact overshoot peak (clear band up to the much larger
#             G = 4 peak), moved off the impact dip where the dark G = 4 curve used to
#             cross through and hide the "1" of "1.5"
#   G = 0.5 : above its post-impact plateau where it is the topmost curve (the band above
#             it is empty; moved off the impact dip where the steep G = 4 curve crossed it)
#   G = 0   : below the flat baseline tail (the band under it is empty; the gust curves
#             have all risen above +1 by then)
# "va" sets which side of the anchor point the text sits ("bottom" => label above the
# point, "top" => label below it). Keyed by G to survive --cases reordering.
LABEL_ANCHOR = {
    0.0: (100, "top"),
    0.5: (90, "bottom"),
    1.5: (51, "bottom"),
    4.0: (36, "bottom"),
}

# Fixed frames shared by every row. Labelled at the bottom of panel (b) by t/c.
DEFAULT_FRAMES = [20, 40, 60, 80]

# Blue gradient for the three gust curves: light (G = 0.5) to dark (G = 4). Anchored on
# the paper POD-family blue so the gust shading matches the rest of the figure set.
GUST_BLUES = ["#9ecae1", "#4292c6", "#08519c"]
BASELINE_GREY = "0.55"


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
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


def build_figure(cases, frames, root: Path, pipe):
    fs.use_style()
    n_rows = len(cases)
    n_cols = len(frames)

    # The vorticity panels are equal-aspect 2:1 (192 x 96). The grid is taller than wide
    # (4 rows of half-height panels), and panel (a) sits on top, so the figure is roughly
    # square. Tuned so the grid cells match the panel shape with little vertical slack.
    fig = plt.figure(figsize=fs.figure_size(1.0, aspect=1.04))

    # Outer split: panel (a) on top, the vorticity grid + colourbar below.
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 2.55],
        left=0.085,
        right=0.975,
        top=0.965,
        bottom=0.075,
        hspace=0.26,
    )

    # ---- panel (a): C_L vs t/c for all cases on one axis ----------------------------
    ax_cl = fig.add_subplot(outer[0])
    curves = []
    for cid, _row_label, curve_label, gval in cases:
        _omega_raw, cl = load_encounter(root, cid, ENCOUNTER)
        n_t = cl.shape[0]
        t = np.arange(n_t) * DT_TC
        if gval == 0.0:
            color = BASELINE_GREY
            lw = 1.0
            z = 2
        else:
            color = GUST_BLUES[[c[3] for c in cases if c[3] > 0].index(gval)]
            lw = 1.2
            z = 3
        ax_cl.plot(t, cl, color=color, lw=lw, zorder=z)
        curves.append((curve_label, gval, t, cl, color))

    ax_cl.set_xlim(0.0, (n_t - 1) * DT_TC)
    ax_cl.set_xlabel(r"$t/c$", labelpad=2)
    ax_cl.set_ylabel(r"$C_L$", labelpad=2)
    ax_cl.tick_params(labelsize=7)

    # inline curve labels (Fukami labels curves, not a legend box). Each label is anchored
    # at a per-curve frame (LABEL_ANCHOR) chosen where that curve is well separated from
    # the others, nudged to the side recorded in LABEL_ANCHOR so it clears the line.
    y0, y1 = ax_cl.get_ylim()
    pad = 0.05 * (y1 - y0)
    # impact-time marker (Carlos's assessment, fig 1): t_imp is the leading-edge
    # crossing at t/c = 2.0; t/c = 0 is the gust release.
    t_imp = IMPACT_FRAME * DT_TC
    ax_cl.axvline(t_imp, color="0.55", lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax_cl.text(t_imp, y1 - pad, r"$t_{\mathrm{imp}}$", fontsize=7, color="0.35",
               ha="center", va="top")
    for curve_label, gval, t, cl, color in curves:
        frame, va = LABEL_ANCHOR.get(gval, (len(cl) - 1, "bottom"))
        j = min(frame, len(cl) - 1)
        nudge = pad if va == "bottom" else -pad
        ax_cl.annotate(
            curve_label,
            xy=(t[j], cl[j]),
            xytext=(t[j], cl[j] + nudge),
            color=color,
            fontsize=7,
            ha="center",
            va=va,
            zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.5),
        )

    # ---- panel (b): vorticity grid (rows = cases, cols = fixed frames) ---------------
    grid = outer[1].subgridspec(
        n_rows,
        n_cols,
        wspace=0.06,
        hspace=0.10,
    )

    im = None
    for r, (cid, row_label, _curve_label, _gval) in enumerate(cases):
        omega_raw, _cl = load_encounter(root, cid, ENCOUNTER)
        # Raw physical vorticity (NO pipeline clip/scale). The colorbar is fixed at
        # +/-3 so the field saturates into a high-contrast sign map, per the figure
        # request. Training uses the 3-sigma-normalised field; this figure does not.
        omega_disp = np.asarray(omega_raw, dtype=np.float64)
        n_t = omega_disp.shape[0]
        for c, frame in enumerate(frames):
            if frame < 0 or frame >= n_t:
                raise ValueError(f"frame {frame} out of range [0, {n_t}) for {cid}")
            ax = fig.add_subplot(grid[r, c])
            im = fs.vort_panel(ax, omega_disp[frame])
            if c == 0:
                ax.text(
                    -0.06,
                    0.5,
                    row_label,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="right",
                    va="center",
                    fontsize=8,
                )
            if r == n_rows - 1:
                ax.set_xlabel(rf"$t/c = {frame * DT_TC:.1f}$", fontsize=7.5, labelpad=3)

    # one shared horizontal colourbar spanning the grid, below the bottom row.
    cax = fig.add_axes([0.32, 0.022, 0.40, 0.016])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-3, -2, -1, 0, 1, 2, 3])
    cb.set_label(r"$\omega_z$", fontsize=7.5, labelpad=2)
    cb.ax.tick_params(labelsize=6.5)

    picks = [
        {"case_id": cid, "row_label": rl, "G": gv, "encounter": ENCOUNTER}
        for cid, rl, _cl, gv in cases
    ]
    return fig, picks


def parse_case(spec: str):
    """`--cases CID:row_label:curve_label:G` (only CID required)."""
    parts = spec.split(":")
    cid = parts[0]
    row_label = parts[1] if len(parts) > 1 else cid
    curve_label = parts[2] if len(parts) > 2 else row_label
    gval = float(parts[3]) if len(parts) > 3 else 0.0
    return (cid, row_label, curve_label, gval)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="case specs CID[:row_label[:curve_label[:G]]]; default is the 4-case "
        "gust-strength sweep at D = 0.5, Y = -0.1.",
    )
    p.add_argument(
        "--frames",
        nargs="+",
        type=int,
        default=None,
        help="fixed column frames shared by every row; default 20 40 60 80.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="validate cache + frames and print the plan; do not render.",
    )
    args = p.parse_args(argv)

    cases = [parse_case(s) for s in args.cases] if args.cases else DEFAULT_CASES
    frames = list(args.frames) if args.frames else DEFAULT_FRAMES

    cmd = "python " + " ".join([sys.argv[0]] + (argv if argv is not None else sys.argv[1:]))
    sha = git_sha()
    utc = dt.datetime.now(dt.timezone.utc).isoformat()
    print("=" * 78)
    print("fig_staging_v2p1: DNS gust-strength sweep (Fukami Fig-5 format)")
    print(f"  git sha : {sha}")
    print(f"  cmd     : {cmd}")
    print(f"  utc     : {utc}")
    print(f"  cache   : {cache_root()}")
    print(f"  rows    : {[c[0] for c in cases]}")
    print(f"  frames  : {frames}  (t/c = {[round(f * DT_TC, 2) for f in frames]})")
    print("=" * 78)

    root = cache_root()
    pipe = load_pipeline()

    # Validate every case and frame up front (fail loudly).
    for cid, _row_label, _curve_label, _gval in cases:
        omega_raw, cl = load_encounter(root, cid, ENCOUNTER)
        n_t = omega_raw.shape[0]
        for frame in frames:
            if frame < 0 or frame >= n_t:
                raise ValueError(f"frame {frame} out of range [0,{n_t}) for {cid}")
        print(
            f"  OK {cid:22s} omega{omega_raw.shape} C_L[{cl.min():+.2f},{cl.max():+.2f}] "
            f"C_L@impact(f{IMPACT_FRAME})={cl[IMPACT_FRAME]:+.2f}"
        )

    if args.dry_run:
        print("dry-run OK: all cases/frames validated against the cache; no figure rendered.")
        return 0

    fig, picks = build_figure(cases, frames, root, pipe)

    w_in, h_in = fig.get_size_inches()
    PDF_OUT.parent.mkdir(parents=True, exist_ok=True)
    PNG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF_OUT)
    fig.savefig(PNG_OUT, dpi=200)
    plt.close(fig)

    sidecar = {
        "figure": "fig_staging_v2p1",
        "section": "Session 29.3 section 3.3 (DNS gust-strength sweep, Fukami Fig-5 format)",
        "git_sha": sha,
        "cmd": cmd,
        "utc": utc,
        "pdf": str(PDF_OUT),
        "png": str(PNG_OUT),
        "omega_manifest": str(OMEGA_MANIFEST),
        "omega_manifest_sha256": file_sha256(OMEGA_MANIFEST),
        "dt_tc": DT_TC,
        "impact_frame": IMPACT_FRAME,
        "encounter": ENCOUNTER,
        "cases": picks,
        "case_ids": [c[0] for c in cases],
        "frames": frames,
        "frames_tc": [round(f * DT_TC, 3) for f in frames],
        "figure_size_in": [round(float(w_in), 3), round(float(h_in), 3)],
        "convention": "figstyle.vort_panel RdBu_r vmin/vmax=-3/+3 black NACA0012, "
        "extent (-1.5,4.5)x(-1.5,1.5)",
    }
    JSON_OUT.write_text(json.dumps(sidecar, indent=2))

    print(f"figure size: {w_in:.3f} x {h_in:.3f} in")
    print(f"wrote {PDF_OUT}")
    print(f"wrote {PNG_OUT}")
    print(f"wrote {JSON_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
