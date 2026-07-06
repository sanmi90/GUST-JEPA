"""F7: decoded C_L traces per Track C cell, representative encounter (v4 manuscript).

Panel (a): truth C_L(t) plus frozen-linear-probe decoded C_L(t) for the CLN,
CLW and CL cells (three greens, one line style each) and the collapsed CW cell
(red dashed) over the full representative encounter
(case G-0.50_D1.00_Y-0.40, lowest test_b encounter index; the da_phase_eval
default rep-case, a representative low-error case per project convention).
The impact + relaxation window (window_mask from the latent cache; see
src/evaluation/rom_eval.py and represent.frame_window_mask) is shaded.

Panel (b): per-encounter phase-lag distributions on test_b (box + strip) for
the same four cells, ``lift_phase_lag`` per encounter (max_lag 20 frames,
dt 0.05), with the tau threshold |tau| = 0.1 t/c from trackc_gates marked.

Probe regime: PER-frame z_gap -> C_L (state-descriptor readout), exactly the
trackc_lift_eval convention (fit_linear_probe on train, predict on test_b).
Every number is read from the Track C latent caches at build time.

Run (CPU):
    OMP_NUM_THREADS=4 .venv/bin/python scripts/session35/fig_cell_traces_v4.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "session21"))

import figstyle as fs  # noqa: E402
from scripts.session34.trackc_lift_eval import (  # noqa: E402
    DT_TC,
    MAX_LAG_FRAMES,
    group_encounters,
    load_cache,
)
from src.evaluation.lift_metrics import lift_phase_lag  # noqa: E402
from src.evaluation.represent import fit_linear_probe  # noqa: E402

CACHE_DIR = REPO_ROOT / "outputs" / "session34" / "trackc_latents"
OUT_DIR = REPO_ROOT / "paper" / "sections" / "figures" / "results"
OUT_STEM = "fig_cell_traces_v4"

REP_CASE = "G-0.50_D1.00_Y-0.40"  # da_phase_eval default rep-case (low-error, representative)
TAU_THRESH_TC = 0.1  # trackc_gates.py:84, gate on median |lag|
SEED_TAG = "s0"

# cell label -> (run_name, color, linestyle); CELLS mapping from trackc_cells.py, seed 0.
_G = fs.FAMILY_COLOR["jepa"]  # base green


def _lighten(hex_color: str, frac: float) -> tuple:
    import matplotlib as mpl

    rgb = np.array(mpl.colors.to_rgb(hex_color))
    return tuple(rgb + (1.0 - rgb) * frac)


CELL_STYLE = {
    "CLN": {"run": "jepa_pool_ln_s0", "color": _G, "ls": "-"},
    "CLW": {"run": "jepa_pool_vec", "color": _lighten(_G, 0.35), "ls": "--"},
    "CL": {"run": "jepa_nowake_pool_vec", "color": _lighten(_G, 0.60), "ls": "-."},
    "CW": {"run": "jepa_pool_w_s0", "color": fs.FAMILY_COLOR["fukami"], "ls": (0, (2, 2))},
}


def decode_cell(run_name: str) -> dict:
    """Fit the frozen linear probe on train z_gap -> C_L, predict test_b."""
    tr = load_cache(CACHE_DIR, run_name, "train")
    tb = load_cache(CACHE_DIR, run_name, "test_b")
    probe = fit_linear_probe(tr["z_gap"], tr["cl"])
    pred = np.asarray(probe.predict(tb["z_gap"]), dtype=np.float64)
    return {"tb": tb, "pred": pred}


def rep_encounter_rows(tb_cache_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Row indices (frame-ordered) of the representative encounter + window mask."""
    z = np.load(tb_cache_path, allow_pickle=True)
    case_mask = z["case_id"] == REP_CASE
    if not case_mask.any():
        raise RuntimeError(f"rep case {REP_CASE} not in test_b cache {tb_cache_path}")
    k = int(min(set(z["encounter_index"][case_mask].tolist())))
    m = case_mask & (z["encounter_index"] == k)
    rows = np.where(m)[0]
    rows = rows[np.argsort(z["frame"][rows])]
    return rows, z["window_mask"][rows]


def main() -> int:
    fs.use_style()

    cells = {label: decode_cell(sty["run"]) for label, sty in CELL_STYLE.items()}

    # ---- representative encounter (rows identical across runs; take CLN's) --
    ref_run = CELL_STYLE["CLN"]["run"]
    rows, wmask = rep_encounter_rows(CACHE_DIR / f"latents_{ref_run}_test_b.npz")
    tb_ref = cells["CLN"]["tb"]
    frames = tb_ref["frame"][rows]
    t_tc = frames.astype(np.float64) * DT_TC
    truth = tb_ref["cl"][rows].astype(np.float64)
    rep_k = int(tb_ref["encounter_index"][rows][0])

    # ---- per-encounter phase lags on test_b, per cell --------------------
    lags: dict[str, np.ndarray] = {}
    for label, cell in cells.items():
        tb = cell["tb"]
        encs = group_encounters(tb)
        cell_lags = []
        for enc in encs:
            r = enc["rows"]
            lag_tc, _corr = lift_phase_lag(
                cell["pred"][r],
                tb["cl"][r].astype(np.float64),
                dt=DT_TC,
                max_lag=MAX_LAG_FRAMES,
            )
            cell_lags.append(lag_tc)
        lags[label] = np.asarray(cell_lags)
        print(
            f"[F7] {label:4s} ({CELL_STYLE[label]['run']}): n_enc={len(cell_lags)}, "
            f"median lag={np.median(lags[label]):+.3f} t/c, "
            f"median |lag|={np.median(np.abs(lags[label])):.3f} t/c"
        )

    # ---- figure -----------------------------------------------------------
    fig_w = fs.TEXTWIDTH_IN
    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(fig_w, fig_w * 0.40),
        gridspec_kw={"width_ratios": [1.7, 1.0], "wspace": 0.32},
    )

    # (a) traces
    w_idx = np.where(wmask)[0]
    ax_a.axvspan(
        t_tc[w_idx.min()], t_tc[w_idx.max()], color="0.88", zorder=0, label="impact + relax. window"
    )
    ax_a.plot(t_tc, truth, color="black", lw=1.2, label="DNS", zorder=5)
    for label, sty in CELL_STYLE.items():
        rows_c, _ = rep_encounter_rows(CACHE_DIR / f"latents_{sty['run']}_test_b.npz")
        tb = cells[label]["tb"]
        order_frames = tb["frame"][rows_c]
        assert np.array_equal(order_frames, frames), f"frame mismatch for {label}"
        ax_a.plot(
            order_frames * DT_TC,
            cells[label]["pred"][rows_c],
            color=sty["color"],
            ls=sty["ls"],
            lw=1.0,
            label=label,
            zorder=4,
        )
    ax_a.set_xlabel(r"$t/c$")
    ax_a.set_ylabel(r"$C_L$")
    ax_a.set_xlim(t_tc.min(), t_tc.max())
    ax_a.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=3,
        handlelength=1.9,
        columnspacing=0.9,
        borderaxespad=0.0,
    )
    ax_a.text(
        0.02,
        0.03,
        f"{REP_CASE}, enc {rep_k} (test B)",
        transform=ax_a.transAxes,
        fontsize=6.5,
        color="0.35",
    )
    ax_a.text(-0.16, 1.24, "(a)", transform=ax_a.transAxes, fontsize=8.5, fontweight="bold")

    # (b) phase-lag distributions
    labels = list(CELL_STYLE)
    data = [lags[lb] for lb in labels]
    positions = np.arange(len(labels))
    bp = ax_b.boxplot(
        data,
        positions=positions,
        widths=0.52,
        showfliers=False,
        medianprops={"color": "black", "lw": 1.1},
        boxprops={"lw": 0.8},
        whiskerprops={"lw": 0.8},
        capprops={"lw": 0.8},
        patch_artist=True,
    )
    rng = np.random.default_rng(0)
    for i, lb in enumerate(labels):
        c = CELL_STYLE[lb]["color"]
        bp["boxes"][i].set_facecolor(c)
        bp["boxes"][i].set_alpha(0.30)
        bp["boxes"][i].set_edgecolor(c)
        for part in ("whiskers", "caps"):
            for artist in (bp[part][2 * i], bp[part][2 * i + 1]):
                artist.set_color(c)
        x = positions[i] + rng.uniform(-0.13, 0.13, size=lags[lb].size)
        ax_b.scatter(x, lags[lb], s=4.5, color=c, alpha=0.65, lw=0, zorder=4)
    for y in (TAU_THRESH_TC, -TAU_THRESH_TC):
        ax_b.axhline(y, color="0.45", ls=(0, (4, 3)), lw=0.8, zorder=1)
    ax_b.text(
        -0.42,
        TAU_THRESH_TC + 0.02,
        r"$|\tau| = 0.1\,t/c$",
        fontsize=6.5,
        color="0.35",
        va="bottom",
        ha="left",
    )
    ax_b.set_xticks(positions)
    ax_b.set_xticklabels(labels)
    ax_b.set_ylabel(r"phase lag $\tau$ ($t/c$)")
    ax_b.text(
        0.03,
        0.03,
        f"test B, seed {SEED_TAG}, n = {lags[labels[0]].size} enc.",
        transform=ax_b.transAxes,
        fontsize=6.5,
        color="0.35",
    )
    ax_b.text(-0.28, 1.24, "(b)", transform=ax_b.transAxes, fontsize=8.5, fontweight="bold")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / f"{OUT_STEM}.pdf"
    png = OUT_DIR / f"{OUT_STEM}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    print(f"[F7] wrote {pdf}")
    print(f"[F7] wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
