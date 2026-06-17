"""SESSION29: "what information the JEPA latent carries" figure (v2.1).

Per-observable LATENT-READABILITY panel on the frozen v2.1 production latents,
framed honestly as COLLECTIVE (whole-latent) readability, NOT per-coordinate
semantics. For each family (predictive JEPA, reconstructive Fukami AE, POD), we
fit ONE linear readout (StandardScaler + Ridge) of the full 64-D latent to each
physical observable on the TRAIN per-frame rows, then report held-out R^2 on
test_b (and a faint test_c overlay).

HONESTY FRAMING (CLAUDE.md "Probe methodology" + subspace-overlap null):
    The bars measure how well the WHOLE 64-D latent linearly predicts each
    observable. They say nothing about individual coordinates: the PLS/PCA bases
    of this code are seed-arbitrary (pairwise cos^2 ~ 0.05 ~ K/d, indistinguishable
    from random rotations), so the wake information is a DISTRIBUTED code with no
    single informative axis. We never assign physical meaning to a latent dimension.

REGIME: PER-FRAME state-descriptor probe. Every valid (encounter, frame) row in a
split is pooled; the linear probe maps z_t (64,) -> observable_t. This is the
per-frame state-descriptor regime (CLAUDE.md): the latent's instantaneous flow
state is read out, NOT the gust parameters. The parameters (G, D, Y) are DELIBERATELY
EXCLUDED: their per-frame R^2 is known to be negative (the parameter content lives
at the IMPACT frame, not per frame), so plotting a per-frame parameter bar would
be misleading. Parameter readability is reported elsewhere as an impact-frame number.

PROBE CLASS: linear Ridge with StandardScaler, alpha selected by GroupKFold-by-case
on the TRAIN rows (same probe class as the paper's closure/probe headline). Linear
is the honest "collective readability" readout: it cannot exploit per-coordinate
nonlinearity, so a high bar means the information is linearly available in the
distributed code, not memorised by a flexible probe.

NB ON THE HEADLINE: the paper's headline JEPA wake-enstrophy R^2 (0.796) is a
POOLED-frame ridge at the readout frame impact+H=16 on the CANONICAL DNS metric
(dns_physical_metrics.npz). This figure's per-frame number (~0.62) is a DIFFERENT,
broader regime (all 120 frames, per_frame_targets observable) and is not the
headline. The two are not in conflict; they answer different questions. The staging
numbers JSON labels every key with this regime so nothing here can be confused with
the closure headline.

Inputs (frozen v2.1):
    outputs/session28/latents/<tag>/{train,test_b,test_c}.npz   (z_full + alignment)
    outputs/session28/exp2/per_frame_targets/{train,test_b,test_c}.npz  (DNS obs)
Outputs:
    paper/sections/figures/results/fig_latent_readability_v2p1.pdf
    outputs/session29/latent_readability_part.json   (STAGING numbers part)

CPU-only sklearn. No GPU, no encoder run (latents are pre-extracted).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "session29"))
sys.path.insert(0, str(REPO / "scripts" / "session21"))
import _s29_common as cm  # noqa: E402
import figstyle as fs  # noqa: E402

FIG_OUT = REPO / "paper" / "sections" / "figures" / "results" / "fig_latent_readability_v2p1.pdf"
JSON_OUT = REPO / "outputs" / "session29" / "latent_readability_part.json"

# Display order: state descriptors only. Parameters (G, D, Y) are excluded by design
# (per-frame parameter R^2 is negative; parameter content is an impact-frame quantity).
OBSERVABLES = [
    "wake_enstrophy",
    "circulation_pos",
    "circulation_neg",
    "peak_pos_omega",
    "peak_neg_omega",
    "centroid_x",
    "centroid_y",
    "C_L",
    "C_D",
]

# Compact axis labels (mathtext to match the body font); reuse figstyle where it has one.
OBS_LABEL = {
    "wake_enstrophy": r"$\Omega_w$",
    "circulation_pos": r"$\Gamma^{+}$",
    "circulation_neg": r"$\Gamma^{-}$",
    "peak_pos_omega": r"$\omega^{+}_{\max}$",
    "peak_neg_omega": r"$\omega^{-}_{\max}$",
    "centroid_x": r"$x_c$",
    "centroid_y": r"$y_c$",
    "C_L": r"$C_L$",
    "C_D": r"$C_D$",
}

# macro stems for the staging numbers JSON (camel-case, no I_y).
OBS_MACRO_STEM = {
    "wake_enstrophy": "WakeEnst",
    "circulation_pos": "CircPos",
    "circulation_neg": "CircNeg",
    "peak_pos_omega": "PeakPosOmega",
    "peak_neg_omega": "PeakNegOmega",
    "centroid_x": "CentroidX",
    "centroid_y": "CentroidY",
    "C_L": "Clift",
    "C_D": "Cdrag",
}

FAMILIES = ["jepa", "fukami", "pod"]  # canonical figstyle keys
# figstyle family key -> session29 latent tag.
FAMILY_TAG = {
    "jepa": "jepa_tf_noc_d64_s42",
    "fukami": "fukami_d64_s42",
    "pod": "pod_d64",
}


def per_frame_xy(tag: str, split: str, observable: str):
    """Pool every valid (encounter, frame) row: X = z_t (64,), y = DNS obs_t.

    Returns (X (N, d), y (N,), groups (N,) case_ids). One row per finite frame,
    grouped by case_id so the CV / bootstrap respect case clustering.
    """
    fam = cm.load_family(tag, split)
    obs_map, _ = cm.load_dns_observable(split, observable)
    z, cid, enc = fam["z_full"], fam["case_ids"], fam["encounter_indices"]
    n, T, d = z.shape
    X, y, groups = [], [], []
    for i in range(n):
        key = (cid[i], int(enc[i]))
        if key not in obs_map:
            continue
        ov = obs_map[key]
        finite = np.isfinite(ov)
        for t in range(T):
            if finite[t]:
                X.append(z[i, t, :])
                y.append(float(ov[t]))
                groups.append(cid[i])
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64), np.asarray(groups)


def fit_linear_probe(Xtr, ytr, gtr, seed: int = 0):
    """Linear collective readout: StandardScaler + Ridge, alpha by GroupKFold-by-case."""
    pipe = Pipeline([("scale", StandardScaler()), ("model", Ridge())])
    n_groups = len(np.unique(gtr))
    cv = GroupKFold(n_splits=min(5, n_groups))
    gs = GridSearchCV(
        pipe,
        {"model__alpha": [0.1, 1.0, 10.0, 100.0]},
        scoring="r2",
        cv=cv,
        n_jobs=-1,
    )
    gs.fit(Xtr, ytr, groups=gtr)
    return gs


def compute_all(seed: int = 0):
    """Returns results[obs][family] = dict(r2 split keys + CI on test_b)."""
    results: dict[str, dict[str, dict]] = {}
    for obs in OBSERVABLES:
        results[obs] = {}
        for fam in FAMILIES:
            tag = FAMILY_TAG[fam]
            Xtr, ytr, gtr = per_frame_xy(tag, "train", obs)
            Xtb, ytb, gtb = per_frame_xy(tag, "test_b", obs)
            Xtc, ytc, gtc = per_frame_xy(tag, "test_c", obs)
            gs = fit_linear_probe(Xtr, ytr, gtr, seed=seed)
            r2_tr = float(r2_score(ytr, gs.predict(Xtr)))
            yp_tb = gs.predict(Xtb)
            r2_tb = float(r2_score(ytb, yp_tb))
            r2_tc = float(r2_score(ytc, gs.predict(Xtc)))
            _, ci_lo, ci_hi = cm.case_clustered_r2_ci(ytb, yp_tb, gtb, seed=seed)
            results[obs][fam] = {
                "train": r2_tr,
                "test_b": r2_tb,
                "test_c": r2_tc,
                "test_b_ci_lo": float(ci_lo),
                "test_b_ci_hi": float(ci_hi),
                "n_test_b": int(len(ytb)),
                "n_cases_test_b": int(len(np.unique(gtb))),
                "best_alpha": float(gs.best_params_["model__alpha"]),
            }
            print(
                f"{obs:18s} {fam:7s} test_b={r2_tb:+.3f} "
                f"[{ci_lo:+.3f},{ci_hi:+.3f}] test_c={r2_tc:+.3f} train={r2_tr:+.3f}"
            )
    return results


def make_figure(results: dict) -> None:
    fs.use_style()
    w, h = fs.figure_size(1.0, aspect=0.50)
    fig, ax = plt.subplots(figsize=(w, h))

    n_obs = len(OBSERVABLES)
    xs = np.arange(n_obs)
    bar_w = 0.26

    for k, fam in enumerate(FAMILIES):
        tb = [results[o][fam]["test_b"] for o in OBSERVABLES]
        lo = [results[o][fam]["test_b_ci_lo"] for o in OBSERVABLES]
        hi = [results[o][fam]["test_b_ci_hi"] for o in OBSERVABLES]
        tc = [results[o][fam]["test_c"] for o in OBSERVABLES]
        offs = (k - 1) * bar_w
        # clamp lower error to the bar so the asymmetric (often very wide) lower CI
        # does not blow up the y-range; record full CI in JSON regardless.
        yerr_lo = [max(0.0, tb[i] - lo[i]) for i in range(n_obs)]
        yerr_hi = [max(0.0, hi[i] - tb[i]) for i in range(n_obs)]
        ax.bar(
            xs + offs,
            tb,
            bar_w,
            color=fs.FAMILY_COLOR[fam],
            edgecolor="white",
            linewidth=0.3,
            label=fs.FAMILY_LABEL[fam],
            zorder=2,
        )
        ax.errorbar(
            xs + offs,
            tb,
            yerr=[yerr_lo, yerr_hi],
            fmt="none",
            ecolor="#333333",
            elinewidth=0.5,
            capsize=1.2,
            zorder=3,
        )
        # faint test_c overlay: open marker at the bar centre
        ax.scatter(
            xs + offs,
            tc,
            marker=fs.FAMILY_MARKER[fam],
            s=10,
            facecolors="none",
            edgecolors=fs.FAMILY_COLOR[fam],
            linewidths=0.7,
            alpha=0.85,
            zorder=4,
        )

    ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.4, zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([OBS_LABEL[o] for o in OBSERVABLES])
    ax.set_ylabel(r"linear readout $R^2$ (full 64-D latent)")
    ax.set_ylim(-0.05, 1.0)
    ax.set_xlim(-0.6, n_obs - 0.4)
    ax.grid(True, axis="y", alpha=0.25, zorder=0)

    fam_handles = [plt.Rectangle((0, 0), 1, 1, color=fs.FAMILY_COLOR[f]) for f in FAMILIES]
    fam_labels = [fs.FAMILY_LABEL[f] for f in FAMILIES]
    tc_handle = plt.Line2D(
        [],
        [],
        marker="o",
        linestyle="none",
        markerfacecolor="none",
        markeredgecolor="#333333",
        markersize=4,
        label=r"test C ($|G|=4$)",
    )
    ax.legend(
        fam_handles + [tc_handle],
        fam_labels + [r"test C ($|G|=4$) overlay"],
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.14),
        columnspacing=1.0,
        handlelength=1.4,
        handletextpad=0.4,
    )

    fig.tight_layout()
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT)
    plt.close(fig)
    print(f"wrote {FIG_OUT.relative_to(REPO)}")


def write_numbers(results: dict, seed: int) -> None:
    """Staging numbers part (NOT under outputs/session28/numbers_parts)."""
    inputs = []
    for fam in FAMILIES:
        for sp in ("train", "test_b", "test_c"):
            inputs.append(cm.LATENTS / FAMILY_TAG[fam] / f"{sp}.npz")
    for sp in ("train", "test_b", "test_c"):
        inputs.append(cm.TARGETS / f"{sp}.npz")

    regime_note = (
        "PER-FRAME state-descriptor regime: linear collective readout "
        "(StandardScaler + Ridge, alpha by GroupKFold-by-case) of the FULL 64-D "
        "latent, pooled over all 120 frames, fit on train cases and evaluated on "
        "held-out test_b. COLLECTIVE whole-latent readability, NOT per-coordinate "
        "semantics. NB: distinct from the closure headline (pooled-frame ridge at "
        "readout frame impact+H=16 on the canonical DNS metric); do not confuse."
    )

    numbers: dict[str, dict] = {}

    def add(key, macro, value, ci_lo=None, ci_hi=None, obs=None, extra_note=""):
        rec = {
            "macro": macro,
            "value": float(value),
            "fmt": "%.2f",
            "split": "test_b",
            "source": "scripts/session29/make_fig_latent_readability_v2p1.py",
            "note": regime_note + (" " + extra_note if extra_note else ""),
        }
        if ci_lo is not None:
            rec["ci_lo"] = float(ci_lo)
            rec["ci_hi"] = float(ci_hi)
        if obs is not None:
            rec["observable"] = obs
            rec["n"] = int(results[obs]["jepa"]["n_test_b"])
        numbers[key] = rec

    # JEPA test_b R^2 for every observable (collective readability).
    for obs in OBSERVABLES:
        r = results[obs]["jepa"]
        stem = OBS_MACRO_STEM[obs]
        add(
            f"latread_jepa_{obs}",
            f"NumLatRead{stem}Jepa",
            r["test_b"],
            ci_lo=r["test_b_ci_lo"],
            ci_hi=r["test_b_ci_hi"],
            obs=obs,
        )

    # Weakest JEPA observable (the floor of the readability profile).
    weak_obs = min(OBSERVABLES, key=lambda o: results[o]["jepa"]["test_b"])
    rw = results[weak_obs]["jepa"]
    add(
        "latread_jepa_weakest",
        "NumLatReadWeakestJepa",
        rw["test_b"],
        ci_lo=rw["test_b_ci_lo"],
        ci_hi=rw["test_b_ci_hi"],
        obs=weak_obs,
        extra_note=f"Weakest collective-readability observable for JEPA on test_b: {weak_obs}.",
    )
    numbers["latread_jepa_weakest_name"] = {
        "macro": "NumLatReadWeakestJepaObs",
        "value": weak_obs,
        "fmt": "%s",
        "split": "test_b",
        "source": "scripts/session29/make_fig_latent_readability_v2p1.py",
        "note": "Name of the weakest collective-readability observable for JEPA on test_b.",
    }

    # Combined-circulation convenience macro (mean of Gamma+ and Gamma-).
    circ_mean = 0.5 * (
        results["circulation_pos"]["jepa"]["test_b"] + results["circulation_neg"]["jepa"]["test_b"]
    )
    add(
        "latread_jepa_circulation",
        "NumLatReadCirculationJepa",
        circ_mean,
        obs="circulation_pos",
        extra_note="Mean of the Gamma+ and Gamma- collective-readability R^2 for JEPA on test_b.",
    )

    payload = {
        "part": "latent_readability",
        "_staging": (
            "STAGING ONLY. Not under outputs/session28/numbers_parts to avoid racing "
            "the GO-gate job. Promote manually after review."
        ),
        "_provenance": cm.provenance(inputs, seed=seed),
        "config": {
            "regime": "per-frame state-descriptor, linear collective readout of full 64-D latent",
            "probe_class": "StandardScaler + Ridge (linear), alpha by GroupKFold-by-case",
            "observables": OBSERVABLES,
            "parameters_excluded": ["G", "D", "Y"],
            "parameters_excluded_reason": (
                "per-frame parameter R^2 is negative; parameter content is an "
                "impact-frame quantity, not a per-frame state descriptor."
            ),
            "families": {f: FAMILY_TAG[f] for f in FAMILIES},
        },
        "full_table": results,
        "numbers": numbers,
    }
    cm.write_artifact(JSON_OUT, payload)
    print(f"wrote {JSON_OUT.relative_to(REPO)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    results = compute_all(seed=args.seed)
    make_figure(results)
    write_numbers(results, seed=args.seed)


if __name__ == "__main__":
    main()
