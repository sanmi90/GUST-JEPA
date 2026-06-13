"""Physics Track P4: LEV circulation budget and gust-sign asymmetry (v2.1).

READ-ONLY analysis (CPU numpy/scipy; reuses the v2.1 DNS cache and the matched
decoded fields; no GPU, no training). Master plan Phase C, Physics Track P4;
extends D146 (the Session 23 LEV-tracking result, scripts/session23/
exp_lev_tracking.py) onto the v2.1 unconditioned rebuild with MATCHED decoders.

What this computes
------------------
PART A -- DNS LEV budget (the physics). Per test_b and test_c encounter, on the
large-scale DNS omega_z field (Gaussian band sigma/c = 0.05 = 1.6 px, EXACTLY
matching exp_lev_tracking / exp_scale_decomposition), identify the dominant
suction-side leading-edge vortex (thresholded NEGATIVE-vorticity connected
component attached to the suction side, x/c < 1; LEV-ID machinery REUSED verbatim
from scripts/session23/exp_lev_tracking.lev_from_field) per frame over the impact
window [25, 55], giving Gamma_LEV(t), the centroid path, and a detachment time.

  (i)  GATE GP4: held-out correlation between peak |Delta C_L| (the already-
       computed dcl_peak_simple from outputs/session28/physics/
       per_encounter_physics.npz) and peak Gamma_LEV across the envelope
       (test_b + test_c). |corr| >= 0.6 -> main-text S4.4 paragraph + one panel;
       below -> appendix. Pearson AND Spearman, case-clustered bootstrap CI, plus
       a case-permutation p (VALID here: this is a correlation / trend, not a
       paired location test; the B6 paired-location degeneracy does NOT apply).
  (ii) G-SIGN ASYMMETRY: positive vs negative gusts interact oppositely with the
       pre-existing suction-side vorticity (PRF describes split-and-merge
       qualitatively; we QUANTIFY the budget). Compare Gamma_LEV growth, peak,
       and detachment time for G > 0 vs G < 0 at matched |G|, with a rank test.
  (iii) Y MODULATION of (i) and (ii): stratify by |Y| (and Y sign).

PART B -- latent image (which family's decode tracks the LEV; regenerates D146 on
v2.1 with MATCHED decoders, removing the old-decoder confound). Per family
(tf_noc, fukami, pod) and per encounter, compute Gamma_LEV(t) from the family's
MATCHED decoded field over [25, 55] and correlate per-encounter with the DNS
Gamma_LEV(t). Report per family the fraction of encounters whose decoded
Gamma_LEV(t) tracks DNS (correlation above a stated threshold). The recon decode
(decode of the true encoded latent) is the headline -- the cleanest test of
whether the family's latent CARRIES the LEV; the forecast decode is noted
separately. Significance via sign test + case-clustered CI, NOT
case_permutation_p (that is for the trend test in (i), not the paired family
comparison).

HARD CONSTRAINT (D167 stands): NO impulse-theorem claim anywhere. We report only
the DIRECT CORRELATION between peak |Delta C_L| and peak Gamma_LEV; we never
assert the LEV circulation "explains" lift via an impulse argument.

Scale convention
----------------
exp_lev_tracking thresholds the large-scale PIPELINE-NORMALISED field at an
absolute iso-level OMEGA_THR = 0.5 (3-sigma normalised units). The v2.1 DNS cache
holds raw (mask + clip) omega_z, and the matched decoded fields are RAW
(pipeline-unnormalised; see omega_scale attr). We bring both onto the normalised
scale by dividing by 3 * train_std (train_std = 3.6337 for split_v2p1, from
outputs/data_pipeline/v2p1/manifest.json) before filtering and thresholding, so
the LEV definition is byte-for-byte the exp_lev_tracking convention.

Output
------
outputs/session28/p4_lev/p4_results.npz   (per-encounter arrays)
outputs/session28/p4_lev/p4_results.json  (all numbers + config + GP4 gate)
outputs/session28/p4_lev/fig_lev_budget.{pdf,png}  (3 panels)
outputs/session28/p4_lev/README.md
outputs/session28/numbers_parts/p4_lev.json        (eval_all macros)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "session28"))
sys.path.insert(0, str(REPO / "scripts" / "session23"))

import stats_lib  # noqa: E402

# REUSE the exp_lev_tracking LEV-ID machinery verbatim (D146). We import its
# lev_from_field, large_scale, and grid constants rather than reinventing them.
import exp_lev_tracking as elt  # noqa: E402

# ---- paths (all absolute under the repo) ------------------------------------
PHYSICS_NPZ = REPO / "outputs" / "session28" / "physics" / "per_encounter_physics.npz"
DECODE_ROOT = REPO / "outputs" / "session28" / "decode"
SPLIT_PATH = REPO / "configs" / "splits" / "split_v2p1.json"
PIPELINE_MANIFEST = REPO / "outputs" / "data_pipeline" / "v2p1" / "manifest.json"
OUT_DIR = REPO / "outputs" / "session28" / "p4_lev"
NUMBERS_PART = REPO / "outputs" / "session28" / "numbers_parts" / "p4_lev.json"

CACHE_ROOT = (
    Path(
        os.environ.get(
            "VORTEX_JEPA_CACHE",
            str(
                Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
                / "data"
                / "processed"
                / "vortex-jepa"
            ),
        )
    )
    / "v2p1"
)

# ---- frame window (around impact frame ~40); the LEV growth / detachment window.
WINDOW_FRAMES = np.arange(25, 56)  # [25, 55] inclusive == 31 frames
IMPACT_FRAME = 40
DT_TC = 0.05

# ---- families for the latent-image part (matched decoders). ------------------
# (decode subdir, canonical family label, figstyle family tag).
FAMILIES = [
    ("tf_noc", "tf_noc", "jepa"),
    ("fukami", "fukami", "fukami"),
    ("pod", "pod", "pod"),
]

# Per-encounter Gamma_LEV(t) tracking threshold for "the decode tracks DNS".
TRACK_CORR_THR = 0.5

# GP4 main/appendix branch threshold (master plan: |corr| >= 0.6).
GP4_THR = 0.6


# =============================================================================
# LEV budget on a single field stack (window of frames).
# =============================================================================
def lev_series(
    field_window_norm: np.ndarray,
    airfoil: np.ndarray,
    thr: float = elt.OMEGA_THR,
) -> dict:
    """Gamma_LEV(t), centroid path, peak |omega|, and detachment time.

    field_window_norm: (T, 192, 96) large-scale-FILTERABLE NORMALISED omega_z
    (raw scale already divided by 3 * train_std; filtering happens here, matching
    elt.large_scale). The suction-side LEV is the NEGATIVE-vorticity component.

    Returns a dict with arrays of length T:
      gamma (signed, negative when present, 0.0 when no lobe detected),
      x, y  (centroid in physical chords; NaN when no lobe),
      peak  (lobe |omega| max; 0.0 when no lobe),
      detected (bool),
    plus scalars:
      peak_gamma_abs (max |gamma| over the window),
      peak_gamma_frame (absolute frame index of the peak),
      detach_frame (circulation half-life detachment clock: first frame strictly
        after the peak where the lobe is lost OR |Gamma_LEV| < 0.5 * peak; NaN if
        the LEV is still strong at the end of the window),
      n_detected.
    """
    fL = elt.large_scale(field_window_norm)  # (T, 192, 96)
    T = fL.shape[0]
    gamma = np.zeros(T, dtype=np.float64)
    xs = np.full(T, np.nan)
    ys = np.full(T, np.nan)
    peak = np.zeros(T, dtype=np.float64)
    det = np.zeros(T, dtype=bool)
    for t in range(T):
        lobe = elt.lev_from_field(fL[t], airfoil, thr=thr)
        if lobe is None:
            continue
        det[t] = True
        gamma[t] = lobe["Gamma"]
        xs[t] = lobe["x"]
        ys[t] = lobe["y"]
        peak[t] = lobe["peak"]

    abs_g = np.abs(gamma)
    if det.any():
        peak_idx = int(np.argmax(abs_g))
        peak_gamma_abs = float(abs_g[peak_idx])
    else:
        peak_idx = -1
        peak_gamma_abs = 0.0

    # Detachment / shedding clock. The LEV-ID region is confined to x/c < 1 (the
    # search zeroes everything at TE_X_INDEX), so the centroid can never cross
    # x/c = 1 by construction; detachment must be read from the circulation budget
    # collapsing instead. We define detachment as the first frame strictly after
    # the peak where EITHER the lobe is lost (the LEV has pinched off / left the
    # LE region) OR |Gamma_LEV| falls below 50% of its peak (the LE circulation
    # has shed downstream). This is a robust "circulation half-life" proxy within
    # the window; reported in t/c relative to impact. NaN if neither happens (the
    # LEV is still strong at frame 55).
    detach_frame = float("nan")
    detach_idx = -1
    if peak_idx >= 0:
        half = 0.5 * peak_gamma_abs
        for t in range(peak_idx + 1, T):
            lobe_lost = not det[t]
            below_half = abs_g[t] < half
            if lobe_lost or below_half:
                detach_idx = t
                break
        if detach_idx >= 0:
            detach_frame = float(WINDOW_FRAMES[detach_idx])

    detach_tc = (
        float("nan") if not np.isfinite(detach_frame) else (detach_frame - IMPACT_FRAME) * DT_TC
    )

    return {
        "gamma": gamma,
        "x": xs,
        "y": ys,
        "peak": peak,
        "detected": det,
        "peak_gamma_abs": peak_gamma_abs,
        "peak_gamma_frame": int(WINDOW_FRAMES[peak_idx]) if peak_idx >= 0 else -1,
        "detach_frame": detach_frame,
        "detach_tc": detach_tc,
        "n_detected": int(det.sum()),
    }


# =============================================================================
# Data loading.
# =============================================================================
def load_pipeline_norm_divisor() -> float:
    """3 * train_std for split_v2p1 (the 3-sigma normalisation divisor)."""
    m = json.loads(PIPELINE_MANIFEST.read_text())
    return 3.0 * float(m["train_stats"]["std"])


def load_split() -> dict:
    return json.loads(SPLIT_PATH.read_text())


def gdy_lookup(split: dict) -> dict:
    """case_id -> (G, D, Y)."""
    out = {}
    for cid, c in split["cases"].items():
        out[cid] = (float(c["G"]), float(c["D"]), float(c["Y"]))
    return out


def load_physics_peak_dcl() -> dict:
    """(case_id, encounter_index) -> dcl_peak_simple (headline peak |Delta C_L|)."""
    d = np.load(PHYSICS_NPZ, allow_pickle=True)
    cid = d["case_id"]
    eidx = d["encounter_index"]
    dcl = d["dcl_peak_simple"]
    dclpm = d["dcl_peak_phase_matched"]
    out = {}
    for i in range(len(cid)):
        out[(str(cid[i]), int(eidx[i]))] = (float(dcl[i]), float(dclpm[i]))
    return out


def load_dns_window(case_id: str, enc: int, divisor: float) -> np.ndarray:
    """Pipeline-normalised DNS omega_z over WINDOW_FRAMES, shape (31, 192, 96).

    The v2.1 cache holds raw (mask + clip) omega_z. We divide by 3 * train_std to
    bring it onto the normalised scale exp_lev_tracking thresholds at OMEGA_THR.
    """
    import h5py

    path = CACHE_ROOT / case_id / f"encounter_{enc:02d}.h5"
    with h5py.File(path, "r") as h:
        om = h["omega_z"][WINDOW_FRAMES[0] : WINDOW_FRAMES[-1] + 1]  # (31, 192, 96)
    return om.astype(np.float64) / divisor


def load_decode(family_dir: str, endpoint: str, split: str) -> dict | None:
    """Load a matched decode npz; return None if missing.

    endpoint in {"recon", "forecast"}. Returns dict with omega_decoded_window
    (n, 31, 192, 96) RAW, case_ids, encounter_indices, window_frames.
    """
    path = DECODE_ROOT / family_dir / f"{endpoint}_{split}.npz"
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    return {
        "omega": d["omega_decoded_window"].astype(np.float64),
        "case_ids": np.array([str(c) for c in d["case_ids"]]),
        "enc": np.array([int(e) for e in d["encounter_indices"]]),
        "window_frames": np.array([int(f) for f in d["window_frames"]]),
    }


# =============================================================================
# PART A: DNS LEV budget.
# =============================================================================
def compute_dns_budget(
    splits: tuple[str, ...], gdy: dict, peak_dcl: dict, divisor: float, airfoil: np.ndarray
) -> dict:
    """Per-encounter DNS LEV budget over test_b + test_c.

    Iterates over the cache encounters of every test_b / test_c case, computing
    the DNS Gamma_LEV(t) series and the matched peak |Delta C_L|.
    """
    split_def = load_split()
    rows: list[dict] = []
    # which cases are in each requested split
    case_split = {}
    for s in splits:
        key = "test_b_cases" if s == "test_b" else "test_c_cases"
        for cid in split_def[key]:
            case_split[cid] = s

    for cid, s in case_split.items():
        G, D, Y = gdy[cid]
        cdir = CACHE_ROOT / cid
        enc_files = sorted(cdir.glob("encounter_*.h5"))
        for ef in enc_files:
            enc = int(ef.stem.split("_")[1])
            key = (cid, enc)
            if key not in peak_dcl:
                continue  # only encounters with a computed peak |Delta C_L|
            field = load_dns_window(cid, enc, divisor)
            ser = lev_series(field, airfoil)
            dcl_simple, dcl_pm = peak_dcl[key]
            rows.append(
                {
                    "case_id": cid,
                    "enc": enc,
                    "split": s,
                    "G": G,
                    "D": D,
                    "Y": Y,
                    "dcl_peak_simple": dcl_simple,
                    "dcl_peak_phase_matched": dcl_pm,
                    "peak_gamma_abs": ser["peak_gamma_abs"],
                    "peak_gamma_frame": ser["peak_gamma_frame"],
                    "detach_tc": ser["detach_tc"],
                    "n_detected": ser["n_detected"],
                    "gamma_series": ser["gamma"],
                }
            )
    return {"rows": rows}


def gp4_correlation(rows: list[dict], rng) -> dict:
    """GATE GP4: corr(peak |Delta C_L|, peak Gamma_LEV) over the held-out envelope.

    Uses peak |Delta C_L| = dcl_peak_simple (headline) and the DNS peak
    |Gamma_LEV|. Reports Pearson AND Spearman, a case-clustered bootstrap CI on
    each coefficient, and a case-permutation p (valid trend test).
    """
    # encounters with a detected LEV (peak_gamma_abs > 0) are the support.
    sub = [r for r in rows if r["n_detected"] > 0]
    x = np.array([r["dcl_peak_simple"] for r in sub])  # peak |Delta C_L|
    y = np.array([r["peak_gamma_abs"] for r in sub])  # peak |Gamma_LEV|
    cids = np.array([r["case_id"] for r in sub])

    pear_r, pear_p = pearsonr(x, y)
    spear_r, spear_p = spearmanr(x, y)

    # case-clustered bootstrap CI on each coefficient (resample CASES).
    def _boot_corr(corr_fn, n_boot=10000):
        uc = np.array(sorted(set(cids.tolist())))
        by = {c: np.where(cids == c)[0] for c in uc}
        vals = []
        for _ in range(n_boot):
            pick = uc[rng.integers(0, len(uc), len(uc))]
            idx = np.concatenate([by[c] for c in pick])
            if np.std(x[idx]) < 1e-12 or np.std(y[idx]) < 1e-12:
                continue
            vals.append(corr_fn(x[idx], y[idx])[0])
        vals = np.asarray(vals)
        if vals.size < 10:
            return [float("nan"), float("nan")]
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

    pear_ci = _boot_corr(pearsonr)
    spear_ci = _boot_corr(spearmanr)

    # case-permutation p (Pearson + Spearman) -- VALID trend test (not B6 paired).
    perm_pear = stats_lib.case_permutation_p(
        x,
        y,
        cids,
        lambda a, b: pearsonr(a, b)[0],
        n_perm=10000,
        rng=np.random.default_rng(12345),
        alternative="two-sided",
    )
    perm_spear = stats_lib.case_permutation_p(
        x,
        y,
        cids,
        lambda a, b: spearmanr(a, b)[0],
        n_perm=10000,
        rng=np.random.default_rng(54321),
        alternative="two-sided",
    )

    headline = abs(pear_r)
    branch = "main_text" if headline >= GP4_THR else "appendix"
    return {
        "n": len(sub),
        "n_cases": int(len(set(cids.tolist()))),
        "pearson_r": float(pear_r),
        "pearson_p_raw": float(pear_p),
        "pearson_ci": pear_ci,
        "pearson_perm_p": perm_pear["p"],
        "spearman_r": float(spear_r),
        "spearman_p_raw": float(spear_p),
        "spearman_ci": spear_ci,
        "spearman_perm_p": perm_spear["p"],
        "gp4_threshold": GP4_THR,
        "abs_corr_headline": float(headline),
        "branch": branch,
        "headline_metric": "abs(pearson_r) of dcl_peak_simple vs peak |Gamma_LEV|",
    }


def gsign_asymmetry(rows: list[dict], rng) -> dict:
    """G-SIGN ASYMMETRY: G>0 vs G<0 at matched |G|.

    Pairs encounters by |G| bucket. Reports peak Gamma_LEV, detachment time, and
    growth (peak Gamma_LEV / window) for each sign, the asymmetry delta, and a
    Mann-Whitney-style rank test plus a case-clustered CI on the per-case mean
    difference within matched |G|.
    """
    from scipy.stats import mannwhitneyu

    sub = [r for r in rows if r["n_detected"] > 0 and abs(r["G"]) > 1e-9]
    absG = np.array([round(abs(r["G"]), 4) for r in sub])
    matched = sorted(set(absG[np.array([_has_both_signs(sub, g) for g in absG])].tolist()))

    pos = [r for r in sub if r["G"] > 0 and round(abs(r["G"]), 4) in matched]
    neg = [r for r in sub if r["G"] < 0 and round(abs(r["G"]), 4) in matched]

    def _arr(group, key):
        return np.array([r[key] for r in group], dtype=np.float64)

    res = {"matched_absG": matched, "n_pos": len(pos), "n_neg": len(neg)}
    for metric in ("peak_gamma_abs", "detach_tc"):
        gp = _arr(pos, metric)
        gn = _arr(neg, metric)
        gp_v = gp[np.isfinite(gp)]
        gn_v = gn[np.isfinite(gn)]
        if gp_v.size >= 2 and gn_v.size >= 2:
            u, p = mannwhitneyu(gp_v, gn_v, alternative="two-sided")
        else:
            u, p = float("nan"), float("nan")
        res[metric] = {
            "pos_mean": float(np.nanmean(gp)) if gp.size else float("nan"),
            "neg_mean": float(np.nanmean(gn)) if gn.size else float("nan"),
            "pos_median": float(np.nanmedian(gp)) if gp.size else float("nan"),
            "neg_median": float(np.nanmedian(gn)) if gn.size else float("nan"),
            "asymmetry_pos_minus_neg": (
                float(np.nanmean(gp) - np.nanmean(gn)) if gp.size and gn.size else float("nan")
            ),
            "mannwhitney_u": float(u),
            "mannwhitney_p_two_sided": float(p),
            "n_pos": int(gp_v.size),
            "n_neg": int(gn_v.size),
        }

    # Case-clustered CI: per matched-|G| paired difference (pos case-mean minus
    # neg case-mean) on peak Gamma_LEV, bootstrapped over the matched |G| buckets.
    diffs = []
    for g in matched:
        gp = _arr([r for r in pos if round(abs(r["G"]), 4) == g], "peak_gamma_abs")
        gn = _arr([r for r in neg if round(abs(r["G"]), 4) == g], "peak_gamma_abs")
        if gp.size and gn.size:
            diffs.append(float(gp.mean() - gn.mean()))
    diffs = np.asarray(diffs)
    if diffs.size >= 2:
        boot = np.array(
            [diffs[rng.integers(0, diffs.size, diffs.size)].mean() for _ in range(10000)]
        )
        ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        mean_diff = float(diffs.mean())
    else:
        ci = [float("nan"), float("nan")]
        mean_diff = float(diffs.mean()) if diffs.size else float("nan")
    res["matched_paired_peak_gamma_diff_mean"] = mean_diff
    res["matched_paired_peak_gamma_diff_ci"] = ci
    res["n_matched_buckets"] = int(diffs.size)
    return res


def _has_both_signs(sub: list[dict], g: float) -> bool:
    g = round(g, 4)
    has_pos = any(r["G"] > 0 and round(abs(r["G"]), 4) == g for r in sub)
    has_neg = any(r["G"] < 0 and round(abs(r["G"]), 4) == g for r in sub)
    return has_pos and has_neg


def y_modulation(rows: list[dict]) -> dict:
    """Y MODULATION of (i) peak Gamma_LEV and (ii) the DeltaC_L-Gamma correlation."""
    sub = [r for r in rows if r["n_detected"] > 0]
    absY = np.array([round(abs(r["Y"]), 4) for r in sub])
    out = {"by_absY": {}, "by_Ysign": {}}
    for ay in sorted(set(absY.tolist())):
        grp = [r for r in sub if round(abs(r["Y"]), 4) == ay]
        x = np.array([r["dcl_peak_simple"] for r in grp])
        y = np.array([r["peak_gamma_abs"] for r in grp])
        r_pear = (
            float(pearsonr(x, y)[0])
            if len(grp) >= 3 and x.std() > 0 and y.std() > 0
            else float("nan")
        )
        out["by_absY"][f"{ay:.2f}"] = {
            "n": len(grp),
            "mean_peak_gamma": float(y.mean()),
            "corr_dcl_gamma": r_pear,
        }
    for sign, sel in (
        ("Y_pos", lambda r: r["Y"] > 1e-9),
        ("Y_neg", lambda r: r["Y"] < -1e-9),
        ("Y_zero", lambda r: abs(r["Y"]) <= 1e-9),
    ):
        grp = [r for r in sub if sel(r)]
        if not grp:
            continue
        x = np.array([r["dcl_peak_simple"] for r in grp])
        y = np.array([r["peak_gamma_abs"] for r in grp])
        r_pear = (
            float(pearsonr(x, y)[0])
            if len(grp) >= 3 and x.std() > 0 and y.std() > 0
            else float("nan")
        )
        out["by_Ysign"][sign] = {
            "n": len(grp),
            "mean_peak_gamma": float(y.mean()),
            "corr_dcl_gamma": r_pear,
        }
    return out


# =============================================================================
# PART B: latent image (matched-decoder LEV tracking).
# =============================================================================
def compute_latent_tracking(
    splits: tuple[str, ...], divisor: float, airfoil: np.ndarray, dns_rows: list[dict]
) -> dict:
    """Per family, fraction of encounters whose decoded Gamma_LEV(t) tracks DNS.

    For each (family, endpoint, split) we recompute Gamma_LEV(t) over [25, 55]
    from the matched decoded field and correlate (Pearson) the time series with
    the DNS Gamma_LEV(t) of the SAME encounter. "Tracks" = corr > TRACK_CORR_THR.
    Significance via sign test (family vs family) + case-clustered CI on the
    per-encounter tracking-correlation; NOT case_permutation_p.
    """
    # DNS gamma series keyed by (case_id, enc).
    dns_gamma = {(r["case_id"], r["enc"]): r["gamma_series"] for r in dns_rows}

    results: dict = {"endpoints": {}}
    per_family_corr: dict = {}  # (endpoint) -> family -> {(cid,enc): corr}

    for endpoint in ("recon", "forecast"):
        results["endpoints"][endpoint] = {}
        per_family_corr[endpoint] = {fam: {} for _, fam, _ in FAMILIES}
        for fam_dir, fam, _tag in FAMILIES:
            corrs = []
            cids_track = []
            n_dns_detected = 0
            n_tracked = 0
            n_total = 0
            for split in splits:
                blob = load_decode(fam_dir, endpoint, split)
                if blob is None:
                    continue
                om_all = blob["omega"] / divisor  # normalise to 3-sigma scale
                for i in range(len(blob["case_ids"])):
                    cid = blob["case_ids"][i]
                    enc = int(blob["enc"][i])
                    key = (cid, enc)
                    if key not in dns_gamma:
                        continue
                    dns_g = dns_gamma[key]
                    if np.count_nonzero(dns_g) < 3:
                        continue  # DNS LEV not present enough to define a series
                    n_dns_detected += 1
                    ser = lev_series(om_all[i], airfoil)
                    dec_g = ser["gamma"]
                    n_total += 1
                    # Pearson over the window; needs variation in both series.
                    if np.std(dec_g) < 1e-12 or np.std(dns_g) < 1e-12:
                        c = float("nan")
                    else:
                        c = float(pearsonr(dec_g, dns_g)[0])
                    corrs.append(c)
                    cids_track.append(cid)
                    per_family_corr[endpoint][fam][key] = c
                    if np.isfinite(c) and c > TRACK_CORR_THR:
                        n_tracked += 1
            corrs_arr = np.array(corrs, dtype=np.float64)
            finite = corrs_arr[np.isfinite(corrs_arr)]
            results["endpoints"][endpoint][fam] = {
                "n_encounters": n_total,
                "n_dns_lev_present": n_dns_detected,
                "n_tracked": n_tracked,
                "tracking_fraction": (n_tracked / n_total) if n_total else float("nan"),
                "tracking_fraction_str": f"{n_tracked}/{n_total}",
                "mean_corr": float(np.nanmean(finite)) if finite.size else float("nan"),
                "median_corr": float(np.nanmedian(finite)) if finite.size else float("nan"),
                "track_corr_threshold": TRACK_CORR_THR,
            }

    # Sign-test + case-clustered CI on the recon paired family comparison
    # (jepa tf_noc vs fukami): per-encounter tracking correlation difference.
    paired = _paired_family_sign(per_family_corr["recon"], "tf_noc", "fukami")
    results["recon_paired_tf_vs_fukami"] = paired
    return results


def _paired_family_sign(corr_by_fam: dict, fam_a: str, fam_b: str) -> dict:
    """Sign test that fam_a tracking-corr > fam_b on shared encounters, + CI."""
    keys = sorted(set(corr_by_fam[fam_a]) & set(corr_by_fam[fam_b]))
    a = np.array([corr_by_fam[fam_a][k] for k in keys], dtype=np.float64)
    b = np.array([corr_by_fam[fam_b][k] for k in keys], dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    keys = [k for k, m in zip(keys, ok) if m]
    cids = np.array([k[0] for k in keys])
    if a.size == 0:
        return {"n": 0}
    # sign test: a "better" (higher tracking corr) than b
    k, n_eff, p = stats_lib.sign_test_one_sided(-a, -b)  # err_a < err_b == a > b
    delta = a - b
    rng = np.random.default_rng(2026)
    cci = (
        stats_lib.case_cluster_bootstrap(delta, cids, n_boot=10000, rng=rng)
        if len(set(cids.tolist())) >= 2
        else None
    )
    return {
        "n": int(a.size),
        "fam_a": fam_a,
        "fam_b": fam_b,
        "a_higher_corr_count": int(k),
        "n_effective": int(n_eff),
        "sign_p_one_sided": float(p),
        "mean_delta_corr": float(delta.mean()),
        "case_cluster_ci": (cci["enc_mean_ci"] if cci else None),
        "case_cluster_case_mean_ci": (cci["case_mean_ci"] if cci else None),
    }


# =============================================================================
# Figure (representative median encounter; Fig-3 field convention).
# =============================================================================
def pick_representative(dns_rows: list[dict]) -> dict:
    """A representative MEDIAN encounter (not the worst) for the field panel.

    Among test_b encounters with a detected LEV, pick the one whose peak
    |Gamma_LEV| is closest to the test_b median (the representative-not-worst
    showcase rule).
    """
    sub = [r for r in dns_rows if r["split"] == "test_b" and r["n_detected"] > 0]
    if not sub:
        sub = [r for r in dns_rows if r["n_detected"] > 0]
    vals = np.array([r["peak_gamma_abs"] for r in sub])
    med = np.median(vals)
    return sub[int(np.argmin(np.abs(vals - med)))]


def make_figure(
    dns_rows: list[dict], gp4: dict, latent: dict, divisor: float, airfoil: np.ndarray
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(REPO / "scripts" / "session21"))
    import figstyle as fs  # noqa: E402

    fs.use_style()
    fig, axes = plt.subplots(1, 3, figsize=(fs.TEXTWIDTH_IN, fs.TEXTWIDTH_IN * 0.36))

    tloc = (WINDOW_FRAMES - IMPACT_FRAME) * DT_TC  # t/c relative to impact

    # Panel (a): Gamma_LEV(t) growth curves by gust sign (DNS, test_b+test_c).
    ax = axes[0]
    sub = [r for r in dns_rows if r["n_detected"] > 0]
    pos = [r["gamma_series"] for r in sub if r["G"] > 1e-9]
    neg = [r["gamma_series"] for r in sub if r["G"] < -1e-9]

    def _band(ax, series, color, label):
        if not series:
            return
        arr = np.array(series)  # (n, T) signed gamma
        m = np.nanmean(arr, axis=0)
        lo = np.nanpercentile(arr, 25, axis=0)
        hi = np.nanpercentile(arr, 75, axis=0)
        ax.fill_between(tloc, lo, hi, color=color, alpha=0.18, lw=0)
        ax.plot(tloc, m, color=color, lw=1.2, label=label)

    _band(ax, pos, "#b2182b", r"$G>0$")
    _band(ax, neg, "#2166ac", r"$G<0$")
    ax.axvline(0.0, color="0.4", lw=0.6, ls=":")
    ax.set_xlabel(r"$t/c$ from impact")
    ax.set_ylabel(r"$\Gamma_{\rm LEV}(t)$ (norm.)")
    ax.set_title("(a) LEV growth by gust sign", fontsize=7.5)
    ax.legend(fontsize=6.5, loc="lower left")

    # Panel (b): peak |Delta C_L| vs peak Gamma_LEV scatter coloured by G + corr.
    ax = axes[1]
    x = np.array([r["dcl_peak_simple"] for r in sub])
    y = np.array([r["peak_gamma_abs"] for r in sub])
    gcol = np.array([r["G"] for r in sub])
    sc = ax.scatter(x, y, c=gcol, cmap="coolwarm", s=14, edgecolors="none", alpha=0.85)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$G$", fontsize=7)
    ax.set_xlabel(r"peak $|\Delta C_L|$")
    ax.set_ylabel(r"peak $|\Gamma_{\rm LEV}|$")
    ax.set_title(
        f"(b) Pearson $r={gp4['pearson_r']:.2f}$ ({gp4['branch'].split('_')[0]})",
        fontsize=7.5,
    )

    # Panel (c): per-family decoded-vs-DNS Gamma_LEV tracking (recon, fractions).
    ax = axes[2]
    fams = [f for _, f, _ in FAMILIES]
    tags = {f: t for _, f, t in FAMILIES}
    recon = latent["endpoints"]["recon"]
    fr = [recon[f]["tracking_fraction"] for f in fams]
    cols = [fs.FAMILY_COLOR[tags[f]] for f in fams]
    labels = [fs.FAMILY_LABEL[tags[f]] for f in fams]
    bars = ax.bar(range(len(fams)), fr, color=cols, width=0.6)
    for i, f in enumerate(fams):
        ax.text(
            i,
            fr[i] + 0.02,
            recon[f]["tracking_fraction_str"],
            ha="center",
            va="bottom",
            fontsize=6.0,
        )
    ax.axhline(0.5, color="0.5", lw=0.6, ls="--")
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels(labels, fontsize=6.0, rotation=12, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel(r"frac. tracking DNS $\Gamma_{\rm LEV}$")
    ax.set_title(f"(c) recon decode (corr $>{TRACK_CORR_THR}$)", fontsize=7.5)
    del bars

    fig.tight_layout(pad=0.5, w_pad=1.4)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig_lev_budget.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[p4] wrote figure {OUT_DIR / 'fig_lev_budget.png'}")

    # Also save the representative-encounter field panel (Fig-3 convention) as a
    # small inset companion (RdBu_r, vmin=-3/vmax=+3, black airfoil).
    rep = pick_representative(dns_rows)
    field = load_dns_window(rep["case_id"], rep["enc"], divisor)
    # peak-Gamma frame within the window
    pk = rep["peak_gamma_frame"]
    fidx = int(np.clip(pk - WINDOW_FRAMES[0], 0, len(WINDOW_FRAMES) - 1)) if pk >= 0 else 15
    fL = elt.large_scale(field[fidx])
    fig2, ax2 = plt.subplots(figsize=(fs.TEXTWIDTH_IN * 0.42, fs.TEXTWIDTH_IN * 0.42 * 0.5))
    fs.vort_panel(ax2, fL)
    ax2.set_title(
        f"DNS large-scale $\\omega_z$ (representative: {rep['case_id']} enc{rep['enc']}, "
        f"frame {WINDOW_FRAMES[fidx]})",
        fontsize=6.0,
    )
    for ext in ("png", "pdf"):
        fig2.savefig(OUT_DIR / f"fig_lev_repfield.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"[p4] wrote representative field panel {OUT_DIR / 'fig_lev_repfield.png'}")


# =============================================================================
# Serialisation: results json/npz + eval_all numbers part + README.
# =============================================================================
def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items() if not k.startswith("gamma_series")}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def write_outputs(
    dns_rows: list[dict], gp4: dict, gsign: dict, ymod: dict, latent: dict, config: dict
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- npz (per-encounter arrays) ----
    keys = [
        "case_id",
        "enc",
        "split",
        "G",
        "D",
        "Y",
        "dcl_peak_simple",
        "dcl_peak_phase_matched",
        "peak_gamma_abs",
        "peak_gamma_frame",
        "detach_tc",
        "n_detected",
    ]
    npz = {k: np.array([r[k] for r in dns_rows]) for k in keys}
    npz["gamma_series"] = np.array([r["gamma_series"] for r in dns_rows])
    npz["window_frames"] = WINDOW_FRAMES
    np.savez(OUT_DIR / "p4_results.npz", **npz)
    print(f"[p4] wrote {OUT_DIR / 'p4_results.npz'}")

    # ---- json (all numbers) ----
    serial = {
        "config": config,
        "no_impulse_claim_note": (
            "D167 stands: we report ONLY the direct correlation between peak "
            "|Delta C_L| and peak |Gamma_LEV|. No impulse-theorem argument; we do "
            "NOT claim the LEV circulation explains lift."
        ),
        "gp4_gate": _clean(gp4),
        "gsign_asymmetry": _clean(gsign),
        "y_modulation": _clean(ymod),
        "latent_tracking": _clean(latent),
        "n_encounters": len(dns_rows),
    }
    (OUT_DIR / "p4_results.json").write_text(json.dumps(serial, indent=2))
    print(f"[p4] wrote {OUT_DIR / 'p4_results.json'}")

    # ---- eval_all numbers part ----
    recon = latent["endpoints"]["recon"]
    numbers = {
        "p4_gp4_pearson_r": {
            "macro": "NumPfourGPfourPearson",
            "value": gp4["pearson_r"],
            "fmt": "%.2f",
            "ci_lo": gp4["pearson_ci"][0],
            "ci_hi": gp4["pearson_ci"][1],
            "n": gp4["n"],
            "split": "test_b+test_c",
            "observable": "peak|dCL| vs peak|Gamma_LEV|",
            "note": (
                f"case-permutation two-sided p = {gp4['pearson_perm_p']:.4g}; "
                f"branch = {gp4['branch']} (GP4 threshold {GP4_THR}); "
                "direct correlation only, no impulse claim (D167)."
            ),
        },
        "p4_gp4_spearman_r": {
            "macro": "NumPfourGPfourSpearman",
            "value": gp4["spearman_r"],
            "fmt": "%.2f",
            "ci_lo": gp4["spearman_ci"][0],
            "ci_hi": gp4["spearman_ci"][1],
            "n": gp4["n"],
            "split": "test_b+test_c",
            "observable": "peak|dCL| vs peak|Gamma_LEV|",
            "note": f"case-permutation two-sided p = {gp4['spearman_perm_p']:.4g}",
        },
        "p4_gsign_peak_gamma_diff": {
            "macro": "NumPfourGsignPeakGammaDiff",
            "value": gsign["peak_gamma_abs"]["asymmetry_pos_minus_neg"],
            "fmt": "%.3f",
            "n": gsign["n_pos"] + gsign["n_neg"],
            "split": "test_b+test_c",
            "observable": "peak|Gamma_LEV| (G>0 minus G<0, matched |G|)",
            "note": (
                f"Mann-Whitney two-sided p = "
                f"{gsign['peak_gamma_abs']['mannwhitney_p_two_sided']:.4g}; "
                f"matched-bucket paired diff CI = "
                f"[{gsign['matched_paired_peak_gamma_diff_ci'][0]:.3f}, "
                f"{gsign['matched_paired_peak_gamma_diff_ci'][1]:.3f}]"
            ),
        },
        "p4_track_frac_tf_noc_recon": {
            "macro": "NumPfourTrackTfNocRecon",
            "value": recon["tf_noc"]["tracking_fraction"],
            "fmt": "%.2f",
            "n": recon["tf_noc"]["n_encounters"],
            "split": "test_b+test_c",
            "endpoint": "recon",
            "observable": "decoded Gamma_LEV(t) tracks DNS",
            "note": (
                f"{recon['tf_noc']['tracking_fraction_str']} encounters "
                f"(corr > {TRACK_CORR_THR}); D146 regenerated on matched decoders."
            ),
        },
        "p4_track_frac_fukami_recon": {
            "macro": "NumPfourTrackFukamiRecon",
            "value": recon["fukami"]["tracking_fraction"],
            "fmt": "%.2f",
            "n": recon["fukami"]["n_encounters"],
            "split": "test_b+test_c",
            "endpoint": "recon",
            "observable": "decoded Gamma_LEV(t) tracks DNS",
            "note": f"{recon['fukami']['tracking_fraction_str']} encounters",
        },
        "p4_track_frac_pod_recon": {
            "macro": "NumPfourTrackPodRecon",
            "value": recon["pod"]["tracking_fraction"],
            "fmt": "%.2f",
            "n": recon["pod"]["n_encounters"],
            "split": "test_b+test_c",
            "endpoint": "recon",
            "observable": "decoded Gamma_LEV(t) tracks DNS",
            "note": f"{recon['pod']['tracking_fraction_str']} encounters",
        },
    }
    NUMBERS_PART.parent.mkdir(parents=True, exist_ok=True)
    NUMBERS_PART.write_text(json.dumps({"part": "p4_lev", "numbers": numbers}, indent=2))
    print(f"[p4] wrote {NUMBERS_PART}")

    _write_readme(gp4, gsign, ymod, latent, config)


def _write_readme(gp4: dict, gsign: dict, ymod: dict, latent: dict, config: dict) -> None:
    recon = latent["endpoints"]["recon"]
    fc = latent["endpoints"]["forecast"]
    lines = []
    lines.append("# Physics Track P4: LEV circulation budget and gust-sign asymmetry (v2.1)\n")
    lines.append(
        "Read-only CPU analysis. Extends D146 (Session 23 LEV tracking) onto the "
        "v2.1 unconditioned rebuild with MATCHED decoders. Master plan Phase C, "
        "Physics Track P4.\n"
    )
    lines.append("## LEV definition (REUSED from scripts/session23/exp_lev_tracking.py)\n")
    lines.append(
        "The dominant suction-side leading-edge vortex is the strongest connected "
        "NEGATIVE-vorticity component (omega_z < 0, clockwise; convention "
        "du/dy - dv/dx) in the leading-edge / wing region x/c < 1 (streamwise pixel "
        f"index < {elt.TE_X_INDEX}), with the frozen 140-cell airfoil-adjacent mask "
        "removed, on the LARGE-SCALE band (Gaussian filter sigma/c = 0.05 = "
        f"{elt.SIGMA_PX:.1f} px, mode='nearest', matching exp_scale_decomposition). "
        "Identification uses an ABSOLUTE iso-level |omega_L| > "
        f"{elt.OMEGA_THR} in 3-sigma normalised units (fair across families). "
        "Gamma_LEV = sum_lobe omega_z * dx * dy (signed, normalised); the centroid "
        "is the |omega|-weighted center_of_mass. The detachment clock is a "
        "circulation half-life proxy: the first frame strictly after the peak "
        "where the lobe is lost OR |Gamma_LEV| falls below 50% of its peak (the "
        "LE-region search confines the centroid to x/c < 1 by construction, so the "
        "shedding signal must be read from the circulation budget collapsing, not "
        "from a centroid crossing). The exp_lev_tracking lev_from_field and "
        "large_scale functions are imported verbatim; only the per-frame time "
        "series, the detachment clock, and the DNS-vs-decode budget around them "
        "are new.\n"
    )
    lines.append(
        "Scale: the v2.1 DNS cache holds raw (mask + clip) omega_z and the matched "
        "decoded fields are raw (pipeline-unnormalised). Both are divided by "
        f"3 * train_std = {config['norm_divisor']:.4f} (split_v2p1, "
        "outputs/data_pipeline/v2p1/manifest.json) before filtering and "
        "thresholding, so the LEV iso-level matches exp_lev_tracking exactly.\n"
    )
    lines.append("## NO impulse claim (D167 stands)\n")
    lines.append(
        "We report ONLY the DIRECT CORRELATION between peak |Delta C_L| and peak "
        "|Gamma_LEV|. There is NO impulse-theorem argument anywhere; we do NOT "
        "assert that the LEV circulation explains lift.\n"
    )
    lines.append("## (i) GATE GP4 (held-out trend)\n")
    lines.append(
        f"Pearson r = {gp4['pearson_r']:.3f} "
        f"(95% case-clustered CI [{gp4['pearson_ci'][0]:.3f}, {gp4['pearson_ci'][1]:.3f}], "
        f"case-permutation two-sided p = {gp4['pearson_perm_p']:.4g}); "
        f"Spearman rho = {gp4['spearman_r']:.3f} "
        f"(95% CI [{gp4['spearman_ci'][0]:.3f}, {gp4['spearman_ci'][1]:.3f}], "
        f"perm p = {gp4['spearman_perm_p']:.4g}); "
        f"n = {gp4['n']} encounters across {gp4['n_cases']} cases (test_b + test_c). "
        f"|corr| headline = {gp4['abs_corr_headline']:.3f} vs GP4 threshold {GP4_THR} "
        f"-> **{gp4['branch'].upper()}**. The case-permutation p is VALID here: this "
        "is a correlation / trend test, not a paired location test, so the B6 "
        "paired-location degeneracy does not apply.\n"
    )
    lines.append("## (ii) G-sign asymmetry (positive vs negative gusts, matched |G|)\n")
    pg = gsign["peak_gamma_abs"]
    dt = gsign["detach_tc"]
    lines.append(
        f"Matched |G| buckets: {gsign['matched_absG']} "
        f"(n_pos = {gsign['n_pos']}, n_neg = {gsign['n_neg']}). "
        f"Peak |Gamma_LEV|: G>0 mean {pg['pos_mean']:.3f} vs G<0 mean {pg['neg_mean']:.3f} "
        f"(pos - neg = {pg['asymmetry_pos_minus_neg']:.3f}, Mann-Whitney two-sided p = "
        f"{pg['mannwhitney_p_two_sided']:.4g}); matched-bucket paired difference "
        f"{gsign['matched_paired_peak_gamma_diff_mean']:.3f} "
        f"(95% CI [{gsign['matched_paired_peak_gamma_diff_ci'][0]:.3f}, "
        f"{gsign['matched_paired_peak_gamma_diff_ci'][1]:.3f}], "
        f"{gsign['n_matched_buckets']} buckets). "
        f"Detachment time t/c: G>0 mean {dt['pos_mean']:.3f} vs G<0 mean {dt['neg_mean']:.3f} "
        f"(MW p = {dt['mannwhitney_p_two_sided']:.4g}). The PRF text describes the "
        "split-and-merge of the gust with the pre-existing suction-side vorticity "
        "qualitatively; this quantifies the budget asymmetry.\n"
    )
    lines.append("## (iii) Y modulation\n")
    lines.append(
        "By |Y|: "
        + "; ".join(
            f"|Y|={k}: mean peak |Gamma_LEV|={v['mean_peak_gamma']:.3f}, "
            f"corr(dCL,Gamma)={v['corr_dcl_gamma']:.2f} (n={v['n']})"
            for k, v in ymod["by_absY"].items()
        )
        + ".\n"
    )
    lines.append(
        "By Y sign: "
        + "; ".join(
            f"{k}: mean peak |Gamma_LEV|={v['mean_peak_gamma']:.3f}, "
            f"corr(dCL,Gamma)={v['corr_dcl_gamma']:.2f} (n={v['n']})"
            for k, v in ymod["by_Ysign"].items()
        )
        + ".\n"
    )
    lines.append("## PART B: matched-decoder LEV tracking (D146 regenerated)\n")
    lines.append(
        "Fraction of encounters whose decoded Gamma_LEV(t) tracks the DNS "
        f"Gamma_LEV(t) over [25, 55] (Pearson corr > {TRACK_CORR_THR}). "
        "RECON decode is the headline (decode of the true encoded latent: does the "
        "family latent CARRY the LEV?); forecast is noted separately.\n"
    )
    for fam in ("tf_noc", "fukami", "pod"):
        lines.append(
            f"- {fam}: recon {recon[fam]['tracking_fraction_str']} "
            f"(mean corr {recon[fam]['mean_corr']:.2f}); "
            f"forecast {fc[fam]['tracking_fraction_str']} "
            f"(mean corr {fc[fam]['mean_corr']:.2f})."
        )
    paired = latent.get("recon_paired_tf_vs_fukami", {})
    if paired.get("n", 0):
        lines.append(
            f"\nPaired recon tf_noc vs fukami (sign test, NOT case-permutation): "
            f"tf_noc higher tracking-corr in {paired['a_higher_corr_count']}/"
            f"{paired['n_effective']} effective encounters, one-sided sign p = "
            f"{paired['sign_p_one_sided']:.4g}; mean delta corr "
            f"{paired['mean_delta_corr']:.3f} "
            f"(case-clustered 95% CI {paired['case_cluster_ci']}).\n"
        )
    lines.append(
        "\nD146 reported predictive 42/42 vs reconstructive 36/42 on the OLD "
        "(confounded) decoder. The numbers above are the HONEST regeneration on the "
        "matched decoders; they may differ and are reported as measured, weak or "
        "strong.\n"
    )
    (OUT_DIR / "README.md").write_text("\n".join(lines))
    print(f"[p4] wrote {OUT_DIR / 'README.md'}")


# =============================================================================
# Driver.
# =============================================================================
def run(splits=("test_b", "test_c"), seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    divisor = load_pipeline_norm_divisor()
    airfoil = elt.load_masks()[0]
    split = load_split()
    gdy = gdy_lookup(split)
    peak_dcl = load_physics_peak_dcl()

    print(
        f"[p4] norm divisor 3*train_std = {divisor:.4f}; LEV iso-level "
        f"|omega_L|>{elt.OMEGA_THR}; sigma {elt.SIGMA_PX:.1f}px; window {WINDOW_FRAMES[0]}"
        f"-{WINDOW_FRAMES[-1]}; splits {splits}"
    )

    budget = compute_dns_budget(splits, gdy, peak_dcl, divisor, airfoil)
    rows = budget["rows"]
    print(
        f"[p4] DNS budget: {len(rows)} encounters; "
        f"{sum(r['n_detected'] > 0 for r in rows)} with detected LEV"
    )

    gp4 = gp4_correlation(rows, rng)
    print(
        f"[p4] GP4: Pearson r={gp4['pearson_r']:.3f} (CI {gp4['pearson_ci']}), "
        f"Spearman rho={gp4['spearman_r']:.3f}; branch={gp4['branch']}"
    )

    gsign = gsign_asymmetry(rows, rng)
    print(
        f"[p4] G-sign peak|Gamma| pos-neg = "
        f"{gsign['peak_gamma_abs']['asymmetry_pos_minus_neg']:.3f} "
        f"(MW p={gsign['peak_gamma_abs']['mannwhitney_p_two_sided']:.4g})"
    )

    ymod = y_modulation(rows)

    latent = compute_latent_tracking(splits, divisor, airfoil, rows)
    for fam in ("tf_noc", "fukami", "pod"):
        r = latent["endpoints"]["recon"][fam]
        print(
            f"[p4] recon track {fam}: {r['tracking_fraction_str']} "
            f"(mean corr {r['mean_corr']:.2f})"
        )

    config = {
        "norm_divisor": divisor,
        "omega_iso_level": elt.OMEGA_THR,
        "sigma_px": elt.SIGMA_PX,
        "sigma_over_c": elt.SIGMA_OVER_C,
        "window_frames": [int(WINDOW_FRAMES[0]), int(WINDOW_FRAMES[-1])],
        "impact_frame": IMPACT_FRAME,
        "dt_tc": DT_TC,
        "te_x_index": elt.TE_X_INDEX,
        "track_corr_threshold": TRACK_CORR_THR,
        "gp4_threshold": GP4_THR,
        "peak_dcl_source": "dcl_peak_simple (headline) from per_encounter_physics.npz",
        "lev_machinery": "scripts/session23/exp_lev_tracking.py (lev_from_field, large_scale)",
        "splits": list(splits),
        "seed": seed,
    }
    make_figure(rows, gp4, latent, divisor, airfoil)
    write_outputs(rows, gp4, gsign, ymod, latent, config)
    return {"gp4": gp4, "gsign": gsign, "ymod": ymod, "latent": latent, "n": len(rows)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--splits",
        nargs="+",
        default=["test_b", "test_c"],
        help="held-out splits to evaluate (default: test_b test_c)",
    )
    args = p.parse_args()
    run(splits=tuple(args.splits), seed=args.seed)


if __name__ == "__main__":
    main()
