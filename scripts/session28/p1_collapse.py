"""Physics Track P1: similarity collapse of the interaction and its latent image (v2.1).

READ-ONLY analysis (CPU numpy/scipy/sklearn; reuses the v2.1 pre-computed physics
table and the predictive-family latents; no GPU, no training, no DNS re-read).
Master plan Phase C, Physics Track P1; the flagship physics line. D190.

The question
------------
The vortex-gust lineage characterises the response case-by-case in (G, D, Y); no
quantitative one-parameter SIMILARITY variable is published for the interaction
amplitude (verified against the PRF text; re-verified via the L1 literature pass).
We pre-registered, BEFORE fitting, four candidate one-parameter scalings (already
stored per encounter by physics_prep.py in
outputs/session28/physics/per_encounter_physics.npz; we LOAD them, never recompute):

    s1 = G                          Kussner-like gust ratio
    s2 = G * D                      proportional to the Taylor-profile gust circulation
    s3 = Gamma_g(G, D)              the exact Taylor-vortex CORE circulation, integrated
                                    numerically from the implemented profile (the honest
                                    version of s2; s3 is exactly proportional to s2 with
                                    the numerically pinned prefactor ~3.81)
    s4 = MMF induced-velocity ratio Martinez-Muriel and Flores (2020) max induced
                                    vertical-velocity ratio at the miss distance |Y|;
                                    the only candidate that carries the geometry (Y)

The candidates are SIGNED interaction-strength variables (G < 0 is allowed); the
response amplitudes are non-negative magnitudes. The similarity hypothesis is
therefore amplitude ~ f(|s|). We fit and score against |s| and DOCUMENT this.

Response amplitudes per encounter (all pre-computed, all on v2.1):
    (a) force:    peak |Delta C_L| from the undisturbed phase-matched cycle
                  (physics_prep dcl_peak_simple).
    (b) wake:     peak large-scale wake-enstrophy excursion (denstrophy_peak_post).
    (c) latent:   peak post-impact Mahalanobis excursion of the PREDICTIVE family
                  (jepa_tf_noc_d64_s42) latent z(t) from the undisturbed-Baseline
                  limit-cycle tube. The orbit + Mahalanobis geometry is REUSED
                  verbatim from scripts/session28/s46_regen.py (settled_baseline_orbit
                  + _orbit_geometry + _min_maha_to_cloud). The headline-distance
                  caveat from s46 stands: the settled-Baseline orbit is ~80 points
                  with effective dimension ~3 of 64, so its sample covariance is
                  rank-deficient and the Mahalanobis whitening inflates off-orbit
                  directions; we report the orbit effective dimension and condition
                  number alongside the excursion so the caveat is auditable. (The OT
                  / transport response was cut from the paper; we do not compute it.)

Fit + score (the gate)
-----------------------
TRAIN cases fit; test_b is the held-out collapse set (test_c |G|=4 is reported as a
secondary extrapolation check, NEVER used for fitting or selection). For each
(response, candidate) we fit two one-parameter models on the TRAIN encounters:

    linear     y = a |s| + b                      (OLS)
    power-law  log y = p log|s| + log k  ->  y = k |s|^p   (OLS in log space)

and score COLLAPSE two ways, both honest about the held-out scatter:
    held-out R^2  = 1 - SSE_model / SST  on test_b, SST about the test_b own mean
                    (closure_matrix.r2_heldout convention; the train-fit params are
                    frozen, so a negative R^2 is possible and reported as-is).
    VRR           = variance-reduction ratio = 1 - Var(test_b residual)/Var(test_b y),
                    the fraction of the UNSCALED held-out scatter the one-variable
                    fit removes. Close to but NOT identical to held-out R^2: VRR
                    absorbs the held-out constant bias into the residual mean, R^2
                    charges it against the fit, so VRR >= R^2 and the gap is the
                    held-out calibration bias. We report both per the plan.

We do this per-|Y| stratum AND pooled (Y as the secondary axis). Case-clustered
95% CIs on the held-out R^2 come from stats_lib.case_cluster_bootstrap (resampling
TEST_B CASES), via the squared-error decomposition.

Gate GP1 (the branch is reported explicitly)
---------------------------------------------
STRONG  one candidate collapses BOTH the force (a) AND the latent excursion (c)
        with held-out test_b R^2 >= 0.8 AND the SAME power-law exponent within CI:
        the flagship line "the learned latent inherits the similarity scaling of
        the interaction".
MEDIUM  collapse holds for the force but the latent follows with a DIFFERENT
        exponent: report both exponents; the difference IS the finding (the latent
        weights the interaction geometry, not just its strength).
WEAK    no single-variable collapse (Y modulation dominates): report the
        Y-stratified result and the honest negative; the figure moves to the
        appendix per the plan.

We are adversarial: if no candidate clears R^2 0.8 on the small held-out set
(likely, given the Y modulation), we report WEAK and do not inflate.

Output
------
outputs/session28/p1/results.json            (all numbers + config + GP1 gate)
outputs/session28/p1/README.md               (human summary)
outputs/session28/p1/fig_p1_collapse.{pdf,png}  (response vs winning candidate)
outputs/session28/numbers_parts/p1.json       (eval_all macros; alphabetic)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "session28"))
sys.path.insert(0, str(REPO / "scripts" / "session21"))

import stats_lib  # noqa: E402
from closure_matrix import r2_heldout  # noqa: E402

# ---- paths (all absolute under the repo) ------------------------------------
PHYSICS_NPZ = REPO / "outputs" / "session28" / "physics" / "per_encounter_physics.npz"
PRED_LATENT_DIR = REPO / "outputs" / "session28" / "latents" / "jepa_tf_noc_d64_s42"
SPLIT_MANIFEST = REPO / "configs" / "splits" / "split_v2p1.json"
OUT_DIR = REPO / "outputs" / "session28" / "p1"
PART_PATH = REPO / "outputs" / "session28" / "numbers_parts" / "p1.json"

# ---- pre-registered candidates and responses (NPZ column names) -------------
CANDIDATES = {
    "s1_G": "s1",
    "s2_GD": "s2",
    "s3_Gamma_g": "s3",
    "s4_MMF": "s4",
}
CANDIDATE_DESC = {
    "s1_G": "G (Kussner-like gust ratio)",
    "s2_GD": "G*D (proportional to Taylor-profile gust circulation)",
    "s3_Gamma_g": "Taylor-vortex core circulation Gamma_g(G,D), numeric",
    "s4_MMF": "Martinez-Muriel & Flores (2020) induced-velocity ratio",
}
RESPONSES = {
    "force_dcl": "dcl_peak_simple",
    "wake_enstrophy": "denstrophy_peak_post",
    "latent_maha": "__latent_maha__",  # computed below from the predictive latents
}
RESPONSE_DESC = {
    "force_dcl": "peak |Delta C_L| (phase-matched-cycle simple peak)",
    "wake_enstrophy": "peak large-scale wake-enstrophy excursion (post-impact)",
    "latent_maha": "peak post-impact latent Mahalanobis excursion (predictive family)",
}

R2_STRONG = 0.8  # GP1 collapse threshold (held-out test_b R^2)
RESPONSE_WINDOW = 40  # frames after impact for the latent excursion peak (matches P1 force/wake)
N_BOOT_CASE = stats_lib.N_BOOT_CASE
EXPONENT_CI_TOL = 0.0  # "same exponent within CI": CIs must overlap (no extra slack)


# =====================================================================================
# Pure fit / score primitives (unit-tested on synthetic data; no I/O)
# =====================================================================================
@dataclass
class FitResult:
    """A one-parameter scaling fit and its held-out scores."""

    model: str  # "linear" | "powerlaw"
    params: dict  # {"a","b"} or {"k","p"}
    exponent: float  # power-law exponent (NaN for linear)
    r2_train: float
    r2_heldout: float
    vrr_heldout: float  # variance-reduction ratio on held-out (1 - Var(resid)/Var(y))
    n_train: int
    n_heldout: int


def _finite_positive(s: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mask of rows usable for a log-log fit: finite, |s| > 0, y > 0."""
    return np.isfinite(s) & np.isfinite(y) & (np.abs(s) > 0) & (y > 0)


def fit_linear(s_tr: np.ndarray, y_tr: np.ndarray) -> dict:
    """OLS of y = a |s| + b on the training rows. Returns {"a","b"}."""
    x = np.abs(np.asarray(s_tr, dtype=np.float64))
    y = np.asarray(y_tr, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return {"a": float("nan"), "b": float("nan")}
    a, b = np.polyfit(x[m], y[m], 1)
    return {"a": float(a), "b": float(b)}


def predict_linear(params: dict, s: np.ndarray) -> np.ndarray:
    return params["a"] * np.abs(np.asarray(s, dtype=np.float64)) + params["b"]


def fit_powerlaw(s_tr: np.ndarray, y_tr: np.ndarray) -> dict:
    """OLS of log y = p log|s| + log k on positive training rows. Returns {"k","p"}.

    The exponent p is the headline number. SE(p) is the OLS standard error of the
    log-log slope, carried so callers can form an exponent CI.
    """
    s = np.asarray(s_tr, dtype=np.float64)
    y = np.asarray(y_tr, dtype=np.float64)
    m = _finite_positive(s, y)
    if m.sum() < 3:
        return {"k": float("nan"), "p": float("nan"), "se_p": float("nan"), "n": int(m.sum())}
    lx = np.log(np.abs(s[m]))
    ly = np.log(y[m])
    p, logk = np.polyfit(lx, ly, 1)
    resid = ly - (p * lx + logk)
    n = lx.size
    dof = max(n - 2, 1)
    sxx = float(((lx - lx.mean()) ** 2).sum())
    se_p = float(np.sqrt((resid**2).sum() / dof / max(sxx, 1e-12)))
    return {"k": float(np.exp(logk)), "p": float(p), "se_p": se_p, "n": int(n)}


def predict_powerlaw(params: dict, s: np.ndarray) -> np.ndarray:
    """y_hat = k |s|^p. Zeros of |s| map to 0 (the response of a zero-strength gust)."""
    x = np.abs(np.asarray(s, dtype=np.float64))
    out = np.full(x.shape, np.nan, dtype=np.float64)
    pos = x > 0
    out[pos] = params["k"] * np.power(x[pos], params["p"])
    out[~pos] = 0.0
    return out


def variance_reduction_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """1 - Var(residual)/Var(y_true), the fraction of UNSCALED scatter removed.

    Variances are about each array's own mean (population, ddof=0). This is the
    "how much tighter is the scatter after scaling" number the plan asks for. It is
    DISTINCT from held-out R^2 (closure_matrix.r2_heldout, SSE/SST about the held-out
    mean): the two coincide only when the held-out residual is mean-zero. On a
    held-out set scored with FROZEN train-fit params the residual carries a small
    constant bias, so VRR (which absorbs that bias into its own mean) is slightly
    higher than R^2 (which charges the bias against the fit). We report both; the gap
    between them is the held-out calibration bias of the scaling.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    m = np.isfinite(yt) & np.isfinite(yp)
    if m.sum() < 2:
        return float("nan")
    var_y = float(np.var(yt[m]))
    var_r = float(np.var(yt[m] - yp[m]))
    return 1.0 - var_r / max(var_y, 1e-12)


def fit_and_score(
    s_tr: np.ndarray,
    y_tr: np.ndarray,
    s_te: np.ndarray,
    y_te: np.ndarray,
    model: str,
) -> FitResult:
    """Fit one model on (s_tr, y_tr), score on the held-out (s_te, y_te)."""
    if model == "linear":
        params = fit_linear(s_tr, y_tr)
        yhat_tr = predict_linear(params, s_tr)
        yhat_te = predict_linear(params, s_te)
        exponent = float("nan")
        out_params = {"a": params["a"], "b": params["b"]}
    elif model == "powerlaw":
        params = fit_powerlaw(s_tr, y_tr)
        yhat_tr = predict_powerlaw(params, s_tr)
        yhat_te = predict_powerlaw(params, s_te)
        exponent = params["p"]
        out_params = {"k": params["k"], "p": params["p"], "se_p": params["se_p"]}
    else:
        raise ValueError(f"unknown model {model!r}")

    mt = np.isfinite(s_tr) & np.isfinite(y_tr) & np.isfinite(yhat_tr)
    me = np.isfinite(s_te) & np.isfinite(y_te) & np.isfinite(yhat_te)
    r2_tr = r2_heldout(yhat_tr[mt], y_tr[mt]) if mt.sum() >= 2 else float("nan")
    r2_te = r2_heldout(yhat_te[me], y_te[me]) if me.sum() >= 2 else float("nan")
    vrr_te = variance_reduction_ratio(y_te[me], yhat_te[me]) if me.sum() >= 2 else float("nan")
    return FitResult(
        model=model,
        params=out_params,
        exponent=exponent,
        r2_train=r2_tr,
        r2_heldout=r2_te,
        vrr_heldout=vrr_te,
        n_train=int(mt.sum()),
        n_heldout=int(me.sum()),
    )


def heldout_r2_ci(
    s_tr: np.ndarray,
    y_tr: np.ndarray,
    s_te: np.ndarray,
    y_te: np.ndarray,
    case_te: np.ndarray,
    model: str,
    n_boot: int = N_BOOT_CASE,
    seed: int = 0,
) -> tuple[float, float]:
    """Case-clustered 95% CI on the held-out R^2 (train-fit params frozen).

    The fit is done ONCE on the (un-resampled) train set; the bootstrap resamples
    TEST_B CASES (stats_lib convention) and recomputes the held-out R^2 on each
    resample, so the CI reflects held-out CASE sampling, not refit noise.
    """
    if model == "linear":
        params = fit_linear(s_tr, y_tr)
        yhat = predict_linear(params, s_te)
    else:
        params = fit_powerlaw(s_tr, y_tr)
        yhat = predict_powerlaw(params, s_te)
    yt = np.asarray(y_te, dtype=np.float64)
    m = np.isfinite(yt) & np.isfinite(yhat)
    yt, yhat, cids = yt[m], yhat[m], np.asarray(case_te)[m]
    if yt.size < 2 or len(set(cids.tolist())) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    uc = np.array(sorted(set(cids.tolist())))
    by_case = {c: np.where(cids == c)[0] for c in uc}
    boot = np.empty(n_boot)
    for b in range(n_boot):
        pick = uc[rng.integers(0, len(uc), size=len(uc))]
        idx = np.concatenate([by_case[c] for c in pick])
        boot[b] = r2_heldout(yhat[idx], yt[idx])
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def exponent_ci(params_se: dict) -> tuple[float, float]:
    """Approximate 95% CI on a power-law exponent from its OLS log-slope SE."""
    p = params_se.get("p", float("nan"))
    se = params_se.get("se_p", float("nan"))
    if not (np.isfinite(p) and np.isfinite(se)):
        return float("nan"), float("nan")
    return p - 1.96 * se, p + 1.96 * se


def cis_overlap(ci_a: tuple[float, float], ci_b: tuple[float, float]) -> bool:
    """Do two CIs overlap (the 'same exponent within CI' test)?"""
    lo_a, hi_a = ci_a
    lo_b, hi_b = ci_b
    if not all(np.isfinite(v) for v in (lo_a, hi_a, lo_b, hi_b)):
        return False
    return (lo_a <= hi_b) and (lo_b <= hi_a)


# =====================================================================================
# Latent Mahalanobis excursion (reuses the s46_regen orbit geometry verbatim)
# =====================================================================================
def _load_latents() -> dict:
    out = {}
    for split in ("train", "test_b", "test_c"):
        blob = np.load(PRED_LATENT_DIR / f"{split}.npz", allow_pickle=True)
        out[split] = {
            "z_full": blob["z_full"],
            "case_ids": blob["case_ids"],
            "encounter_indices": blob["encounter_indices"].astype(int),
            "impact_frame": blob["impact_frame"].astype(int),
            "G": blob["G"],
            "D": blob["D"],
            "Y": blob["Y"],
        }
    return out


def compute_latent_maha_excursion(
    latents: dict,
    window: int = RESPONSE_WINDOW,
) -> dict:
    """Peak post-impact Mahalanobis excursion per encounter, keyed (case_id, enc_idx).

    The undisturbed-Baseline limit-cycle tube and its Mahalanobis whitening are built
    with s46_regen.settled_baseline_orbit + _orbit_geometry; the per-frame distance is
    s46_regen._min_maha_to_cloud. Excursion = max over [impact, impact+window] of the
    minimum Mahalanobis distance of z(t) to the settled-Baseline orbit cloud.
    """
    import s46_regen as s46  # local import: keeps synthetic unit tests free of s46

    manifest = json.loads(SPLIT_MANIFEST.read_text())
    baseline_cid = s46.baseline_case_id(manifest)
    orbit = s46.settled_baseline_orbit(latents["train"], baseline_cid)
    geom = s46._orbit_geometry(orbit)
    orbit_w = (orbit - geom.mean) @ geom.whiten

    # condition number of the orbit covariance (caveat auditing)
    cov = np.cov(orbit, rowvar=False) + 1e-6 * np.eye(orbit.shape[1])
    evals = np.clip(np.linalg.eigvalsh(cov), 1e-12, None)
    cond = float(evals.max() / evals.min())

    excursion: dict[tuple, float] = {}
    for split in ("train", "test_b", "test_c"):
        d = latents[split]
        z = d["z_full"]
        n, t_len, _ = z.shape
        imp = np.clip(d["impact_frame"], 0, t_len - 1)
        for i in range(n):
            lo = int(imp[i])
            hi = min(lo + window, t_len - 1)
            seg = z[i, lo : hi + 1, :]
            dist = s46._min_maha_to_cloud(seg, geom, orbit_w)
            key = (str(d["case_ids"][i]), int(d["encounter_indices"][i]))
            excursion[key] = float(np.max(dist))
    meta = {
        "baseline_case": baseline_cid,
        "orbit_n_pts": int(geom.n_pts),
        "orbit_effective_dim": float(geom.effective_dim),
        "orbit_cond_number": cond,
        "latent_dim": int(orbit.shape[1]),
        "window_frames": int(window),
        "caveat": (
            "Mahalanobis on a rank-deficient (~80-pt, eff-dim ~3 of 64) orbit cloud "
            "inflates off-orbit directions; reported per the master-plan P1 spec "
            "(the s46 HEADLINE return metric is Euclidean). Treat the latent excursion "
            "as ordinal across encounters, not as an absolute whitened distance."
        ),
    }
    return {"excursion": excursion, "meta": meta}


# =====================================================================================
# Data assembly
# =====================================================================================
@dataclass
class P1Data:
    split: np.ndarray
    case_id: np.ndarray
    enc_idx: np.ndarray
    G: np.ndarray
    D: np.ndarray
    Y: np.ndarray
    candidates: dict  # name -> (n,) signed candidate value
    responses: dict  # name -> (n,) response amplitude (NaN where unavailable)


def load_p1_data() -> tuple[P1Data, dict]:
    """Assemble the per-encounter (candidate, response) table on v2.1."""
    d = np.load(PHYSICS_NPZ, allow_pickle=True)
    split = d["split"]
    case_id = d["case_id"]
    enc_idx = d["encounter_index"].astype(int)

    candidates = {name: np.asarray(d[col], dtype=np.float64) for name, col in CANDIDATES.items()}

    responses = {}
    responses["force_dcl"] = np.asarray(d[RESPONSES["force_dcl"]], dtype=np.float64)
    responses["wake_enstrophy"] = np.asarray(d[RESPONSES["wake_enstrophy"]], dtype=np.float64)

    latents = _load_latents()
    maha = compute_latent_maha_excursion(latents)
    lat = np.full(case_id.shape, np.nan, dtype=np.float64)
    for i in range(case_id.shape[0]):
        lat[i] = maha["excursion"].get((str(case_id[i]), int(enc_idx[i])), np.nan)
    responses["latent_maha"] = lat

    data = P1Data(
        split=split,
        case_id=case_id,
        enc_idx=enc_idx,
        G=np.asarray(d["G"], dtype=np.float64),
        D=np.asarray(d["D"], dtype=np.float64),
        Y=np.asarray(d["Y"], dtype=np.float64),
        candidates=candidates,
        responses=responses,
    )
    return data, maha["meta"]


# =====================================================================================
# Per-(response, candidate, model, stratum) scoring
# =====================================================================================
def _stratum_masks(data: P1Data) -> dict:
    """Pooled + per-|Y| strata, on the FIT-eligible encounters (train vs test_b)."""
    absy = np.round(np.abs(data.Y), 3)
    strata = {"pooled": np.ones(absy.shape, dtype=bool)}
    for yv in sorted(set(absy.tolist())):
        strata[f"absY_{yv:.2f}"] = absy == yv
    return strata


def score_response_candidate(
    data: P1Data,
    response: str,
    candidate: str,
    stratum_mask: np.ndarray,
    seed: int = 0,
) -> dict:
    """Both models, train-fit / test_b-held-out, with CIs, for one stratum."""
    is_tr = data.split == "train"
    is_te = data.split == "test_b"
    is_tc = data.split == "test_c"
    s = data.candidates[candidate]
    y = data.responses[response]

    def slice_(mask):
        m = mask & stratum_mask & np.isfinite(s) & np.isfinite(y)
        return s[m], y[m], data.case_id[m]

    s_tr, y_tr, _ = slice_(is_tr)
    s_te, y_te, c_te = slice_(is_te)
    s_tc, y_tc, c_tc = slice_(is_tc)

    out = {"n_train": int(s_tr.size), "n_test_b": int(s_te.size), "n_test_c": int(s_tc.size)}
    if s_tr.size < 3 or s_te.size < 2:
        out["insufficient"] = True
        return out

    for model in ("linear", "powerlaw"):
        fr = fit_and_score(s_tr, y_tr, s_te, y_te, model)
        rec = {
            "params": fr.params,
            "exponent": fr.exponent,
            "r2_train": fr.r2_train,
            "r2_heldout": fr.r2_heldout,
            "vrr_heldout": fr.vrr_heldout,
            "n_train": fr.n_train,
            "n_heldout": fr.n_heldout,
        }
        lo, hi = heldout_r2_ci(s_tr, y_tr, s_te, y_te, c_te, model, seed=seed)
        rec["r2_heldout_ci"] = [lo, hi]
        if model == "powerlaw":
            elo, ehi = exponent_ci(fr.params)
            rec["exponent_ci"] = [elo, ehi]
            # test_c extrapolation check (secondary; never used for selection)
            if s_tc.size >= 2:
                fr_tc = fit_and_score(s_tr, y_tr, s_tc, y_tc, model)
                rec["r2_test_c"] = fr_tc.r2_heldout
        out[model] = rec
    return out


def run_scoring(data: P1Data, seed: int = 0) -> dict:
    strata = _stratum_masks(data)
    results: dict = {}
    for response in RESPONSES:
        results[response] = {}
        for candidate in CANDIDATES:
            results[response][candidate] = {}
            for sname, smask in strata.items():
                results[response][candidate][sname] = score_response_candidate(
                    data, response, candidate, smask, seed=seed
                )
    return results


# =====================================================================================
# GP1 gate logic
# =====================================================================================
def _pooled_powerlaw(results: dict, response: str, candidate: str) -> dict | None:
    rec = results.get(response, {}).get(candidate, {}).get("pooled", {})
    return rec.get("powerlaw")


def decide_gp1(results: dict) -> dict:
    """Pick the winning candidate and the GP1 branch (pooled, power-law headline).

    Selection rule (pre-registered, adversarial):
      * winner = the candidate with the highest pooled held-out test_b R^2 on the
        FORCE response (the force is the physically primary amplitude).
      * STRONG if that candidate's force AND latent pooled held-out R^2 are both
        >= 0.8 AND the force / latent power-law exponent CIs overlap.
      * MEDIUM if the force clears 0.8 but the latent exponent differs (CIs do not
        overlap, or the latent R^2 < 0.8 while the force fit is real).
      * WEAK otherwise (no single-variable force collapse on the held-out set).
    """
    force = "force_dcl"
    latent = "latent_maha"

    scored = []
    for candidate in CANDIDATES:
        fl = _pooled_powerlaw(results, force, candidate)
        if fl is None:
            continue
        scored.append((candidate, fl.get("r2_heldout", float("nan"))))
    scored = [(c, r) for c, r in scored if np.isfinite(r)]
    if not scored:
        return {"branch": "WEAK", "winner": None, "reason": "no fittable force collapse"}

    scored.sort(key=lambda cr: cr[1], reverse=True)
    winner, force_r2 = scored[0]

    fl = _pooled_powerlaw(results, force, winner)
    ll = _pooled_powerlaw(results, latent, winner)
    force_exp = fl.get("exponent", float("nan"))
    force_exp_ci = tuple(fl.get("exponent_ci", [float("nan"), float("nan")]))
    latent_r2 = ll.get("r2_heldout", float("nan")) if ll else float("nan")
    latent_exp = ll.get("exponent", float("nan")) if ll else float("nan")
    latent_exp_ci = (
        tuple(ll.get("exponent_ci", [float("nan"), float("nan")]))
        if ll
        else (
            float("nan"),
            float("nan"),
        )
    )

    same_exp = cis_overlap(force_exp_ci, latent_exp_ci)
    if force_r2 >= R2_STRONG and np.isfinite(latent_r2) and latent_r2 >= R2_STRONG and same_exp:
        branch = "STRONG"
        reason = (
            f"force R2={force_r2:.2f} and latent R2={latent_r2:.2f} both >= {R2_STRONG}; "
            f"exponents agree within CI (force {force_exp:.2f}, latent {latent_exp:.2f})"
        )
    elif force_r2 >= R2_STRONG:
        branch = "MEDIUM"
        reason = (
            f"force R2={force_r2:.2f} >= {R2_STRONG} but latent does not match: "
            f"latent R2={latent_r2:.2f}, exponents (force {force_exp:.2f}, "
            f"latent {latent_exp:.2f}) same_within_CI={same_exp}"
        )
    else:
        branch = "WEAK"
        reason = (
            f"best force collapse (candidate {winner}) only reaches held-out "
            f"R2={force_r2:.2f} < {R2_STRONG}; Y modulation dominates"
        )

    return {
        "branch": branch,
        "winner": winner,
        "winner_desc": CANDIDATE_DESC[winner],
        "force_r2_heldout": force_r2,
        "force_exponent": force_exp,
        "force_exponent_ci": list(force_exp_ci),
        "latent_r2_heldout": latent_r2,
        "latent_exponent": latent_exp,
        "latent_exponent_ci": list(latent_exp_ci),
        "exponents_same_within_ci": same_exp,
        "reason": reason,
        "all_force_r2": {c: r for c, r in scored},
    }


# =====================================================================================
# Figure
# =====================================================================================
def make_figure(data: P1Data, results: dict, gate: dict, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import figstyle

        figstyle.use_style()
        figw, figh = figstyle.figure_size(width_fraction=1.0, aspect=0.42)
    except Exception:
        figw, figh = (6.0, 2.6)

    winner = gate["winner"]
    if winner is None:
        winner = "s1_G"

    fig, axes = plt.subplots(1, 2, figsize=(figw, figh))
    panels = [("force_dcl", axes[0]), ("latent_maha", axes[1])]
    s = data.candidates[winner]
    absy = np.round(np.abs(data.Y), 3)
    y_strata = sorted(set(absy.tolist()))
    cmap = plt.get_cmap("viridis")
    colors = {yv: cmap(i / max(len(y_strata) - 1, 1)) for i, yv in enumerate(y_strata)}

    for response, ax in panels:
        y = data.responses[response]
        is_tr = data.split == "train"
        is_te = data.split == "test_b"
        for yv in y_strata:
            for mask, marker, alpha, lbl in (
                (is_tr, "o", 0.45, "train"),
                (is_te, "s", 0.95, "test_b"),
            ):
                m = mask & (absy == yv) & np.isfinite(s) & np.isfinite(y) & (np.abs(s) > 0)
                if not m.any():
                    continue
                ax.scatter(
                    np.abs(s[m]),
                    y[m],
                    s=18 if marker == "s" else 11,
                    marker=marker,
                    color=colors[yv],
                    alpha=alpha,
                    edgecolors="k" if marker == "s" else "none",
                    linewidths=0.4,
                    label=f"|Y|={yv:.2f} {lbl}",
                )
        rec = _pooled_powerlaw(results, response, winner)
        if rec is not None and np.isfinite(rec.get("exponent", np.nan)):
            xs = np.linspace(
                max(np.abs(s[np.isfinite(s) & (np.abs(s) > 0)]).min(), 1e-3),
                np.abs(s[np.isfinite(s)]).max(),
                100,
            )
            k = rec["params"]["k"]
            p = rec["params"]["p"]
            ax.plot(xs, k * np.power(xs, p), "k-", lw=1.2, zorder=5)
            r2 = rec.get("r2_heldout", float("nan"))
            ax.set_title(
                f"{RESPONSE_DESC[response].split('(')[0].strip()}\n"
                f"p={p:.2f}, test_b R$^2$={r2:.2f}",
                fontsize=7,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"|{CANDIDATE_DESC[winner].split('(')[0].strip()}|", fontsize=7)
        ax.tick_params(labelsize=6)

    axes[0].set_ylabel("response amplitude", fontsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    # de-duplicate the legend
    seen = {}
    for h, lab in zip(handles, labels):
        seen.setdefault(lab, h)
    fig.legend(
        list(seen.values()),
        list(seen.keys()),
        loc="lower center",
        ncol=4,
        fontsize=5.2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(f"P1 similarity collapse vs {winner} (GP1: {gate['branch']})", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig_p1_collapse.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_p1_collapse.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# =====================================================================================
# Outputs: results.json, README.md, numbers_parts/p1.json
# =====================================================================================
def _winner_letter(candidate: str) -> str:
    return {"s1_G": "G", "s2_GD": "GD", "s3_Gamma_g": "GammaG", "s4_MMF": "MMF"}.get(
        candidate, "None"
    )


def emit_part(results: dict, gate: dict, meta: dict, out_path: Path) -> dict:
    """eval_all part with ALPHABETIC macros for the winning candidate's headline numbers."""
    winner = gate.get("winner")
    branch = gate["branch"]
    fl = _pooled_powerlaw(results, "force_dcl", winner) if winner else None
    ll = _pooled_powerlaw(results, "latent_maha", winner) if winner else None

    def g(rec, key, default=float("nan")):
        return rec.get(key, default) if rec else default

    numbers = {}
    # GP1 branch as a string macro
    numbers["p1_gp1_branch"] = {
        "macro": "NumPoneBranch",
        "value": branch,
        "fmt": "%s",
        "note": gate.get("reason", ""),
        "source": "p1_collapse.py",
    }
    numbers["p1_winning_candidate"] = {
        "macro": "NumPoneWinner",
        "value": _winner_letter(winner) if winner else "none",
        "fmt": "%s",
        "note": gate.get("winner_desc", "no fittable collapse"),
        "source": "p1_collapse.py",
    }
    if fl is not None:
        flo, fhi = fl.get("r2_heldout_ci", [float("nan"), float("nan")])
        elo, ehi = fl.get("exponent_ci", [float("nan"), float("nan")])
        numbers["p1_force_r2_heldout"] = {
            "macro": "NumPoneForceRtwo",
            "value": float(g(fl, "r2_heldout")),
            "fmt": "%.2f",
            "ci_lo": float(flo),
            "ci_hi": float(fhi),
            "n": int(g(fl, "n_heldout", 0)),
            "split": "test_b",
            "observable": "peak|dCL| vs winning candidate (power-law, pooled)",
            "note": f"winner={winner}; held-out R2 (train-fit frozen); case-clustered CI.",
            "source": "p1_collapse.py",
        }
        numbers["p1_force_exponent"] = {
            "macro": "NumPoneForceExp",
            "value": float(g(fl, "exponent")),
            "fmt": "%.2f",
            "ci_lo": float(elo),
            "ci_hi": float(ehi),
            "split": "train",
            "observable": "power-law exponent, peak|dCL| vs winning candidate",
            "source": "p1_collapse.py",
        }
    if ll is not None:
        llo, lhi = ll.get("r2_heldout_ci", [float("nan"), float("nan")])
        elo2, ehi2 = ll.get("exponent_ci", [float("nan"), float("nan")])
        numbers["p1_latent_r2_heldout"] = {
            "macro": "NumPoneLatentRtwo",
            "value": float(g(ll, "r2_heldout")),
            "fmt": "%.2f",
            "ci_lo": float(llo),
            "ci_hi": float(lhi),
            "n": int(g(ll, "n_heldout", 0)),
            "split": "test_b",
            "endpoint": "predictive",
            "observable": "peak latent Mahalanobis excursion vs winning candidate",
            "note": (
                f"winner={winner}; predictive family jepa_tf_noc_d64_s42; "
                f"orbit eff-dim {meta['orbit_effective_dim']:.1f}/{meta['latent_dim']} "
                f"(rank-deficient; Mahalanobis cross-check per P1 spec)."
            ),
            "source": "p1_collapse.py",
        }
        numbers["p1_latent_exponent"] = {
            "macro": "NumPoneLatentExp",
            "value": float(g(ll, "exponent")),
            "fmt": "%.2f",
            "ci_lo": float(elo2),
            "ci_hi": float(ehi2),
            "split": "train",
            "endpoint": "predictive",
            "observable": "power-law exponent, latent excursion vs winning candidate",
            "source": "p1_collapse.py",
        }
    part = {"part": "p1", "numbers": numbers}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(part, indent=2, sort_keys=True))
    return part


def _fmt(v, nd=3):
    try:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "n/a"
        return f"{v:.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def write_readme(results: dict, gate: dict, meta: dict, out_dir: Path) -> None:
    lines = []
    lines.append("# Physics Track P1: similarity collapse and its latent image (v2.1)")
    lines.append("")
    lines.append(f"Generated {datetime.now(timezone.utc).isoformat()} (CPU, read-only).")
    lines.append("")
    lines.append(f"## GP1 verdict: **{gate['branch']}**")
    lines.append("")
    lines.append(f"{gate.get('reason','')}")
    lines.append("")
    lines.append(
        f"Winning candidate (highest pooled held-out force R^2): "
        f"**{gate.get('winner')}** ({gate.get('winner_desc','')})."
    )
    lines.append("")
    lines.append("Pooled power-law held-out test_b R^2 of |dCL| vs each candidate:")
    for c, r in gate.get("all_force_r2", {}).items():
        lines.append(f"  - {c} ({CANDIDATE_DESC[c]}): {_fmt(r,2)}")
    lines.append("")
    lines.append("## Headline numbers (winning candidate, pooled, power-law)")
    fl = _pooled_powerlaw(results, "force_dcl", gate.get("winner"))
    ll = _pooled_powerlaw(results, "latent_maha", gate.get("winner"))
    if fl:
        lines.append(
            f"  force |dCL|: exponent p = {_fmt(fl.get('exponent'),2)} "
            f"CI {[_fmt(x,2) for x in fl.get('exponent_ci',[None,None])]}, "
            f"held-out R^2 = {_fmt(fl.get('r2_heldout'),2)} "
            f"CI {[_fmt(x,2) for x in fl.get('r2_heldout_ci',[None,None])]}, "
            f"test_c R^2 = {_fmt(fl.get('r2_test_c'),2)}"
        )
    if ll:
        lines.append(
            f"  latent excursion: exponent p = {_fmt(ll.get('exponent'),2)} "
            f"CI {[_fmt(x,2) for x in ll.get('exponent_ci',[None,None])]}, "
            f"held-out R^2 = {_fmt(ll.get('r2_heldout'),2)} "
            f"CI {[_fmt(x,2) for x in ll.get('r2_heldout_ci',[None,None])]}, "
            f"test_c R^2 = {_fmt(ll.get('r2_test_c'),2)}"
        )
    lines.append("")
    lines.append("## Full grid (pooled, both models): held-out R^2 [VRR] / exponent")
    lines.append("")
    header = "| response | model | " + " | ".join(CANDIDATES) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (2 + len(CANDIDATES)))
    for response in RESPONSES:
        for model in ("linear", "powerlaw"):
            cells = []
            for candidate in CANDIDATES:
                rec = results[response][candidate]["pooled"].get(model)
                if rec is None:
                    cells.append("n/a")
                    continue
                r2 = _fmt(rec.get("r2_heldout"), 2)
                vrr = _fmt(rec.get("vrr_heldout"), 2)
                if model == "powerlaw":
                    cells.append(f"{r2} [{vrr}] / p={_fmt(rec.get('exponent'),2)}")
                else:
                    cells.append(f"{r2} [{vrr}]")
            lines.append(f"| {response} | {model} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Per-|Y| stratified (power-law held-out R^2, winning candidate)")
    winner = gate.get("winner")
    if winner:
        for response in RESPONSES:
            strat = results[response][winner]
            cells = []
            for sname in sorted(strat):
                if sname == "pooled":
                    continue
                rec = strat[sname].get("powerlaw")
                if rec is None:
                    cells.append(f"{sname}: n/a")
                else:
                    cells.append(
                        f"{sname}: R2={_fmt(rec.get('r2_heldout'),2)} (n_tb={rec.get('n_heldout')})"
                    )
            lines.append(f"  {response}: " + "; ".join(cells))
    lines.append("")
    lines.append("## Latent Mahalanobis orbit geometry (caveat)")
    lines.append(f"  baseline case: {meta['baseline_case']}; orbit points: {meta['orbit_n_pts']}")
    lines.append(
        f"  orbit effective dim: {_fmt(meta['orbit_effective_dim'],2)} / {meta['latent_dim']}; "
        f"cov condition number: {_fmt(meta['orbit_cond_number'],1)}"
    )
    lines.append(f"  {meta['caveat']}")
    lines.append("")
    lines.append("## Method notes")
    lines.append(
        "  - Candidates are SIGNED strength variables; responses are non-negative "
        "magnitudes; fits use |s| (documented in the module docstring)."
    )
    lines.append(
        "  - s3 (Gamma_g) is exactly proportional to s2 (G*D); their power-law "
        "exponents are identical and their held-out R^2 coincide. The exponent "
        "differs from s1 (G) only through the D leverage."
    )
    lines.append(
        "  - test_c (|G|=4) is reported as a secondary extrapolation R^2; it is NEVER "
        "used for fitting or candidate selection."
    )
    lines.append(
        "  - Held-out R^2 uses closure_matrix.r2_heldout (SST about the held-out mean); "
        "CIs are case-clustered (stats_lib convention, resampling test_b cases)."
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--no-fig", action="store_true", help="skip the figure (faster).")
    args = ap.parse_args(argv)

    t0 = datetime.now()
    print("[p1] loading physics table + predictive latents ...", flush=True)
    data, meta = load_p1_data()
    print(
        f"[p1] orbit: {meta['baseline_case']} {meta['orbit_n_pts']} pts, "
        f"eff-dim {meta['orbit_effective_dim']:.2f}/{meta['latent_dim']}, "
        f"cond {meta['orbit_cond_number']:.1f}",
        flush=True,
    )
    print("[p1] scoring all (response x candidate x stratum x model) ...", flush=True)
    results = run_scoring(data, seed=args.seed)
    gate = decide_gp1(results)
    print(f"[p1] GP1 branch = {gate['branch']}; winner = {gate['winner']}", flush=True)
    print(f"[p1] {gate['reason']}", flush=True)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "physics_npz": str(PHYSICS_NPZ),
            "predictive_latents": str(PRED_LATENT_DIR),
            "split_manifest": str(SPLIT_MANIFEST),
            "candidates": CANDIDATE_DESC,
            "responses": RESPONSE_DESC,
            "r2_strong_threshold": R2_STRONG,
            "response_window_frames": RESPONSE_WINDOW,
            "n_boot_case": N_BOOT_CASE,
            "seed": args.seed,
        },
        "latent_orbit_meta": meta,
        "results": results,
        "gp1_gate": gate,
        "generated_iso": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[p1] wrote {out_dir/'results.json'}", flush=True)

    emit_part(results, gate, meta, PART_PATH)
    print(f"[p1] wrote {PART_PATH}", flush=True)

    write_readme(results, gate, meta, out_dir)
    print(f"[p1] wrote {out_dir/'README.md'}", flush=True)

    if not args.no_fig:
        make_figure(data, results, gate, out_dir)
        print(f"[p1] wrote {out_dir/'fig_p1_collapse.pdf'}", flush=True)

    print(f"[p1] done in {(datetime.now()-t0).total_seconds():.1f}s", flush=True)


if __name__ == "__main__":
    main()
