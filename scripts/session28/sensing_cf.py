"""Sparse wall-pressure OBSERVABILITY (master plan C-F; D195), v2.1, MAIN result.

PROBE REGIME DECLARATION (CLAUDE.md probe methodology): the pressure -> latent
map fitted here is a STATE-RECOVERY map. It reads the encoder's PER-encounter
latent at the readout frame impact + READOUT_H from a causal window of K
wall-pressure taps ending AT that frame (wall pressure available up to the
readout instant); it is NOT a (G, D, Y) parameter probe and not a per-frame
state-descriptor probe. The downstream wake probe IS the frozen closure-matrix
wake-enstrophy probe (a STATE-DESCRIPTOR ridge on PER-frame z), reused verbatim
and never refit on pressure. The readout frame and primary endpoint match the
closure matrix (impact + 16). Y attribution (master plan C-F vi) uses the same
window and reports pressure -> Y R^2 directly.

This rebuilds the sparse-pressure observability track on the v2.1 unconditioned
rebuild and elevates it to a MAIN result (author decision 2026-06-13): the same
predictive latent that makes the wake linearly readable (closure matrix) is the
state most recoverable from a few wall-pressure taps. It ports and adapts:
  scripts/session27_exp_pressure_v2_tcsi_uncond.py  (single-family uncond study),
  scripts/session14_tcsi_pilot.py                   (TCSI / qDEIM / uniform),
  scripts/session28/closure_matrix.py               (frozen wake probe, folds),
  scripts/session28/stats_lib.py                    (case-clustered bootstrap CI).

Six requirements (master plan C-F, closes referees B5, M7, M11, M12):
  (i)   PLACEMENT: target-blind qDEIM is the PRIMARY cross-family placement;
        per-family TCSI is secondary. Picks regenerated at K = 2/4/8/16 on the
        v2p1 cache (physical taps 0-191). qDEIM and TCSI picks are both reported.
  (ii)  METRIC: a variance-weighted state-recovery R^2 is reported ALONGSIDE the
        mean canonical correlation over min(d, K*W) directions, so anisotropy is
        visible. Both are defined by an equation in the README. W is the
        pressure window length (WINDOW below; the causal window ends at readout).
  (iii) DEPLOYMENT PANEL: pressure (K taps over the causal window) -> estimated
        readout latent z(impact+H) (KernelRidge-RBF on the flattened K*W vector,
        the established map) -> FROZEN closure wake probe -> wake enstrophy. R^2
        vs K per family, WITH the direct pressure -> wake regression as the
        no-latent baseline (dashed).
  (iv)  THREE-PANEL FIGURE with ALL families (jepa_tf_noc, fukami, pod, bvae) and
        direct baselines dashed: (a) state recovery R^2 vs K, (b) wake-via-latent
        R^2 vs K, (c) lift-via-latent R^2 vs K.
  (v)   CAUSALITY: the per-encounter percentile clip of the omega pipeline uses
        within-encounter statistics; we run a causal-clip variant of the pressure
        window once and report the (expected negligible) delta on the headline.
  (vi)  Y ATTRIBUTION: pressure -> Y R^2 is reported to soften the Y limitation.

Statistics: 5-fold CASE-level CV on every readout (no case straddles folds, the
closure fold construction); case-clustered bootstrap CI on the headline
pressure -> wake R^2 per family via stats_lib.case_cluster_bootstrap; paired
family differences use a case-clustered CI + one-sided sign test, NOT
stats_lib.case_permutation_p (degenerate for paired location tests, the B6
lesson).

CPU-only sklearn work (KernelRidge / ridge); no GPU is touched.

Usage:
    nice -n 10 .venv/bin/python scripts/session28/sensing_cf.py
    nice -n 10 .venv/bin/python scripts/session28/sensing_cf.py --quick   # K=8 only
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "session28"))

import stats_lib  # noqa: E402
from closure_matrix import apply_probe, fit_ridge, make_case_folds  # noqa: E402

# --------------------------------------------------------------------- constants

PREVENT = Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
CACHE = (
    Path(os.environ.get("VORTEX_JEPA_CACHE", str(PREVENT / "data/processed/vortex-jepa"))) / "v2p1"
)
LAT_ROOT = REPO / "outputs" / "session28" / "latents"
DNS_PATH = REPO / "outputs" / "session28" / "exp2" / "dns_physical_metrics.npz"
SPLIT_MANIFEST = REPO / "configs" / "splits" / "split_v2p1.json"
OUT = REPO / "outputs" / "session28" / "sensing"
PARTS = REPO / "outputs" / "session28" / "numbers_parts"
AIRFOIL_PATH = REPO / "outputs" / "session21" / "airfoil_xy.npy"

K_LIST = [2, 4, 8, 16]
WINDOW = 30  # W: pressure window length, frames; ends AT the readout instant
N_PRESSURE = 192  # surface taps (already spanwise-averaged per CLAUDE.md)
HEADLINE_K = 8
PRIMARY_OBS = "wake_enstrophy"
# Readout horizon H past impact (matches the closure-matrix primary endpoint
# impact + 16). The latent/observable read are at frame impact + H; the pressure
# window is the causal [impact + H - W, impact + H], i.e. wall pressure available
# up to the readout instant. H = 0 recovers the impact-frame readout.
READOUT_H = 16

# KRR map hyperparameters (the established pressure -> latent map; verbatim s27).
KRR_ALPHA = 0.1
KRR_GAMMA = 0.01

# Families present in the MAIN figure. macro_family drives the LaTeX macro suffix.
# "bvae" headlines the matched-head (lift + wake) Solera-Rico variant: the faithful
# lift-only variant cannot read the wake at all (closure repr R^2 = -0.19), so the
# matched-head is the credible reconstructive VAE for an observability comparison.


@dataclass
class Family:
    name: str  # display / key name
    objective: str  # predictive | reconstructive
    macro: str  # LaTeX macro suffix
    color: str  # figure colour
    marker: str  # figure marker
    seed_tags: list[str]  # latents dirs, lead seed first


FAMILIES: list[Family] = [
    Family(
        "jepa_tf_noc",
        "predictive",
        "JepaTf",
        "#1b7837",
        "o",
        ["jepa_tf_noc_d64_s42", "jepa_tf_noc_d64_s0", "jepa_tf_noc_d64_s1", "jepa_tf_noc_d64_s2"],
    ),
    Family(
        "fukami",
        "reconstructive",
        "Fukami",
        "#c0392b",
        "s",
        ["fukami_d64_s42", "fukami_d64_s0", "fukami_d64_s1", "fukami_d64_s2"],
    ),
    Family("pod", "reconstructive", "Pod", "#2166ac", "^", ["pod_d64"]),
    Family(
        "bvae",
        "reconstructive",
        "Bvae",
        "#8c6bb1",
        "v",
        ["bvae_match_d64_s42", "bvae_match_d64_s0", "bvae_match_d64_s1"],
    ),
]
SPLITS = ["train", "test_b", "test_c"]


# --------------------------------------------------------------------- selectors


def selector_uniform(K: int, n: int = N_PRESSURE) -> list[int]:
    """Uniformly spaced taps (port of session14_tcsi_pilot.selector_uniform)."""
    K = min(K, n)
    return sorted({int(round(i * n / K)) % n for i in range(K)})


def selector_qdeim(snapshot: np.ndarray, K: int) -> list[int]:
    """Target-blind qDEIM placement via QR with column pivoting (Drmac & Gugercin
    2016; Manohar et al. 2018). ``snapshot`` is (n_enc, n_taps); returns K sorted,
    distinct tap indices. Deterministic.
    """
    from scipy.linalg import qr

    A = np.asarray(snapshot, dtype=np.float64)
    n_enc, n_tap = A.shape
    K_eff = min(K, n_tap)
    r = min(K_eff, n_enc, n_tap)
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    Vr = Vt[:r, :].T  # (n_tap, r) leading right singular vecs
    _, _, piv = qr(Vr.T, pivoting=True, mode="economic")
    return sorted(int(p) for p in piv[:K_eff])


def selector_tcsi(
    X_window: np.ndarray, target: np.ndarray, K: int, alpha: float = 1.0
) -> list[int]:
    """Target-aware greedy TCSI selection on the joint (sensor x W) feature vector.

    Verbatim objective of session14_tcsi_pilot.greedy_forward_selection: at each
    step add the tap that most reduces residual variance of ``target`` regressed
    on the selected taps' windows (with the same G/S_preq/H_res/Eff combination).
    ``X_window`` is (n, n_taps, W); ``target`` is (n,). Returns taps in greedy
    (most-informative-first) order.
    """
    n, n_tap, _ = X_window.shape
    y = np.asarray(target, dtype=np.float64)
    yc = y - y.mean()
    L_null = float(np.mean(yc**2))
    Xc = X_window - X_window.mean(axis=0, keepdims=True)
    selected: list[int] = []
    while len(selected) < min(K, n_tap):
        S = Xc[:, selected, :].reshape(n, -1) if selected else np.zeros((n, 0))
        best_j, best_score = -1, -np.inf
        for j in range(n_tap):
            if j in selected:
                continue
            X_try = np.concatenate([S, Xc[:, j, :]], axis=1)
            d = X_try.shape[1]
            gram = X_try.T @ X_try + alpha * np.eye(d)
            try:
                w = np.linalg.solve(gram, X_try.T @ yc)
            except np.linalg.LinAlgError:
                continue
            L_final = float(np.mean((yc - X_try @ w) ** 2))
            G = float(n) * max(0.0, L_null - L_final)
            S_preq = max(0.0, L_null - L_final)
            H_res = float(n) * L_final
            Eff = G / (S_preq + 1e-6)
            score = 1.0 * G - 0.5 * S_preq - 0.5 * H_res + 0.5 * Eff
            if score > best_score:
                best_score, best_j = score, j
        if best_j < 0:
            break
        selected.append(best_j)
    return selected


# --------------------------------------------------------------------- metrics


def variance_weighted_r2(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Variance-weighted multi-output state-recovery R^2 (METRIC eq. 1).

        R^2_vw = 1 - sum_j SSE_j / sum_j SST_j,
        SSE_j = sum_i (yhat_ij - y_ij)^2,  SST_j = sum_i (y_ij - mean_i y_ij)^2.

    SST about the held-out sample's own per-dimension mean. Weighting each
    latent dimension by its own variance (the pooled-SSE / pooled-SST form)
    means high-variance directions dominate, so a single scalar can hide
    anisotropy; report it WITH the canonical correlation (METRIC eq. 2).
    """
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    sse = float(((y_pred - y_true) ** 2).sum())
    sst = float(((y_true - y_true.mean(axis=0, keepdims=True)) ** 2).sum())
    return 1.0 - sse / max(sst, 1e-12)


def mean_canonical_correlation(
    y_pred: np.ndarray, y_true: np.ndarray, n_dir: int, reg: float = 1e-6
) -> tuple[float, list[float]]:
    """Mean canonical correlation over the first ``n_dir`` directions (METRIC eq. 2).

    Canonical correlations rho_1 >= ... >= rho_r between predicted and true
    held-out latents are the singular values of the whitened cross-covariance
        M = Sigma_pp^{-1/2} Sigma_pt Sigma_tt^{-1/2},
    so a near-isotropic recovery gives many large rho while an anisotropic one
    gives a few. We report the mean of the leading n_dir = min(d, K*W) values
    (n_dir caps at the recoverable rank K*W of the pressure feature).

    Returns (mean_rho, [rho_1, ..., rho_{n_dir}]).
    """
    P = np.asarray(y_pred, dtype=np.float64)
    T = np.asarray(y_true, dtype=np.float64)
    P = P - P.mean(axis=0, keepdims=True)
    T = T - T.mean(axis=0, keepdims=True)
    n = P.shape[0]
    Spp = P.T @ P / n + reg * np.eye(P.shape[1])
    Stt = T.T @ T / n + reg * np.eye(T.shape[1])
    Spt = P.T @ T / n

    def inv_sqrt(S: np.ndarray) -> np.ndarray:
        w, V = np.linalg.eigh(S)
        w = np.clip(w, reg, None)
        return V @ np.diag(1.0 / np.sqrt(w)) @ V.T

    M = inv_sqrt(Spp) @ Spt @ inv_sqrt(Stt)
    rho = np.clip(np.linalg.svd(M, compute_uv=False), 0.0, 1.0)
    k = max(1, min(n_dir, rho.size))
    return float(rho[:k].mean()), [float(x) for x in rho[:k]]


# --------------------------------------------------------------------- loaders


def load_pressure_window(
    case_id: str,
    k: int,
    readout: int,
    causal: bool = False,
    clip_hi: np.ndarray | None = None,
    clip_lo: np.ndarray | None = None,
) -> np.ndarray:
    """Causal (W, 192) wall-pressure window ending AT the readout instant.

    The window is frames [readout - W, readout): wall pressure available up to
    (and not beyond) the readout frame, so the deployment map is causal. With
    readout = impact + READOUT_H this is "sparse pressure observed up to the
    impact+H readout". ``causal`` + per-tap clip bounds apply the causal-clip
    variant (master plan C-F v): each tap is winsorised to [clip_lo, clip_hi]
    computed from the TRAIN window distribution only.
    """
    p = CACHE / case_id / f"encounter_{int(k):02d}.h5"
    with h5py.File(p, "r") as f:
        p_wall = np.asarray(f["p_wall"], dtype=np.float32)  # (120, 192)
    t0 = max(0, readout - WINDOW)
    win = p_wall[t0:readout]
    if win.shape[0] < WINDOW:  # left-pad short
        pad = np.zeros((WINDOW - win.shape[0], N_PRESSURE), dtype=np.float32)
        win = np.concatenate([pad, win], axis=0)
    if causal and clip_hi is not None:
        win = np.clip(win, clip_lo[None, :], clip_hi[None, :])
    return win  # (W, 192)


@dataclass
class SplitBundle:
    X: np.ndarray  # (n, W, 192) causal pressure windows ending at the readout
    z: np.ndarray  # (n, d) latent at the readout frame (impact + READOUT_H)
    case_ids: np.ndarray  # (n,) str
    enc_idx: np.ndarray  # (n,) encounter index (stable per-encounter key)
    wake: np.ndarray  # (n,) DNS wake_enstrophy at the readout frame
    cl: np.ndarray  # (n,) DNS C_L at the readout frame
    Y: np.ndarray  # (n,) gust offset (frame-independent)
    z_perframe: np.ndarray = field(default=None)  # (n, T, d) for frozen-probe fit
    wake_perframe: np.ndarray = field(default=None)  # (n, T) DNS wake, frozen-probe fit
    impact: np.ndarray = field(default=None)  # (n,) impact frames
    readout: np.ndarray = field(default=None)  # (n,) readout frames (impact + H)


def _dns_index(dns, split: str, cids: np.ndarray, eis: np.ndarray) -> np.ndarray:
    lut = {
        (str(c), int(e)): i
        for i, (c, e) in enumerate(zip(dns[f"{split}_case_id"], dns[f"{split}_encounter_index"]))
    }
    return np.array([lut.get((str(c), int(e)), -1) for c, e in zip(cids, eis)])


def load_family_split(tag: str, split: str, dns, causal_clip: tuple | None = None) -> SplitBundle:
    """Assemble the (causal pressure window, readout latent, DNS wake/CL/Y) bundle.

    The readout frame is impact + READOUT_H (clamped to the trajectory). The
    pressure window ends at the readout (causal), the target latent and DNS
    observables are read at the readout, matching the closure-matrix endpoint.
    """
    npz = np.load(LAT_ROOT / tag / f"{split}.npz", allow_pickle=True)
    cids = np.array([str(c) for c in npz["case_ids"]])
    eis = npz["encounter_indices"].astype(int)
    imp = npz["impact_frame"].astype(int)
    z_full = npz["z_full"].astype(np.float64)  # (n, T, d)
    Y = npz["Y"].astype(np.float64)
    T = z_full.shape[1]
    rd = np.clip(imp + READOUT_H, 0, T - 1)  # readout frame per encounter
    z_rd = np.stack([z_full[i, int(rd[i])] for i in range(len(cids))])

    di = _dns_index(dns, split, cids, eis)
    keep = di >= 0
    cids, eis, imp, rd, z_full, z_rd, Y, di = (
        cids[keep],
        eis[keep],
        imp[keep],
        rd[keep],
        z_full[keep],
        z_rd[keep],
        Y[keep],
        di[keep],
    )
    wake = dns[f"{split}_{PRIMARY_OBS}"][di]  # (n, 120)
    cl = dns[f"{split}_C_L"][di]
    wake_rd = np.array([wake[i, int(rd[i])] for i in range(len(cids))])
    cl_rd = np.array([cl[i, int(rd[i])] for i in range(len(cids))])

    clip_hi = clip_lo = None
    causal = causal_clip is not None
    if causal:
        clip_lo, clip_hi = causal_clip
    X = np.zeros((len(cids), WINDOW, N_PRESSURE), dtype=np.float64)
    for i in range(len(cids)):
        X[i] = load_pressure_window(
            cids[i], int(eis[i]), int(rd[i]), causal=causal, clip_hi=clip_hi, clip_lo=clip_lo
        )
    return SplitBundle(
        X=X,
        z=z_rd,
        case_ids=cids,
        enc_idx=eis,
        wake=wake_rd,
        cl=cl_rd,
        Y=Y,
        z_perframe=z_full,
        wake_perframe=wake,
        impact=imp,
        readout=rd,
    )


# --------------------------------------------------------------------- KRR map


def feats(X: np.ndarray, taps: list[int]) -> np.ndarray:
    """Flatten the (n, W, K) sub-window into the (n, K*W) regressor vector."""
    return X[:, :, taps].reshape(len(X), -1)


def fit_krr(Xtr: np.ndarray, ytr: np.ndarray) -> tuple:
    """KernelRidge(RBF) with standardised in/out (verbatim s27 pressure->z map)."""
    ytr = ytr.reshape(len(ytr), -1)
    sx = StandardScaler().fit(Xtr)
    sy = StandardScaler().fit(ytr)
    m = KernelRidge(alpha=KRR_ALPHA, kernel="rbf", gamma=KRR_GAMMA)
    m.fit(sx.transform(Xtr), sy.transform(ytr))
    return m, sx, sy


def krr_pred(model: tuple, X: np.ndarray) -> np.ndarray:
    m, sx, sy = model
    out = sy.inverse_transform(m.predict(sx.transform(X)))
    return out


# --------------------------------------------------------------------- CV readout


def cv_r2(
    X: np.ndarray, y: np.ndarray, case_ids: np.ndarray, metric, n_dir_fn=None
) -> tuple[float, np.ndarray, np.ndarray]:
    """5-fold CASE-level CV of the KRR map X -> y, scored by ``metric``.

    Returns (mean_metric_over_folds, oof_pred (n, ...), fold_id (n,)). The
    out-of-fold predictions feed the case-clustered bootstrap. ``y`` may be
    (n,) or (n, m). ``n_dir_fn`` (optional) is called per fold to choose the
    canonical-correlation direction count.
    """
    folds = make_case_folds(case_ids, n_folds=5, seed=0)
    y2 = y.reshape(len(y), -1)
    oof = np.full_like(y2, np.nan, dtype=np.float64)
    scores = []
    for f in range(5):
        te = folds == f
        tr = ~te
        if te.sum() == 0 or tr.sum() == 0:
            continue
        model = fit_krr(X[tr], y2[tr])
        pred = krr_pred(model, X[te]).reshape(te.sum(), -1)
        oof[te] = pred
        if n_dir_fn is not None:
            scores.append(metric(pred, y2[te], n_dir_fn(X.shape[1]))[0])
        else:
            scores.append(metric(pred, y2[te]))
    return float(np.mean(scores)), oof, folds


# --------------------------------------------------------------------- main eval


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="K=8 only (smoke).")
    ap.add_argument(
        "--no-tcsi", action="store_true", help="skip the secondary per-family TCSI placement."
    )
    args = ap.parse_args()
    k_list = [HEADLINE_K] if args.quick else K_LIST

    OUT.mkdir(parents=True, exist_ok=True)
    PARTS.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    dns = np.load(DNS_PATH, allow_pickle=True)

    print("[sensing_cf] loading families (pressure windows + impact latents)...", flush=True)
    data: dict[str, dict[str, SplitBundle]] = {}
    for fam in FAMILIES:
        tag = fam.seed_tags[0]  # lead seed for placement + headline
        data[fam.name] = {sp: load_family_split(tag, sp, dns) for sp in SPLITS}
        n = {sp: len(data[fam.name][sp].z) for sp in SPLITS}
        print(
            f"    {fam.name:12s} d={data[fam.name]['train'].z.shape[1]} " f"n_enc={n}", flush=True
        )

    # ---- (i) PLACEMENT: qDEIM (primary, target-blind) + TCSI (secondary) ----
    # qDEIM is target-blind: the snapshot matrix is the window-mean pressure over
    # the TRAIN encounters, identical for every family. The qDEIM picks are
    # therefore family-independent (the primary cross-family placement).
    ref = data[FAMILIES[0].name]["train"]
    snapshot = ref.X.mean(axis=1)  # (n_train, 192) window-mean
    qdeim = {K: selector_qdeim(snapshot, K) for K in K_LIST}
    uniform = {K: selector_uniform(K) for K in K_LIST}
    print(
        f"[sensing_cf] qDEIM picks (target-blind, primary): " f"{ {K: qdeim[K] for K in K_LIST} }",
        flush=True,
    )

    tcsi: dict[str, dict[int, list[int]]] = {}
    if not args.no_tcsi:
        for fam in FAMILIES:
            tr = data[fam.name]["train"]
            target = tr.z[:, 0]  # first latent coord as proxy
            # PCA-1 of the latent is a stronger univariate proxy than coord 0:
            zc = tr.z - tr.z.mean(0)
            _, _, vt = np.linalg.svd(zc, full_matrices=False)
            target = zc @ vt[0]
            Xw = tr.X.transpose(0, 2, 1)  # (n, 192, W)
            ranked = selector_tcsi(Xw, target, max(K_LIST))
            tcsi[fam.name] = {K: sorted(ranked[:K]) for K in K_LIST}
        print(
            f"[sensing_cf] TCSI picks (per-family, secondary): "
            f"{ {f: tcsi[f][HEADLINE_K] for f in tcsi} }",
            flush=True,
        )

    json.dump(
        {
            "qDEIM": {str(K): qdeim[K] for K in K_LIST},
            "uniform": {str(K): uniform[K] for K in K_LIST},
            "TCSI": {f: {str(K): tcsi[f][K] for K in K_LIST} for f in tcsi},
            "window_W": WINDOW,
            "n_taps": N_PRESSURE,
            "qdeim_note": "target-blind window-mean snapshot; family-independent",
        },
        open(OUT / "sensor_picks_v2p1.json", "w"),
        indent=2,
    )

    # ---- causal-clip bounds (master plan C-F v), from TRAIN windows only ----
    # The omega pipeline's per-encounter p99.99 clip uses within-encounter stats;
    # the causal variant winsorises each tap to the TRAIN causal-window
    # [p0.1, p99.9], i.e. global TRAIN statistics rather than within-encounter
    # ones. We rebuild the JEPA headline pressure->wake under this clip and
    # report the (expected negligible) delta.
    Xtr_ref = ref.X  # (n, W, 192) train windows
    clip_lo = np.percentile(Xtr_ref.reshape(-1, N_PRESSURE), 0.1, axis=0)
    clip_hi = np.percentile(Xtr_ref.reshape(-1, N_PRESSURE), 99.9, axis=0)

    # ---- (ii)-(iv) per-family curves under qDEIM (primary) ----
    rows = []
    headline = {}  # fam -> {K: dict with oof + cc bootstrap}
    rng = np.random.default_rng(0)

    for fam in FAMILIES:
        tr = data[fam.name]["train"]
        # FROZEN closure wake probe: ridge on ALL train per-frame latents -> wake.
        # Fitted ONCE per family, never refit on pressure (the "fixed linear probe").
        # This is the identical fit_ridge(per-frame z -> per-frame wake) of
        # closure_matrix.py; the deployment panel reuses it without refitting.
        zf = tr.z_perframe.reshape(-1, tr.z_perframe.shape[2])  # (n*T, d)
        wake_pf = tr.wake_perframe.reshape(-1)  # (n*T,)
        wake_probe = fit_ridge(zf, wake_pf, alpha=1.0)
        # latent -> C_L probe (train), the lift readout used by the via-latent panel
        cl_probe = fit_ridge(tr.z, tr.cl, alpha=1.0)

        headline[fam.name] = {}
        for K in k_list:
            taps = qdeim[K]
            Xtr = feats(tr.X, taps)
            # pressure -> z (impact latent) CV map, used for all three panels
            r2_state_vw, z_oof_tr, _ = cv_r2(Xtr, tr.z, tr.case_ids, variance_weighted_r2)
            n_dir = min(tr.z.shape[1], K * WINDOW)
            cc_mean, _, _ = cv_r2(
                Xtr, tr.z, tr.case_ids, mean_canonical_correlation, n_dir_fn=lambda w, nd=n_dir: nd
            )

            for sp in ("test_b", "test_c"):
                te = data[fam.name][sp]
                Xte = feats(te.X, taps)
                # refit pressure->z on full train (held-out split eval)
                pz = fit_krr(Xtr, tr.z)
                zhat = krr_pred(pz, Xte)  # (n, d)
                r2_state = variance_weighted_r2(zhat, te.z)
                cc, rhos = mean_canonical_correlation(zhat, te.z, n_dir)
                # wake via latent: pressure->z->frozen wake probe
                wake_hat = apply_probe(zhat, wake_probe)
                r2_wake_via = variance_weighted_r2(wake_hat, te.wake)
                # lift via latent: pressure->z->C_L probe
                cl_hat = apply_probe(zhat, cl_probe)
                ok = ~np.isnan(te.cl)
                r2_cl_via = variance_weighted_r2(cl_hat[ok], te.cl[ok])
                # direct (no-latent) baselines
                pwake = fit_krr(Xtr, tr.wake)
                pcl = fit_krr(Xtr, tr.cl)
                wake_dir = krr_pred(pwake, Xte).ravel()
                cl_dir = krr_pred(pcl, Xte).ravel()
                r2_wake_dir = variance_weighted_r2(wake_dir, te.wake)
                r2_cl_dir = variance_weighted_r2(cl_dir[ok], te.cl[ok])
                # pressure -> Y (M11 softening)
                py = fit_krr(Xtr, tr.Y)
                r2_y = variance_weighted_r2(krr_pred(py, Xte).ravel(), te.Y)

                rows.append(
                    dict(
                        family=fam.name,
                        objective=fam.objective,
                        K=K,
                        split=sp,
                        placement="qDEIM",
                        r2_state_vw=r2_state,
                        cc_mean=cc,
                        r2_wake_via=r2_wake_via,
                        r2_wake_direct=r2_wake_dir,
                        r2_cl_via=r2_cl_via,
                        r2_cl_direct=r2_cl_dir,
                        r2_y=r2_y,
                        cv_r2_state_vw=r2_state_vw,
                        cv_cc_mean=cc_mean,
                        n_dir=n_dir,
                        n_enc=len(te.z),
                    )
                )

                # ---- headline: case-clustered CI on per-encounter pressure->wake ----
                if sp == "test_b":
                    # per-encounter squared-error-based R^2 contribution; we
                    # bootstrap the held-out R^2 by resampling CASES (stats_lib).
                    headline[fam.name][K] = dict(
                        wake_hat=wake_hat,
                        wake_true=te.wake,
                        case_ids=te.case_ids,
                        keys=np.array([f"{c}|{e}" for c, e in zip(te.case_ids, te.enc_idx)]),
                        r2=r2_wake_via,
                        r2_direct=r2_wake_dir,
                    )

    # ---- case-clustered bootstrap CI on the headline pressure->wake R^2 ----
    def boot_r2_ci(wake_hat, wake_true, case_ids, n_boot=2000):
        uc = np.array(sorted(set(case_ids.tolist())))
        idx_by_case = {c: np.where(case_ids == c)[0] for c in uc}
        boots = []
        for _ in range(n_boot):
            pick = uc[rng.integers(0, len(uc), size=len(uc))]
            sel = np.concatenate([idx_by_case[c] for c in pick])
            boots.append(variance_weighted_r2(wake_hat[sel], wake_true[sel]))
        return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

    headline_ci = {}
    for fam in FAMILIES:
        headline_ci[fam.name] = {}
        for K in k_list:
            h = headline[fam.name][K]
            ci = boot_r2_ci(h["wake_hat"], h["wake_true"], h["case_ids"])
            headline_ci[fam.name][K] = dict(
                r2=h["r2"], ci_lo=ci[0], ci_hi=ci[1], r2_direct=h["r2_direct"]
            )

    # ---- paired family difference (JEPA vs each reconstructive) at headline K ----
    # case-clustered CI + one-sided sign test, NOT case_permutation_p (B6 lesson).
    paired = {}
    jep = FAMILIES[0].name
    hj = headline[jep][HEADLINE_K] if HEADLINE_K in headline[jep] else headline[jep][k_list[0]]
    Kp = HEADLINE_K if HEADLINE_K in headline[jep] else k_list[0]
    jpos = {k: i for i, k in enumerate(hj["keys"])}
    for fam in FAMILIES[1:]:
        ho = headline[fam.name][Kp]
        # Align on the (case_id|enc_idx) key so per-encounter pairing is exact
        # even if a family's npz ordering or DNS-match drops differ.
        common = [k for k in ho["keys"] if k in jpos]
        ji = np.array([jpos[k] for k in common])
        oi = np.array([np.where(ho["keys"] == k)[0][0] for k in common])
        e_j = (hj["wake_hat"][ji] - hj["wake_true"][ji]) ** 2
        e_o = (ho["wake_hat"][oi] - ho["wake_true"][oi]) ** 2
        case_aligned = hj["case_ids"][ji]
        # sign test that JEPA error < other error (jepa better)
        k_better, n_eff, sign_p = stats_lib.sign_test_one_sided(e_j, e_o)
        # case-clustered CI on the per-encounter error reduction (other - jepa)
        delta = e_o - e_j
        cc = stats_lib.case_cluster_bootstrap(
            delta, case_aligned, n_boot=10000, rng=np.random.default_rng(0)
        )
        paired[f"{jep}_vs_{fam.name}"] = dict(
            K=Kp,
            n_paired=len(common),
            sign_k=k_better,
            sign_n=n_eff,
            sign_p=sign_p,
            enc_mean_delta=cc["enc_mean"],
            enc_mean_ci=cc["enc_mean_ci"],
            case_mean_delta=cc["case_mean"],
            case_mean_ci=cc["case_mean_ci"],
        )

    # ---- (v) CAUSALITY: JEPA headline pressure->wake under causal-clip variant ----
    jfam = FAMILIES[0]
    causal_clip = (clip_lo, clip_hi)
    tr_c = load_family_split(jfam.seed_tags[0], "train", dns, causal_clip=causal_clip)
    tb_c = load_family_split(jfam.seed_tags[0], "test_b", dns, causal_clip=causal_clip)
    zf_c = tr_c.z_perframe.reshape(-1, tr_c.z_perframe.shape[2])
    wake_pf_c = tr_c.wake_perframe.reshape(-1)
    wake_probe_c = fit_ridge(zf_c, wake_pf_c, alpha=1.0)
    taps_c = qdeim[HEADLINE_K if HEADLINE_K in qdeim else k_list[0]]
    pz_c = fit_krr(feats(tr_c.X, taps_c), tr_c.z)
    zhat_c = krr_pred(pz_c, feats(tb_c.X, taps_c))
    r2_wake_causal = variance_weighted_r2(apply_probe(zhat_c, wake_probe_c), tb_c.wake)
    r2_wake_base = headline_ci[jfam.name][
        HEADLINE_K if HEADLINE_K in headline_ci[jfam.name] else k_list[0]
    ]["r2"]
    causal_delta = r2_wake_causal - r2_wake_base

    # ---------------------------------------------------------------- write CSV/NPZ
    with open(OUT / "cf_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # figure source arrays: per family, qDEIM curves on test_b
    def series(fam_name, col):
        d = {r["K"]: r[col] for r in rows if r["family"] == fam_name and r["split"] == "test_b"}
        return [d.get(K, np.nan) for K in k_list]

    save_arrays = {
        "K_list": np.asarray(k_list, dtype=np.int64),
        "families": np.asarray([f.name for f in FAMILIES]),
        "colors": np.asarray([f.color for f in FAMILIES]),
        "markers": np.asarray([f.marker for f in FAMILIES]),
    }
    for fam in FAMILIES:
        for col, key in (
            ("r2_state_vw", "state_r2"),
            ("cc_mean", "cc"),
            ("r2_wake_via", "wake_via"),
            ("r2_wake_direct", "wake_direct"),
            ("r2_cl_via", "cl_via"),
            ("r2_cl_direct", "cl_direct"),
            ("r2_y", "y_r2"),
        ):
            save_arrays[f"{fam.name}_{key}"] = np.asarray(series(fam.name, col), dtype=np.float64)
    np.savez(OUT / "cf_results.npz", **save_arrays)

    wall = time.time() - t_start
    cf_json = dict(
        window_W=WINDOW,
        K_list=k_list,
        headline_K=HEADLINE_K,
        primary_observable=PRIMARY_OBS,
        placement_primary="qDEIM",
        qdeim_picks={str(K): qdeim[K] for K in K_LIST},
        tcsi_picks={f: {str(K): tcsi[f][K] for K in K_LIST} for f in tcsi},
        uniform_picks={str(K): uniform[K] for K in K_LIST},
        rows=rows,
        headline_ci=headline_ci,
        paired=paired,
        causal=dict(
            r2_wake_base=r2_wake_base,
            r2_wake_causal=r2_wake_causal,
            delta=causal_delta,
            note="JEPA pressure->wake at headline K; causal-clip = TRAIN "
            "causal-window [p0.1,p99.9] per-tap winsorisation",
        ),
        wall_seconds=wall,
    )

    def _json_default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return float(o)

    json.dump(cf_json, open(OUT / "cf_results.json", "w"), indent=2, default=_json_default)

    # ---------------------------------------------------------------- numbers_part
    emit_numbers_part(
        headline_ci, paired, qdeim, tcsi, rows, causal_delta, r2_wake_base, r2_wake_causal, k_list
    )

    # ---------------------------------------------------------------- figure
    try:
        make_three_panel(rows, headline_ci, qdeim, k_list)
    except Exception as exc:  # figure is non-load-bearing; never fail the run
        print(f"[sensing_cf] WARNING figure skipped: {exc}", flush=True)

    print(
        f"[sensing_cf] wrote {OUT/'cf_results.json'}, .csv, .npz "
        f"({len(rows)} rows); wall {wall:.1f}s",
        flush=True,
    )
    print_report(headline_ci, paired, rows, causal_delta, r2_wake_base, r2_wake_causal, k_list)


def make_three_panel(rows, headline_ci, qdeim, k_list) -> None:
    """The MAIN three-panel sensing figure (master plan C-F iv).

    (a) state recovery R^2 vs K, (b) wake-via-latent R^2 vs K, (c) lift-via-latent
    R^2 vs K. ALL families present (referee B5 fix); the direct no-latent
    pressure->observable regression is the dashed family-coloured baseline.
    qDEIM (target-blind) placement.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(REPO / "scripts" / "session21"))
    import figstyle as fs  # noqa: E402

    fs.use_style()
    fig, axes = plt.subplots(1, 3, figsize=fs.figure_size(1.0, aspect=0.38))

    def series(fam_name, col):
        d = {r["K"]: r[col] for r in rows if r["family"] == fam_name and r["split"] == "test_b"}
        return [d.get(K, np.nan) for K in k_list]

    panels = [
        (axes[0], "r2_state_vw", None, "(a) state recovery", r"state recovery $R^2$"),
        (
            axes[1],
            "r2_wake_via",
            "r2_wake_direct",
            "(b) wake via latent",
            r"wake $\Omega_w$ recovery $R^2$",
        ),
        (
            axes[2],
            "r2_cl_via",
            "r2_cl_direct",
            "(c) lift via latent",
            r"impact $C_L$ recovery $R^2$",
        ),
    ]
    for ax, via_col, dir_col, title, ylab in panels:
        for fam in FAMILIES:
            y = series(fam.name, via_col)
            ax.plot(
                k_list,
                y,
                marker=fam.marker,
                ms=3.8,
                lw=1.2,
                color=fam.color,
                label=fam.name.replace("_", " "),
            )
            if dir_col is not None:
                yd = series(fam.name, dir_col)
                ax.plot(k_list, yd, color=fam.color, lw=1.0, ls=(0, (4, 2)), alpha=0.7, zorder=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(k_list)
        ax.set_xticklabels([str(k) for k in k_list])
        ax.set_xlabel("sensors $K$")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=8)
        ax.axhline(0, color="0.85", lw=0.6, zorder=0)
    # one legend: families (solid) + a dashed proxy for "direct (no latent)"
    handles = [
        plt.Line2D([], [], color=f.color, marker=f.marker, lw=1.2, label=f.name.replace("_", " "))
        for f in FAMILIES
    ]
    handles.append(
        plt.Line2D([], [], color="0.4", lw=1.0, ls=(0, (4, 2)), label="direct (no latent)")
    )
    axes[0].legend(handles=handles, loc="lower right", fontsize=6, handletextpad=0.4, borderpad=0.2)
    fig.tight_layout()
    pdf = OUT / "fig_sensing_three_panel.pdf"
    fig.savefig(pdf)
    fig.savefig(OUT / "fig_sensing_three_panel.png", dpi=200)
    plt.close(fig)
    print(f"[sensing_cf] wrote {pdf}", flush=True)


def emit_numbers_part(
    headline_ci, paired, qdeim, tcsi, rows, causal_delta, r2_wake_base, r2_wake_causal, k_list
) -> None:
    """Write outputs/session28/numbers_parts/sensing_cf.json (eval_all format)."""
    nums: dict[str, dict] = {}
    Kh = HEADLINE_K if HEADLINE_K in k_list else k_list[0]
    macro = {f.name: f.macro for f in FAMILIES}

    def row_at(fam, K, col):
        return [r for r in rows if r["family"] == fam and r["split"] == "test_b" and r["K"] == K][
            0
        ][col]

    # HEADLINE (the result that holds): variance-weighted state recovery R^2 per
    # family at K=8, plus the mean canonical correlation (anisotropy). The
    # predictive latent is the most pressure-recoverable.
    for fam in FAMILIES:
        nums[f"sensing_state_K{Kh}_testb_{fam.name}"] = dict(
            macro=f"SenseState{macro[fam.name]}",
            value=float(row_at(fam.name, Kh, "r2_state_vw")),
            fmt="%.2f",
            split="test_b",
            endpoint="deployment",
            probe="krr_rbf",
            observable="state_recovery",
            note=f"variance-weighted state-recovery R^2, {Kh} qDEIM taps, d=64 "
            f"lead seed; readout at impact+{READOUT_H}",
        )
        nums[f"sensing_cc_K{Kh}_testb_{fam.name}"] = dict(
            macro=f"SenseCc{macro[fam.name]}",
            value=float(row_at(fam.name, Kh, "cc_mean")),
            fmt="%.2f",
            split="test_b",
            endpoint="deployment",
            observable="canonical_correlation",
            note=f"mean canonical correlation over min(d,K*W) dirs, {Kh} qDEIM "
            f"taps; reported WITH the vw-R^2 to expose anisotropy",
        )
    # DEPLOYMENT PANEL: pressure->wake R^2 at K=8 per family, with case-clustered CI.
    for fam in FAMILIES:
        h = headline_ci[fam.name][Kh]
        nums[f"sensing_pwake_K{Kh}_testb_{fam.name}"] = dict(
            macro=f"SensePWake{macro[fam.name]}",
            value=float(h["r2"]),
            fmt="%.2f",
            ci_lo=float(h["ci_lo"]),
            ci_hi=float(h["ci_hi"]),
            n=int(row_at(fam.name, Kh, "n_enc")),
            split="test_b",
            endpoint="deployment",
            probe="krr_rbf->ridge",
            observable="wake_enstrophy",
            note=f"pressure({Kh} qDEIM taps,W={WINDOW})->z(impact+{READOUT_H})->"
            f"frozen wake probe; case-clustered CI; d=64 lead seed",
        )
    # direct no-latent baseline (JEPA window; representation-independent direct map)
    hj = headline_ci[FAMILIES[0].name][Kh]
    nums[f"sensing_pwake_direct_K{Kh}_testb"] = dict(
        macro="SensePWakeDirect",
        value=float(hj["r2_direct"]),
        fmt="%.2f",
        split="test_b",
        endpoint="deployment",
        observable="wake_enstrophy",
        note=f"direct pressure({Kh} qDEIM taps)->wake regression, no latent",
    )
    # PLACEMENT (i): qDEIM is the primary cross-family placement; the headline
    # state recovery above uses qDEIM. We expose the primary qDEIM JEPA number
    # under its own macro for the placement sentence (qDEIM is target-blind, so
    # its picks are family-independent; TCSI picks are reported in the JSON).
    jep = FAMILIES[0].name
    nums[f"sensing_state_qdeim_K{Kh}_testb_{jep}"] = dict(
        macro="SenseStateQdeim",
        value=float(row_at(jep, Kh, "r2_state_vw")),
        fmt="%.2f",
        split="test_b",
        endpoint="deployment",
        observable="state_recovery",
        note=f"variance-weighted state recovery R^2, {Kh} target-blind qDEIM "
        f"taps (primary placement), JEPA; TCSI picks in sensor_picks_v2p1.json",
    )
    # pressure -> Y R^2 (M11 softening) JEPA at K=8
    nums[f"sensing_pY_K{Kh}_testb_{jep}"] = dict(
        macro="SensePY",
        value=float(row_at(jep, Kh, "r2_y")),
        fmt="%.2f",
        split="test_b",
        endpoint="deployment",
        observable="Y",
        note=f"pressure({Kh} qDEIM taps)->Y R^2; M11 Y-attribution softening",
    )
    # lift via latent vs direct, JEPA at K=8 (the C_L deployment panel)
    nums[f"sensing_pcl_via_K{Kh}_testb_{jep}"] = dict(
        macro="SensePClVia",
        value=float(row_at(jep, Kh, "r2_cl_via")),
        fmt="%.2f",
        split="test_b",
        endpoint="deployment",
        observable="C_L",
        note=f"pressure({Kh} qDEIM taps)->z(impact+{READOUT_H})->C_L probe R^2, "
        f"JEPA; direct={row_at(jep, Kh, 'r2_cl_direct'):.3f}",
    )
    # causal-clip delta on the JEPA headline pressure->wake
    nums["sensing_causal_clip_delta"] = dict(
        macro="SenseCausalDelta",
        value=float(causal_delta),
        fmt="%.3f",
        split="test_b",
        endpoint="deployment",
        observable="wake_enstrophy",
        note=f"R^2 delta (causal-clip - base) on JEPA pressure->wake at K={Kh}; "
        f"base={r2_wake_base:.3f} causal={r2_wake_causal:.3f}",
    )
    # paired JEPA-vs-recon sign test p at headline K (worst-case = max p)
    if paired:
        worst = max(paired.items(), key=lambda kv: kv[1]["sign_p"])
        nums["sensing_pwake_jepa_vs_best_recon_signp"] = dict(
            macro="SensePWakeSignP",
            value=float(worst[1]["sign_p"]),
            fmt="%.3f",
            split="test_b",
            endpoint="deployment",
            observable="wake_enstrophy",
            note=f"max over recon families of one-sided sign-test p (JEPA "
            f"pressure->wake error < recon), {worst[0]}, K={Kh}",
        )
    json.dump(
        {"part": "sensing_cf", "numbers": nums}, open(PARTS / "sensing_cf.json", "w"), indent=2
    )


def print_report(headline_ci, paired, rows, causal_delta, r2_base, r2_causal, k_list) -> None:
    print("\n========== C-F HEADLINE: pressure -> wake R^2 (qDEIM, test_b) ==========")
    print(
        f"{'family':14s} "
        + "  ".join(f"K={K:<2d}" for K in k_list)
        + "   (CI at K="
        + str(HEADLINE_K if HEADLINE_K in k_list else k_list[0])
        + ")"
    )
    Kh = HEADLINE_K if HEADLINE_K in k_list else k_list[0]
    for fam in FAMILIES:
        vals = "  ".join(f"{headline_ci[fam.name][K]['r2']:+.2f}" for K in k_list)
        ci = headline_ci[fam.name][Kh]
        print(f"{fam.name:14s} {vals}   [{ci['ci_lo']:+.2f},{ci['ci_hi']:+.2f}]")
    print(
        f"\ndirect (no-latent) pressure->wake at K={Kh}: "
        f"{headline_ci[FAMILIES[0].name][Kh]['r2_direct']:+.3f} (JEPA window)"
    )
    print(
        f"causal-clip delta on JEPA pressure->wake (K={Kh}): "
        f"{causal_delta:+.4f}  (base {r2_base:+.3f} -> causal {r2_causal:+.3f})"
    )
    print("\npaired JEPA vs reconstructive (sign test + case-clustered CI):")
    for k, v in paired.items():
        print(
            f"  {k:32s} sign {v['sign_k']}/{v['sign_n']} p={v['sign_p']:.3f} "
            f"case-delta={v['case_mean_delta']:+.3f} "
            f"CI[{v['case_mean_ci'][0]:+.3f},{v['case_mean_ci'][1]:+.3f}]"
        )


if __name__ == "__main__":
    main()
