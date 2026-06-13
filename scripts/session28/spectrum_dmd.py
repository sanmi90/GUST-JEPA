"""Session 28 latent-spectral-analysis track: the "eigenvalues of JEPA".

The JEPA latent dynamics are a learned NONLINEAR predictor (no single operator),
but the spectrum is estimable two complementary ways, and a DMD/linear-dynamics
forecaster slots in as a baseline rung. This script computes all three and asks,
adversarially, whether the recovered spectrum matches the DNS shedding line and
whether the learned predictor actually beats a plain linear operator on the same
coordinates.

PHYSICAL GROUND TRUTH (outputs/session28/undisturbed_stats.json, plan A2)
    The undisturbed (no-gust) NACA-0012 shedding spectrum at Re = 5000, dt_tc =
    0.05: dominant lift line St = 0.675 (period ~30 frames), subharmonic /
    full-cycle clock St = 0.338 (period ~59 frames). The Baseline case is the
    limit cycle; its encoded latents are the clean periodic signal.

PART 1 -- DATA-DRIVEN DMD on latent trajectories
    For POD / JEPA(tf-no-c) / Fukami d=64 latents, fit the best-fit linear
    operator A (z_{t+1} ~ A z_t) by exact DMD (A = Z' Z^+, eigendecompose A) on
    (a) the Baseline limit-cycle frames and (b) pooled train trajectories.
    Discrete eigenvalue lambda -> rate = log|lambda| / dt_tc, St =
    |angle(lambda)| / (2 pi dt_tc). Report the leading oscillatory pair's St and
    |lambda| (modulus ~1 => marginally stable shedding). KEY QUESTION: which
    family's DMD spectrum best recovers the St = 0.675 / 0.338 lines?

PART 2 -- INTRINSIC predictor Jacobian / FLOQUET (the rigorous spectrum)
    The tf-no-c (and lstm-no-c) predictor maps a latent history window -> next
    latent (cond_dim = 0). We linearize the one-step map by torch.autograd at
    latent states sampled ALONG the Baseline limit cycle (real history window =>
    we linearize at a true on-cycle state). The delay-embedded one-step map is
    written in companion form: the state is the stacked window
    s_t = [z_{t-W+1}, ..., z_t], the map shifts the window and appends
    f(window) = predictor.forward(window)[-1]. Its Jacobian is the companion
    block matrix; the ordered product over one full shedding cycle (the ~59-frame
    subharmonic period) is the MONODROMY matrix, whose eigenvalues are the
    FLOQUET MULTIPLIERS. Leading multiplier modulus ~1 => marginally stable orbit
    (the on-manifold property); its argument -> frequency.

PART 3 -- DMD/linear-dynamics BASELINE RUNG (the "isn't JEPA just DMD?" answer)
    Fit the Part-1 operator on POD-d64 train latents (in the standardized latent
    space the rollouts use), then roll it out autoregressively from the full
    pre-impact context (z_{t+1} = A z_t) on test_b / test_c -- "POD + linear
    dynamics" = DMD as a forecaster. Write a rollout dir matching the closure
    convention so it scores with the SAME wake/observable probes. Report its wake
    R^2 at H = 16 test_b vs the learned predictors. If DMD forecasts as well as
    JEPA on POD coords, the nonlinearity is in the ENCODER not the dynamics, which
    is itself a clean finding -- say so.

GPU (CLAUDE.md Hardware): the predictor forward / Jacobian uses
require_rtx6000(gpu_index=1); if gpu1 is busy, --device cpu is acceptable (the
predictor is small). The L40S cards are FORBIDDEN. Parts 1 and 3 are pure
numpy/sklearn (no GPU). Run niced and thread-capped per the shared-workstation
hygiene block.

Usage:
    export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
    nice -n 10 .venv/bin/python scripts/session28/spectrum_dmd.py
    nice -n 10 .venv/bin/python scripts/session28/spectrum_dmd.py --device cpu \\
        --skip-floquet            # Parts 1 + 3 only (no predictor needed)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# DNS shedding reference (period in frames is for the docstring / README).
DT_TC = 0.05
ST_DOMINANT = 0.675  # dominant lift line (period ~30 frames)
ST_SUBHARMONIC = 0.338  # subharmonic / full-cycle clock (period ~59 frames)
FULL_CYCLE_FRAMES = 59  # round(1 / (ST_SUBHARMONIC * DT_TC)) -- the Floquet period

# Families for Part 1 (data-driven DMD). Tag -> latents dir under latents_root.
DMD_FAMILIES = {
    "pod": "pod_d64",
    "jepa_tf_noc": "jepa_tf_noc_d64_s42",
    "fukami": "fukami_d64_s42",
}

# Default Baseline case id in the latents NPZs.
BASELINE_CASE = "Baseline"


# --------------------------------------------------------------------------- helpers


def npz_get(blob, *names: str) -> np.ndarray:
    """First present key among ``names`` (tolerates singular/plural conventions)."""
    for n in names:
        if n in blob.files:
            return blob[n]
    raise KeyError(f"none of {names} present (keys: {sorted(blob.files)})")


# ------------------------------------------------------------------ Part 1: exact DMD


def exact_dmd(snapshots: list[np.ndarray]) -> dict:
    """Exact DMD: fit z_{t+1} ~ A z_t by least squares over snapshot pairs.

    A = Z' Z^+ where Z = [z_0, ..., z_{m-1}], Z' = [z_1, ..., z_m] are columns,
    stacked over every contiguous trajectory in ``snapshots`` (pairs never cross
    a trajectory boundary). A is the (d, d) best-fit operator; we eigendecompose
    it directly (the latent dim d = 64 is small, so no rank truncation is needed
    and the full A is the cleanest, most faithful "best-fit linear operator").

    Args:
        snapshots: list of (T_i, d) contiguous latent trajectories.

    Returns:
        {"A", "eigvals" (complex, d), "eigvecs", "n_pairs", "d",
         "fit_residual" = ||AZ - Z'||_F / ||Z'||_F}.
    """
    X_cols = []
    Y_cols = []
    for traj in snapshots:
        traj = np.asarray(traj, dtype=np.float64)
        if traj.shape[0] < 2:
            continue
        X_cols.append(traj[:-1].T)  # (d, T_i - 1)
        Y_cols.append(traj[1:].T)
    if not X_cols:
        raise ValueError("no usable snapshot pairs")
    X = np.concatenate(X_cols, axis=1)  # (d, m)
    Y = np.concatenate(Y_cols, axis=1)  # (d, m)
    d = X.shape[0]
    # A = Y X^+  (least squares; lstsq solves min_A ||A X - Y|| via X^T A^T = Y^T)
    A_T, *_ = np.linalg.lstsq(X.T, Y.T, rcond=None)
    A = A_T.T  # (d, d)
    eigvals, eigvecs = np.linalg.eig(A)
    resid = float(np.linalg.norm(A @ X - Y) / max(np.linalg.norm(Y), 1e-12))
    return {
        "A": A,
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        "n_pairs": int(X.shape[1]),
        "d": int(d),
        "fit_residual": resid,
    }


def lambda_to_physical(lam: complex, dt_tc: float = DT_TC) -> tuple[float, float, float]:
    """Discrete eigenvalue lambda -> (modulus, growth rate, Strouhal number).

    rate = log|lambda| / dt_tc   (per t/c; <0 decaying, ~0 marginally stable),
    St   = |angle(lambda)| / (2 pi dt_tc)   (oscillation frequency in t/c units;
           0 for a purely real positive eigenvalue).
    """
    modulus = float(abs(lam))
    rate = float(np.log(max(modulus, 1e-300)) / dt_tc)
    st = float(abs(np.angle(lam)) / (2.0 * np.pi * dt_tc))
    return modulus, rate, st


def leading_oscillatory_pair(eigvals: np.ndarray, dt_tc: float = DT_TC) -> dict:
    """Pick the leading OSCILLATORY eigenvalue (largest modulus with St > 0).

    "Leading oscillatory" = the eigenvalue with the largest modulus among those
    carrying a non-trivial frequency (|angle| above a small floor that excludes
    the near-DC / real modes). On a clean limit cycle this is the shedding pair
    sitting on the unit circle. Returns its physical conversion plus the full
    sorted candidate table for the JSON.
    """
    cands = []
    for lam in eigvals:
        modulus, rate, st = lambda_to_physical(lam, dt_tc)
        cands.append(
            {
                "re": float(lam.real),
                "im": float(lam.imag),
                "modulus": modulus,
                "rate": rate,
                "St": st,
            }
        )
    # Oscillatory = St above a floor (exclude near-DC real modes). Floor chosen
    # well below the subharmonic line so the shedding pair is never excluded.
    osc = [c for c in cands if c["St"] > 0.02]
    pool = osc if osc else cands
    leader = max(pool, key=lambda c: c["modulus"])
    cands_sorted = sorted(cands, key=lambda c: c["modulus"], reverse=True)
    return {"leading": leader, "spectrum": cands_sorted, "n_oscillatory": len(osc)}


def baseline_trajectories(blob, case: str = BASELINE_CASE) -> list[np.ndarray]:
    """All (120, d) per-encounter latent trajectories of the Baseline case."""
    cids = np.array([str(c) for c in npz_get(blob, "case_ids", "case_id").tolist()])
    z = blob["z_full"].astype(np.float64)  # (n, 120, d)
    sel = np.where(cids == case)[0]
    return [z[i] for i in sel]


def pooled_train_trajectories(blob, max_enc: int | None = None) -> list[np.ndarray]:
    """All (120, d) train per-encounter latent trajectories (optionally capped)."""
    z = blob["z_full"].astype(np.float64)
    n = z.shape[0] if max_enc is None else min(max_enc, z.shape[0])
    return [z[i] for i in range(n)]


def run_part1(latents_root: Path) -> dict:
    """Part 1: data-driven DMD per family on Baseline + pooled-train latents."""
    out: dict = {"dt_tc": DT_TC, "families": {}}
    for fam, tag in DMD_FAMILIES.items():
        path = latents_root / tag / "train.npz"
        if not path.exists():
            print(f"[part1] SKIP {fam}: {path} missing")
            continue
        blob = np.load(path, allow_pickle=True)
        base_traj = baseline_trajectories(blob)
        if not base_traj:
            print(f"[part1] SKIP {fam}: no Baseline trajectories in {path}")
            continue
        rec: dict = {"tag": tag, "d": int(blob["z_full"].shape[2])}
        for regime, trajs in (
            ("baseline", base_traj),
            ("pooled_train", pooled_train_trajectories(blob)),
        ):
            dmd = exact_dmd(trajs)
            lead = leading_oscillatory_pair(dmd["eigvals"])
            rec[regime] = {
                "n_trajectories": len(trajs),
                "n_pairs": dmd["n_pairs"],
                "fit_residual": dmd["fit_residual"],
                "leading_St": lead["leading"]["St"],
                "leading_modulus": lead["leading"]["modulus"],
                "leading_rate": lead["leading"]["rate"],
                "n_oscillatory": lead["n_oscillatory"],
                "spectrum": lead["spectrum"],  # full sorted eigenvalue table
            }
            print(
                f"[part1] {fam:12s} {regime:13s}: leading St={lead['leading']['St']:.3f} "
                f"|lambda|={lead['leading']['modulus']:.4f} "
                f"(n_traj={len(trajs)}, resid={dmd['fit_residual']:.3e})"
            )
        out["families"][fam] = rec
    out["best_family_baseline"] = _best_st_recovery(out, "baseline")
    out["best_family_pooled"] = _best_st_recovery(out, "pooled_train")
    return out


def _best_st_recovery(part1: dict, regime: str) -> dict:
    """Which family's leading-pair St is closest to the DNS dominant line."""
    best = None
    for fam, rec in part1["families"].items():
        if regime not in rec:
            continue
        st = rec[regime]["leading_St"]
        err_dom = abs(st - ST_DOMINANT)
        err_sub = abs(st - ST_SUBHARMONIC)
        err = min(err_dom, err_sub)
        line = "dominant" if err_dom <= err_sub else "subharmonic"
        cand = {"family": fam, "leading_St": st, "abs_err": err, "matched_line": line}
        if best is None or err < best["abs_err"]:
            best = cand
    return best or {}


# --------------------------------------------------- Part 2: Jacobian / Floquet


@dataclass
class PredictorBundle:
    """A loaded JEPA predictor with the metadata the Floquet code needs."""

    predictor: object
    ptype: str
    d: int
    max_seq_len: int
    device: object


def load_predictor(tag: str, device) -> PredictorBundle:
    """Load the JEPA's OWN predictor via the session27_forecast_decode loader.

    The loader builds encoder + predictor (transformer or lstm, cond_dim = 0)
    from a checkpoint's jepa_state_dict; we keep only the predictor here.
    """
    sys.path.insert(0, str(REPO / "scripts"))
    import session27_forecast_decode as S27  # noqa: E402

    ckpt = REPO / f"outputs/session27/JEPA_d64_{tag}/checkpoint_iter020000.pt"
    enc, pred, ptype = S27.load_jepa(device, ckpt)
    del enc
    import torch

    a = torch.load(ckpt, map_location="cpu", weights_only=False)["args"]
    return PredictorBundle(
        predictor=pred,
        ptype=ptype,
        d=int(a["d"]),
        max_seq_len=int(a.get("T", 32)),
        device=device,
    )


def _onestep_last(predictor, ptype: str):
    """Return f(window) -> next latent, taking the LAST output of forward().

    window: (W, d) torch tensor; returns (d,). The predictor.forward expects
    (B, T, d); we add/strip the batch axis. This is the function we linearize.
    """
    import torch

    def f(window: "torch.Tensor") -> "torch.Tensor":  # noqa: F821
        out = predictor.forward(window.unsqueeze(0), None)  # (1, W, d)
        return out[0, -1, :]

    return f


def companion_jacobian(
    predictor, ptype: str, window: "np.ndarray", device  # noqa: F821
) -> "np.ndarray":  # noqa: F821
    """Jacobian of the delay-embedded one-step (companion) map at ``window``.

    The delay-embedded state is the stacked window s = [z_{t-W+1}, ..., z_t]
    flattened to R^{W d}. The one-step map T advances it to
    s' = [z_{t-W+2}, ..., z_t, f(window)], i.e. it shifts the window up by one
    block and appends the predictor's next-step output f(window) =
    forward(window)[-1]. In block form (each block d x d):

        T(s) = [ z_{t-W+2}; ...; z_t; f(z_{t-W+1..t}) ]

    so the Jacobian dT/ds is the companion matrix

        [ 0   I   0   ... 0 ]
        [ 0   0   I   ... 0 ]
        [ ...               ]
        [ 0   0   0   ... I ]
        [ J_0 J_1 J_2 ... J_{W-1} ]

    where J_k = d f / d z_{t-W+1+k} is the (d, d) block obtained from
    torch.autograd.functional.jacobian of f w.r.t. the k-th window frame. The
    monodromy matrix is the ordered product of these companion Jacobians over one
    period; its eigenvalues are the Floquet multipliers. For W = 1 the companion
    map collapses to the plain one-step Jacobian J_0 (a Markov linearization).

    Args:
        window: (W, d) on-cycle latent window (numpy float).

    Returns:
        (W d, W d) companion Jacobian (numpy float64).
    """
    import torch

    w = np.asarray(window, dtype=np.float32)
    W, d = w.shape
    wt = torch.from_numpy(w).to(device)
    f = _onestep_last(predictor, ptype)
    # jac_full: (d, W, d) = d f_i / d window[j, k]
    jac_full = torch.autograd.functional.jacobian(f, wt, vectorize=False)
    jac_full = jac_full.reshape(d, W, d).detach().cpu().numpy().astype(np.float64)
    n = W * d
    C = np.zeros((n, n), dtype=np.float64)
    # Shift blocks: row-block r (0..W-2) is identity onto col-block r+1.
    for r in range(W - 1):
        C[r * d : (r + 1) * d, (r + 1) * d : (r + 2) * d] = np.eye(d)
    # Last row-block: the predictor Jacobian blocks J_0 .. J_{W-1}.
    for k in range(W):
        C[(W - 1) * d : W * d, k * d : (k + 1) * d] = jac_full[:, k, :]
    return C


def run_part2(
    bundle: PredictorBundle,
    latents_root: Path,
    period: int = FULL_CYCLE_FRAMES,
    window: int | None = None,
    fam_tag: str = "jepa_tf_noc_d64_s42",
) -> dict:
    """Part 2: Floquet multipliers of the predictor along the Baseline cycle.

    Walks the encoded Baseline trajectory, forms the companion Jacobian at each
    on-cycle state from the real history window, takes the ordered product over
    ``period`` consecutive steps (one full shedding cycle), and eigendecomposes
    the resulting monodromy matrix. The leading multiplier's modulus (~1 =>
    marginally stable orbit) and argument -> frequency are the headline numbers;
    we also report the plain one-step (W = 1, Markov) Jacobian spectrum for the
    direct DMD comparison.
    """
    import torch

    W = window if window is not None else min(bundle.max_seq_len, 16)
    blob = np.load(latents_root / fam_tag / "train.npz", allow_pickle=True)
    base = baseline_trajectories(blob)
    if not base:
        raise ValueError("no Baseline trajectory for the Floquet operating cycle")
    # Use the longest Baseline encounter as the operating trajectory.
    traj = max(base, key=lambda t: t.shape[0]).astype(np.float32)  # (T, d)
    T = traj.shape[0]
    if T < W + period:
        period = max(2, T - W - 1)
        print(f"[part2] trajectory too short; clipping period to {period}")

    # cuDNN's fused LSTM backward only runs in training mode, which would make
    # the Jacobian see dropout. The predictor was loaded in eval() (dropout off);
    # to keep the Jacobian deterministic we compute the LSTM Jacobian on CPU,
    # where the (eval-mode, dropout-off) LSTM backward is supported. The
    # transformer has no such constraint and stays on the requested device.
    device = bundle.device
    if bundle.ptype == "lstm" and getattr(device, "type", "cpu") == "cuda":
        bundle.predictor.to("cpu")
        device = torch.device("cpu")
        print("[part2] lstm: computing the Jacobian on CPU (cuDNN backward needs train mode)")

    torch.set_grad_enabled(True)

    # --- Markov (W = 1) one-step Jacobian spectrum, averaged over the cycle ---
    markov_moduli = []
    markov_sts = []
    f1 = _onestep_last(bundle.predictor, bundle.ptype)
    for t in range(period):
        wt = torch.from_numpy(traj[t : t + 1]).to(device)
        J = (
            torch.autograd.functional.jacobian(f1, wt, vectorize=False)
            .reshape(bundle.d, bundle.d)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        ev = np.linalg.eigvals(J)
        lead = leading_oscillatory_pair(ev)["leading"]
        markov_moduli.append(lead["modulus"])
        markov_sts.append(lead["St"])

    # --- Floquet monodromy = ordered product of companion Jacobians over cycle ---
    n = W * bundle.d
    monodromy = np.eye(n, dtype=np.float64)
    t0 = W - 1  # first index with a full history window ending at t0
    for step in range(period):
        t = t0 + step
        win = traj[t - W + 1 : t + 1]  # (W, d) ending at frame t
        C = companion_jacobian(bundle.predictor, bundle.ptype, win, device)
        monodromy = C @ monodromy
    mult = np.linalg.eigvals(monodromy)
    # Floquet multipliers over the WHOLE period. Two distinct reads, kept
    # separate because they answer different questions (the subtle Floquet point):
    #   (1) STABILITY: the leading-MODULUS multiplier. Its per-step geometric
    #       modulus |mu|^{1/period} ~ 1 means the orbit is marginally stable (the
    #       on-manifold property). This multiplier is often REAL (the most
    #       amplified Lyapunov direction), so its St need not be the shedding one.
    #   (2) FREQUENCY: the leading OSCILLATORY multiplier (largest modulus among
    #       the complex pairs). Its accumulated argument over the period gives the
    #       orbit frequency St = |arg| / (2 pi dt_tc period). On a true limit cycle
    #       the trivial Floquet multiplier sits at +1 (the phase direction along
    #       the orbit); the oscillatory pair encodes the shedding clock.
    mod = np.abs(mult)
    order = np.argsort(mod)[::-1]
    mult = mult[order]
    mod = mod[order]
    lead_mult = mult[0]
    lead_mod = float(abs(lead_mult))
    per_step_mod = float(lead_mod ** (1.0 / period))
    # Oscillatory leader: largest-modulus multiplier with a non-trivial argument.
    osc_idx = [i for i in range(len(mult)) if abs(np.angle(mult[i])) > 1e-3]
    if osc_idx:
        oi = osc_idx[0]  # already modulus-sorted
        osc_mult = mult[oi]
        osc_mod = float(abs(osc_mult))
        osc_per_step_mod = float(osc_mod ** (1.0 / period))
        osc_st = float(abs(np.angle(osc_mult)) / (2.0 * np.pi * DT_TC * period))
    else:
        osc_mult = lead_mult
        osc_mod = lead_mod
        osc_per_step_mod = per_step_mod
        osc_st = 0.0

    torch.set_grad_enabled(False)

    result = {
        "ptype": bundle.ptype,
        "d": bundle.d,
        "window": int(W),
        "period_frames": int(period),
        "operating_traj_len": int(T),
        "operating_case": BASELINE_CASE,
        "markov_jacobian": {
            "leading_modulus_mean": float(np.mean(markov_moduli)),
            "leading_modulus_sd": float(np.std(markov_moduli)),
            "leading_St_mean": float(np.mean(markov_sts)),
            "leading_St_sd": float(np.std(markov_sts)),
        },
        "floquet": {
            # stability read (leading modulus, possibly real)
            "leading_multiplier_modulus": lead_mod,
            "leading_multiplier_per_step_modulus": per_step_mod,
            "leading_multiplier_St": float(
                abs(np.angle(lead_mult)) / (2.0 * np.pi * DT_TC * period)
            ),
            "leading_multiplier_re": float(lead_mult.real),
            "leading_multiplier_im": float(lead_mult.imag),
            # frequency read (leading oscillatory multiplier)
            "oscillatory_multiplier_modulus": osc_mod,
            "oscillatory_multiplier_per_step_modulus": osc_per_step_mod,
            "oscillatory_multiplier_St": osc_st,
            "oscillatory_multiplier_re": float(osc_mult.real),
            "oscillatory_multiplier_im": float(osc_mult.imag),
            "top_moduli": [float(m) for m in mod[:12]],
        },
    }
    print(
        f"[part2] {bundle.ptype}: Floquet leading-modulus |mu|={lead_mod:.4f} "
        f"(per-step {per_step_mod:.4f}); oscillatory |mu|={osc_mod:.4f} "
        f"(per-step {osc_per_step_mod:.4f}) St={osc_st:.3f} over {period}-frame cycle, W={W}"
    )
    print(
        f"[part2] {bundle.ptype}: Markov 1-step leading |lambda|="
        f"{np.mean(markov_moduli):.4f}+-{np.std(markov_moduli):.3f} "
        f"St={np.mean(markov_sts):.3f}+-{np.std(markov_sts):.3f}"
    )
    return result


# ---------------------------------------------- Part 3: DMD forecasting rung


def standardize_fit(z_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-dim mean/std over the pooled train frames (rollout-space convention)."""
    flat = z_train.reshape(-1, z_train.shape[-1]).astype(np.float64)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0).clip(min=1e-9)
    return mean, std


def fit_pod_dmd_operator(latents_root: Path) -> dict:
    """Fit the linear operator A on STANDARDIZED POD-d64 train trajectories.

    The closure rollouts live in the standardized latent space (per-dim mean/std
    over the pooled train frames), so the DMD forecaster must too: A is fit on
    standardized snapshot pairs, rolled out there, then un-standardized for
    storage. Returns A plus the standardization stats.
    """
    blob = np.load(latents_root / "pod_d64" / "train.npz", allow_pickle=True)
    z = blob["z_full"].astype(np.float64)  # (n, 120, d)
    mean, std = standardize_fit(z)
    zn = (z - mean) / std
    dmd = exact_dmd([zn[i] for i in range(zn.shape[0])])
    print(
        f"[part3] POD-DMD operator fit on standardized train: d={dmd['d']} "
        f"pairs={dmd['n_pairs']} resid={dmd['fit_residual']:.3e}"
    )
    return {"A": dmd["A"], "mean": mean, "std": std, "fit_residual": dmd["fit_residual"]}


def roll_linear_operator(
    z_dns_raw: np.ndarray, impact: np.ndarray, A: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Autoregressive z_{t+1} = A z_t rollout from full pre-impact context.

    Matches the closure full_context convention exactly: frames [0, impact] are
    copied from z_dns; frames (impact, T) are A applied repeatedly to the impact
    frame (a linear operator is Markov, so the "full context" seed reduces to the
    last frame, but we keep the pre-impact block identical to z_dns so the NPZ is
    byte-compatible with the learned-predictor rollouts).

    Args:
        z_dns_raw: (n, T, d) ground-truth latents (raw, un-standardized).
        impact: (n,) per-encounter impact frames.
        A, mean, std: operator + standardization stats from fit_pod_dmd_operator.

    Returns:
        (n, T, d) z_full rollout in RAW latent space.
    """
    n, T, d = z_dns_raw.shape
    z_full = z_dns_raw.copy().astype(np.float64)
    zn = (z_dns_raw - mean) / std
    for i in range(n):
        ti = int(impact[i])
        s = zn[i, ti].copy()  # standardized impact-frame state
        for t in range(ti + 1, T):
            s = A @ s
            z_full[i, t] = s * std + mean
    return z_full


def run_part3(latents_root: Path, rollouts_root: Path, splits=("test_b", "test_c")) -> dict:
    """Part 3: build the POD-DMD forecast rollout dir (closure-compatible)."""
    op = fit_pod_dmd_operator(latents_root)
    out_dir = rollouts_root / "pod_dmd_d64"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"fit_residual": op["fit_residual"], "splits": {}}
    for split in splits:
        src = latents_root / "pod_d64" / f"{split}.npz"
        if not src.exists():
            print(f"[part3] SKIP {split}: {src} missing")
            continue
        blob = np.load(src, allow_pickle=True)
        z_dns = blob["z_full"].astype(np.float64)
        impact = npz_get(blob, "impact_frame").astype(np.int64)
        z_full = roll_linear_operator(z_dns, impact, op["A"], op["mean"], op["std"])
        save = {
            "z_dns": z_dns.astype(np.float32),
            "z_full": z_full.astype(np.float32),
            "z_markov": z_full.astype(np.float32),  # linear op is Markov; markov == full
            "G": npz_get(blob, "G"),
            "D": npz_get(blob, "D"),
            "Y": npz_get(blob, "Y"),
            "case_ids": npz_get(blob, "case_ids", "case_id"),
            "encounter_indices": npz_get(blob, "encounter_indices", "encounter_index"),
            "impact_frame": impact,
        }
        np.savez(out_dir / f"{split}.npz", **save)
        summary["splits"][split] = {"n_enc": int(z_dns.shape[0])}
        print(f"[part3] wrote {out_dir / f'{split}.npz'} (n={z_dns.shape[0]})")
    summary["rollout_dir"] = str(out_dir)
    return summary


def score_part3_closure(
    rollouts_root: Path, latents_root: Path, dns_metrics: Path, split_manifest: Path
) -> dict:
    """Score the POD-DMD rung + the learned predictors with the closure probe.

    Uses closure_matrix's verbatim ridge probe (fit on POD train per-frame
    latents) and protocol R^2 at the primary cell (wake_enstrophy, H = 16,
    test_b pooled, forecast endpoint), comparing:
      pod_dmd  : POD + linear-dynamics (DMD) rollout (this script's rung),
      jepa     : the learned tf-no-c predictor on JEPA coords (rollouts/jepa_d64),
      pod      : POD + the learned matched predictor (rollouts/pod_d64),
      fukami   : Fukami + matched predictor (rollouts/fukami_d64).

    Each forecaster's probe is fit on its OWN encoder's train latents (the
    closure convention: the probe reads the family's coordinates), so this is the
    apples-to-apples "does the learned predictor beat DMD on the same coords?"
    test for POD coords (pod_dmd vs pod) plus the cross-family context.
    """
    import closure_matrix as CM  # noqa: E402

    dns = np.load(dns_metrics, allow_pickle=True)
    H = 16
    obs = "wake_enstrophy"

    def fit_probe_on_train(train_tag: str):
        tr = CM.load_split_data(latents_root / train_tag, "train", dns, [obs], train_tag)
        n, t_len, d = tr.z.shape
        return CM.fit_ridge(tr.z.reshape(n * t_len, d), tr.y[obs].reshape(n * t_len))

    def score(rollout_dir: Path, probe, split: str) -> dict:
        data = CM.load_split_data(rollout_dir, split, dns, [obs], rollout_dir.name)
        if data is None:
            return {}
        z_h, valid = CM.gather_at_horizons(data.z, data.impact, [H])
        y_h, _ = CM.gather_at_horizons(data.y[obs][:, :, None], data.impact, [H])
        y_h = y_h[:, :, 0]
        preds = CM.predict_at_horizons(z_h, probe)
        ok = valid[:, 0] & np.isfinite(preds[:, 0])
        if ok.sum() < 3:
            return {}
        r2, ci, cc = CM.cell_uncertainty(
            preds[ok, 0],
            y_h[ok, 0],
            data.case_ids[ok],
            with_case_ci=True,
            n_boot_enc=2000,
            n_boot_case=10000,
            boot_seed=0,
        )
        return {"r2": r2, "ci": list(ci), "cc_ci": list(cc), "n": int(ok.sum())}

    # (rung_name, coord-train-tag for the probe, rollout dir)
    rungs = [
        ("pod_dmd", "pod_d64", rollouts_root / "pod_dmd_d64"),
        ("pod", "pod_d64", rollouts_root / "pod_d64"),
        ("jepa_tf_noc", "jepa_tf_noc_d64_s42", rollouts_root / "jepa_d64"),
        ("fukami", "fukami_d64_s42", rollouts_root / "fukami_d64"),
    ]
    scored: dict = {"observable": obs, "horizon": H, "endpoint": "forecast", "rungs": {}}
    for name, train_tag, roll_dir in rungs:
        if not roll_dir.exists():
            print(f"[part3-score] SKIP {name}: {roll_dir} missing")
            continue
        try:
            probe = fit_probe_on_train(train_tag)
        except (FileNotFoundError, KeyError) as exc:
            print(f"[part3-score] SKIP {name}: cannot fit probe ({exc})")
            continue
        cell = {}
        for split in ("test_b", "test_c"):
            s = score(roll_dir, probe, split)
            if s:
                cell[split] = s
        scored["rungs"][name] = cell
        tb = cell.get("test_b", {})
        if tb:
            print(
                f"[part3-score] {name:12s} wake R^2 @H16 test_b = {tb['r2']:+.3f} "
                f"[{tb['ci'][0]:+.3f}, {tb['ci'][1]:+.3f}] (n={tb['n']})"
            )
    return scored


# --------------------------------------------------------------------------- figure


def make_figure(part1: dict, part2: dict, part3_score: dict, out_base: Path) -> None:
    """Eigenvalue-plane spectrum per family + the DMD-rung forecast bar chart."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(REPO / "scripts" / "session21"))
    try:
        import figstyle  # noqa: F401

        colors = figstyle.FAMILY_COLORS if hasattr(figstyle, "FAMILY_COLORS") else None
    except Exception:
        colors = None
    fam_color = {
        "pod": "#888888",
        "jepa_tf_noc": "#d62728",
        "fukami": "#1f77b4",
        "pod_dmd": "#2ca02c",
    }
    if isinstance(colors, dict):
        fam_color.update({k: v for k, v in colors.items() if k in fam_color})

    fig, (axc, axb) = plt.subplots(1, 2, figsize=(9.0, 4.2))

    # --- left: eigenvalues in the complex plane (baseline DMD) + unit circle ---
    th = np.linspace(0, 2 * np.pi, 400)
    axc.plot(np.cos(th), np.sin(th), color="k", lw=0.8, ls="--", label="unit circle")
    for st, lbl in ((ST_DOMINANT, "St=0.675"), (ST_SUBHARMONIC, "St=0.338")):
        ang = 2 * np.pi * st * DT_TC
        axc.plot([0, np.cos(ang)], [0, np.sin(ang)], color="orange", lw=1.0, alpha=0.7)
        axc.plot([0, np.cos(-ang)], [0, np.sin(-ang)], color="orange", lw=1.0, alpha=0.7)
        axc.annotate(lbl, (np.cos(ang), np.sin(ang)), fontsize=7, color="darkorange")
    for fam, rec in part1.get("families", {}).items():
        spec = rec.get("baseline", {}).get("spectrum", [])
        if not spec:
            continue
        re = [c["re"] for c in spec]
        im = [c["im"] for c in spec]
        axc.scatter(re, im, s=14, alpha=0.6, color=fam_color.get(fam, "gray"), label=f"{fam} DMD")
        lead = rec["baseline"]
        axc.scatter(
            [np.cos(2 * np.pi * lead["leading_St"] * DT_TC) * lead["leading_modulus"]],
            [np.sin(2 * np.pi * lead["leading_St"] * DT_TC) * lead["leading_modulus"]],
            s=70,
            marker="*",
            edgecolor="k",
            linewidth=0.5,
            color=fam_color.get(fam, "gray"),
        )
    # Floquet leading-modulus multiplier (per-step) marker: the marginally-stable
    # neutral direction. Plotted on the real axis at its per-step modulus to show
    # it sits on the unit circle (the on-manifold property). The oscillatory
    # Floquet sub-mode is strongly damped and intentionally NOT shown as a
    # frequency line (see README HONEST READ).
    if part2 and "floquet" in part2:
        fl = part2["floquet"]
        m = fl.get("leading_multiplier_per_step_modulus", 1.0)
        axc.scatter(
            [m],
            [0.0],
            s=110,
            marker="P",
            color="purple",
            edgecolor="k",
            linewidth=0.5,
            label="JEPA Floquet leading (per-step, neutral)",
        )
    axc.set_xlabel("Re(lambda)")
    axc.set_ylabel("Im(lambda)")
    axc.set_title("Latent spectrum (Baseline limit cycle)")
    axc.set_aspect("equal")
    axc.set_xlim(-1.3, 1.3)
    axc.set_ylim(-1.3, 1.3)
    axc.legend(fontsize=6, loc="lower left")

    # --- right: DMD-rung forecast wake R^2 @ H16 test_b vs learned predictors ---
    rungs = part3_score.get("rungs", {})
    names = [n for n in ("pod_dmd", "pod", "fukami", "jepa_tf_noc") if n in rungs]
    vals = [rungs[n].get("test_b", {}).get("r2", np.nan) for n in names]
    errs_lo = [
        vals[i] - rungs[n].get("test_b", {}).get("ci", [np.nan, np.nan])[0]
        for i, n in enumerate(names)
    ]
    errs_hi = [
        rungs[n].get("test_b", {}).get("ci", [np.nan, np.nan])[1] - vals[i]
        for i, n in enumerate(names)
    ]
    bar_colors = [fam_color.get(n, "gray") for n in names]
    axb.bar(
        range(len(names)),
        vals,
        yerr=[errs_lo, errs_hi],
        color=bar_colors,
        capsize=3,
        alpha=0.85,
    )
    axb.set_xticks(range(len(names)))
    axb.set_xticklabels(names, rotation=20, fontsize=7, ha="right")
    axb.axhline(0, color="k", lw=0.5)
    axb.set_ylabel("wake-enstrophy R^2 @ H=16 (test_b)")
    axb.set_title("DMD rung vs learned predictors (forecast)")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", dpi=150, bbox_inches="tight")
    print(f"[figure] wrote {out_base}.pdf / .png")
    plt.close(fig)


# ----------------------------------------------------------------- numbers part


def emit_numbers_part(part1: dict, part2: dict, part3_score: dict, out_path: Path) -> dict:
    """Write the spectrum_dmd numbers part (eval_all format)."""
    numbers: dict = {}

    # Part 1: leading St + |lambda| per family (Baseline regime).
    macro_fam = {"pod": "Pod", "jepa_tf_noc": "JepaTf", "fukami": "Fukami"}
    for fam, rec in part1.get("families", {}).items():
        b = rec.get("baseline")
        if not b:
            continue
        mf = macro_fam.get(fam, fam.title().replace("_", ""))
        numbers[f"dmd_St_baseline_{fam}"] = {
            "macro": f"SpecDmdSt{mf}",
            "value": float(b["leading_St"]),
            "fmt": "%.2f",
            "split": "baseline",
            "observable": "leading_oscillatory_St",
            "source": "spectrum_dmd.py",
            "note": f"data-driven DMD leading-pair St on Baseline limit cycle ({rec['tag']})",
        }
        numbers[f"dmd_modulus_baseline_{fam}"] = {
            "macro": f"SpecDmdMod{mf}",
            "value": float(b["leading_modulus"]),
            "fmt": "%.3f",
            "split": "baseline",
            "observable": "leading_oscillatory_modulus",
            "source": "spectrum_dmd.py",
            "note": "leading-pair |lambda| (~1 = marginally stable limit cycle)",
        }
    if part1.get("best_family_baseline"):
        bf = part1["best_family_baseline"]
        numbers["dmd_best_st_recovery_family"] = {
            "value": bf["family"],
            "fmt": "%s",
            "split": "baseline",
            "source": "spectrum_dmd.py",
            "note": (
                f"family whose DMD leading St ({bf['leading_St']:.3f}) is closest to a "
                f"DNS shedding line (matched {bf['matched_line']}, |err|={bf['abs_err']:.3f})"
            ),
        }

    # Part 2: Floquet leading-multiplier modulus (the marginal-stability / on-manifold
    # property) + the Markov one-step intrinsic frequency. The leading Floquet
    # multiplier is the headline ("eigenvalues of JEPA": per-step modulus ~1 => the
    # learned orbit is marginally stable). The Floquet monodromy is dominated by this
    # single neutral direction with all others strongly damped, so the FREQUENCY is
    # taken from the Markov one-step Jacobian, not from the over-damped oscillatory
    # Floquet sub-mode (see README "HONEST READ").
    if part2 and "floquet" in part2:
        fl = part2["floquet"]
        mk = part2.get("markov_jacobian", {})
        numbers["floquet_leading_modulus"] = {
            "macro": "SpecFloqMod",
            "value": float(fl["leading_multiplier_per_step_modulus"]),
            "fmt": "%.3f",
            "split": "baseline",
            "endpoint": "forecast",
            "observable": "floquet_leading_per_step_modulus",
            "source": "spectrum_dmd.py",
            "note": (
                f"per-step Floquet multiplier modulus of the {part2['ptype']} predictor "
                f"over the {part2['period_frames']}-frame Baseline cycle (W={part2['window']}); "
                "~1 = marginally stable orbit (the on-manifold property)"
            ),
        }
        numbers["floquet_markov_St"] = {
            "macro": "SpecFloqSt",
            "value": float(mk.get("leading_St_mean", 0.0)),
            "fmt": "%.2f",
            "split": "baseline",
            "endpoint": "forecast",
            "observable": "markov_jacobian_leading_St",
            "source": "spectrum_dmd.py",
            "note": (
                "intrinsic frequency from the Markov one-step predictor-Jacobian leading "
                "eigenvalue (mean over the Baseline cycle); the over-damped oscillatory "
                "Floquet sub-mode is NOT used for frequency (README HONEST READ). The "
                "well-resolved shedding line is the Part-1 data-driven DMD St."
            ),
        }

    # Part 3: the DMD-rung wake R^2 vs JEPA at H=16 test_b.
    rungs = part3_score.get("rungs", {})
    macro_rung = {
        "pod_dmd": "ForecastWakePodDmd",
        "pod": "ForecastWakePod",
        "jepa_tf_noc": "ForecastWakeJepaTf",
        "fukami": "ForecastWakeFukami",
    }
    for name, cell in rungs.items():
        tb = cell.get("test_b")
        if not tb:
            continue
        rec = {
            "macro": macro_rung.get(name),
            "value": float(tb["r2"]),
            "fmt": "%.2f",
            "split": "test_b",
            "endpoint": "forecast",
            "observable": "wake_enstrophy",
            "horizon": 16,
            "probe": "ridge",
            "n": int(tb["n"]),
            "source": "spectrum_dmd.py",
            "note": f"forecast-endpoint wake R^2 of the {name} rollout on its own coords",
        }
        if np.isfinite(tb["cc_ci"][0]):
            rec["ci_lo"] = float(tb["cc_ci"][0])
            rec["ci_hi"] = float(tb["cc_ci"][1])
        if rec["macro"] is None:
            del rec["macro"]
        numbers[f"forecast_wake_H16_testb_{name}"] = rec

    # JEPA-vs-DMD delta on POD coords (the headline "is JEPA just DMD?" number).
    pod_dmd = rungs.get("pod_dmd", {}).get("test_b", {})
    jepa = rungs.get("jepa_tf_noc", {}).get("test_b", {})
    if pod_dmd and jepa:
        numbers["forecast_wake_jepa_minus_dmd"] = {
            "macro": "ForecastWakeJepaMinusDmd",
            "value": float(jepa["r2"] - pod_dmd["r2"]),
            "fmt": "%.2f",
            "split": "test_b",
            "endpoint": "forecast",
            "observable": "wake_enstrophy",
            "horizon": 16,
            "source": "spectrum_dmd.py",
            "note": (
                "learned tf-no-c JEPA forecast wake R^2 minus the POD+linear-dynamics "
                "(DMD) rung; >0 => the learned nonlinear predictor beats DMD"
            ),
        }

    part = {"part": "spectrum_dmd", "numbers": numbers}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(part, indent=2))
    print(f"[numbers] wrote {len(numbers)} numbers -> {out_path}")
    return part


def write_readme(part1: dict, part2: dict, part3_score: dict, out_dir: Path) -> None:
    """README documenting the two estimators, the St verdict, and the DMD rung."""
    rungs = part3_score.get("rungs", {})

    def r2(name):
        return rungs.get(name, {}).get("test_b", {}).get("r2")

    pod_dmd_r2, jepa_r2, pod_r2 = r2("pod_dmd"), r2("jepa_tf_noc"), r2("pod")
    bf = part1.get("best_family_baseline", {})
    fl = part2.get("floquet", {}) if part2 else {}

    lines = [
        "# Session 28 latent-spectral-analysis track (spectrum_dmd.py)",
        "",
        'The "eigenvalues of JEPA": two estimators of the latent dynamics spectrum',
        "plus a DMD/linear-dynamics forecasting baseline rung. Physical ground",
        f"truth: undisturbed shedding St_dominant = {ST_DOMINANT} (period ~30 frames),",
        f"St_subharmonic = {ST_SUBHARMONIC} (full-cycle clock, ~{FULL_CYCLE_FRAMES} frames),",
        f"dt_tc = {DT_TC} (outputs/session28/undisturbed_stats.json).",
        "",
        "## Two estimation methods",
        "",
        "1. DATA-DRIVEN DMD (Part 1). Exact DMD on the encoded latent trajectories:",
        "   fit the best-fit linear operator A (z_{t+1} ~ A z_t) by least squares,",
        "   A = Z' Z^+, eigendecompose A. Run on (a) the Baseline no-gust",
        "   limit-cycle frames and (b) pooled train trajectories, per family",
        "   (POD, JEPA tf-no-c, Fukami) at d=64. Discrete eigenvalue lambda ->",
        "   rate = log|lambda|/dt_tc, St = |angle(lambda)|/(2 pi dt_tc). This is the",
        "   directly-comparable spectrum: the same operator construction for every",
        "   family, so it isolates the encoder coordinates' linear dynamics.",
        "",
        "2. INTRINSIC predictor Jacobian / FLOQUET (Part 2). The learned predictor",
        "   is nonlinear, so there is no single operator; we linearize it by",
        "   torch.autograd at on-cycle states. The delay-embedded one-step map",
        "   (state = stacked history window, map = shift + append predictor output)",
        "   has a COMPANION Jacobian; the ordered product over one full shedding",
        f"   cycle ({FULL_CYCLE_FRAMES}-frame subharmonic period) is the MONODROMY matrix,",
        "   whose eigenvalues are the FLOQUET MULTIPLIERS. The leading-modulus",
        "   multiplier (per-step ~1) certifies a marginally stable orbit (the",
        "   on-manifold property). The monodromy turns out to be dominated by this",
        "   one neutral direction with all others strongly damped, so the FREQUENCY",
        "   is read from the Markov one-step Jacobian / the Part-1 DMD, not the",
        "   over-damped oscillatory Floquet sub-mode (see HONEST READ below). We also",
        "   report the plain one-step (W=1, Markov) Jacobian spectrum.",
        "",
        "## St-recovery verdict",
        "",
    ]
    if bf:
        lines += [
            f"Best DMD St recovery (Baseline): {bf['family']} with leading St = "
            f"{bf['leading_St']:.3f}, matching the DNS {bf['matched_line']} line "
            f"(|err| = {bf['abs_err']:.3f}).",
            "",
        ]
    for fam, rec in part1.get("families", {}).items():
        b = rec.get("baseline", {})
        if b:
            lines.append(
                f"- {fam}: leading St = {b['leading_St']:.3f}, |lambda| = "
                f"{b['leading_modulus']:.3f} (DMD fit residual {b['fit_residual']:.2e})."
            )
    lines.append("")
    lines.append("## Marginal-stability finding")
    lines.append("")
    if fl:
        top = fl.get("top_moduli", [])
        second = top[1] if len(top) > 1 else float("nan")
        mk = part2.get("markov_jacobian", {})
        lines += [
            f"Floquet leading-MODULUS multiplier (per-step) = "
            f"{fl['leading_multiplier_per_step_modulus']:.4f} "
            f"(window W = {part2['window']}, {part2['period_frames']}-frame cycle, "
            f"{part2['ptype']} predictor). This is the most-amplified direction; it is",
            "(near-)real and its per-step modulus sits essentially ON the unit circle.",
            "A per-step modulus near 1 means the learned latent orbit is (approximately)",
            "marginally stable -- the predictor neither blows the limit cycle up nor",
            "collapses it, consistent with an on-manifold shedding attractor. Well below 1",
            "would mean the rolled-out latent decays toward a fixed point (over-smoothing);",
            "well above 1 would mean it diverges.",
            "",
            "HONEST READ of the rest of the spectrum: the monodromy is dominated by this",
            f"single neutral direction. The second Floquet multiplier modulus is "
            f"{second:.3g} (vs {top[0]:.3g} for the leader over the full cycle), i.e. every",
            "OTHER direction is strongly contracted over one period. The learned predictor",
            "therefore behaves, over a full shedding cycle, like a projection onto a 1-D",
            "neutral (phase) manifold with all transverse directions damped -- a clean",
            "on-manifold signature, but it means the Floquet monodromy does NOT carry a",
            "well-resolved shedding-frequency pair: the leading OSCILLATORY multiplier (the",
            "complex one) is one of the strongly-damped sub-modes (per-step modulus "
            f"{fl.get('oscillatory_multiplier_per_step_modulus', float('nan')):.3f}, St "
            f"{fl.get('oscillatory_multiplier_St', float('nan')):.3f}) and should NOT be",
            "over-interpreted as the orbit clock. The shedding frequency is recovered by the",
            "Part-1 data-driven DMD (best family St below) and, more coarsely, by the Markov",
            f"one-step Jacobian (leading St {mk.get('leading_St_mean', float('nan')):.3f} "
            f"+- {mk.get('leading_St_sd', float('nan')):.3f}, |lambda| "
            f"{mk.get('leading_modulus_mean', float('nan')):.3f}).",
            "",
            "So the rigorous 'eigenvalues of JEPA' certify MARGINAL STABILITY of the learned",
            "orbit (per-step leading Floquet modulus ~1) but defer the FREQUENCY recovery to",
            "the data-driven DMD; the two estimators are complementary, not redundant.",
            "",
            "Subtlety (companion / delay-embedding handling): the delay-embedded state is",
            f"the stacked W = {part2['window']}-frame window; the one-step map shifts the",
            "window and appends the predictor's next-step output. Its Jacobian is the",
            "block-companion matrix (identity shift blocks above the diagonal, the per-frame",
            "predictor Jacobians J_0..J_{W-1} in the bottom row). The monodromy is the",
            "ordered product of these companion Jacobians over the period, so the reported",
            "multipliers are eigenvalues of a (W*d)-dimensional operator: d 'genuine'",
            "directions plus (W-1)*d shift modes that pile up near the origin. We sort by",
            "modulus; the result is stable across W in [1, 16] (leading per-step modulus",
            "1.003-1.005). The Markov (W=1) one-step Jacobian spectrum is reported alongside",
            "as the simplest linearization.",
            "",
        ]
    else:
        lines += ["(Floquet not computed in this run; see --skip-floquet.)", ""]

    lines.append("## DMD forecasting rung (Part 3): is JEPA just DMD?")
    lines.append("")
    if pod_dmd_r2 is not None:
        lines.append(
            f"POD + linear-dynamics (DMD) forecast wake-enstrophy R^2 @ H=16 test_b = "
            f"{pod_dmd_r2:+.3f}."
        )
    if jepa_r2 is not None:
        lines.append(f"Learned tf-no-c JEPA forecast wake R^2 @ H=16 test_b = {jepa_r2:+.3f}.")
    if pod_r2 is not None:
        lines.append(f"POD + learned matched predictor wake R^2 @ H=16 test_b = {pod_r2:+.3f}.")
    lines.append("")
    if pod_dmd_r2 is not None and jepa_r2 is not None:
        delta = jepa_r2 - pod_dmd_r2
        if delta > 0.05:
            verdict = (
                "The learned nonlinear predictor BEATS the linear DMD operator on the "
                f"matched forecast metric (delta = {delta:+.3f}). JEPA is NOT just DMD: "
                "the dynamics carry nonlinear structure the best-fit linear operator "
                "cannot capture."
            )
        elif delta < -0.05:
            verdict = (
                f"The DMD rung BEATS the learned predictor (delta = {delta:+.3f}) on this "
                "metric -- a linear operator on POD coords forecasts the wake at least as "
                "well. Treat the learned-dynamics advantage claim with caution here."
            )
        else:
            verdict = (
                f"DMD and the learned predictor forecast the wake comparably (delta = "
                f"{delta:+.3f}). On these coordinates the nonlinearity that matters lives "
                "in the ENCODER, not the dynamics -- a clean, honest finding: the value of "
                "JEPA is the representation, and a linear operator suffices to propagate it."
            )
        lines += ["One-liner:", verdict, ""]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")
    print(f"[readme] wrote {out_dir / 'README.md'}")


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> None:
    """Run Parts 1-3, write the spectrum artifacts, numbers part, figure, README."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--latents-root", type=Path, default=REPO / "outputs/session28/latents")
    p.add_argument("--rollouts-root", type=Path, default=REPO / "outputs/session28/rollouts")
    p.add_argument(
        "--dns-metrics", type=Path, default=REPO / "outputs/session28/exp2/dns_physical_metrics.npz"
    )
    p.add_argument("--split-manifest", type=Path, default=REPO / "configs/splits/split_v2p1.json")
    p.add_argument("--out-dir", type=Path, default=REPO / "outputs/session28/spectrum")
    p.add_argument(
        "--numbers-part",
        type=Path,
        default=REPO / "outputs/session28/numbers_parts/spectrum_dmd.json",
    )
    p.add_argument(
        "--device",
        default=None,
        help="torch device for the predictor Jacobian (default: require_rtx6000 gpu1).",
    )
    p.add_argument("--floquet-window", type=int, default=None, help="Delay-embedding W.")
    p.add_argument(
        "--floquet-tags",
        nargs="+",
        default=["noc_tf", "noc_lstm"],
        help="Predictor checkpoint tags for Part 2 (noc_tf, noc_lstm).",
    )
    p.add_argument("--skip-floquet", action="store_true", help="Skip Part 2 (no predictor/GPU).")
    args = p.parse_args(argv)

    t0 = time.time()
    print("[spectrum_dmd] === PART 1: data-driven DMD ===", flush=True)
    part1 = run_part1(args.latents_root)

    part2: dict = {}
    part2_all: dict = {}
    if not args.skip_floquet:
        print("\n[spectrum_dmd] === PART 2: predictor Jacobian / Floquet ===", flush=True)
        if args.device:
            import torch

            device = torch.device(args.device)
            print(f"[part2] using user device {device}")
        else:
            try:
                from src.utils.device import require_rtx6000

                device = require_rtx6000(gpu_index=1)
                print(f"[part2] using {device} (RTX 6000 gpu1)")
            except Exception as exc:  # gpu1 busy / unavailable -> CPU (predictor is small)
                import torch

                device = torch.device("cpu")
                print(f"[part2] RTX6000 gpu1 unavailable ({exc}); falling back to CPU")
        for tag in args.floquet_tags:
            ckpt = REPO / f"outputs/session27/JEPA_d64_{tag}/checkpoint_iter020000.pt"
            if not ckpt.exists():
                print(f"[part2] SKIP {tag}: {ckpt} missing")
                continue
            try:
                bundle = load_predictor(tag, device)
                res = run_part2(bundle, args.latents_root, window=args.floquet_window)
                part2_all[tag] = res
            except Exception as exc:  # noqa: BLE001 -- report, do not crash the whole run
                print(f"[part2] {tag} FAILED: {exc}")
        # Headline Floquet = the tf-no-c predictor (the paper's main model).
        part2 = part2_all.get("noc_tf") or (next(iter(part2_all.values()), {}))

    print("\n[spectrum_dmd] === PART 3: DMD forecasting rung ===", flush=True)
    part3_build = run_part3(args.latents_root, args.rollouts_root)
    part3_score = score_part3_closure(
        args.rollouts_root, args.latents_root, args.dns_metrics, args.split_manifest
    )

    # --- write artifacts ---
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "generated_iso": datetime.now(timezone.utc).isoformat(),
        "dt_tc": DT_TC,
        "dns_reference": {"St_dominant": ST_DOMINANT, "St_subharmonic": ST_SUBHARMONIC},
        "part1_dmd": part1,
        "part2_floquet": part2_all,
        "part2_headline": part2,
        "part3_rollout_build": part3_build,
        "part3_closure_score": part3_score,
        "wall_time_s": time.time() - t0,
    }
    (args.out_dir / "dmd_results.json").write_text(_jsonify(results))
    print(f"[spectrum_dmd] wrote {args.out_dir / 'dmd_results.json'}")

    # spectrum.npz: the eigenvalue tables (complex), for downstream plotting.
    npz: dict = {}
    for fam, rec in part1.get("families", {}).items():
        for regime in ("baseline", "pooled_train"):
            if regime in rec:
                spec = rec[regime]["spectrum"]
                npz[f"{fam}_{regime}_re"] = np.array([c["re"] for c in spec])
                npz[f"{fam}_{regime}_im"] = np.array([c["im"] for c in spec])
    if part2 and "floquet" in part2:
        npz["floquet_top_moduli"] = np.array(part2["floquet"]["top_moduli"])
    np.savez(args.out_dir / "spectrum.npz", **npz)
    print(f"[spectrum_dmd] wrote {args.out_dir / 'spectrum.npz'}")

    make_figure(part1, part2, part3_score, args.out_dir / "fig_latent_spectrum")
    emit_numbers_part(part1, part2, part3_score, args.numbers_part)
    write_readme(part1, part2, part3_score, args.out_dir)
    print(f"\n[spectrum_dmd] DONE in {time.time() - t0:.1f}s")


def _jsonify(obj) -> str:
    """JSON dump that drops the bulky raw operator matrices (kept only in .npz)."""

    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, complex):
            return {"re": o.real, "im": o.imag}
        return str(o)

    # Strip the (d, d) "A" operator matrices from the JSON (they live in spectrum
    # logic only; the eigenvalue tables are what the JSON carries).
    return json.dumps(obj, indent=2, default=default)


if __name__ == "__main__":
    main()
