"""Track I: measured interventional test of the conditional predictor.

Earns (or softens) the world-model framing by MEASURING it. For each base point
(G, D, Y) that has a +dG partner at identical (D, Y), we form:

  PREDICTED response: take each base encounter's impact-frame latent, roll the
  conditional predictor (markov, the closure protocol) to H=16 under its true
  c = (G, D, Y) and again under c' = (G + dG, D, Y), map both to the six closure
  observables with the train-fit ridge probe, and difference. Same starting
  latent both times, so phase is controlled by construction.

  MEASURED response: the DNS group-mean of each observable at impact+H over the
  partner encounters at (G + dG, D, Y) minus the group-mean over the base
  encounters at (G, D, Y).

We then correlate predicted vs measured response across base points, per
observable. High positive correlation -> the predictor's response to a parameter
intervention tracks the simulation, and the world-model language is earned.

Reuses the verified rollout helpers from scripts/session18/eval_baseline_rollouts.
Method reference: Wang, Kou, Noack, Zhang (JFM 1035 A18, 2026), Granger-causal
intervention template.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/carlos/GUST-JEPA")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "session18"))

from src.utils.device import require_rtx6000  # noqa: E402
from eval_baseline_rollouts import load_predictor, rollout_markov  # noqa: E402

OBS = ("C_L", "C_D", "I_y", "wake_enstrophy", "circulation_pos", "circulation_neg")
LAT = REPO / "outputs/session18/exp_b1/latents_jepa_d64_test1_noBN"
CKPT = REPO / "outputs/session18/exp_b1/predictor_jepa_d64/checkpoint_iter006000.pt"
DNS = REPO / "outputs/session17/exp2/dns_physical_metrics.npz"
OUT = REPO / "outputs/session23/intervention"
H = 16
delta = 0.5


def fit_ridge(Z, y, alpha=1.0):
    Z = Z.astype(np.float64)
    y = y.astype(np.float64)
    mu = Z.mean(0)
    sd = Z.std(0).clip(min=1e-9)
    Zn = (Z - mu) / sd
    A = Zn.T @ Zn + alpha * np.eye(Zn.shape[1])
    W = np.linalg.solve(A, Zn.T @ (y - y.mean()))
    return {"W": W, "mu": mu, "sd": sd, "b": float(y.mean())}


def apply_probe(z, p):
    return (z.astype(np.float64) - p["mu"]) / p["sd"] @ p["W"] + p["b"]


def key(cid, ei):
    return {(str(c), int(e)): i for i, (c, e) in enumerate(zip(cid, ei))}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=float, default=0.5)
    delta = ap.parse_args().delta
    device = require_rtx6000(gpu_index=0)
    OUT.mkdir(parents=True, exist_ok=True)
    rng_round = lambda a: np.round(a.astype(float), 3)  # noqa: E731

    # ---- latents (train: fit probe + base points) ----
    tr = np.load(LAT / "train.npz", allow_pickle=True)
    z_full = tr["z_full"].astype(np.float32)          # (n,120,d)
    G, D, Y = tr["G"].astype(float), tr["D"].astype(float), tr["Y"].astype(float)
    cid = np.array([str(c) for c in tr["case_id"]])
    ei = tr["encounter_index"].astype(int)
    imp = tr["impact_frame"].astype(int)
    n, T, d = z_full.shape

    # ---- DNS metrics, aligned to the latent encounter order ----
    dns = np.load(DNS, allow_pickle=True)
    dkey = key([str(c) for c in dns["train_case_id"]], dns["train_encounter_index"].astype(int))
    di = np.array([dkey.get((c, int(e)), -1) for c, e in zip(cid, ei)])
    assert (di >= 0).all(), f"{(di < 0).sum()} latent encounters unmatched in DNS"
    dns_obs = {o: dns[f"train_{o}"][di].astype(np.float64) for o in OBS}  # each (n,120)
    dns_imp = dns["train_impact_frame"][di].astype(int)

    # ---- fit ridge probe per observable on ALL train frames (raw latent) ----
    Zf = z_full.reshape(n * T, d)
    probes = {o: fit_ridge(Zf, dns_obs[o].reshape(n * T)) for o in OBS}

    # ---- predictor ----
    pred, mean_t, std_t = load_predictor(CKPT, d, device)

    def roll_to_H(z_imp_raw, cond_vec):
        z0 = (torch.from_numpy(z_imp_raw[None, None]).to(device) - mean_t) / std_t
        c = torch.tensor(cond_vec, dtype=torch.float32, device=device)[None]
        zr = rollout_markov(pred, z0, c, steps=H, device=device)  # (1,H+1,d) normed
        return (zr[0, H] * std_t + mean_t).cpu().numpy()          # raw latent at +H

    # ---- enumerate base points (G,D,Y) with a +delta partner at same (D,Y) ----
    gdy = list(zip(rng_round(G), rng_round(D), rng_round(Y)))
    uniq = sorted(set(gdy))
    have = set(uniq)
    pairs = [(g, dd, y) for (g, dd, y) in uniq if (round(g + delta, 3), dd, y) in have]
    print(f"d={d} n_train_enc={n}  base points with +{delta} G partner: {len(pairs)}")

    rows = {o: [] for o in OBS}  # (predicted, measured) per base point
    for (g, dd, y) in pairs:
        base = np.where(np.isclose(G, g) & np.isclose(D, dd) & np.isclose(Y, y))[0]
        part = np.where(np.isclose(G, g + delta) & np.isclose(D, dd) & np.isclose(Y, y))[0]
        # predicted: per base encounter, counterfactual c -> c'
        pred_d = {o: [] for o in OBS}
        for e in base:
            zc = roll_to_H(z_full[e, imp[e]], [g, dd, y])
            zc2 = roll_to_H(z_full[e, imp[e]], [g + delta, dd, y])
            for o in OBS:
                pred_d[o].append(apply_probe(zc2, probes[o]) - apply_probe(zc, probes[o]))
        # measured: DNS group-mean at impact+H, partner minus base
        for o in OBS:
            mb = np.mean([dns_obs[o][e, dns_imp[e] + H] for e in base])
            mp = np.mean([dns_obs[o][e, dns_imp[e] + H] for e in part])
            rows[o].append((float(np.mean(pred_d[o])), float(mp - mb)))

    # ---- correlations ----
    from scipy.stats import pearsonr, spearmanr
    print(f"\n{'observable':16} {'n':>3} {'pearson':>8} {'spearman':>9} {'sign-agree':>10}")
    summary = {}
    all_pred_z, all_meas_z = [], []
    for o in OBS:
        arr = np.array(rows[o])
        p, m = arr[:, 0], arr[:, 1]
        pr = pearsonr(p, m)[0] if len(p) > 2 and p.std() > 0 and m.std() > 0 else float("nan")
        sr = spearmanr(p, m)[0] if len(p) > 2 else float("nan")
        sign = float(np.mean(np.sign(p) == np.sign(m)))
        summary[o] = {"n": len(p), "pearson": pr, "spearman": sr, "sign_agree": sign}
        print(f"{o:16} {len(p):>3} {pr:>8.3f} {sr:>9.3f} {sign:>10.2f}")
        if p.std() > 0 and m.std() > 0:
            all_pred_z.append((p - p.mean()) / p.std())
            all_meas_z.append((m - m.mean()) / m.std())
    pooled = pearsonr(np.concatenate(all_pred_z), np.concatenate(all_meas_z))[0]
    print(f"\npooled (per-observable z-scored) pearson = {pooled:.3f}")

    np.savez(OUT / "intervention_response.npz",
             rows={o: np.array(rows[o]) for o in OBS}, summary=summary,
             pooled=pooled, delta=delta, H=H, n_pairs=len(pairs))
    print(f"wrote {OUT / 'intervention_response.npz'}")
    return summary, pooled, rows


if __name__ == "__main__":
    main()
