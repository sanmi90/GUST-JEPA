"""Track O2 -- wall-visibility spectrum of the jepa_pool latent directions.

Which latent directions of the pooled coefficient state does the wall pressure
see once they are propagated through the frozen dynamics? We perturb the pooled
d=32 state along its own latent PCs (delta scaled to the per-direction std) at
base states sampled across the three gust phases, propagate H in {1,4,8} steps
through the matched AR-transformer on the pooled state (the SAME dynamics the
Track B filter advances, D229), and measure the normalised signal energy at a
linear pressure head h: z_pool -> p_K at the model's OWN OSP K8 taps (what the
filter senses):

    E_j(H) = mean_over_basestates  sum_{t=1..H} || h(z_t) - h(z~_t) ||^2 / delta_j^2

Each direction's loading on the five observable probes (z_pool -> observable
linear weights) is overlaid so a "visible"/"hidden" direction can be read in
physical terms. The pressure head is a self-contained frozen linear map fit on
TRAIN (the same kind of head Track B uses); it is NOT an observable probe.

REVISION (coordinator): the propagation is now the matched AR-transformer on the
pooled d=32 state (fit_matched_transformer), REPLACING the broadcast->ResUNet->GAP
path, and the pressure head reads the per-model OSP K8 taps
(osp[model]["K8"]) instead of the shared qDEIM taps -- so O2 describes the exact
dynamics and sensors of the estimation-tier filter. The prior ResUNet+qDEIM result
is retained at outputs/session32/track_o2_visibility_resunet_baseline.json.

Run (GPU, RTX 6000):
    taskset -c 8-15 python -m scripts.session32.track_o2_visibility
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OBSERVABLES = ("C_L", "C_D", "wake_enstrophy", "circulation_pos", "circulation_neg")


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def load_pooled_cache(cache_dir: Path, model: str, split: str) -> dict:
    d = np.load(cache_dir / f"latents_{model}_{split}.npz", allow_pickle=True)
    return {
        "z_gap": np.asarray(d["z_gap"], dtype=np.float32),
        "case_id": np.asarray(d["case_id"]).astype(str),
        "encounter_index": np.asarray(d["encounter_index"]).astype(int),
        "frame": np.asarray(d["frame"]).astype(int),
        "window_mask": np.asarray(d["window_mask"], dtype=bool),
        "targets": {o: np.asarray(d[f"target_{o}"], dtype=np.float64) for o in OBSERVABLES},
    }


def sample_base_states(
    cache: dict, windows: dict, cl: int, n_per_phase: int, seed: int
) -> list[dict]:
    """Sample anchor contexts (cl pooled frames) across lead_in/impact/relaxation."""
    rng = np.random.default_rng(seed)
    cid = cache["case_id"]
    enc = cache["encounter_index"]
    frame = cache["frame"]
    keys = np.array([f"{c}/{int(e):02d}" for c, e in zip(cid, enc)])
    z = cache["z_gap"]
    # per-encounter frame -> row lookup
    by_key: dict[str, dict[int, int]] = {}
    for i, k in enumerate(keys):
        by_key.setdefault(k, {})[int(frame[i])] = i

    phase_pools: dict[str, list[tuple[str, int]]] = {"lead_in": [], "impact": [], "relaxation": []}
    for k, fr2row in by_key.items():
        w = windows.get(k)
        if w is None:
            continue
        for phase in ("lead_in", "impact", "relaxation"):
            lo, hi = w[phase]
            for f in range(int(lo), int(hi)):
                if (
                    f in fr2row
                    and (f - cl + 1) >= 0
                    and all((f - cl + 1 + t) in fr2row for t in range(cl))
                ):
                    phase_pools[phase].append((k, f))

    bases: list[dict] = []
    for phase, pool in phase_pools.items():
        if not pool:
            continue
        idx = rng.choice(len(pool), size=min(n_per_phase, len(pool)), replace=False)
        for j in idx:
            k, f = pool[j]
            fr2row = by_key[k]
            ctx_rows = [fr2row[f - cl + 1 + t] for t in range(cl)]
            bases.append(
                {"key": k, "anchor_frame": int(f), "phase": phase, "ctx": z[ctx_rows].copy()}
            )  # (cl, d)
    return bases


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Track O2 wall-visibility spectrum (jepa_pool)")
    ap.add_argument("--model", default="jepa_pool")
    ap.add_argument("--runs-base", default="outputs/runs/session31")
    ap.add_argument("--checkpoint-name", default="checkpoint_iter010000.pt")
    ap.add_argument("--cache-dir", default="outputs/session31/q1_latents")
    ap.add_argument(
        "--osp-taps",
        default="outputs/session32/osp_taps_v2p2.json",
        help="Per-model OSP taps; the pressure head uses osp[model]['K8'].",
    )
    ap.add_argument(
        "--context-length",
        type=int,
        default=2,
        help="Seed frames for the AR-transformer rollout (matches base-state sampling).",
    )
    ap.add_argument("--transformer-steps", type=int, default=4000)
    ap.add_argument("--windows", default="outputs/session31/windows_v2p2.json")
    ap.add_argument("--out", default="outputs/session32/track_o2_visibility.json")
    ap.add_argument("--horizons", nargs="+", type=int, default=[1, 4, 8])
    ap.add_argument("--n-per-phase", type=int, default=40)
    ap.add_argument("--delta-scale", type=float, default=1.0, help="Perturbation in std units.")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    import torch

    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression, Ridge

    from src.evaluation.rollout import (
        encounter_trajectories,
        fit_matched_transformer,
        load_latent_split,
    )
    from src.evaluation.rom_eval import load_windows
    from src.utils.device import require_rtx6000

    device = require_rtx6000(gpu_index=args.gpu)
    gpu_name = torch.cuda.get_device_name(device.index)
    if "RTX" not in gpu_name or "6000" not in gpu_name:
        raise RuntimeError(f"hardware policy: gpu_name={gpu_name!r} is not an RTX 6000")
    print(f"[o2] device={device} ({gpu_name})", flush=True)

    cache_dir = _resolve(args.cache_dir)
    windows = load_windows(_resolve(args.windows))
    # Pressure head uses the model's OWN OSP K8 taps (what the Track B filter senses).
    taps = json.loads(_resolve(args.osp_taps).read_text())[args.model]["K8"]

    tr = load_pooled_cache(cache_dir, args.model, "train")
    tb = load_pooled_cache(cache_dir, args.model, "test_b")
    d = tr["z_gap"].shape[1]

    # --- latent PCs of the pooled state (fit on train) ---
    pca = PCA(n_components=d, random_state=args.seed).fit(tr["z_gap"])
    pcs = pca.components_.astype(np.float32)  # (d, d) rows = directions
    pc_std = np.sqrt(pca.explained_variance_).astype(np.float32)  # per-direction std
    evr = pca.explained_variance_ratio_.astype(float)

    # --- frozen pressure head h: z_pool -> p_K (K=8 OSP taps), fit on train ---
    pres_tr = np.load(cache_dir / "pressure_train.npz", allow_pickle=True)
    p_tr = np.asarray(pres_tr["p_wall"], dtype=np.float32)[:, taps]  # (N, 8)
    # align pressure rows to train latent rows
    assert np.array_equal(np.asarray(pres_tr["case_id"]).astype(str), tr["case_id"])
    assert np.array_equal(np.asarray(pres_tr["frame"]).astype(int), tr["frame"])
    head = Ridge(alpha=1.0).fit(tr["z_gap"], p_tr)  # 32 -> 8
    H_w = head.coef_.astype(np.float32)  # (8, 32)

    # --- observable probe weights (z_pool -> observable), for loadings ---
    obs_w = {}
    for obs in OBSERVABLES:
        reg = LinearRegression().fit(tr["z_gap"], tr["targets"][obs])
        obs_w[obs] = reg.coef_.astype(np.float32)  # (32,)

    # --- pooled-state AR-transformer (SAME dynamics the Track B filter advances) ---
    # fit_matched_transformer GAP-pools the (broadcast) spatial train trajectories to
    # the pooled d=32 state and trains an unconditioned AutoregressivePredictor.
    tr_split = load_latent_split(cache_dir / f"latents_{args.model}_train.npz")
    trajs = encounter_trajectories(tr_split)
    predictor = fit_matched_transformer(
        trajs,
        device=device,
        latent_dim=d,
        steps=args.transformer_steps,
        seed=args.seed,
    )
    cl = int(args.context_length)
    bases = sample_base_states(tb, windows, cl, args.n_per_phase, args.seed)
    n_base = len(bases)
    ctx_stack = np.stack([b["ctx"] for b in bases], axis=0)  # (B, cl, d)
    print(
        f"[o2] PCs fit; head fit; AR-transformer fit; {n_base} base states "
        f"(phases: {[ (ph, sum(1 for b in bases if b['phase']==ph)) for ph in ('lead_in','impact','relaxation')]})",
        flush=True,
    )

    def roll_pooled(ctx_pooled: torch.Tensor, horizon: int) -> torch.Tensor:
        """ctx_pooled (B, cl, d) -> rolled pooled traj (B, horizon, d) via the AR-transformer."""
        z_full = predictor.rollout(ctx_pooled, None, horizon)  # (B, cl + horizon, d)
        return z_full[:, ctx_pooled.shape[1] :]  # drop the seed context frames

    max_H = max(args.horizons)
    ctx_t = torch.from_numpy(ctx_stack).to(device)
    Hw_t = torch.from_numpy(H_w).to(device)  # (8, 32)

    with torch.no_grad():
        base_traj = roll_pooled(ctx_t, max_H)  # (B, max_H, d)
        # energy per direction per horizon
        energy = np.zeros((d, len(args.horizons)), dtype=np.float64)
        for j in range(d):
            delta = float(args.delta_scale) * float(pc_std[j])
            if delta <= 0:
                continue
            pert = torch.from_numpy(pcs[j] * delta).to(device)  # (d,)
            ctx_pert = ctx_t.clone()
            ctx_pert[:, -1, :] = ctx_pert[:, -1, :] + pert  # perturb anchor frame
            pert_traj = roll_pooled(ctx_pert, max_H)  # (B, max_H, d)
            diff = pert_traj - base_traj  # (B, max_H, d)
            # pressure-head signal difference: (B, max_H, 8)
            p_diff = torch.einsum("bhd,kd->bhk", diff, Hw_t)
            se = (p_diff**2).sum(dim=2)  # (B, max_H)  per-frame energy
            cum = torch.cumsum(se, dim=1)  # (B, max_H) cumulative over t
            for hi, Hh in enumerate(args.horizons):
                energy[j, hi] = float(cum[:, Hh - 1].mean().item()) / (delta**2)

    # --- observable loadings per direction ---
    loadings = {}
    for obs in OBSERVABLES:
        w = obs_w[obs]
        wn = w / (np.linalg.norm(w) + 1e-12)
        loadings[obs] = [float(np.dot(pcs[j], wn)) for j in range(d)]  # cos-like signed

    # rank by visibility at max horizon
    vis_maxH = energy[:, -1]
    order = np.argsort(vis_maxH)[::-1]
    top_visible = int(order[0])
    top_hidden = int(order[-1])

    # Ranking restricted to physically-meaningful directions. The raw ranking is
    # dominated by near-null PCs (evr ~ 0) that the AR-transformer amplifies and
    # the 1/delta^2 normalisation inflates -- their dynamics are unconstrained by
    # training data (never excited), so their "visibility" is not trustworthy.
    evr_thresh = 0.02
    meaningful = [j for j in range(d) if float(evr[j]) >= evr_thresh]
    mrank = sorted(meaningful, key=lambda j: energy[j, -1], reverse=True)
    corr_evr_energy = float(np.corrcoef(evr, vis_maxH)[0, 1])

    def dir_summary(j: int) -> dict:
        return {
            "direction": int(j),
            "explained_var_ratio": float(evr[j]),
            "pc_std": float(pc_std[j]),
            "energy_by_H": {str(Hh): float(energy[j, hi]) for hi, Hh in enumerate(args.horizons)},
            "observable_loadings": {o: loadings[o][j] for o in OBSERVABLES},
            "dominant_observable": max(OBSERVABLES, key=lambda o: abs(loadings[o][j])),
        }

    payload = {
        "task": "SESSION 32 Track O2 -- wall-visibility spectrum (jepa_pool pooled latent)",
        "params": {
            "model": args.model,
            "horizons": list(args.horizons),
            "n_base_states": n_base,
            "n_per_phase": args.n_per_phase,
            "context_length": cl,
            "delta_scale_std": args.delta_scale,
            "predictor": (
                "matched AR-transformer on the pooled d=32 state "
                "(fit_matched_transformer; the SAME dynamics the Track B filter advances, "
                "D229) -- REPLACES the broadcast->ResUNet->GAP path of the baseline"
            ),
            "transformer_steps": int(args.transformer_steps),
            "K_taps": taps,
            "K_taps_source": f"osp[{args.model}]['K8'] (per-model TCSI OSP, not qDEIM)",
            "pressure_head": (
                "Ridge(alpha=1.0) z_pool(32)->p_K(8 OSP taps), frozen on train; "
                "same tap set the Track B filter senses"
            ),
            "energy_definition": "sum_t ||h(z_t)-h(z~_t)||^2 / delta_j^2 (cumulative over horizon)",
            "observable_loading": "cos-like signed dot(PC_j, unit probe weight)",
            "baseline_file": "outputs/session32/track_o2_visibility_resunet_baseline.json "
            "(ResUNet propagation + qDEIM K8 head)",
            "seed": args.seed,
            "gpu_name": gpu_name,
        },
        "ranking_at_maxH": {
            "horizon": int(max_H),
            "directions_by_visibility_desc": [int(x) for x in order],
            "top_visible": dir_summary(top_visible),
            "top_hidden": dir_summary(top_hidden),
            "corr_evr_vs_energy": corr_evr_energy,
            "note": (
                "raw ranking; negative corr_evr_vs_energy means near-null PCs score "
                "highest because the AR-transformer amplifies them and 1/delta^2 "
                "inflates their tiny natural scale (data-unconstrained). See "
                "ranking_meaningful_directions for the physically-interpretable ranking."
            ),
        },
        "ranking_meaningful_directions": {
            "evr_threshold": evr_thresh,
            "n_meaningful": len(meaningful),
            "cum_evr": float(sum(float(evr[j]) for j in meaningful)),
            "directions_by_visibility_desc": [int(x) for x in mrank],
            "top_visible": dir_summary(int(mrank[0])),
            "top_hidden": dir_summary(int(mrank[-1])),
            "note": (
                "restricted to PCs carrying real variance (evr>=0.02); this is the "
                "robust wall-visibility statement under the filter's AR-transformer dynamics."
            ),
        },
        "per_direction": [dir_summary(j) for j in range(d)],
    }
    out_path = _resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[o2] wrote {out_path}", flush=True)
    print(
        f"[o2] TOP VISIBLE dir {top_visible} (evr={evr[top_visible]:.3f}) "
        f"energy@maxH={vis_maxH[top_visible]:.3e} dom_obs={dir_summary(top_visible)['dominant_observable']}",
        flush=True,
    )
    print(
        f"[o2] TOP HIDDEN  dir {top_hidden} (evr={evr[top_hidden]:.3f}) "
        f"energy@maxH={vis_maxH[top_hidden]:.3e} dom_obs={dir_summary(top_hidden)['dominant_observable']}",
        flush=True,
    )
    print(
        f"[o2] corr(evr, energy@maxH)={corr_evr_energy:+.3f} "
        f"(negative => near-null PCs inflated; see meaningful ranking)",
        flush=True,
    )
    mv, mh = int(mrank[0]), int(mrank[-1])
    print(
        f"[o2] MEANINGFUL (evr>={evr_thresh}, {len(meaningful)} dirs, cum_evr="
        f"{sum(float(evr[j]) for j in meaningful):.2f}): "
        f"MOST-visible dir {mv} (evr={evr[mv]:.3f}) dom={dir_summary(mv)['dominant_observable']}; "
        f"LEAST-visible dir {mh} (evr={evr[mh]:.3f}) dom={dir_summary(mh)['dominant_observable']}",
        flush=True,
    )
    # print top-5 visible with dominant observable
    print("[o2] top-5 visible directions (dir, evr, energy@maxH, dom_obs):", flush=True)
    for j in order[:5]:
        s = dir_summary(int(j))
        print(
            f"    dir {j:2d}  evr={evr[j]:.3f}  E={energy[j,-1]:.3e}  dom={s['dominant_observable']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
