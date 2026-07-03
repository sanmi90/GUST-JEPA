"""Session 32 Track P gate P3: near-null departure mechanism on the pooled tier.

Pre-registered claim (plan ~/.claude/plans/session-32-v2-binary-meteor.md, P3;
HANDOFF D231/D233): "regAE_pool rollout departs along near-null directions of the
encoded covariance while the anti-collapse latents stay in distribution." This is
the v2.1 geometric mechanism, REPLICATED on the pooled v2.2 matrix.

Cast (each on its OWN frozen pooled d=32 latent):
- ``regAE_pool``           -- anti-collapse ON, no heads (the mechanism CANDIDATE);
- ``jepa_wake_pool``       -- = Session 31 ``jepa_pool`` (D226), anti-collapse + wake head;
- ``supervised_only_pool`` -- anti-collapse OFF, lift+wake heads (supervised comparator).

Method:
1. Frozen pooled latents come from the Session 31/32 Q1 caches (z_gap is the pooled
   d=32 state -- z_spatial is a uniform broadcast for pooled models, verified).
2. Forecast operator (D229): the matched AR-transformer on the pooled state
   (:func:`src.evaluation.rollout.fit_matched_transformer`, the ``transformer_matched``
   estimation-tier operator), fit fresh on each model's frozen TRAIN pooled latents.
3. Roll each Test B encounter open-loop from the PRE-IMPACT context (2-frame seed
   ending at ``t_impact - 1``) forward through impact + relaxation onset; form the
   DEPARTURE ``d_t = rolled z_t - encoded-true z_t``.
4. Estimate the TRAIN pooled-latent covariance under TWO estimators: (a) empirical +
   1e-6 jitter (matches ``metrics.py`` convention), (b) Ledoit-Wolf shrinkage (sklearn).
   Eigendecompose; the smallest-eigenvalue eigenvectors are the NEAR-NULL directions.
5. Measure, per model, under BOTH estimators: (1) the fraction of departure ENERGY on
   the bottom-quartile near-null PCs (the pre-registered ROBUST direction measure);
   (2) the departure Mahalanobis ratio (secondary; magnitudes collapse under shrinkage,
   per plan -- reported with the caveat, NOT used for the verdict).

Verdict: P3 PASS if regAE_pool's departure concentrates on the near-null directions
markedly more than BOTH anti-collapse+supervised comparators, under BOTH covariance
estimators for the DIRECTION measure. Reported as measured (honesty rule).

No training, no Test C, no edits to src/estimation/ or Track B/O files (metrics.py is
READ/reused only).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

S31 = REPO / "outputs" / "session31"
S32 = REPO / "outputs" / "session32"

CANDIDATE = "regAE_pool"
COMPARATORS = ("jepa_wake_pool", "supervised_only_pool")
MODELS = (CANDIDATE,) + COMPARATORS

CONTEXT_LENGTH = 2
# fit_matched_transformer builds the AR predictor with max_seq_len=32; the open-loop
# rollout's longest forward pass is (context_length + horizon - 1), so horizon is
# capped at 30 to cover the full impact window + relaxation onset within capacity.
DEFAULT_HORIZON = 30
DEFAULT_K = 8  # bottom quartile of d=32
MARGIN = 0.05  # "markedly more" point margin (project tie tolerance)


# --------------------------------------------------------------------------- loading
def latent_path(model: str, split: str) -> Path:
    """Frozen pooled-latent npz path. D250: the predictive flagship jepa_wake_pool
    aliases the native-vector jepa_pool_vec cache (encoder geometry re-derived on the
    vec pipeline); the reconstructive and objective-free comparators are unchanged."""
    if model == "jepa_wake_pool":
        vec_dir = REPO / "outputs" / "session33" / "q1_vec_latents"
        return vec_dir / f"latents_jepa_pool_vec_{split}.npz"
    return S32 / "q1_pool_latents" / f"latents_{model}_{split}.npz"


def load_pooled(model: str, split: str) -> dict:
    """Load the pooled d=32 state (z_gap) + alignment for one model/split."""
    from src.evaluation.pressure_infer import load_gap_split

    g = load_gap_split(latent_path(model, split))
    return {
        "z_gap": np.asarray(g["z_gap"], dtype=np.float32),
        "case_id": np.asarray(g["case_id"]).astype(str),
        "encounter_index": np.asarray(g["encounter_index"]),
        "frame": np.asarray(g["frame"]),
    }


def group_pooled(gap: dict) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Group flat pooled rows into per-encounter (key, z(L,d), frames(L,)) trajectories."""
    cid, enc, fr, z = gap["case_id"], gap["encounter_index"], gap["frame"], gap["z_gap"]
    keys = np.array([f"{c}/{int(e):02d}" for c, e in zip(cid, enc)])
    out: list[tuple[str, np.ndarray, np.ndarray]] = []
    for k in sorted(np.unique(keys)):
        rows = np.where(keys == k)[0]
        rows = rows[np.argsort(fr[rows])]
        out.append((k, z[rows].astype(np.float32), fr[rows]))
    return out


# --------------------------------------------------------------------------- operator
def fit_operator(train_encs, device, *, seed: int = 0, steps: int = 4000):
    """Fit the D229 matched AR-transformer on frozen TRAIN pooled trajectories.

    The pooled state is fed as a degenerate (L, d, 1, 1) "spatial" trajectory so that
    :func:`fit_matched_transformer`'s internal GAP-pool returns the pooled vector
    unchanged (exact for pooled models: z_spatial is a uniform broadcast of z_gap).
    """
    from src.evaluation.rollout import fit_matched_transformer

    d = int(train_encs[0][1].shape[1])
    trajs = [z[:, :, None, None] for (_, z, _) in train_encs]  # (L, d, 1, 1)
    return fit_matched_transformer(
        trajs, device=device, latent_dim=d, steps=steps, seed=seed, verbose=True
    )


def roll_departures(predictor, test_encs, windows, *, context_length: int, horizon: int, device):
    """Open-loop rollouts from the PRE-IMPACT context; return departures + alignment.

    One rollout per Test B encounter: seed = the ``context_length`` frames ending at
    ``t_impact - 1`` (pre-impact), roll ``horizon`` steps (clamped to the encounter),
    departure ``d_h = rolled_h - z_true[a+h]``. Rows are emitted in a deterministic
    (encounter, horizon) order so the departure matrix is ROW-ALIGNED across models
    (the geometry -- t_impact, anchor, horizon set -- is model-independent).
    """
    import torch

    deps: list[np.ndarray] = []
    labels: list[str] = []
    hs: list[int] = []
    per_enc: dict[str, int] = {}
    # the AR predictor's longest forward is (context_length + h - 1); stay within capacity
    max_seq = int(getattr(predictor, "max_seq_len", context_length + horizon))
    horizon_cap = min(horizon, max_seq - context_length + 1)
    for key, z, fr in test_encs:
        w = windows[key]
        ti = int(w["t_impact"])
        pos = np.where(fr == ti - 1)[0]
        if pos.size == 0:
            continue
        a = int(pos[0])
        if a - (context_length - 1) < 0:
            continue
        h_eff = min(horizon_cap, z.shape[0] - 1 - a)
        if h_eff < 1:
            continue
        lo, hi = a - context_length + 1, a + 1
        z_init = torch.from_numpy(z[lo:hi][None].astype(np.float32)).to(device)
        with torch.no_grad():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    z_full = predictor.rollout(z_init, None, h_eff)
            else:
                z_full = predictor.rollout(z_init, None, h_eff)
        rolled = z_full[0, context_length:].float().cpu().numpy()  # (h_eff, d)
        for h in range(1, h_eff + 1):
            deps.append((rolled[h - 1] - z[a + h]).astype(np.float64))
            labels.append(key)
            hs.append(h)
        per_enc[key] = h_eff
    return (
        np.asarray(deps, dtype=np.float64),
        np.asarray(labels),
        np.asarray(hs, dtype=int),
        per_enc,
    )


# --------------------------------------------------------------------------- geometry
def covariances(z_train: np.ndarray, jitter: float = 1e-6) -> dict:
    """TRAIN pooled-latent covariance under the two P3 estimators."""
    from sklearn.covariance import LedoitWolf

    d = z_train.shape[1]
    c_emp = np.cov(z_train.T) + jitter * np.eye(d)  # empirical (ddof=1) + tiny jitter
    lw = LedoitWolf().fit(z_train)  # centers internally
    return {
        "empirical": c_emp,
        "ledoit_wolf": np.asarray(lw.covariance_, dtype=np.float64),
        "lw_shrinkage": float(lw.shrinkage_),
    }


def eig_ascending(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecompose a symmetric covariance; eigenvalues ascending (near-null first)."""
    evals, evecs = np.linalg.eigh(cov)
    return evals, evecs


def near_null_energy_fraction(dep: np.ndarray, evecs: np.ndarray, k: int) -> float:
    """Energy-weighted fraction of departure energy on the bottom-k near-null PCs.

    ``sum_i ||V_null^T d_i||^2 / sum_i ||d_i||^2`` with ``V_null`` the k smallest-
    eigenvalue eigenvectors (orthonormal). Isotropic-null expectation = k / d.
    """
    v_null = evecs[:, :k]
    proj = dep @ v_null
    return float((proj**2).sum() / (dep**2).sum())


def per_departure_near_null_fraction(dep: np.ndarray, evecs: np.ndarray, k: int) -> np.ndarray:
    """Per-departure near-null energy fraction (equal-weight companion to the aggregate)."""
    v_null = evecs[:, :k]
    proj = dep @ v_null
    num = (proj**2).sum(axis=1)
    den = (dep**2).sum(axis=1)
    return num / np.clip(den, 1e-30, None)


def departure_mahalanobis(dep: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Per-departure Mahalanobis ratio sqrt(d^T C^-1 d / dim).

    Same normalized-by-dimension convention as
    :func:`src.estimation.metrics.analysis_mahalanobis_ratio` (validated below on a
    subsample). Secondary diagnostic: magnitudes collapse under shrinkage.
    """
    dim = dep.shape[1]
    sol = np.linalg.solve(cov, dep.T).T
    m2 = (dep * sol).sum(axis=1)
    return np.sqrt(np.clip(m2, 0.0, None) / dim)


def crosscheck_metrics_primitive(dep: np.ndarray, z_train: np.ndarray, n: int = 16) -> float:
    """Genuine reuse check: match departure_mahalanobis (empirical) to metrics.py.

    Constructs, per sampled departure, an ensemble = the full TRAIN cloud and a truth
    state = mu - d, so :func:`analysis_mahalanobis_ratio` returns sqrt(d^T C^-1 d /dim)
    with C the empirical+1e-6-jitter covariance -- identical to our empirical branch.
    Returns the max abs difference (expected ~1e-9).
    """
    from src.estimation.metrics import analysis_mahalanobis_ratio

    dim = z_train.shape[1]
    mu = z_train.mean(axis=0)
    ens = z_train.astype(np.float64)[None]  # (1, N, d)
    idx = np.arange(min(n, dep.shape[0]))
    ref = np.array(
        [analysis_mahalanobis_ratio(ens, (mu - dep[i])[None], jitter=1e-6)[0] for i in idx]
    )
    c_emp = np.cov(z_train.T) + 1e-6 * np.eye(dim)
    mine = departure_mahalanobis(dep[idx], c_emp)
    return float(np.max(np.abs(ref - mine)))


def paired_delta_ci(
    dep_cand: np.ndarray,
    dep_comp: np.ndarray,
    labels: np.ndarray,
    evecs_cand: np.ndarray,
    evecs_comp: np.ndarray,
    k: int,
    *,
    n_boot: int,
    seed: int,
) -> dict:
    """Case-clustered (per-encounter) paired bootstrap CI of the near-null fraction delta.

    Resample encounters with replacement; recompute each model's AGGREGATE near-null
    energy fraction over the resampled departures (row-aligned across models) and take
    the delta (candidate - comparator). Reports the observed delta and its 95% CI.
    """
    uc = np.array(sorted(set(labels.tolist())))
    by_case = {c: np.where(labels == c)[0] for c in uc}
    nc = len(uc)
    rng = np.random.default_rng(seed)
    obs = near_null_energy_fraction(dep_cand, evecs_cand, k) - near_null_energy_fraction(
        dep_comp, evecs_comp, k
    )
    boot = np.empty(n_boot)
    for b in range(n_boot):
        pick = uc[rng.integers(0, nc, size=nc)]
        idx = np.concatenate([by_case[c] for c in pick])
        boot[b] = near_null_energy_fraction(
            dep_cand[idx], evecs_cand, k
        ) - near_null_energy_fraction(dep_comp[idx], evecs_comp, k)
    return {
        "delta_cand_minus_comp": float(obs),
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
    }


# --------------------------------------------------------------------------- runner
def run(
    *,
    horizon: int = DEFAULT_HORIZON,
    k: int = DEFAULT_K,
    predictor_steps: int = 4000,
    n_boot: int = 10000,
    seed: int = 0,
    gpu: int = 0,
    device_override: str | None = None,
    windows_path: str | Path = S31 / "windows_v2p2.json",
    out_json: str | Path = S32 / "track_p3_mechanism.json",
) -> dict:
    import torch

    if device_override is not None:
        device = torch.device(device_override)
        gpu_name = device_override
    else:
        from src.utils.device import require_rtx6000

        device = require_rtx6000(gpu_index=gpu)
        gpu_name = torch.cuda.get_device_name(device.index)
        if "RTX" not in gpu_name or "6000" not in gpu_name:
            raise RuntimeError(f"hardware policy: gpu_name={gpu_name!r} is not an RTX 6000")

    torch.manual_seed(seed)
    np.random.seed(seed)

    windows = json.loads(Path(windows_path).read_text())["windows"]
    k_variants = sorted({4, k, 16})
    estimators = ("empirical", "ledoit_wolf")

    payload: dict = {
        "task": "SESSION 32 Track P gate P3 -- near-null departure mechanism (pooled tier)",
        "pre_registered_claim": (
            "regAE_pool rollout departs along near-null directions of the encoded "
            "covariance while the anti-collapse+supervised latents stay in distribution "
            "(v2.1 geometric mechanism replicated on the pooled v2.2 matrix)."
        ),
        "params": {
            "candidate": CANDIDATE,
            "comparators": list(COMPARATORS),
            "forecast_operator": (
                "matched AR-transformer on the pooled d=32 state (D229; "
                "fit_matched_transformer, teacher-forced next-step, steps="
                f"{predictor_steps}, seed={seed}), fit fresh per model on frozen TRAIN latents"
            ),
            "rollout": (
                f"open-loop from the pre-impact context (2-frame seed ending at t_impact-1), "
                f"horizon={horizon} steps clamped to the encounter; departure d_h = "
                "rolled_h - encoded-true z_{a+h}, on Test B (Test C untouched)"
            ),
            "covariance_estimators": {
                "empirical": "np.cov(TRAIN pooled latents) + 1e-6 * I",
                "ledoit_wolf": "sklearn.covariance.LedoitWolf on TRAIN pooled latents",
            },
            "near_null": (
                f"bottom-k smallest-eigenvalue eigenvectors of the TRAIN covariance; "
                f"primary k={k} (bottom quartile of d=32); isotropic-null fraction = k/d"
            ),
            "robust_measure": (
                "near-null DEPARTURE-ENERGY fraction (direction). The Mahalanobis ratio is "
                "SECONDARY and collapses under shrinkage (plan P3 / D231); not used for verdict."
            ),
            "margin": MARGIN,
            "n_boot": n_boot,
            "gpu_name": gpu_name,
            "context_length": CONTEXT_LENGTH,
            "d": 32,
        },
        "models": {},
    }

    # ---- per-model: fit operator, roll departures, build covariances/eigvecs ----
    dep_by_model: dict[str, np.ndarray] = {}
    labels_ref: np.ndarray | None = None
    evecs_by_model: dict[str, dict[str, np.ndarray]] = {}
    z_train_by_model: dict[str, np.ndarray] = {}
    for model in MODELS:
        print(f"\n[p3] === {model} ===", flush=True)
        tr = load_pooled(model, "train")
        tb = load_pooled(model, "test_b")
        train_encs = group_pooled(tr)
        test_encs = group_pooled(tb)
        z_train = tr["z_gap"].astype(np.float64)
        z_train_by_model[model] = z_train

        predictor = fit_operator(train_encs, device, seed=seed, steps=predictor_steps)
        dep, labels, hs, per_enc = roll_departures(
            predictor,
            test_encs,
            windows,
            context_length=CONTEXT_LENGTH,
            horizon=horizon,
            device=device,
        )
        del predictor
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if labels_ref is None:
            labels_ref = labels
        elif not np.array_equal(labels, labels_ref):
            raise RuntimeError(f"departure rows for {model} not aligned with {MODELS[0]}")
        dep_by_model[model] = dep

        covs = covariances(z_train)
        evecs_by_model[model] = {}
        model_out: dict = {
            "n_departures": int(dep.shape[0]),
            "n_encounters_rolled": int(len(per_enc)),
            "lw_shrinkage": covs["lw_shrinkage"],
            "estimators": {},
        }
        for est in estimators:
            evals, evecs = eig_ascending(covs[est])
            evecs_by_model[model][est] = evecs
            maha = departure_mahalanobis(dep, covs[est])
            frac_by_k = {str(kk): near_null_energy_fraction(dep, evecs, kk) for kk in k_variants}
            # per-departure (equal-weight) companion at primary k
            per_dep_frac = per_departure_near_null_fraction(dep, evecs, k)
            model_out["estimators"][est] = {
                "eigenvalues_ascending": [float(x) for x in evals],
                "eig_condition_number": float(evals[-1] / max(evals[0], 1e-30)),
                "near_null_energy_fraction": {kk: float(v) for kk, v in frac_by_k.items()},
                "near_null_energy_fraction_primary_k": float(frac_by_k[str(k)]),
                "per_departure_frac_primary_k_mean": float(per_dep_frac.mean()),
                "mahalanobis_ratio_mean": float(np.mean(maha)),
                "mahalanobis_ratio_median": float(np.median(maha)),
            }
        # metrics.py reuse cross-check (empirical branch)
        model_out["metrics_primitive_crosscheck_maxabsdiff"] = crosscheck_metrics_primitive(
            dep, z_train
        )
        payload["models"][model] = model_out
        parts = []
        for est in estimators:
            e = model_out["estimators"][est]
            parts.append(
                f"{est}_nnfrac(k={k})={e['near_null_energy_fraction_primary_k']:.3f} "
                f"maha={e['mahalanobis_ratio_mean']:.2f}"
            )
        print(f"[p3] {model}: n_dep={dep.shape[0]} " + " ".join(parts), flush=True)

    # ---- pairwise deltas + paired case-cluster bootstrap (direction measure) ----
    isotropic_null = k / 32.0
    payload["isotropic_null_fraction"] = float(isotropic_null)
    comparisons: dict = {}
    for est in estimators:
        comparisons[est] = {}
        cand_frac = payload["models"][CANDIDATE]["estimators"][est][
            "near_null_energy_fraction_primary_k"
        ]
        comp_fracs = {}
        for comp in COMPARATORS:
            comp_frac = payload["models"][comp]["estimators"][est][
                "near_null_energy_fraction_primary_k"
            ]
            comp_fracs[comp] = comp_frac
            ci = paired_delta_ci(
                dep_by_model[CANDIDATE],
                dep_by_model[comp],
                labels_ref,
                evecs_by_model[CANDIDATE][est],
                evecs_by_model[comp][est],
                k,
                n_boot=n_boot,
                seed=seed,
            )
            comparisons[est][f"{CANDIDATE}_vs_{comp}"] = {
                "cand_frac": float(cand_frac),
                "comp_frac": float(comp_frac),
                **ci,
                "exceeds_margin": bool(cand_frac - comp_frac > MARGIN),
                "ci_excludes_0": bool(ci["ci95"][0] > 0.0),
            }
        worst_comp_frac = max(comp_fracs.values())
        margin_ok = all(
            comparisons[est][f"{CANDIDATE}_vs_{c}"]["exceeds_margin"] for c in COMPARATORS
        )
        ci_ok = all(comparisons[est][f"{CANDIDATE}_vs_{c}"]["ci_excludes_0"] for c in COMPARATORS)
        comparisons[est]["_summary"] = {
            "cand_frac": float(cand_frac),
            "max_comparator_frac": float(worst_comp_frac),
            "cand_minus_max_comparator": float(cand_frac - worst_comp_frac),
            "all_exceed_margin": bool(margin_ok),
            "all_ci_exclude_0": bool(ci_ok),
            "direction_pass": bool(margin_ok and ci_ok),
        }
    payload["comparisons"] = comparisons

    # ---- P3 verdict (direction measure, BOTH estimators) ----
    direction_pass_both = all(comparisons[est]["_summary"]["direction_pass"] for est in estimators)
    payload["verdict"] = {
        "P3": "PASS" if direction_pass_both else "FAIL",
        "basis": (
            "near-null departure-energy fraction (direction); regAE_pool must exceed BOTH "
            f"comparators by > {MARGIN} with a paired case-clustered CI excluding 0, under "
            "BOTH covariance estimators. Mahalanobis ratios reported but NOT decisive "
            "(collapse under shrinkage, per plan P3 / D231)."
        ),
        "direction_pass_empirical": comparisons["empirical"]["_summary"]["direction_pass"],
        "direction_pass_ledoit_wolf": comparisons["ledoit_wolf"]["_summary"]["direction_pass"],
    }

    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[p3] wrote {out_path}", flush=True)
    _print_summary(payload)
    return payload


def _print_summary(payload: dict) -> None:
    print("\n[p3] ===== P3 near-null departure mechanism (pooled tier) =====")
    print(f"[p3] isotropic-null fraction (k/d) = {payload['isotropic_null_fraction']:.3f}")
    for est in ("empirical", "ledoit_wolf"):
        print(f"[p3] -- estimator: {est} --")
        for model in MODELS:
            e = payload["models"][model]["estimators"][est]
            print(
                f"[p3]   {model:<22} near-null-frac(k={DEFAULT_K})="
                f"{e['near_null_energy_fraction_primary_k']:.3f}  "
                f"maha_mean={e['mahalanobis_ratio_mean']:.2f}"
            )
        s = payload["comparisons"][est]["_summary"]
        print(
            f"[p3]   delta(cand - max comparator) = {s['cand_minus_max_comparator']:+.3f}  "
            f"direction_pass={s['direction_pass']}"
        )
    print(f"[p3] VERDICT P3 = {payload['verdict']['P3']}")
    print("[p3] ==========================================================\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Session 32 Track P gate P3 (near-null departure)")
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    p.add_argument("--near-null-k", type=int, default=DEFAULT_K)
    p.add_argument("--predictor-steps", type=int, default=4000)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--windows", default=str(S31 / "windows_v2p2.json"))
    p.add_argument("--out", default=str(S32 / "track_p3_mechanism.json"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(
        horizon=args.horizon,
        k=args.near_null_k,
        predictor_steps=args.predictor_steps,
        n_boot=args.n_boot,
        seed=args.seed,
        gpu=args.gpu,
        device_override=args.device,
        windows_path=args.windows,
        out_json=args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
